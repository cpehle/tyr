/-
  Tyr/GPU/Kernels/RingAttn.lean

  Ring attention surfaces split into the same coarse phases the vendored
  ThunderKittens kernel exposes:

  1. partial attention against the current KV ring buffer,
  2. communication to advance K/V into the next ping-pong buffer,
  3. numerically stable reduction of running O/L statistics.

  The Lean DSL still cannot encode the exact multi-launch schedule or peer index
  arithmetic from `ring_attn_h100.cu`, so the kernels below treat the caller's
  pointers as already-partitioned current/next views.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

/-- Ring phase 1: compute attention for the current K/V shard and emit a partial
output tile plus its per-row log-sum-exp summary. -/
@[gpu_kernel .SM90]
def ringAttnPartial (Q_ptr : GPtr GpuFloat.BFloat16) (K_curr_ptr : GPtr GpuFloat.BFloat16)
    (V_curr_ptr : GPtr GpuFloat.BFloat16) (O_partial_ptr : GPtr GpuFloat.Float32)
    (L_partial_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  comment "Ring partial phase: stationary Q, current-ring K/V"
  rawLines #[
    "using Q_tile = st_bf<64, 128>;",
    "using K_tile = st_bf<128, 128>;",
    "using V_tile = st_bf<128, 128>;",
    "using O_tile = st_fl<64, 128>;",
    "using L_vec = col_vec<st_fl<64, 128>>;",
    s!"auto &Q = {Q_ptr.id.toIdent};",
    s!"auto &K_curr = {K_curr_ptr.id.toIdent};",
    s!"auto &V_curr = {V_curr_ptr.id.toIdent};",
    s!"auto &O_block = {O_partial_ptr.id.toIdent};",
    s!"auto &L_block = {L_partial_ptr.id.toIdent};",
    s!"const int ring_stage = {ring_stage.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    s!"const int world_size = {world_size.id.toIdent};",
    s!"(void){seq_len.id.toIdent};",
    s!"(void){head_dim.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "Q_tile (&Q_smem)[3] = al.allocate<Q_tile, 3>();",
    "K_tile (&K_smem)[2] = al.allocate<K_tile, 2>();",
    "V_tile (&V_smem)[2] = al.allocate<V_tile, 2>();",
    "L_vec (&L_smem)[3] = al.allocate<L_vec, 3>();",
    "O_tile (&O_smem)[3] = al.allocate<O_tile, 3>();",
    "__shared__ semaphore Q_arrived[3], K_arrived[2], V_arrived[2], compute_done[2];",
    "if (threadIdx.x == 0) {",
    "  Q.template prefetch_tma<Q_tile>();",
    "  K_curr.template prefetch_tma<K_tile>();",
    "  V_curr.template prefetch_tma<V_tile>();",
    "  O_block.template prefetch_tma<O_tile>();",
    "  for (int i = 0; i < 3; i++) init_semaphore(Q_arrived[i], 0, 1);",
    "  for (int i = 0; i < 2; i++) { init_semaphore(K_arrived[i], 0, 1); init_semaphore(V_arrived[i], 0, 1); init_semaphore(compute_done[i], 3, 0); }",
    "}",
    "__syncthreads();",
    "const int batch_idx = blockIdx.x / Q.depth();",
    "const int head_idx = blockIdx.x % Q.depth();",
    "const int qo_idx = (blockIdx.y * 3 + warpgroup::groupid()) * 64;",
    "if (warpgroup::groupid() == 3) {",
    "  warpgroup::decrease_registers<32>();",
    "  for (int kv_idx = 0; kv_idx < K_curr.rows() / 128; kv_idx++) {",
    "    wait(compute_done[kv_idx % 2], (kv_idx / 2 + 1) % 2);",
    "    warpgroup::tma::expect_bytes(K_arrived[kv_idx % 2], sizeof(K_tile));",
    "    warpgroup::tma::load_async(K_smem[kv_idx % 2], K_curr, {batch_idx, head_idx, kv_idx, 0}, K_arrived[kv_idx % 2]);",
    "    warpgroup::tma::expect_bytes(V_arrived[kv_idx % 2], sizeof(V_tile));",
    "    warpgroup::tma::load_async(V_smem[kv_idx % 2], V_curr, {batch_idx, head_idx, kv_idx, 0}, V_arrived[kv_idx % 2]);",
    "  }",
    "} else {",
    "  warpgroup::increase_registers<160>();",
    "  rt_fl<16, 128> att_block, o_reg;",
    "  rt_bf<16, 128> att_block_mma;",
    "  col_vec<rt_fl<16, 128>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;",
    "  warpgroup::tma::expect_bytes(Q_arrived[warpgroup::groupid()], sizeof(Q_tile));",
    "  warpgroup::tma::load_async(Q_smem[warpgroup::groupid()], Q, {batch_idx, head_idx, qo_idx / 64, 0}, Q_arrived[warpgroup::groupid()]);",
    "  warp::zero(norm_vec); warp::zero(o_reg); warp::neg_infty(max_vec);",
    "  wait(Q_arrived[warpgroup::groupid()], 0);",
    "  for (int kv_idx = 0; kv_idx < K_curr.rows() / 128; kv_idx++) {",
    "    wait(K_arrived[kv_idx % 2], (kv_idx / 2) % 2);",
    "    warpgroup::mm_ABt(att_block, Q_smem[warpgroup::groupid()], K_smem[kv_idx % 2]);",
    "    warp::copy(max_vec_last_scaled, max_vec);",
    "    warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.08838834764f);",
    "    warpgroup::mma_async_wait();",
    "    warp::row_max(max_vec, att_block, max_vec);",
    "    warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);",
    "    warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);",
    "    warp::sub_row(att_block, att_block, max_vec_scaled);",
    "    warp::exp2(att_block, att_block);",
    "    warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);",
    "    warp::exp2(max_vec_last_scaled, max_vec_last_scaled);",
    "    warp::mul(norm_vec, norm_vec, max_vec_last_scaled);",
    "    warp::row_sum(norm_vec, att_block, norm_vec);",
    "    warp::copy(att_block_mma, att_block);",
    "    warp::mul_row(o_reg, o_reg, max_vec_last_scaled);",
    "    wait(V_arrived[kv_idx % 2], (kv_idx / 2) % 2);",
    "    warpgroup::mma_AB(o_reg, att_block_mma, V_smem[kv_idx % 2]);",
    "    warpgroup::mma_async_wait();",
    "    warpgroup::arrive(compute_done[kv_idx % 2], 1);",
    "  }",
    "  warp::div_row(o_reg, o_reg, norm_vec);",
    "  warpgroup::store(O_smem[warpgroup::groupid()], o_reg);",
    "  warpgroup::store(L_smem[warpgroup::groupid()], norm_vec);",
    "  warpgroup::sync(warpgroup::groupid() + 4);",
    "  if (warpgroup::laneid() == 0) {",
    "    tma::store_async(O_block, O_smem[warpgroup::groupid()], {batch_idx, head_idx, qo_idx / 64, 0});",
    "    tma::store_async(L_block, L_smem[warpgroup::groupid()], {batch_idx, head_idx, qo_idx / 64});",
    "  }",
    "}",
    "(void)ring_stage; (void)dev_idx; (void)world_size;"
  ]

/-- Ring phase 2: move the current K/V shard into the caller-provided "next"
buffers. This is the explicit communication stage the old monolithic kernel hid. -/
@[gpu_kernel .SM90]
def ringAttnComm (K_next_ptr : GPtr GpuFloat.BFloat16) (V_next_ptr : GPtr GpuFloat.BFloat16)
    (K_curr_ptr : GPtr GpuFloat.BFloat16) (V_curr_ptr : GPtr GpuFloat.BFloat16)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  comment "Ring communication phase: current K/V -> next ping-pong buffer"
  rawLines #[
    "using KV_tile = st_bf<128, 128>;",
    s!"auto &K_next = {K_next_ptr.id.toIdent};",
    s!"auto &V_next = {V_next_ptr.id.toIdent};",
    s!"auto &K_curr = {K_curr_ptr.id.toIdent};",
    s!"auto &V_curr = {V_curr_ptr.id.toIdent};",
    s!"const int ring_stage = {ring_stage.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    s!"const int world_size = {world_size.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "static constexpr int NUM_CHUNKS = 7;",
    "KV_tile (&KV_smem)[NUM_CHUNKS] = al.allocate<KV_tile, NUM_CHUNKS>();",
    "__shared__ semaphore inputs_arrived[NUM_CHUNKS], inputs_finished[NUM_CHUNKS];",
    "if (threadIdx.x == 0) { for (int i = 0; i < NUM_CHUNKS; i++) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 1); } }",
    "__syncthreads();",
    "const int warp_id = warp::groupid();",
    "const int lane_id = laneid();",
    "const int dst_dev_idx = (dev_idx + 1) % world_size;",
    "uint32_t phasebits = 0xFFFF0000;",
    "if (warp_id < NUM_CHUNKS && lane_id == 0) {",
    "  for (int task_id = NUM_CHUNKS * blockIdx.x + warp_id; task_id < K_curr.batch() * K_curr.depth() * (K_curr.rows() / 128); task_id += NUM_CHUNKS * gridDim.x) {",
    "    int batch_idx = task_id / (K_curr.depth() * (K_curr.rows() / 128));",
    "    int head_idx = (task_id % (K_curr.depth() * (K_curr.rows() / 128))) / (K_curr.rows() / 128);",
    "    int kv_idx = task_id % (K_curr.rows() / 128);",
    "    wait(inputs_finished[warp_id], get_phasebit<1>(phasebits, 0));",
    "    update_phasebit<1>(phasebits, 0);",
    "    tma::expect_bytes(inputs_arrived[warp_id], sizeof(KV_tile));",
    "    if (blockIdx.x % 2 == 0) tma::load_async(KV_smem[warp_id], K_curr, {batch_idx, head_idx, kv_idx, 0}, inputs_arrived[warp_id]);",
    "    else tma::load_async(KV_smem[warp_id], V_curr, {batch_idx, head_idx, kv_idx, 0}, inputs_arrived[warp_id]);",
    "  }",
    "} else if (NUM_CHUNKS <= warp_id && warp_id < 2 * NUM_CHUNKS && lane_id == 0) {",
    "  int chunk_id = warp_id - NUM_CHUNKS;",
    "  for (int task_id = NUM_CHUNKS * blockIdx.x + chunk_id; task_id < K_curr.batch() * K_curr.depth() * (K_curr.rows() / 128); task_id += NUM_CHUNKS * gridDim.x) {",
    "    int batch_idx = task_id / (K_curr.depth() * (K_curr.rows() / 128));",
    "    int head_idx = (task_id % (K_curr.depth() * (K_curr.rows() / 128))) / (K_curr.rows() / 128);",
    "    int kv_idx = task_id % (K_curr.rows() / 128);",
    "    wait(inputs_arrived[chunk_id], get_phasebit<0>(phasebits, 0));",
    "    update_phasebit<0>(phasebits, 0);",
    "    if (blockIdx.x % 2 == 0) tma::store_async(K_next, KV_smem[chunk_id], {batch_idx, head_idx, kv_idx, 0});",
    "    else tma::store_async(V_next, KV_smem[chunk_id], {batch_idx, head_idx, kv_idx, 0});",
    "    tma::store_async_read_wait();",
    "    arrive(inputs_finished[chunk_id]);",
    "  }",
    "}",
    "(void)ring_stage;"
  ]
  Support.barrierAllDevices "ring K/V epilogue" 1

/-- Ring phase 3: merge the running `(O, L)` state with the current partial
contribution using the standard log-sum-exp reduction. -/
@[gpu_kernel .SM90]
def ringAttnReduce (O_running_ptr : GPtr GpuFloat.Float32) (O_block_ptr : GPtr GpuFloat.Float32)
    (L_running_ptr : GPtr GpuFloat.Float32) (L_block_ptr : GPtr GpuFloat.Float32)
    (O_out_ptr : GPtr GpuFloat.BFloat16) (L_out_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  comment "Ring reduction phase: stable merge of running and current partial tiles"
  rawLines #[
    "using O_tile = st_fl<128, 128>;",
    "using L_vec = col_vec<st_fl<128, 128>>;",
    s!"auto &O_running = {O_running_ptr.id.toIdent};",
    s!"auto &O_block = {O_block_ptr.id.toIdent};",
    s!"auto &L_running = {L_running_ptr.id.toIdent};",
    s!"auto &L_block = {L_block_ptr.id.toIdent};",
    s!"auto &O_out = {O_out_ptr.id.toIdent};",
    s!"auto &L_out = {L_out_ptr.id.toIdent};",
    s!"(void){seq_len.id.toIdent};",
    s!"(void){head_dim.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "O_tile (&O_block_smem)[3] = al.allocate<O_tile, 3>();",
    "O_tile (&O_smem)[3] = al.allocate<O_tile, 3>();",
    "L_vec (&L_block_smem)[3] = al.allocate<L_vec, 3>();",
    "L_vec (&L_smem)[3] = al.allocate<L_vec, 3>();",
    "__shared__ semaphore inputs_arrived[3];",
    "if (threadIdx.x == 0) {",
    "  for (int i = 0; i < 3; i++) {",
    "    init_semaphore(inputs_arrived[i], 0, 1);",
    "    tma::expect_bytes(inputs_arrived[i], (sizeof(L_vec) + sizeof(O_tile)) * 2);",
    "    tma::load_async(L_smem[i], L_running, {blockIdx.z, blockIdx.y, blockIdx.x * 3 + i}, inputs_arrived[i]);",
    "    tma::load_async(O_smem[i], O_running, {blockIdx.z, blockIdx.y, blockIdx.x * 3 + i, 0}, inputs_arrived[i]);",
    "    tma::load_async(L_block_smem[i], L_block, {blockIdx.z, blockIdx.y, blockIdx.x * 3 + i}, inputs_arrived[i]);",
    "    tma::load_async(O_block_smem[i], O_block, {blockIdx.z, blockIdx.y, blockIdx.x * 3 + i, 0}, inputs_arrived[i]);",
    "  }",
    "}",
    "__syncthreads();",
    "if (warpgroup::groupid() < 3) {",
    "  wait(inputs_arrived[warpgroup::groupid()], 0);",
    "  rt_fl<32, 128> O_reg, O_block_reg;",
    "  col_vec<rt_fl<32, 128>> L_reg, L_block_reg, L_new_reg;",
    "  warpgroup::load(L_reg, L_smem[warpgroup::groupid()]);",
    "  warpgroup::load(L_block_reg, L_block_smem[warpgroup::groupid()]);",
    "  warp::sub(L_new_reg, L_block_reg, L_reg);",
    "  warp::exp(L_new_reg, L_new_reg);",
    "  warp::add(L_new_reg, L_new_reg, 1.f);",
    "  warp::log(L_new_reg, L_new_reg);",
    "  warp::add(L_new_reg, L_new_reg, L_reg);",
    "  warpgroup::store(L_smem[warpgroup::groupid()], L_new_reg);",
    "  warp::sub(L_reg, L_reg, L_new_reg);",
    "  warp::exp(L_reg, L_reg);",
    "  warp::sub(L_block_reg, L_block_reg, L_new_reg);",
    "  warp::exp(L_block_reg, L_block_reg);",
    "  warpgroup::load(O_reg, O_smem[warpgroup::groupid()]);",
    "  warp::mul_row(O_reg, O_reg, L_reg);",
    "  warpgroup::load(O_block_reg, O_block_smem[warpgroup::groupid()]);",
    "  warp::mul_row(O_block_reg, O_block_reg, L_block_reg);",
    "  warp::add(O_reg, O_reg, O_block_reg);",
    "  warpgroup::store(O_smem[warpgroup::groupid()], O_reg);",
    "  warpgroup::sync(warpgroup::groupid() + 4);",
    "  if (warpgroup::laneid() == 0) {",
    "    tma::store_async(O_out, O_smem[warpgroup::groupid()], {blockIdx.z, blockIdx.y, blockIdx.x * 3 + warpgroup::groupid(), 0});",
    "    tma::store_async(L_out, L_smem[warpgroup::groupid()], {blockIdx.z, blockIdx.y, blockIdx.x * 3 + warpgroup::groupid()});",
    "  }",
    "}"
  ]

end Tyr.GPU.Kernels.RingAttn
