import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Kernels.Prelude

/-!
# Tyr.GPU.Kernels.MhaH100LCF

ThunderKittens-style load-compute-finish attention kernels based on
`thirdparty/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu`.

The vendored CUDA kernel uses the ThunderKittens `lcf` pipeline template with:

- one stationary 64xD query tile per worker,
- larger streamed KV tiles (192x64 or 128x128),
- online softmax accumulation across the KV stream,
- a multi-worker CTA packing that the current Lean DSL does not model directly.

The Lean surfaces below keep the same tile geometry and benchmark-sized KV
stream lengths from the source (`3072` sequence length, so 16 or 24 KV tiles),
but represent the worker packing as a single logical query tile per kernel
instance. That preserves the source-facing contract without pretending the DSL
already has the full `lcf` template/runtime launch arithmetic.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

private def mhaH100LcfFwd
    {headDim kvTileRows numKvTiles : Nat}
    (banner : String)
    (scale : Float)
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  comment banner
  comment "ThunderKittens lcf pipeline: stationary Q scratch, producer-only KV loads, per-worker online softmax, finish-stage TMA store"
  rawLines #[
    s!"constexpr int D = {headDim};",
    s!"constexpr int NUM_WORKERS = 3;",
    s!"constexpr int KV_TILE_ROWS = {kvTileRows};",
    s!"constexpr int NUM_KV_TILES = {numKvTiles};",
    s!"constexpr float TEMPERATURE_SCALE = {scale}f * 1.44269504089f;",
    "using qo_tile = st_bf<64, D>;",
    "using kv_tile = st_bf<KV_TILE_ROWS, D>;",
    s!"auto &O = {o_ptr.id.toIdent};",
    s!"auto &Q = {q_ptr.id.toIdent};",
    s!"auto &K = {k_ptr.id.toIdent};",
    s!"auto &V = {v_ptr.id.toIdent};",
    "struct input_block { kv_tile k, v; };",
    "struct scratch_block { qo_tile q[NUM_WORKERS]; };",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "input_block (&inputs)[2] = al.allocate<input_block, 2>();",
    "scratch_block &scratch = al.allocate<scratch_block>();",
    "__shared__ semaphore inputs_arrived[2], inputs_finished[2], finish_finished;",
    "if (threadIdx.x == 0) {",
    "  Q.template prefetch_tma<qo_tile>();",
    "  K.template prefetch_tma<kv_tile>();",
    "  V.template prefetch_tma<kv_tile>();",
    "  O.template prefetch_tma<qo_tile>();",
    "  for (int i = 0; i < 2; i++) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, NUM_WORKERS); }",
    "  init_semaphore(finish_finished, 0, 1);",
    "}",
    "__syncthreads();",
    "int task_id = gridDim.x * 0 + blockIdx.x;",
    "int seq_q = (Q.rows() + NUM_WORKERS * qo_tile::rows - 1) / (NUM_WORKERS * qo_tile::rows);",
    "int batch = task_id / (seq_q * K.depth()); task_id -= batch * seq_q * K.depth();",
    "int head = task_id / seq_q; task_id -= head * seq_q;",
    "int seq = task_id;",
    "int num_iters = batch < Q.batch() ? (K.rows() + kv_tile::rows - 1) / kv_tile::rows : -1;",
    "if (warpgroup::groupid() == NUM_WORKERS) {",
    "  warpgroup::producer_registers();",
    "  if (warpgroup::warpid() == 0) {",
    "    for (int iter = 0; iter < num_iters; iter++) {",
    "      warp::tma::expect(inputs_arrived[iter % 2], inputs[iter % 2]);",
    "      warp::tma::load_async(inputs[iter % 2].k, K, {batch, head, iter, 0}, inputs_arrived[iter % 2]);",
    "      warp::tma::load_async(inputs[iter % 2].v, V, {batch, head, iter, 0}, inputs_arrived[iter % 2]);",
    "    }",
    "  } else if (laneid() == 0) {",
    "    for (int iter = 0; iter < num_iters; iter++) arrive(inputs_arrived[iter % 2]);",
    "  }",
    "} else if (warpgroup::groupid() < NUM_WORKERS) {",
    "  warpgroup::consumer_registers<NUM_WORKERS>();",
    "  if ((seq * NUM_WORKERS + warpgroup::groupid()) * qo_tile::rows < Q.rows())",
    "    warpgroup::load(scratch.q[warpgroup::groupid()], Q, {batch, head, seq * NUM_WORKERS + warpgroup::groupid(), 0});",
    "  rt_fl<16, qo_tile::cols> o_reg = 0.f;",
    "  col_vec<rt_fl<16, kv_tile::rows>> max_vec = base_types::constants<float>::neg_infty();",
    "  col_vec<rt_fl<16, kv_tile::rows>> norm_vec = 0.f;",
    "  col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;",
    "  rt_fl<16, kv_tile::rows> att_block;",
    "  rt_bf<16, kv_tile::rows> att_block_mma;",
    "  warpgroup::sync(warpgroup::groupid());",
    "  for (int iter = 0; iter < num_iters; iter++) {",
    "    wait(inputs_arrived[iter % 2], 0);",
    "    warpgroup::mm<transpose::N, transpose::T>(att_block, scratch.q[warpgroup::groupid()], inputs[iter % 2].k);",
    "    max_vec_last_scaled = max_vec * TEMPERATURE_SCALE;",
    "    warpgroup::mma_async_wait();",
    "    warp::right_fill(att_block, att_block, K.rows() - iter * kv_tile::rows, base_types::constants<float>::neg_infty());",
    "    max_vec = warp::max<axis::COL>(att_block, max_vec);",
    "    max_vec_scaled = max_vec * TEMPERATURE_SCALE;",
    "    att_block = warp::exp2((att_block * TEMPERATURE_SCALE) - max_vec_scaled);",
    "    max_vec_last_scaled = warp::exp2(max_vec_last_scaled - max_vec_scaled);",
    "    norm_vec *= max_vec_last_scaled;",
    "    norm_vec = warp::sum<axis::COL>(att_block, norm_vec);",
    "    o_reg *= max_vec_last_scaled;",
    "    att_block_mma = att_block;",
    "    warpgroup::mma<transpose::N, transpose::N>(o_reg, att_block_mma, inputs[iter % 2].v);",
    "    warpgroup::mma_async_wait();",
    "    if (laneid() == 0) arrive(inputs_finished[iter % 2]);",
    "  }",
    "  if ((seq * NUM_WORKERS + warpgroup::groupid()) * 64 < Q.rows()) {",
    "    o_reg /= norm_vec;",
    "    auto &o_smem = reinterpret_cast<qo_tile&>(scratch.q[warpgroup::groupid()]);",
    "    warpgroup::store(o_smem, o_reg);",
    "    warpgroup::sync(warpgroup::groupid());",
    "    if (warpgroup::warpid() == 0) warp::tma::store_async(O, o_smem, {batch, head, seq * NUM_WORKERS + warpgroup::groupid(), 0});",
    "    warp::tma::store_async_read_wait();",
    "  }",
    "  __syncwarp();",
    "  if (laneid() == 0) arrive(finish_finished);",
    "}"
  ]

/-- ThunderKittens `mha_h100_lcf` benchmark-sized forward surface for `D = 64`.

This matches the source tile geometry:

- `Q`: 64x64
- `K/V`: 192x64
- `3072 / 192 = 16` streamed KV tiles
- scale `1 / sqrt(64) = 0.125`
-/
@[gpu_kernel .SM90]
def tkMhaH100LCFFwd64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64)
    (head_dim : KVal UInt64) : KernelM Unit := do
  mhaH100LcfFwd
    (headDim := 64)
    (kvTileRows := 192)
    (numKvTiles := 16)
    "=== ThunderKittens mha_h100_lcf forward (D=64, KV tile 192) ==="
    0.125
    q_ptr k_ptr v_ptr o_ptr seq_len head_dim

/-- ThunderKittens `mha_h100_lcf` benchmark-sized forward surface for `D = 128`.

This matches the source tile geometry:

- `Q`: 64x128
- `K/V`: 128x128
- `3072 / 128 = 24` streamed KV tiles
- scale `1 / sqrt(128)`
-/
@[gpu_kernel .SM90]
def tkMhaH100LCFFwd128
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64)
    (head_dim : KVal UInt64) : KernelM Unit := do
  mhaH100LcfFwd
    (headDim := 128)
    (kvTileRows := 128)
    (numKvTiles := 24)
    "=== ThunderKittens mha_h100_lcf forward (D=128, KV tile 128) ==="
    0.08838834764
    q_ptr k_ptr v_ptr o_ptr seq_len head_dim

end Tyr.GPU.Kernels
