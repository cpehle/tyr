/-
  Tyr/GPU/Kernels/Mamba2.lean

  Mamba2 state-space model forward kernels.
  Based on ThunderKittens patterns:
  - Hillis-Steele prefix sum for cumulative decay
  - Exponential state decay computation
  - Attention with decay masking
  - State accumulation across chunks

  This module owns the source-backed Mamba2 forward surface.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Mamba2

open Tyr.GPU
open Tyr.GPU.Codegen

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

/-! ## Mamba2 Forward Kernel

The Mamba2 architecture uses selective state spaces with:
1. Per-position decay factors (A vector)
2. Input-dependent state updates
3. Causal attention with exponential decay

Key computation flow:
1. Compute cumulative sum of decay factors (log-space)
2. Convert to decay matrix: decay[i,j] = exp(cumsum[i] - cumsum[j])
3. Apply causal mask to decay matrix
4. Compute attention with decay: O = softmax(Q @ K^T * decay) @ V
5. Update running state: KV_state = KV_state * total_decay + K^T @ V
-/

/-- Source-faithful Mamba2 forward surface aligned with
`thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu`. -/
@[gpu_kernel .SM90]
def mamba2Fwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (A_ptr : GPtr GpuFloat.Float32)
    (O_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64) : KernelM Unit := do
  let _ := (seq_len, head_dim)
  comment "=== Mamba2 Forward Pass ==="
  comment "ThunderKittens lcsf pipeline: producer load, consumer compute, producer store, consumer finish"
  rawLines #[
    "using q_tile = st_bf<64, 64>;",
    "using k_tile = st_bf<64, 64>;",
    "using v_tile = st_bf<64, 64>;",
    "using o_tile = st_bf<64, 64>;",
    "using a_vec = sv_fl<64>;",
    s!"auto &Q = {Q_ptr.id.toIdent};",
    s!"auto &K = {K_ptr.id.toIdent};",
    s!"auto &V = {V_ptr.id.toIdent};",
    s!"auto &A = {A_ptr.id.toIdent};",
    s!"auto &O = {O_ptr.id.toIdent};",
    "constexpr int NUM_CONSUMER_WARPS = 8;",
    "constexpr int OUTPUT_PIPE_STAGES = 2;",
    "constexpr int INPUT_PIPE_STAGES = 2;",
    "constexpr int NUM_WORKERS = NUM_CONSUMER_WARPS / 4;",
    "struct input_block { q_tile q; k_tile k; v_tile v[2]; a_vec a[2]; a_vec padding[6]; };",
    "struct output_block { o_tile o[2]; };",
    "struct scratch_block { st_bf<64, 64> kv[2], k[2]; a_vec a_cumsum[2]; };",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "input_block (&inputs)[INPUT_PIPE_STAGES] = al.allocate<input_block, INPUT_PIPE_STAGES>();",
    "output_block (&outputs)[OUTPUT_PIPE_STAGES] = al.allocate<output_block, OUTPUT_PIPE_STAGES>();",
    "scratch_block &scratch = al.allocate<scratch_block>();",
    "__shared__ semaphore inputs_arrived[INPUT_PIPE_STAGES], inputs_finished[INPUT_PIPE_STAGES], outputs_arrived[OUTPUT_PIPE_STAGES], outputs_finished[OUTPUT_PIPE_STAGES], finish_finished;",
    "if (threadIdx.x == 0) {",
    "  Q.template prefetch_tma<q_tile>();",
    "  K.template prefetch_tma<k_tile>();",
    "  V.template prefetch_tma<v_tile>();",
    "  A.template prefetch_tma<a_vec>();",
    "  O.template prefetch_tma<o_tile>();",
    "  for (int i = 0; i < INPUT_PIPE_STAGES; i++) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, NUM_CONSUMER_WARPS / 4); }",
    "  for (int i = 0; i < OUTPUT_PIPE_STAGES; i++) { init_semaphore(outputs_arrived[i], 0, 1); init_semaphore(outputs_finished[i], 0, 1); }",
    "  init_semaphore(finish_finished, 0, 1);",
    "}",
    "__syncthreads();",
    "int task_id = blockIdx.x;",
    "int batch = task_id / (V.depth() / (NUM_CONSUMER_WARPS / 4));",
    "task_id -= batch * (V.depth() / (NUM_CONSUMER_WARPS / 4));",
    "int head = task_id * 2;",
    "int num_iters = batch < Q.batch() ? K.rows() / q_tile::rows : -1;",
    "if (warpgroup::groupid() == NUM_WORKERS) {",
    "  warpgroup::producer_registers();",
    "  if (warpgroup::warpid() == 0 || warpgroup::warpid() == 1 || warpgroup::warpid() == 2 || warpgroup::warpid() == 3) {",
    "    for (int iter = 0; iter < num_iters; iter++) {",
    "      if (warpgroup::warpid() == iter % 4) {",
    "        warp::tma::expect(inputs_arrived[iter % INPUT_PIPE_STAGES], inputs[iter % INPUT_PIPE_STAGES].q, inputs[iter % INPUT_PIPE_STAGES].k, inputs[iter % INPUT_PIPE_STAGES].v[0], inputs[iter % INPUT_PIPE_STAGES].a[0], inputs[iter % INPUT_PIPE_STAGES].v[1], inputs[iter % INPUT_PIPE_STAGES].a[1]);",
    "        warp::tma::load_async(inputs[iter % INPUT_PIPE_STAGES].q, Q, {batch, 0, iter, 0}, inputs_arrived[iter % INPUT_PIPE_STAGES]);",
    "        warp::tma::load_async(inputs[iter % INPUT_PIPE_STAGES].k, K, {batch, 0, iter, 0}, inputs_arrived[iter % INPUT_PIPE_STAGES]);",
    "        for (int i = 0; i < NUM_WORKERS; i++) {",
    "          warp::tma::load_async(inputs[iter % INPUT_PIPE_STAGES].v[i], V, {batch, head + i, iter, 0}, inputs_arrived[iter % INPUT_PIPE_STAGES]);",
    "          warp::tma::load_async(inputs[iter % INPUT_PIPE_STAGES].a[i], A, {batch, head + i, 0, iter}, inputs_arrived[iter % INPUT_PIPE_STAGES]);",
    "        }",
    "      }",
    "    }",
    "  }",
    "  if (warpgroup::warpid() == 0) {",
    "    for (int iter = 0; iter < num_iters; iter++) {",
    "      for (int i = 0; i < NUM_WORKERS; i++) warp::tma::store_async(O, outputs[iter % OUTPUT_PIPE_STAGES].o[i], {batch, head + i, iter, 0});",
    "      warp::tma::store_async_read_wait();",
    "      arrive(outputs_finished[iter % OUTPUT_PIPE_STAGES]);",
    "    }",
    "  }",
    "} else if (warpgroup::groupid() < NUM_WORKERS) {",
    "  warpgroup::consumer_registers<NUM_WORKERS>();",
    "  rt_fl<16, 64> o_reg;",
    "  rt_fl<16, 64> att_block;",
    "  rt_bf<16, 64> att_block_mma;",
    "  rt_fl<16, 64> local_decay;",
    "  rt_bf<16, 64> q_reg, k_reg;",
    "  rt_fl<16, 64> kv;",
    "  warp::zero(kv);",
    "  for (int iter = 0; iter < num_iters; iter++) {",
    "    wait(inputs_arrived[iter % INPUT_PIPE_STAGES], 0);",
    "    warpgroup::copy(scratch.a_cumsum[warpgroup::groupid()], inputs[iter % INPUT_PIPE_STAGES].a[warpgroup::groupid()]);",
    "    warpgroup::sync(warpgroup::groupid());",
    "    if (warpgroup::warpid() <= 1) {",
    "      int tid = warpgroup::laneid();",
    "      for (int offset = 1; offset < 64; offset *= 2) {",
    "        float temp = (tid >= offset) ? scratch.a_cumsum[warpgroup::groupid()][tid - offset] : 0.0f;",
    "        group<2>::sync(warpgroup::groupid() + 2);",
    "        scratch.a_cumsum[warpgroup::groupid()][tid] += temp;",
    "        group<2>::sync(warpgroup::groupid() + 2);",
    "      }",
    "    }",
    "    warpgroup::sync(warpgroup::groupid());",
    "    for (int i = 0; i < 4; i++) {",
    "      int base_row = warpgroup::warpid() * 16 + laneid() / 4;",
    "      int base_col = i * 16 + (laneid() % 4) * 2;",
    "      local_decay.tiles[0][i].data[0].x = scratch.a_cumsum[warpgroup::groupid()][base_row + 0] - scratch.a_cumsum[warpgroup::groupid()][base_col + 0];",
    "      local_decay.tiles[0][i].data[0].y = scratch.a_cumsum[warpgroup::groupid()][base_row + 0] - scratch.a_cumsum[warpgroup::groupid()][base_col + 1];",
    "      local_decay.tiles[0][i].data[1].x = scratch.a_cumsum[warpgroup::groupid()][base_row + 8] - scratch.a_cumsum[warpgroup::groupid()][base_col + 0];",
    "      local_decay.tiles[0][i].data[1].y = scratch.a_cumsum[warpgroup::groupid()][base_row + 8] - scratch.a_cumsum[warpgroup::groupid()][base_col + 1];",
    "      local_decay.tiles[0][i].data[2].x = scratch.a_cumsum[warpgroup::groupid()][base_row + 0] - scratch.a_cumsum[warpgroup::groupid()][base_col + 8];",
    "      local_decay.tiles[0][i].data[2].y = scratch.a_cumsum[warpgroup::groupid()][base_row + 0] - scratch.a_cumsum[warpgroup::groupid()][base_col + 9];",
    "      local_decay.tiles[0][i].data[3].x = scratch.a_cumsum[warpgroup::groupid()][base_row + 8] - scratch.a_cumsum[warpgroup::groupid()][base_col + 8];",
    "      local_decay.tiles[0][i].data[3].y = scratch.a_cumsum[warpgroup::groupid()][base_row + 8] - scratch.a_cumsum[warpgroup::groupid()][base_col + 9];",
    "    }",
    "    warp::exp(local_decay, local_decay);",
    "    for (int i = 0; i < 4; i++) {",
    "      auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(local_decay.tiles[0][i]);",
    "      if (i > warpgroup::warpid()) warp::zero(decay_subtile);",
    "      else if (i == warpgroup::warpid()) warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero());",
    "    }",
    "    warpgroup::load(q_reg, inputs[iter % INPUT_PIPE_STAGES].q);",
    "    warpgroup::mm_ABt(att_block, q_reg, inputs[iter % INPUT_PIPE_STAGES].k);",
    "    warpgroup::mma_async_wait();",
    "    warp::mul(att_block, att_block, local_decay);",
    "    warp::copy(att_block_mma, att_block);",
    "    warpgroup::mm_AB(o_reg, att_block_mma, inputs[iter % INPUT_PIPE_STAGES].v[warpgroup::groupid()]);",
    "    warpgroup::mma_async_wait();",
    "    warpgroup::store(scratch.kv[warpgroup::groupid()], kv);",
    "    warpgroup::sync(warpgroup::groupid());",
    "    warpgroup::mma_AB(o_reg, q_reg, scratch.kv[warpgroup::groupid()]);",
    "    warpgroup::mma_async_wait();",
    "    warpgroup::store(outputs[iter % OUTPUT_PIPE_STAGES].o[warpgroup::groupid()], o_reg);",
    "    warpgroup::sync(warpgroup::groupid());",
    "    float last_decay = scratch.a_cumsum[warpgroup::groupid()][scratch.a_cumsum[warpgroup::groupid()].length - 1];",
    "    float total_decay = expf(last_decay);",
    "    warp::mul(kv, kv, total_decay);",
    "    warpgroup::load(k_reg, inputs[iter % INPUT_PIPE_STAGES].k);",
    "    warpgroup::store(scratch.k[warpgroup::groupid()], k_reg);",
    "    warpgroup::sync(warpgroup::groupid());",
    "    warpgroup::mma_AtB(kv, scratch.k[warpgroup::groupid()], inputs[iter % INPUT_PIPE_STAGES].v[warpgroup::groupid()]);",
    "    warpgroup::mma_async_wait();",
    "    if (warpgroup::laneid() == 0) { arrive(outputs_arrived[iter % OUTPUT_PIPE_STAGES]); arrive(inputs_finished[iter % INPUT_PIPE_STAGES]); }",
    "  }",
    "  if (warpgroup::laneid() == 0) arrive(finish_finished);",
    "}"
  ]

end Tyr.GPU.Kernels.Mamba2
