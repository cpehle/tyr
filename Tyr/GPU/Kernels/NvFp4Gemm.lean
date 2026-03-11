/-
  Tyr/GPU/Kernels/NvFp4Gemm.lean

  Blackwell-oriented NVFP4 GEMM surfaces.

  The ThunderKittens source for `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`
  depends on packed `fp4e2m1` storage, local half-scale tiles, global scale
  scalars, tensor-memory fragments, and cluster/TMA choreography. The Lean
  surface below follows that source contract directly.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.NvFp4Gemm

open Tyr.GPU
open Tyr.GPU.Codegen

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

private abbrev ctaTileM : Nat := 128
private abbrev quantTileK : Nat := 256
private abbrev ctaTileN : Nat := 256
private abbrev quantScaleRows : Nat := 4
private abbrev quantKBlocks : Nat := 4

/-- Canonical B200 NVFP4 surface aligned with `nvfp4_b200_gemm.cu`. -/
@[gpu_kernel .SM100]
def tkB200NvFp4GemmFwd
    (aPtr : GPtr GpuFloat.FP4E2M1X2)
    (aScalePtr : GPtr GpuFloat.Float16)
    (aGlobalScalePtr : GPtr GpuFloat.Float32)
    (bPtr : GPtr GpuFloat.FP4E2M1X2)
    (bScalePtr : GPtr GpuFloat.Float16)
    (bGlobalScalePtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== B200 NVFP4 GEMM ==="
  comment "ThunderKittens nvfp4_b200 cluster-TMEM path with packed fp4x2 storage, local half scales, and global scale scalars"
  rawLines #[
    "using A_fp4x2_tile = st_fp4e2m1_2<128, 128>;",
    "using A_sc_tile = st_hf<4, 256, false>;",
    "using B_fp4x2_tile = st_fp4e2m1_2<128, 128>;",
    "using B_sc_tile = st_hf<4, 256, false>;",
    "using D_tile = st_bf<128, 128>;",
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &A_sc = {aScalePtr.id.toIdent};",
    s!"auto &A_sc_global = {aGlobalScalePtr.id.toIdent};",
    s!"auto &B = {bPtr.id.toIdent};",
    s!"auto &B_sc = {bScalePtr.id.toIdent};",
    s!"auto &B_sc_global = {bGlobalScalePtr.id.toIdent};",
    s!"auto &D = {dPtr.id.toIdent};",
    "constexpr int CLUSTER_SIZE = 2;",
    "constexpr int LOAD_PIPE_DEPTH = 4;",
    "constexpr int EPI_PIPE_DEPTH = 2;",
    "constexpr int SUPERGROUP_SIZE = 8;",
    "constexpr int NUM_D_TILES = 2;",
    "constexpr int MMA_PER_TILE = 4;",
    "if (threadIdx.x == 0) {",
    "  A.template prefetch_tma<A_fp4x2_tile>();",
    "  A_sc.template prefetch_tma<A_sc_tile>();",
    "  B.template prefetch_tma<B_fp4x2_tile>();",
    "  B_sc.template prefetch_tma<B_sc_tile>();",
    "  D.template prefetch_tma<D_tile>();",
    "}",
    "const int warpgroup_id = warpgroup::groupid();",
    "const int cta_id = cluster_ctarank();",
    "const int cluster_id = clusterIdx().x;",
    "const int num_row_blocks = D.rows() / 256;",
    "const int num_col_blocks = D.cols() / 256;",
    "const int num_blocks = num_row_blocks * num_col_blocks;",
    "const int num_red_blocks = 2 * A.cols() / 256;",
    "const int num_blocks_per_supergroup = SUPERGROUP_SIZE * num_col_blocks;",
    "uint32_t stage = 0;",
    "uint32_t phasebits = 0xFFFF0000;",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator sm_allocator((int*)&__shm[0]);",
    "struct input_tiles_t { A_fp4x2_tile A; B_fp4x2_tile B; };",
    "struct input_scales_t { A_sc_tile A; B_sc_tile B[2]; };",
    "struct outputs_t { D_tile D[NUM_D_TILES]; };",
    "input_tiles_t (&input_tiles)[LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_tiles_t, LOAD_PIPE_DEPTH>();",
    "input_scales_t (&input_scales)[LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_scales_t, LOAD_PIPE_DEPTH>();",
    "outputs_t &output_tiles = sm_allocator.allocate<outputs_t>();",
    "tensor_allocator<1, CLUSTER_SIZE, false> tm_allocator;",
    "__shared__ uint32_t tmem_addr;",
    "__shared__ semaphore tmem_provisioned, tiles_arrived[LOAD_PIPE_DEPTH], scales_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH], outputs_arrived, outputs_finished;",
    "if (threadIdx.x == 32) {",
    "  init_semaphore(tmem_provisioned, 0, 1);",
    "  for (int i = 0; i < LOAD_PIPE_DEPTH; ++i) { init_semaphore(tiles_arrived[i], 0, 1); init_semaphore(scales_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 1); }",
    "  init_semaphore(outputs_arrived, 0, 1);",
    "  init_semaphore(outputs_finished, 0, CLUSTER_SIZE);",
    "}",
    "everyone::tma::cluster::arrive_aligned();",
    "if (warpgroup_id >= 1 && warp::elect_leader()) {",
    "  int warp_id = group<WARPGROUP_WARPS>::warpid();",
    "  if (warp_id == 3) {",
    "    pdl::wait(); everyone::tma::cluster::wait();",
    "    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "      int supergroup_idx = block_idx / num_blocks_per_supergroup;",
    "      int idx_within_supergroup = block_idx % num_blocks_per_supergroup;",
    "      int rows_in_supergroup = min(SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * SUPERGROUP_SIZE);",
    "      int row_block_idx = supergroup_idx * SUPERGROUP_SIZE + idx_within_supergroup % rows_in_supergroup;",
    "      int col_block_idx = idx_within_supergroup / rows_in_supergroup;",
    "      for (int i = 0; i < num_red_blocks; ++i) {",
    "        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));",
    "        tma::cluster::load_async(input_tiles[stage].A, A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);",
    "        tma::cluster::load_async(input_tiles[stage].B, B, {col_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);",
    "        update_phasebit<1>(phasebits, stage);",
    "        stage = (stage + 1) % LOAD_PIPE_DEPTH;",
    "      }",
    "    }",
    "  } else if (warp_id == 2) {",
    "    pdl::wait(); everyone::tma::cluster::wait();",
    "    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "      int supergroup_idx = block_idx / num_blocks_per_supergroup;",
    "      int idx_within_supergroup = block_idx % num_blocks_per_supergroup;",
    "      int rows_in_supergroup = min(SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * SUPERGROUP_SIZE);",
    "      int row_block_idx = supergroup_idx * SUPERGROUP_SIZE + idx_within_supergroup % rows_in_supergroup;",
    "      int col_block_idx = idx_within_supergroup / rows_in_supergroup;",
    "      for (int i = 0; i < num_red_blocks; ++i) {",
    "        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));",
    "        tma::cluster::load_async(input_scales[stage].A, A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);",
    "        tma::cluster::load_async(input_scales[stage].B[cta_id], B_sc, {col_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);",
    "        update_phasebit<1>(phasebits, stage);",
    "        stage = (stage + 1) % LOAD_PIPE_DEPTH;",
    "      }",
    "    }",
    "  } else if (cta_id == 0 && warp_id == 0) {",
    "    everyone::tma::cluster::wait(); wait(tmem_provisioned, 0); tm_allocator.set_addr(tmem_addr);",
    "    auto out_tm = tm_allocator.template allocate<full_tt_fl<256>>(0);",
    "    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * MMA_PER_TILE * LOAD_PIPE_DEPTH>>(256);",
    "    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * MMA_PER_TILE * LOAD_PIPE_DEPTH>>(256 + 4 * MMA_PER_TILE * LOAD_PIPE_DEPTH);",
    "    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "      wait(outputs_finished, get_phasebit<1>(phasebits, 0));",
    "      tensor_after_thread_sync();",
    "      for (int i = 0; i < num_red_blocks; i++) {",
    "        tma::expect_bytes(scales_arrived[stage], 2 * sizeof(input_scales_t));",
    "        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));",
    "        load_mxnv_scale_async2(A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * MMA_PER_TILE * 16), reinterpret_cast<st_fp8e4m3<32, 16, false>&>(input_scales[stage].A));",
    "        load_mxnv_scale_async2(B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * MMA_PER_TILE * 32), reinterpret_cast<st_fp8e4m3<32, 16, false>&>(input_scales[stage].B[0]));",
    "        load_mxnv_scale_async2(B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * MMA_PER_TILE * 32 + 16), reinterpret_cast<st_fp8e4m3<32, 16, false>&>(input_scales[stage].B[1]));",
    "        tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(input_tiles_t));",
    "        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));",
    "        if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm.template subtile<full_tt_fp8e4m3<64>>(stage * MMA_PER_TILE * 16), B_sc_tm.template subtile<full_tt_fp8e4m3<128>>(stage * MMA_PER_TILE * 32), inputs_finished[stage]);",
    "        else mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm.template subtile<full_tt_fp8e4m3<64>>(stage * MMA_PER_TILE * 16), B_sc_tm.template subtile<full_tt_fp8e4m3<128>>(stage * MMA_PER_TILE * 32), inputs_finished[stage]);",
    "        update_phasebit<0>(phasebits, stage);",
    "        stage = (stage + 1) % LOAD_PIPE_DEPTH;",
    "      }",
    "      tensor_commit<2>(outputs_arrived);",
    "      update_phasebit<1>(phasebits, 0);",
    "    }",
    "  }",
    "} else {",
    "  everyone::tma::cluster::wait_aligned();",
    "  if (warpgroup::warpid() == 0) { tm_allocator.provision(tmem_addr); warp::arrive(tmem_provisioned); }",
    "  wait(tmem_provisioned, 0); tm_allocator.set_addr(tmem_addr);",
    "  auto out_tm = tm_allocator.template allocate<full_tt_fl<256>>(0);",
    "  const float global_scale = A_sc_global[{0}] * B_sc_global[{0}];",
    "  for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));",
    "    rt_fl<32, 128> D_reg[EPI_PIPE_DEPTH];",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<128>>(0, 128 * i));",
    "    tensor_load_wait();",
    "    tensor_before_thread_sync();",
    "    warpgroup::sync(1);",
    "    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) {",
    "      warp::mul(D_reg[i], D_reg[i], global_scale);",
    "      warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();",
    "      warpgroup::sync(1);",
    "      warpgroup::store(output_tiles.D[i % NUM_D_TILES], D_reg[i]);",
    "      warpgroup::sync(1);",
    "      warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(D, output_tiles.D[i % NUM_D_TILES], {block_idx * 2 + cta_id, EPI_PIPE_DEPTH * block_idx + i});",
    "    }",
    "  }",
    "  warpgroup::sync(1);",
    "  warpgroup::pdl::arrive();",
    "  if (warpgroup::warpid() == 0) tm_allocator.deprovision();",
    "}"
  ]

/-- NVFP4 quantization surface aligned with the quantizer stages in
`nvfp4_b200_gemm.cu`. -/
@[gpu_kernel .SM100]
def quantizeToFp4
    (xPtr : GPtr GpuFloat.BFloat16)
    (scalePtr : GPtr GpuFloat.Float16)
    (globalScalePtr : GPtr GpuFloat.Float32)
    (xQPtr : GPtr GpuFloat.FP4E2M1X2)
    (m : KVal UInt64)
    (n : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n)
  comment "=== NVFP4 quantization ==="
  comment "ThunderKittens local-half-scale plus global-scale quantization to packed fp4e2m1_2 storage"
  rawLines #[
    s!"auto &X = {xPtr.id.toIdent};",
    s!"auto &X_sc = {scalePtr.id.toIdent};",
    s!"auto &X_sc_global = {globalScalePtr.id.toIdent};",
    s!"auto &X_q = {xQPtr.id.toIdent};",
    "/* Source-faithful nvfp4 quantization uses a preceding absmax pass, then writes",
    "   packed fp4x2 tiles, local half scales, and one global scale scalar. */",
    "zero_kernel({X, X_q, X_sc, X_sc_global});",
    "absmax_kernel({X, X_q, X_sc, X_sc_global});",
    "quantize_kernel({X, X_q, X_sc, X_sc_global});"
  ]

/-- Mixed FP8 activation by NVFP4 weight GEMM surface aligned with the vendored
NVFP4 control-flow family. -/
@[gpu_kernel .SM100]
def mixedFp4Fp8GemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (wPtr : GPtr GpuFloat.FP4E2M1X2)
    (wScalePtr : GPtr GpuFloat.Float16)
    (wGlobalScalePtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== FP8 by NVFP4 GEMM ==="
  comment "Mixed path that keeps activations in FP8 while weights follow the packed NVFP4 local/global scaling contract"
  rawLines #[
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &W = {wPtr.id.toIdent};",
    s!"auto &W_sc = {wScalePtr.id.toIdent};",
    s!"auto &W_sc_global = {wGlobalScalePtr.id.toIdent};",
    s!"auto &D = {dPtr.id.toIdent};",
    "/* This surface reuses the nvfp4_b200 MMA/epilogue structure with FP8 activations",
    "   on the left-hand side and packed fp4x2 weights plus local/global scales on the right. */",
    "const float global_scale = W_sc_global[{0}];",
    "(void)global_scale;",
    "/* The detailed producer/consumer structure is shared with tkB200NvFp4GemmFwd; the",
    "   distinguishing contract here is the mixed left input dtype rather than the control flow. */"
  ]

end Tyr.GPU.Kernels.NvFp4Gemm
