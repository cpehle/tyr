/-
  Tyr/GPU/Kernels/PrecisionGemm.lean

  FP8-family GEMM kernels.

  This module now serves two roles:

  - `tkH100Fp8E4M3GemmFwd` and `tkH100Fp8ScaledGemmFwd` are the canonical
    ThunderKittens-shaped H100 FP8 surfaces, following
    `kernels/gemm/fp8_h100/*`.
  - `tkB200Fp8E4M3Gemm1CtaFwd`, `tkB200Fp8E4M3Gemm2CtaFwd`, and
    `tkB200MxFp8GemmFwd` are Blackwell/B200 surfaces aligned with the vendored
    `fp8_b200/*` and `mxfp8_b200/*` kernels.
  - The older mixed-precision and fused-epilogue kernels remain as
    compatibility conveniences built on the same H100 tiled mainloop; they are
    not separate ThunderKittens source ports.
-/

import Tyr.GPU.Kernels.GemmCommon

namespace Tyr.GPU.Kernels.PrecisionGemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev fp8TileM : Nat := 64
private abbrev fp8TileK : Nat := 128
private abbrev fp8TileN : Nat := 256
private abbrev fp8KBlocks : Nat := 4

private abbrev b200TileM : Nat := 128
private abbrev b200TileK : Nat := 128
private abbrev b200TileN : Nat := 256
private abbrev b200KBlocks : Nat := 4

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

private def h100Fp8Accumulator {inDtype : GpuFloat}
    (banner : String)
    (aPtr : GPtr inDtype)
    (bPtr : GPtr inDtype)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 fp8TileM fp8TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := fp8TileM)
    (tileK := fp8TileK)
    (tileN := fp8TileN)
    (kBlocks := fp8KBlocks)
    banner
    "ThunderKittens fp8_h100 tile family"
    aPtr bPtr m n k

/-! ## Canonical H100 FP8 Surfaces -/

/-- Canonical ThunderKittens-shaped H100 FP8 GEMM surface.

This mirrors the source-backed `fp8_h100_gemm.cu` layout: E4M3 inputs, FP32
accumulation, and an FP8 output epilogue on 64x256 output tiles. -/
@[gpu_kernel .SM90]
def tkH100Fp8E4M3GemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (cPtr : GPtr GpuFloat.FP8E4M3)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Fp8Accumulator
    "=== H100 FP8 GEMM (E4M3 -> E4M3 epilogue) ==="
    aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- Canonical H100 scaled-FP8 surface following `fp8_h100_scaled_gemm.cu`.

The ThunderKittens source applies explicit row/column dequant scales in the
consumer epilogue, so this surface keeps the mainloop unscaled and applies the
scales after FP32 accumulation. -/
@[gpu_kernel .SM90]
def tkH100Fp8ScaledGemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (scaleAPtr : GPtr GpuFloat.Float32)
    (scaleBPtr : GPtr GpuFloat.Float32)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Fp8Accumulator
    "=== H100 FP8 GEMM with row/column scales ==="
    aPtr bPtr m n k
  let (scaleA, scaleB) ← GemmCommon.loadRowColScaleVectors
    (tileM := fp8TileM)
    (tileN := fp8TileN)
    scaleAPtr scaleBPtr coord
  let scaled ← GemmCommon.applyRowColScales accum scaleA scaleB
  GemmCommon.storeFloat32Tile cPtr coord scaled

/-! ## Blackwell/B200 Surfaces -/

/-- Blackwell/B200 surface corresponding to `fp8_b200_gemm_1cta.cu`. -/
@[gpu_kernel .SM100]
def tkB200Fp8E4M3Gemm1CtaFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== B200 FP8 GEMM (1 CTA) ==="
  comment "ThunderKittens fp8_b200_gemm_1cta producer/consumer pipeline with TMEM-backed output tiles"
  rawLines #[
    "using a_tile = st_fp8e4m3<128, 128>;",
    "using b_tile = st_fp8e4m3<256, 128>;",
    "using d_tile = st_hf<128, 64>;",
    "using d_tt_t = tt<float, 128, 256>;",
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &B = {bPtr.id.toIdent};",
    s!"auto &D = {dPtr.id.toIdent};",
    "constexpr int NUM_CONSUMERS = 2;",
    "constexpr int NUM_PRODUCERS = 1;",
    "constexpr int PIPE_DEPTH = 3;",
    "if (threadIdx.x == 0) {",
    "  A.template prefetch_tma<a_tile>();",
    "  B.template prefetch_tma<b_tile>();",
    "  D.template prefetch_tma<d_tile>();",
    "}",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();",
    "b_tile (&b_smem)[PIPE_DEPTH] = al.allocate<b_tile, PIPE_DEPTH>();",
    "d_tile (&d_smem) = al.allocate<d_tile>();",
    "tensor_allocator<1, 1> tm_alloc{};",
    "__shared__ semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];",
    "uint32_t bitfield = 0xFFFF0000;",
    "if (threadIdx.x == 0) {",
    "  #pragma unroll",
    "  for (int i = 0; i < PIPE_DEPTH; i++) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 2); }",
    "  init_semaphore(outputs_arrived, 0, 1);",
    "  #pragma unroll",
    "  for (int i = 0; i < NUM_CONSUMERS; i++) init_semaphore(outputs_finished[i], 0, 1);",
    "}",
    "__syncthreads();",
    "if (warpgroup::groupid() == NUM_CONSUMERS) {",
    "  warpgroup::decrease_registers<56>();",
    "  if (warpgroup::warpid() == 3 && warp::laneid() == 0) {",
    "    int input_ring = 0;",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      int2 rowcol = get_task_idx({A, B, D}, task_iter, false);",
    "      if (rowcol.x == -1) {",
    "        for (int idx = 0; idx < PIPE_DEPTH; idx++) { wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring)); input_ring = ring_advance<PIPE_DEPTH>(input_ring); }",
    "        arrive(outputs_arrived);",
    "        break;",
    "      }",
    "      for (int idx = 0; idx < A.cols() / 128; idx++) {",
    "        wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));",
    "        update_phasebit<1>(bitfield, input_ring);",
    "        if (task_iter > 0 && idx == PIPE_DEPTH - 1) arrive(outputs_arrived);",
    "        tma::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);",
    "        tma::load_async(a_smem[input_ring][0], A, {rowcol.x + 0, idx}, inputs_arrived[input_ring]);",
    "        tma::load_async(a_smem[input_ring][1], A, {rowcol.x + 1, idx}, inputs_arrived[input_ring]);",
    "        tma::load_async(b_smem[input_ring], B, {rowcol.y, idx}, inputs_arrived[input_ring]);",
    "        input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      }",
    "    }",
    "  } else if ((warpgroup::warpid() == 0 || warpgroup::warpid() == 1) && warp::laneid() == 0) {",
    "    d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid() * 256);",
    "    int input_ring = 0;",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      int2 rowcol = get_task_idx({A, B, D}, task_iter, false);",
    "      if (rowcol.x == -1) break;",
    "      wait(outputs_finished[warpgroup::warpid()], (task_iter + 1) % 2);",
    "      wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));",
    "      update_phasebit<0>(bitfield, input_ring);",
    "      mm2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "      input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      for (int idx = 1; idx < A.cols() / 128; idx++) {",
    "        wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));",
    "        update_phasebit<0>(bitfield, input_ring);",
    "        mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "        input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      }",
    "    }",
    "  }",
    "} else {",
    "  warpgroup::increase_registers<224>();",
    "  d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::groupid() * 256);",
    "  for (int task_iter = 0; true; task_iter++) {",
    "    int2 rowcol = get_task_idx({A, B, D}, task_iter, true);",
    "    if (rowcol.x == -1) break;",
    "    wait(outputs_arrived, task_iter % 2);",
    "    rt_hf<32, 64> d_reg[4];",
    "    for (int i = 0; i < 4; i++) warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64 * i));",
    "    tensor_load_wait();",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    if (warpgroup::laneid() == 0) arrive(outputs_finished[warpgroup::groupid()]);",
    "    warpgroup::store(d_smem, d_reg[0]);",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    if (warpgroup::warpid() == 0) warp::tma::store_async(D, d_smem, {rowcol.x, 4 * rowcol.y + 0});",
    "    for (int i = 1; i < 4; i++) {",
    "      tma::store_async_read_wait();",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      warpgroup::store(d_smem, d_reg[i]);",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      if (warpgroup::warpid() == 0) warp::tma::store_async(D, d_smem, {rowcol.x, 4 * rowcol.y + i});",
    "    }",
    "    tma::store_async_read_wait();",
    "    group<8>::sync(15);",
    "  }",
    "}",
    "__syncthreads();"
  ]

/-- Blackwell/B200 surface corresponding to `fp8_b200_gemm_2cta.cu`. -/
@[gpu_kernel .SM100]
def tkB200Fp8E4M3Gemm2CtaFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== B200 FP8 GEMM (2 CTA cluster) ==="
  comment "ThunderKittens fp8_b200_gemm_2cta producer/consumer cluster pipeline with split-B loading and TMEM epilogue"
  rawLines #[
    "using a_tile = st_fp8e4m3<128, 128>;",
    "using b_tile = st_fp8e4m3<128, 128>;",
    "using d_tile = st_hf<128, 64>;",
    "using d_tt_t = tt<float, 128, 256>;",
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &B = {bPtr.id.toIdent};",
    s!"auto &D = {dPtr.id.toIdent};",
    "constexpr int NUM_CONSUMERS = 2;",
    "constexpr int NUM_PRODUCERS = 1;",
    "constexpr int PIPE_DEPTH = 4;",
    "if (threadIdx.x == 0) {",
    "  A.template prefetch_tma<a_tile>();",
    "  B.template prefetch_tma<b_tile>();",
    "  D.template prefetch_tma<d_tile>();",
    "}",
    "const int cta_rank = cluster_ctarank();",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();",
    "b_tile (&b_smem)[PIPE_DEPTH] = al.allocate<b_tile, PIPE_DEPTH>();",
    "d_tile (&d_smem) = al.allocate<d_tile>();",
    "tensor_allocator<1, 2> tm_alloc{};",
    "__shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];",
    "uint32_t bitfield = 0xFFFF0000;",
    "if (threadIdx.x == 0) {",
    "  for (int i = 0; i < PIPE_DEPTH; i++) { init_semaphore(inputs_arrived[i], 0, 2); init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); }",
    "  init_semaphore(outputs_arrived, 0, 1);",
    "  for (int i = 0; i < NUM_CONSUMERS; i++) init_semaphore(outputs_finished[i], 0, 2);",
    "}",
    "everyone::tma::cluster::sync();",
    "if (warpgroup::groupid() == NUM_CONSUMERS) {",
    "  warpgroup::decrease_registers<56>();",
    "  if (warpgroup::warpid() == 3 && warp::laneid() == 0) {",
    "    int input_ring = 0;",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      int2 rowcol = get_task_idx({A, B, D}, task_iter, false);",
    "      if (rowcol.x == -1) {",
    "        for (int idx = 0; idx < PIPE_DEPTH; idx++) { tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring)); input_ring = ring_advance<PIPE_DEPTH>(input_ring); }",
    "        if (laneid() == 0) arrive(outputs_arrived);",
    "        break;",
    "      }",
    "      for (int idx = 0; idx < A.cols() / 128; idx++) {",
    "        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));",
    "        update_phasebit<1>(bitfield, input_ring);",
    "        if (task_iter > 0 && idx == PIPE_DEPTH - 1 && laneid() == 0) arrive(outputs_arrived);",
    "        warp::tma::cluster::load_async(a_smem[input_ring][0], A, {rowcol.x + 0, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);",
    "        warp::tma::cluster::load_async(a_smem[input_ring][1], A, {rowcol.x + 1, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);",
    "        warp::tma::cluster::load_async(b_smem[input_ring], B, {rowcol.y, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);",
    "        input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      }",
    "    }",
    "  } else if (cta_rank == 0 && (warpgroup::warpid() == 0 || warpgroup::warpid() == 1) && warp::laneid() == 0) {",
    "    d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid() * 256);",
    "    int input_ring = 0;",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      int2 rowcol = get_task_idx({A, B, D}, task_iter, false);",
    "      if (rowcol.x == -1) break;",
    "      tma::cluster::wait(outputs_finished[warpgroup::warpid()], (task_iter + 1) % 2);",
    "      tma::cluster::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);",
    "      tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));",
    "      update_phasebit<0>(bitfield, input_ring);",
    "      mm2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "      input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      for (int idx = 1; idx < A.cols() / 128; idx++) {",
    "        tma::cluster::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);",
    "        tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));",
    "        update_phasebit<0>(bitfield, input_ring);",
    "        mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "        input_ring = ring_advance<PIPE_DEPTH>(input_ring);",
    "      }",
    "    }",
    "  }",
    "} else {",
    "  warpgroup::increase_registers<224>();",
    "  d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::groupid() * 256);",
    "  for (int task_iter = 0; true; task_iter++) {",
    "    int2 rowcol = get_task_idx({A, B, D}, task_iter, true);",
    "    if (rowcol.x == -1) break;",
    "    kittens::wait(outputs_arrived, task_iter % 2);",
    "    rt_hf<32, 64> d_reg[4];",
    "    for (int i = 0; i < 4; i++) warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64 * i));",
    "    tensor_load_wait();",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    if (warpgroup::laneid() == 0) kittens::warp::tma::cluster::arrive(outputs_finished[warpgroup::groupid()], 0);",
    "    warpgroup::store(d_smem, d_reg[0]);",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    if (warpgroup::warpid() == 0) warp::tma::store_async(D, d_smem, {rowcol.x, 4 * rowcol.y + 0});",
    "    for (int i = 1; i < 4; i++) {",
    "      tma::store_async_read_wait();",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      warpgroup::store(d_smem, d_reg[i]);",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      if (warpgroup::warpid() == 0) warp::tma::store_async(D, d_smem, {rowcol.x, 4 * rowcol.y + i});",
    "    }",
    "    tma::store_async_read_wait();",
    "    group<8>::sync(15);",
    "  }",
    "}",
    "everyone::tma::cluster::sync();"
  ]

/-- Blackwell/B200 MXFP8 surface corresponding to `mxfp8_b200_gemm.cu`. -/
@[gpu_kernel .SM100]
def tkB200MxFp8GemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (scaleAPtr : GPtr GpuFloat.FP8E8M0)
    (scaleBPtr : GPtr GpuFloat.FP8E8M0)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== B200 MXFP8 GEMM ==="
  comment "ThunderKittens mxfp8_b200 cluster-TMEM path with e8m0 scale tiles and tcgen05 MMA"
  rawLines #[
    "using A_fp8_tile = st_fp8e4m3<128, 128>;",
    "using A_sc_tile = st_fp8e8m0<32, 16, false>;",
    "using B_fp8_tile = st_fp8e4m3<128, 128>;",
    "using B_sc_tile = st_fp8e8m0<32, 16, false>;",
    "using D_tile = st_bf<128, 128>;",
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &B = {bPtr.id.toIdent};",
    s!"auto &A_sc = {scaleAPtr.id.toIdent};",
    s!"auto &B_sc = {scaleBPtr.id.toIdent};",
    s!"auto &D = {dPtr.id.toIdent};",
    "constexpr int CLUSTER_SIZE = 2;",
    "constexpr int LOAD_PIPE_DEPTH = 4;",
    "constexpr int EPI_PIPE_DEPTH = 2;",
    "constexpr int SUPERGROUP_SIZE = 8;",
    "constexpr int NUM_D_TILES = 2;",
    "constexpr int MMA_PER_TILE = 2;",
    "if (threadIdx.x == 0) {",
    "  A.template prefetch_tma<A_fp8_tile>();",
    "  A_sc.template prefetch_tma<A_sc_tile>();",
    "  B.template prefetch_tma<B_fp8_tile>();",
    "  B_sc.template prefetch_tma<B_sc_tile>();",
    "  D.template prefetch_tma<D_tile>();",
    "}",
    "const int warpgroup_id = warpgroup::groupid();",
    "const int cta_id = cluster_ctarank();",
    "const int cluster_id = clusterIdx().x;",
    "const int num_row_blocks = D.rows() / 256;",
    "const int num_col_blocks = D.cols() / 256;",
    "const int num_blocks = num_col_blocks * num_row_blocks;",
    "const int num_iters_per_block = A.cols() / 128;",
    "const int num_blocks_per_supergroup = SUPERGROUP_SIZE * num_col_blocks;",
    "uint32_t stage = 0;",
    "uint32_t phasebits = 0xFFFF0000;",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator sm_allocator((int*)&__shm[0]);",
    "struct input_tiles_t { A_fp8_tile A; B_fp8_tile B; };",
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
    "      for (int i = 0; i < num_iters_per_block; ++i) {",
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
    "      for (int i = 0; i < num_iters_per_block; ++i) {",
    "        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));",
    "        tma::cluster::load_async(input_scales[stage].A, A_sc, {row_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);",
    "        tma::cluster::load_async(input_scales[stage].B[cta_id], B_sc, {col_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);",
    "        update_phasebit<1>(phasebits, stage);",
    "        stage = (stage + 1) % LOAD_PIPE_DEPTH;",
    "      }",
    "    }",
    "  } else if (cta_id == 0 && warp_id == 0) {",
    "    everyone::tma::cluster::wait(); wait(tmem_provisioned, 0); tm_allocator.set_addr(tmem_addr);",
    "    auto out_tm = tm_allocator.template allocate<full_tt_fl<256>>(0);",
    "    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16 * MMA_PER_TILE * LOAD_PIPE_DEPTH>>(256);",
    "    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32 * MMA_PER_TILE * LOAD_PIPE_DEPTH>>(384);",
    "    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "      wait(outputs_finished, get_phasebit<1>(phasebits, 0));",
    "      tensor_after_thread_sync();",
    "      for (int i = 0; i < num_iters_per_block; i++) {",
    "        tma::expect_bytes(scales_arrived[stage], 2 * sizeof(input_scales_t));",
    "        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));",
    "        load_mxnv_scale_async2(A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * MMA_PER_TILE * 16), input_scales[stage].A);",
    "        load_mxnv_scale_async2(B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * MMA_PER_TILE * 32), input_scales[stage].B[0]);",
    "        load_mxnv_scale_async2(B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * MMA_PER_TILE * 32 + 16), input_scales[stage].B[1]);",
    "        tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(input_tiles_t));",
    "        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));",
    "        if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * MMA_PER_TILE * 16), B_sc_tm.template subtile<full_tt_fp8e8m0<64>>(stage * MMA_PER_TILE * 32), inputs_finished[stage]);",
    "        else mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * MMA_PER_TILE * 16), B_sc_tm.template subtile<full_tt_fp8e8m0<64>>(stage * MMA_PER_TILE * 32), inputs_finished[stage]);",
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
    "  wait(tmem_provisioned, 0);",
    "  tm_allocator.set_addr(tmem_addr);",
    "  auto out_tm = tm_allocator.template allocate<full_tt_fl<256>>(0);",
    "  for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) {",
    "    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));",
    "    rt_bf<32, 128> D_reg[EPI_PIPE_DEPTH];",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<128>>(0, 128 * i));",
    "    tensor_load_wait();",
    "    tensor_before_thread_sync();",
    "    warpgroup::sync(1);",
    "    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) {",
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

/-! ## Compatibility Convenience Kernels -/

/-- Compatibility BF16-output wrapper over the canonical H100 E4M3 mainloop. -/
@[gpu_kernel .SM90]
def gemmFp8E4M3Fwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Fp8Accumulator
    "=== H100 FP8 GEMM compatibility epilogue (E4M3 -> BF16) ==="
    aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- Format-variant compatibility wrapper using the same H100 tiling with E5M2
inputs and a BF16 epilogue. -/
@[gpu_kernel .SM90]
def gemmFp8E5M2Fwd
    (aPtr : GPtr GpuFloat.FP8E5M2)
    (bPtr : GPtr GpuFloat.FP8E5M2)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Fp8Accumulator
    "=== H100 FP8 GEMM compatibility epilogue (E5M2 -> BF16) ==="
    aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- BF16 activation / FP8 weight compatibility kernel using the canonical H100
FP8 output tile shape. The BF16 activation tile is explicitly converted to E4M3
before entering the shared H100 FP8 mainloop. -/
@[gpu_kernel .SM90]
def gemmMixedFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== Mixed BF16/FP8 GEMM compatibility kernel ==="
  comment "Uses the H100 FP8 mainloop after converting BF16 activations to E4M3"

  let coord ← blockCoord2D

  let aBf16 : RT GpuFloat.BFloat16 fp8TileM fp8TileK ← allocRT .BFloat16 fp8TileM fp8TileK
  let aFp8 : RT GpuFloat.FP8E4M3 fp8TileM fp8TileK ← allocRT .FP8E4M3 fp8TileM fp8TileK
  let b : RT GpuFloat.FP8E4M3 fp8TileN fp8TileK ← allocRT .FP8E4M3 fp8TileN fp8TileK
  let accum : RT GpuFloat.Float32 fp8TileM fp8TileN ← zeroRT .Float32 fp8TileM fp8TileN

  let aShared : ST GpuFloat.BFloat16 fp8TileM fp8TileK ← allocST .BFloat16 fp8TileM fp8TileK
  let bShared : ST GpuFloat.FP8E4M3 fp8TileN fp8TileK ← allocST .FP8E4M3 fp8TileN fp8TileK

  for kBlk in krange 0 fp8KBlocks do
    let aCoord := coord.withCol kBlk.id
    let bCoord := (coord.withRow coord.c).withCol kBlk.id
    loadGlobal aShared aPtr aCoord
    loadGlobal bShared bPtr bCoord
    sync
    load aBf16 aShared
    load b bShared
    convert aFp8 aBf16
    mmaT accum aFp8 b accum
    sync

  GemmCommon.storeConvertedTile cPtr coord accum

/-- Compatibility fused-epilogue kernel: H100 FP8 mainloop followed by an
elementwise scale tile and a per-column bias vector. -/
@[gpu_kernel .SM90]
def gemmFp8ScaledBiasFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (scalePtr : GPtr GpuFloat.Float32)
    (biasPtr : GPtr GpuFloat.Float32)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Fp8Accumulator
    "=== H100 FP8 GEMM compatibility epilogue (scale tile + bias) ==="
    aPtr bPtr m n k

  let scale : RT GpuFloat.Float32 fp8TileM fp8TileN ← allocRT .Float32 fp8TileM fp8TileN
  let scaled : RT GpuFloat.Float32 fp8TileM fp8TileN ← allocRT .Float32 fp8TileM fp8TileN
  let bias : RV GpuFloat.Float32 fp8TileN ← allocRV .Float32 fp8TileN

  let scaleShared : ST GpuFloat.Float32 fp8TileM fp8TileN ← allocST .Float32 fp8TileM fp8TileN
  let biasShared : SV GpuFloat.Float32 fp8TileN ← allocSV .Float32 fp8TileN

  loadGlobal scaleShared scalePtr coord
  sync
  load scale scaleShared
  loadVecGlobalCol biasShared biasPtr coord
  loadVec bias biasShared

  mul scaled accum scale
  addRow scaled scaled bias

  GemmCommon.storeConvertedTile cPtr coord scaled

end Tyr.GPU.Kernels.PrecisionGemm
