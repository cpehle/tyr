import Tyr.GPU.Kernels.GemmCommon

/-!
  Tyr/GPU/Kernels/Bf16Gemm.lean

  BF16 GEMM counterparts for the vendored ThunderKittens GEMM catalog.

  - `tkH100Bf16GemmFwd` is the canonical H100/Hopper BF16 surface aligned with
    `kernels/gemm/bf16_h100/bf16_h100_gemm.cu`.
  - `tkB200Bf16GemmFwd` is the Blackwell/B200 surface aligned with
    `kernels/gemm/bf16_b200/bf16_b200_gemm.cu`.
-/

namespace Tyr.GPU.Kernels.Bf16Gemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev h100TileM : Nat := 128
private abbrev h100TileK : Nat := 64
private abbrev h100TileN : Nat := 256
private abbrev h100KBlocks : Nat := 4

private abbrev b200TileM : Nat := 256
private abbrev b200TileK : Nat := 64
private abbrev b200TileN : Nat := 256
private abbrev b200KBlocks : Nat := 4

private def h100Bf16Accumulator
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 h100TileM h100TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := h100TileM)
    (tileK := h100TileK)
    (tileN := h100TileN)
    (kBlocks := h100KBlocks)
    "=== H100 BF16 GEMM ==="
    "ThunderKittens bf16_h100 producer/consumer tile, expressed as a single CTA-local tiled mainloop"
    aPtr bPtr m n k

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

/-- Canonical H100 BF16 GEMM surface matching the tile geometry used by the
vendored `bf16_h100` kernel family. -/
@[gpu_kernel .SM90]
def tkH100Bf16GemmFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Bf16Accumulator aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- Blackwell/B200 BF16 surface aligned with `bf16_b200`.

This follows the vendored ThunderKittens structure directly: cluster TMA
prefetch, producer/consumer warpgroup split, TMEM-backed tcgen05 MMA, and the
epilogue overlap controls used by the B200 kernel family. -/
@[gpu_kernel .SM100]
def tkB200Bf16GemmFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== B200 BF16 GEMM ==="
  comment "ThunderKittens bf16_b200 producer/consumer + TMEM pipeline on a concrete 256x256x64 configuration"
  rawLines #[
    "using a_tile = st_bf<128, 64>;",
    "using b_tile = st_bf<128, 64>;",
    "using d_tile = st_bf<128, 128>;",
    "using d_tt_t = tt<float, 128, 256>;",
    s!"auto &A = {aPtr.id.toIdent};",
    s!"auto &B = {bPtr.id.toIdent};",
    s!"auto &D = {cPtr.id.toIdent};",
    "if (threadIdx.x == 0) {",
    "  A.template prefetch_tma<a_tile>();",
    "  B.template prefetch_tma<b_tile>();",
    "  D.template prefetch_tma<d_tile>();",
    "}",
    "constexpr int SUPERGROUP_SIZE = 8;",
    "constexpr int LOAD_PIPE_DEPTH = 4;",
    "constexpr int EPI_PIPE_DEPTH = 2;",
    "constexpr int NUM_CONSUMERS = 2;",
    "constexpr int NUM_PRODUCERS = 1;",
    "constexpr int NUM_D_TILES = 2;",
    "constexpr int CLUSTER_SIZE = 2;",
    "const int cta_rank = cluster_ctarank();",
    "const int iters_per_task = A.cols() / 64;",
    "const int rblks = D.rows() / (256 * NUM_CONSUMERS);",
    "const int cblks = D.cols() / 256;",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator al((int*)&__shm[0]);",
    "a_tile (&a_smem)[LOAD_PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, LOAD_PIPE_DEPTH, NUM_CONSUMERS>();",
    "b_tile (&b_smem)[LOAD_PIPE_DEPTH] = al.allocate<b_tile, LOAD_PIPE_DEPTH>();",
    "d_tile (&d_smem)[NUM_CONSUMERS][NUM_D_TILES] = al.allocate<d_tile, NUM_CONSUMERS, NUM_D_TILES>();",
    "tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};",
    "__shared__ uint32_t tmem_addr;",
    "__shared__ clc::handle clc_handle[1];",
    "__shared__ semaphore tmem_provisioned, schedule_arrived[1], schedule_finished[1];",
    "__shared__ semaphore inputs_arrived[LOAD_PIPE_DEPTH], inputs_finished[LOAD_PIPE_DEPTH], outputs_arrived[NUM_CONSUMERS], outputs_finished[2];",
    "uint32_t bitfield = 0xFFFF0000;",
    "if (threadIdx.x == 32) {",
    "  init_semaphore(tmem_provisioned, 0, 1);",
    "  init_semaphore(schedule_arrived[0], 0, 1);",
    "  init_semaphore(schedule_finished[0], 0, (2 + NUM_CONSUMERS) * CLUSTER_SIZE + NUM_CONSUMERS);",
    "  #pragma unroll",
    "  for (int i = 0; i < LOAD_PIPE_DEPTH; i++) {",
    "    init_semaphore(inputs_arrived[i], 0, NUM_CONSUMERS);",
    "    init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS);",
    "  }",
    "  #pragma unroll",
    "  for (int i = 0; i < NUM_CONSUMERS; i++) init_semaphore(outputs_arrived[i], 0, 1);",
    "  #pragma unroll",
    "  for (int i = 0; i < 2; i++) init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE * NUM_CONSUMERS);",
    "}",
    "everyone::tma::cluster::arrive_aligned();",
    "if (warpgroup::groupid() == NUM_CONSUMERS) {",
    "  warpgroup::decrease_registers<56>();",
    "  if (warpgroup::warpid() == 3 && warp::elect_leader()) {",
    "    int input_ring = 0;",
    "    int2 tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x / CLUSTER_SIZE);",
    "    pdl::wait();",
    "    everyone::tma::cluster::wait();",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      for (int idx = 0; idx < iters_per_task; idx++) {",
    "        wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));",
    "        for (int i = 0; i < NUM_CONSUMERS; i++)",
    "          tma::cluster::load_async(a_smem[input_ring][i], A, {(tile_coord.x * 2 + cta_rank) * NUM_CONSUMERS + i, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);",
    "        tma::cluster::load_async(b_smem[input_ring], B, {tile_coord.y * 2 + cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);",
    "        update_phasebit<1>(bitfield, input_ring);",
    "        input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);",
    "      }",
    "      wait(schedule_arrived[0], task_iter % 2);",
    "      auto schedule = clc::query(clc_handle[0]);",
    "      tma::cluster::arrive(schedule_finished[0], 0);",
    "      if (schedule.success) tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x / CLUSTER_SIZE);",
    "      else break;",
    "    }",
    "  } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {",
    "    everyone::tma::cluster::wait();",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      if (cta_rank == 0) {",
    "        wait(schedule_finished[0], (task_iter + 1) % 2);",
    "        clc::schedule(clc_handle[0], schedule_arrived[0]);",
    "      }",
    "      tma::expect_bytes(schedule_arrived[0], sizeof(clc_handle[0]));",
    "      wait(schedule_arrived[0], task_iter % 2);",
    "      auto schedule = clc::query(clc_handle[0]);",
    "      tma::cluster::arrive(schedule_finished[0], 0);",
    "      if (!schedule.success) break;",
    "    }",
    "  } else if (cta_rank == 0 && warpgroup::warpid() < NUM_CONSUMERS && warp::elect_leader()) {",
    "    wait(tmem_provisioned, 0);",
    "    tm_alloc.set_addr(tmem_addr);",
    "    d_tt_t d_tt[2];",
    "    for (int i = 0; i < 2; i++) d_tt[i] = tm_alloc.template allocate<d_tt_t>((i + warpgroup::warpid()) * 256);",
    "    int input_ring = 0;",
    "    for (int task_iter = 0; true; task_iter++) {",
    "      wait(schedule_arrived[0], task_iter % 2);",
    "      auto schedule = clc::query(clc_handle[0]);",
    "      tma::cluster::arrive(schedule_finished[0], 0);",
    "      wait(outputs_finished[task_iter % 2], ((task_iter + 2) / 2) % 2);",
    "      for (int idx = 0; idx < iters_per_task; idx++) {",
    "        tma::expect_bytes(inputs_arrived[input_ring], (CLUSTER_SIZE * NUM_CONSUMERS * sizeof(a_tile) + 2 * sizeof(b_tile)) / NUM_CONSUMERS);",
    "        wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));",
    "        if (idx == 0) mm2_ABt(d_tt[task_iter % 2], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "        else mma2_ABt(d_tt[task_iter % 2], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);",
    "        update_phasebit<0>(bitfield, input_ring);",
    "        input_ring = ring_advance<LOAD_PIPE_DEPTH>(input_ring);",
    "      }",
    "      detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);",
    "      if (!schedule.success) break;",
    "    }",
    "  }",
    "} else {",
    "  using epilogue_group = group<WARPGROUP_WARPS * NUM_CONSUMERS>;",
    "  warpgroup::increase_registers<224>();",
    "  everyone::tma::cluster::wait_aligned();",
    "  if (epilogue_group::warpid() == 0) { tm_alloc.provision(tmem_addr); warp::arrive(tmem_provisioned); }",
    "  wait(tmem_provisioned, 0);",
    "  tm_alloc.set_addr(tmem_addr);",
    "  d_tt_t d_tt[2];",
    "  for (int i = 0; i < 2; i++) d_tt[i] = tm_alloc.template allocate<d_tt_t>((i + warpgroup::groupid()) * 256);",
    "  int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x / CLUSTER_SIZE);",
    "  for (int task_iter = 0; true; task_iter++) {",
    "    tile_coord = next_tile_coord;",
    "    wait(schedule_arrived[0], task_iter % 2);",
    "    auto schedule = clc::query(clc_handle[0]);",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    warpgroup::tma::cluster::arrive(schedule_finished[0], 0);",
    "    if (schedule.success) next_tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x / CLUSTER_SIZE);",
    "    wait(outputs_arrived[warpgroup::groupid()], task_iter % 2);",
    "    rt_bf<32, 128> d_reg[EPI_PIPE_DEPTH];",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) warpgroup::load_async(d_reg[i], d_tt[task_iter % 2].template subtile<tt<float, 128, 128>>(0, 128 * i));",
    "    tensor_load_wait();",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    if (!schedule.success) warpgroup::pdl::arrive();",
    "    warpgroup::tma::cluster::arrive(outputs_finished[task_iter % 2], 0);",
    "    for (int i = 0; i < EPI_PIPE_DEPTH; i++) {",
    "      warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      warpgroup::store(d_smem[warpgroup::groupid()][i % NUM_D_TILES], d_reg[i]);",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(D, d_smem[warpgroup::groupid()][i % NUM_D_TILES], {(2 * tile_coord.x + cta_rank) * NUM_CONSUMERS + warpgroup::groupid(), EPI_PIPE_DEPTH * tile_coord.y + i});",
    "    }",
    "    if (!schedule.success) break;",
    "  }",
    "  epilogue_group::sync(4);",
    "  if (epilogue_group::warpid() == 0) tm_alloc.deprovision();",
    "}"
  ]

end Tyr.GPU.Kernels.Bf16Gemm
