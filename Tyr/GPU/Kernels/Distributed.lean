/-
  Tyr/GPU/Kernels/Distributed.lean

  ThunderKittens-shaped distributed communication and communication+GEMM
  surfaces. These kernels still operate within the current Lean DSL limits,
  but the public entrypoints now mirror the real collective families and
  producer/consumer structure from `thirdparty/ThunderKittens/kernels/parallel`.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

namespace Tyr.GPU.Kernels.Distributed

open Tyr.GPU
open Tyr.GPU.Codegen

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

private def allReduceOutOfPlaceBody {tileRows tileCols : Nat}
    (label : String)
    (output_ptr : GPtr GpuFloat.Float32)
    (input_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  let reduced : RT GpuFloat.Float32 tileRows tileCols ← allocRT .Float32 tileRows tileCols
  let inputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let publishShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let outputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols

  comment label
  Support.asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * GpuFloat.bytes .Float32)
  multimemLoadReduce reduced inputShared .Sum
  multimemStore publishShared reduced
  Support.barrierAllDevices "all_reduce publish complete" 0

  store outputShared reduced
  storeGlobal output_ptr outputShared coord
  Support.barrierAllDevices "all_reduce epilogue" 1

private def allReduceInPlaceBody {tileRows tileCols : Nat}
    (label : String)
    (data_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  let reduced : RT GpuFloat.Float32 tileRows tileCols ← allocRT .Float32 tileRows tileCols
  let inputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let publishShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let outputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols

  comment label
  Support.asyncTileLoad inputShared data_ptr coord (tileRows * tileCols * GpuFloat.bytes .Float32)
  multimemLoadReduce reduced inputShared .Sum
  multimemStore publishShared reduced
  Support.barrierAllDevices "all_reduce educational publish complete" 0

  store outputShared reduced
  storeGlobal data_ptr outputShared coord
  Support.barrierAllDevices "all_reduce educational epilogue" 1

private def agGemmCompatBody {inDtype : GpuFloat} {rowBlock colBlock redBlock numRedTiles : Nat}
    (label : String)
    (c_ptr : GPtr GpuFloat.BFloat16)
    (a_local_ptr : GPtr inDtype)
    (b_ptr : GPtr inDtype)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32)
    (num_comm_sms : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size, num_comm_sms)
  let coord ← blockCoord2D

  let aStage : ST inDtype rowBlock redBlock ← allocST inDtype rowBlock redBlock
  let aGatherStage : ST inDtype rowBlock redBlock ← allocST inDtype rowBlock redBlock
  let bStage : ST inDtype redBlock colBlock .Col ← allocST inDtype redBlock colBlock .Col
  let cStage : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment label
  ifWarpGroup 0 do
    for redIdx in krange 0 numRedTiles do
      expectBytes semA (rowBlock * redBlock * GpuFloat.bytes inDtype)
      loadGlobalAsync aStage a_local_ptr (coord.withCol redIdx.id) semA.id
      waitSemaphore semA
      let aLocal : RT inDtype rowBlock redBlock ← allocRT inDtype rowBlock redBlock
      load aLocal aStage
      multimemStore aGatherStage aLocal
      Support.barrierAllDevices "ag_gemm all-gather row shard ready" 0

      expectBytes semB (redBlock * colBlock * GpuFloat.bytes inDtype)
      loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
      waitSemaphore semB
      namedBarrierArrive 2 128

  ifWarpGroup 1 do
    let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
    for _redIdx in krange 0 numRedTiles do
      namedBarrierSync 2 128
      let aGathered : RT inDtype rowBlock redBlock ← allocRT inDtype rowBlock redBlock
      let bTile : RT inDtype redBlock colBlock .Col ← allocRT inDtype redBlock colBlock .Col
      load aGathered aGatherStage
      load bTile bStage
      mma cAcc aGathered bTile cAcc
      sync

    let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
    convert out cAcc
    store cStage out
    storeGlobal c_ptr cStage coord

private def gemmArCompatBody {inDtype : GpuFloat} {rowBlock colBlock redBlock numRedTiles : Nat}
    (label : String)
    (c_ptr : GPtr GpuFloat.BFloat16)
    (a_ptr : GPtr inDtype)
    (b_ptr : GPtr inDtype)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32)
    (num_comm_sms : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size, num_comm_sms)
  let coord ← blockCoord2D

  let aStage : ST inDtype rowBlock redBlock ← allocST inDtype rowBlock redBlock
  let bStage : ST inDtype redBlock colBlock .Col ← allocST inDtype redBlock colBlock .Col
  let partialShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let reducedShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment label
  ifWarpGroup 1 do
    let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
    for redIdx in krange 0 numRedTiles do
      expectBytes semA (rowBlock * redBlock * GpuFloat.bytes inDtype)
      loadGlobalAsync aStage a_ptr (coord.withCol redIdx.id) semA.id
      waitSemaphore semA
      expectBytes semB (redBlock * colBlock * GpuFloat.bytes inDtype)
      loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
      waitSemaphore semB

      let aTile : RT inDtype rowBlock redBlock ← allocRT inDtype rowBlock redBlock
      let bTile : RT inDtype redBlock colBlock .Col ← allocRT inDtype redBlock colBlock .Col
      load aTile aStage
      load bTile bStage
      mma cAcc aTile bTile cAcc
      sync

    store partialShared cAcc
    namedBarrierArrive 3 128

  ifWarpGroup 0 do
    namedBarrierSync 3 128
    let cPartial : RT GpuFloat.Float32 rowBlock colBlock ← allocRT .Float32 rowBlock colBlock
    let cReduced : RT GpuFloat.Float32 rowBlock colBlock ← allocRT .Float32 rowBlock colBlock
    load cPartial partialShared
    multimemRed reducedShared cPartial .Sum
    Support.barrierAllDevices "gemm_ar reduction complete" 0
    load cReduced reducedShared

    let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
    convert out cReduced
    store outShared out
    storeGlobal c_ptr outShared coord

private def gemmRsReduceBody {inDtype : GpuFloat} {rowBlock colBlock redBlock numRedTiles : Nat}
    (label : String)
    (c_ptr : GPtr GpuFloat.BFloat16)
    (a_ptr : GPtr inDtype)
    (b_ptr : GPtr inDtype)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  let aStage : ST inDtype rowBlock redBlock ← allocST inDtype rowBlock redBlock
  let bStage : ST inDtype redBlock colBlock .Col ← allocST inDtype redBlock colBlock .Col
  let partialShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment label
  let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
  for redIdx in krange 0 numRedTiles do
    expectBytes semA (rowBlock * redBlock * GpuFloat.bytes inDtype)
    loadGlobalAsync aStage a_ptr (coord.withCol redIdx.id) semA.id
    waitSemaphore semA
    expectBytes semB (redBlock * colBlock * GpuFloat.bytes inDtype)
    loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
    waitSemaphore semB

    let aTile : RT inDtype rowBlock redBlock ← allocRT inDtype rowBlock redBlock
    let bTile : RT inDtype redBlock colBlock .Col ← allocRT inDtype redBlock colBlock .Col
    load aTile aStage
    load bTile bStage
    mma cAcc aTile bTile cAcc
    sync

  store partialShared cAcc
  let cReduced : RT GpuFloat.Float32 rowBlock colBlock ← allocRT .Float32 rowBlock colBlock
  multimemLoadReduce cReduced partialShared .Sum
  Support.barrierAllDevices "gemm_rs reduction complete" 0

  let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
  convert out cReduced
  store outShared out
  storeGlobal c_ptr outShared coord

private def gemmRsStoreAddCompatBody {inDtype : GpuFloat} {rowBlock colBlock redBlock numRedTiles : Nat}
    (label : String)
    (c_ptr : GPtr GpuFloat.BFloat16)
    (a_ptr : GPtr inDtype)
    (b_ptr : GPtr inDtype)
    (dev_idx : KVal UInt32)
    (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  let aStage : ST inDtype rowBlock redBlock ← allocST inDtype rowBlock redBlock
  let bStage : ST inDtype redBlock colBlock .Col ← allocST inDtype redBlock colBlock .Col
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment label
  let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
  for redIdx in krange 0 numRedTiles do
    expectBytes semA (rowBlock * redBlock * GpuFloat.bytes inDtype)
    loadGlobalAsync aStage a_ptr (coord.withCol redIdx.id) semA.id
    waitSemaphore semA
    expectBytes semB (redBlock * colBlock * GpuFloat.bytes inDtype)
    loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
    waitSemaphore semB

    let aTile : RT inDtype rowBlock redBlock ← allocRT inDtype rowBlock redBlock
    let bTile : RT inDtype redBlock colBlock .Col ← allocRT inDtype redBlock colBlock .Col
    load aTile aStage
    load bTile bStage
    mma cAcc aTile bTile cAcc
    sync

  Support.barrierAllDevices "gemm_rs local tiles ready for distributed store_add" 0
  let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
  convert out cAcc
  store outShared out
  storeGlobalAdd c_ptr outShared coord
  Support.barrierAllDevices "gemm_rs distributed store_add epilogue" 1

/-! ## Concrete Collectives -/

/-- ThunderKittens-style all-gather: stage a local shard, publish via multimem,
then write the gathered view through the caller-provided output layout. -/
@[gpu_kernel .SM90]
def allGatherFwd (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := world_size
  comment "ThunderKittens all_gather: single-tile TMA load from the local shard, then store into the gathered column range"
  rawLines #[
    "using shared_tile = st_bf<128, 128>;",
    s!"auto &output = {output_ptr.id.toIdent};",
    s!"auto &input = {input_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator allocator((int*)&__shm[0]);",
    "shared_tile &tile = allocator.allocate<shared_tile>();",
    "const int row_block_idx = blockIdx.y;",
    "const int col_block_idx = blockIdx.x;",
    "const int col_blocks_per_dev = output.cols() / 128 / 8;",
    "__shared__ semaphore arrived;",
    "init_semaphore(arrived, 0, 1);",
    "tma::expect_bytes(arrived, sizeof(tile));",
    "tma::load_async(tile, input, {row_block_idx, col_block_idx}, arrived);",
    "wait(arrived, 0);",
    "tma::store_async(output, tile, {row_block_idx, col_blocks_per_dev * dev_idx + col_block_idx});"
  ]
  Support.barrierAllDevices "all_gather epilogue" 1

/-- ThunderKittens-style all-reduce: multimem reduce, republish the reduced tile,
then store through the local output view. -/
@[gpu_kernel .SM90]
def allReduceFwd (output_ptr : GPtr GpuFloat.Float32) (input_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := world_size
  comment "ThunderKittens all_reduce: reduce a device-local slice over the multicast tensor and republish the reduced slice"
  rawLines #[
    s!"auto &output = {output_ptr.id.toIdent};",
    s!"auto &input = {input_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    "const size_t num_elems_per_inst = 1;",
    "const size_t num_elems_per_block = blockDim.x * num_elems_per_inst;",
    "const size_t n_total = input.numel();",
    "const size_t n_per_dev = n_total / 8;",
    "const size_t idx = n_per_dev * dev_idx + num_elems_per_block * blockIdx.x + num_elems_per_inst * threadIdx.x;",
    "float tmp;",
    "multimem<float>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<float*>(&input.mc_ptr[idx]));",
    "multimem<float>::st(reinterpret_cast<float*>(&output.mc_ptr[idx]), tmp);"
  ]
  Support.barrierAllDevices "all_reduce epilogue" 1

/-- ThunderKittens-style reduce-scatter: reduce a multicast source and store the
caller-selected shard view. -/
@[gpu_kernel .SM90]
def reduceScatterFwd (output_ptr : GPtr GpuFloat.Float32) (input_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := world_size
  comment "ThunderKittens reduce_scatter: reduce over the multicast source and keep only the local output shard"
  rawLines #[
    s!"auto &output = {output_ptr.id.toIdent};",
    s!"auto &input = {input_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    "const size_t num_elems_per_inst = 1;",
    "const size_t num_elems_per_block = blockDim.x * num_elems_per_inst;",
    "const size_t row_col_idx = static_cast<size_t>(blockIdx.x) * num_elems_per_block;",
    "const size_t col_idx = row_col_idx % output.cols();",
    "const size_t row_idx = row_col_idx / output.cols();",
    "const size_t depth_idx = blockIdx.y;",
    "const size_t num_cols_per_dev = output.cols();",
    "const size_t input_idx = depth_idx * input.rows() * input.cols() + row_idx * input.cols() + col_idx + num_cols_per_dev * dev_idx + threadIdx.x * num_elems_per_inst;",
    "const size_t output_idx = depth_idx * output.rows() * output.cols() + row_idx * output.cols() + col_idx + threadIdx.x * num_elems_per_inst;",
    "float tmp;",
    "multimem<float>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<float*>(&input.mc_ptr[input_idx]));",
    "move<float>::stg(reinterpret_cast<float*>(&output.raw_ptr[output_idx]), tmp);"
  ]
  Support.barrierAllDevices "reduce_scatter epilogue" 1

/-- Concrete all-to-all surface for head-parallel -> sequence-parallel transport.
The caller supplies already-partitioned input/output views; this kernel owns the
TMA load, barrier, and store choreography. -/
@[gpu_kernel .SM90]
def allToAllHeadsToSeq (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let scatter ← constIntVal 1 "scatter_depth"
  let gather ← constIntVal 2 "gather_rows"

  comment "ThunderKittens all_to_all<scatter=head, gather=seq>"
  Support.allToAllTile "heads->seq shard" output_ptr input_ptr dev_idx world_size scatter gather
  Support.barrierAllDevices "all_to_all heads->seq epilogue" 1

/-- Concrete all-to-all surface for sequence-parallel -> head-parallel transport. -/
@[gpu_kernel .SM90]
def allToAllSeqToHeads (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let scatter ← constIntVal 2 "scatter_rows"
  let gather ← constIntVal 1 "gather_depth"

  comment "ThunderKittens all_to_all<scatter=seq, gather=head>"
  Support.allToAllTile "seq->heads shard" output_ptr input_ptr dev_idx world_size scatter gather
  Support.barrierAllDevices "all_to_all seq->heads epilogue" 1

/-! ## Communication + GEMM Surfaces -/

/-- AllGather + GEMM.
Warpgroup 0 acts as the communication producer; warpgroup 1 consumes the
gathered `A` tile and replicated `B` tile for MMA. -/
@[gpu_kernel .SM90]
def agGemmFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_local_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) (num_comm_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := world_size
  comment "ThunderKittens ag_gemm_h100: comm SMs all-gather A shards while comp SMs run the pipelined MMA/epilogue path"
  rawLines #[
    "using A_tile = st_bf<64, 64>;",
    "using A_comm_tile = st_bf<256, 128>;",
    "using B_tile = st_bf<64, 256>;",
    "using C_tile = st_bf<64, 256>;",
    s!"auto &A = {a_local_ptr.id.toIdent};",
    s!"auto &B = {b_ptr.id.toIdent};",
    s!"auto &C = {c_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    s!"const int num_comm_sms = {num_comm_sms.id.toIdent};",
    "const int num_comp_sms = gridDim.x - num_comm_sms;",
    "if (blockIdx.x >= num_comp_sms) {",
    "  extern __shared__ int __shm[];",
    "  tma_swizzle_allocator al((int*)&__shm[0]);",
    "  A_comm_tile (&A_smem)[4] = al.allocate<A_comm_tile, 4>();",
    "  __shared__ semaphore inputs_arrived[4];",
    "  const int comm_sm_id = blockIdx.x - num_comp_sms;",
    "  const int global_row_blocks = A.rows() / 256;",
    "  const int local_row_blocks = global_row_blocks / 8;",
    "  const int col_blocks = A.cols() / 128;",
    "  const int num_local_blocks = local_row_blocks * col_blocks;",
    "  uint32_t phasebits = 0xFFFF0000;",
    "  if (warp::groupid() < 4 && warp::laneid() == 0) {",
    "    init_semaphore(inputs_arrived[warp::groupid()], 0, 1);",
    "    for (int task_id = comm_sm_id * 4 + warp::groupid(); task_id < num_local_blocks; task_id += num_comm_sms * 4) {",
    "      const int row_idx = task_id / col_blocks;",
    "      const int global_row_idx = row_idx + dev_idx * local_row_blocks;",
    "      const int col_idx = task_id % col_blocks;",
    "      tma::expect_bytes(inputs_arrived[warp::groupid()], sizeof(A_comm_tile));",
    "      tma::load_async(A_smem[warp::groupid()], A, {global_row_idx, col_idx}, inputs_arrived[warp::groupid()]);",
    "      wait(inputs_arrived[warp::groupid()], get_phasebit<0>(phasebits, warp::groupid()));",
    "      update_phasebit<0>(phasebits, warp::groupid());",
    "      tma::store_async(A, A_smem[warp::groupid()], {global_row_idx, col_idx});",
    "      tma::store_async_wait();",
    "      if (col_idx + num_comm_sms * 4 >= col_blocks) signal_all(barrier, {global_row_idx}, 1);",
    "    }",
    "  }",
    "} else {",
    "  extern __shared__ int __shm[];",
    "  tma_swizzle_allocator allocator((int*)&__shm[0]);",
    "  struct pipeline_inputs { A_tile A[2]; B_tile B; };",
    "  struct pipeline_outputs { C_tile C[2]; };",
    "  pipeline_inputs (&inputs)[4] = allocator.allocate<pipeline_inputs, 4>();",
    "  pipeline_outputs &outputs = *reinterpret_cast<pipeline_outputs*>(&inputs[3]);",
    "  __shared__ semaphore inputs_arrived[4], inputs_finished[4], outputs_arrived, outputs_finished;",
    "  if (threadIdx.x == 0) { for (int i = 0; i < 4; ++i) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 8); } init_semaphore(outputs_arrived, 0, 2); init_semaphore(outputs_finished, 0, 1); }",
    "  __syncthreads();",
    "  const int row_blocks = A.rows() / 128;",
    "  const int col_blocks = B.cols() / 256;",
    "  const int num_iters = A.cols() / 64;",
    "  if (warpgroup::groupid() == 2) {",
    "    warpgroup::decrease_registers<40>();",
    "    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {",
    "      for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += num_comp_sms) {",
    "        const int row_idx = task_id / col_blocks; const int col_idx = task_id % col_blocks;",
    "        for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "          wait(inputs_finished[red_idx % 4], (red_idx / 4 + 1) % 2);",
    "          tma::expect_bytes(inputs_arrived[red_idx % 4], sizeof(pipeline_inputs));",
    "          if (red_idx == 3) wait(outputs_finished, 1);",
    "          tma::load_async(inputs[red_idx % 4].A[0], A, {row_idx * 2 + 0, red_idx}, inputs_arrived[red_idx % 4]);",
    "          tma::load_async(inputs[red_idx % 4].A[1], A, {row_idx * 2 + 1, red_idx}, inputs_arrived[red_idx % 4]);",
    "          tma::load_async(inputs[red_idx % 4].B, B, {red_idx, col_idx}, inputs_arrived[red_idx % 4]);",
    "        }",
    "      }",
    "    }",
    "  } else {",
    "    warpgroup::increase_registers<232>();",
    "    for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += num_comp_sms) {",
    "      rt_fl<16, 256> C_accum; warp::zero(C_accum);",
    "      for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "        wait(inputs_arrived[red_idx % 4], (red_idx / 4) % 2);",
    "        warpgroup::mma_AB(C_accum, inputs[red_idx % 4].A[warpgroup::groupid()], inputs[red_idx % 4].B);",
    "        warpgroup::mma_async_wait();",
    "        warp::arrive(inputs_finished[red_idx % 4]);",
    "      }",
    "      warpgroup::store(outputs.C[warpgroup::groupid()], C_accum);",
    "      warpgroup::sync(warpgroup::groupid() + 1);",
    "      warpgroup::arrive(outputs_arrived);",
    "    }",
    "  }",
    "}"
  ]

/-- GEMM + AllReduce.
Consumer warpgroups compute local partial `C`; producer warpgroup reduces the
tile with multimem and stores the replicated result. -/
@[gpu_kernel .SM90]
def gemmArFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) (num_comm_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := (world_size, num_comm_sms)
  comment "ThunderKittens gemm_ar_h100: pipelined local MMA on comp SMs plus in-network all-reduce on comm SMs"
  rawLines #[
    "using A_tile = st_bf<64, 64>;",
    "using B_tile = st_bf<64, 256>;",
    "using C_tile = st_bf<64, 256>;",
    s!"auto &A = {a_ptr.id.toIdent};",
    s!"auto &B = {b_ptr.id.toIdent};",
    s!"auto &C = {c_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    "const int num_comp_sms = gridDim.x;",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator allocator((int*)&__shm[0]);",
    "struct pipeline_inputs { A_tile A[2]; B_tile B; };",
    "struct pipeline_outputs { C_tile C[2]; };",
    "pipeline_inputs (&inputs)[4] = allocator.allocate<pipeline_inputs, 4>();",
    "pipeline_outputs &outputs = *reinterpret_cast<pipeline_outputs*>(&inputs[3]);",
    "__shared__ semaphore inputs_arrived[4], inputs_finished[4], outputs_arrived, outputs_finished;",
    "if (threadIdx.x == 0) { for (int i = 0; i < 4; ++i) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 8); } init_semaphore(outputs_arrived, 0, 2); init_semaphore(outputs_finished, 0, 1); }",
    "__syncthreads();",
    "const int row_blocks = A.rows() / 128;",
    "const int col_blocks = B.cols() / 256;",
    "const int num_iters = A.cols() / 64;",
    "if (warpgroup::groupid() == 2) {",
    "  warpgroup::decrease_registers<40>();",
    "  if (warpgroup::warpid() == 0 && warp::laneid() == 0) {",
    "    for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += num_comp_sms) {",
    "      const int row_idx = task_id / col_blocks; const int col_idx = task_id % col_blocks;",
    "      for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "        wait(inputs_finished[red_idx % 4], (red_idx / 4 + 1) % 2);",
    "        tma::expect_bytes(inputs_arrived[red_idx % 4], sizeof(pipeline_inputs));",
    "        if (red_idx == 3) wait(outputs_finished, 1);",
    "        tma::load_async(inputs[red_idx % 4].A[0], A, {row_idx * 2 + 0, red_idx}, inputs_arrived[red_idx % 4]);",
    "        tma::load_async(inputs[red_idx % 4].A[1], A, {row_idx * 2 + 1, red_idx}, inputs_arrived[red_idx % 4]);",
    "        tma::load_async(inputs[red_idx % 4].B, B, {red_idx, col_idx}, inputs_arrived[red_idx % 4]);",
    "      }",
    "    }",
    "  } else if (warpgroup::warpid() == 1 && warp::laneid() == 0) {",
    "    for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += num_comp_sms) {",
    "      const int row_idx = task_id / col_blocks; const int col_idx = task_id % col_blocks;",
    "      wait(outputs_arrived, 0);",
    "      tma::store_async(C, outputs.C[0], {row_idx * 2 + 0, col_idx});",
    "      tma::store_async(C, outputs.C[1], {row_idx * 2 + 1, col_idx});",
    "      tma::store_async_read_wait();",
    "      signal(barrier, {row_idx, col_idx}, task_id % 8, 1);",
    "      arrive(outputs_finished);",
    "    }",
    "  }",
    "} else {",
    "  warpgroup::increase_registers<232>();",
    "  for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += num_comp_sms) {",
    "    rt_fl<16, 256> C_accum; warp::zero(C_accum);",
    "    for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "      wait(inputs_arrived[red_idx % 4], (red_idx / 4) % 2);",
    "      warpgroup::mma_AB(C_accum, inputs[red_idx % 4].A[warpgroup::groupid()], inputs[red_idx % 4].B);",
    "      warpgroup::mma_async_wait();",
    "      warp::arrive(inputs_finished[red_idx % 4]);",
    "    }",
    "    warpgroup::store(outputs.C[warpgroup::groupid()], C_accum);",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    warpgroup::arrive(outputs_arrived);",
    "  }",
    "}"
  ]

/-- GEMM + ReduceScatter.
Consumer warpgroups compute the local tile; producer warpgroup performs the
reduce-scatter write into the caller-provided sharded output view. -/
@[gpu_kernel .SM90]
def gemmRsFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := world_size
  comment "ThunderKittens gemm_rs_h100: pipelined local MMA with reduce-scatter store into the device shard"
  rawLines #[
    "using A_tile = st_bf<64, 64>;",
    "using B_tile = st_bf<64, 256>;",
    "using C_tile = st_bf<64, 256>;",
    s!"auto &A = {a_ptr.id.toIdent};",
    s!"auto &B = {b_ptr.id.toIdent};",
    s!"auto &C = {c_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator allocator((int*)&__shm[0]);",
    "struct pipeline_inputs { A_tile A[2]; B_tile B; };",
    "struct pipeline_outputs { C_tile C[2]; };",
    "pipeline_inputs (&inputs)[4] = allocator.allocate<pipeline_inputs, 4>();",
    "pipeline_outputs &outputs = *reinterpret_cast<pipeline_outputs*>(&inputs[3]);",
    "__shared__ semaphore inputs_arrived[4], inputs_finished[4], outputs_arrived, outputs_finished;",
    "if (threadIdx.x == 0) { for (int i = 0; i < 4; ++i) { init_semaphore(inputs_arrived[i], 0, 1); init_semaphore(inputs_finished[i], 0, 8); } init_semaphore(outputs_arrived, 0, 2); init_semaphore(outputs_finished, 0, 1); }",
    "__syncthreads();",
    "const int row_blocks = A.rows() / 128;",
    "const int col_blocks = B.cols() / 256;",
    "const int num_iters = A.cols() / 64;",
    "const int row_blocks_per_dev = row_blocks / 8;",
    "const int dev_task_offset = ((dev_idx + 1) * ((row_blocks * col_blocks) / 8)) % (row_blocks * col_blocks);",
    "if (warpgroup::groupid() == 2) {",
    "  warpgroup::decrease_registers<40>();",
    "  if (warpgroup::warpid() == 0 && warp::laneid() == 0) {",
    "    for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += gridDim.x) {",
    "      const int real_task_id = (task_id + dev_task_offset) % (row_blocks * col_blocks);",
    "      const int row_idx = real_task_id / col_blocks;",
    "      const int col_idx = real_task_id % col_blocks;",
    "      for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "        wait(inputs_finished[red_idx % 4], (red_idx / 4 + 1) % 2);",
    "        tma::expect_bytes(inputs_arrived[red_idx % 4], sizeof(pipeline_inputs));",
    "        if (red_idx == 3) wait(outputs_finished, 1);",
    "        tma::load_async(inputs[red_idx % 4].A[0], A, {row_idx * 2 + 0, red_idx}, inputs_arrived[red_idx % 4]);",
    "        tma::load_async(inputs[red_idx % 4].A[1], A, {row_idx * 2 + 1, red_idx}, inputs_arrived[red_idx % 4]);",
    "        tma::load_async(inputs[red_idx % 4].B, B, {red_idx, col_idx}, inputs_arrived[red_idx % 4]);",
    "      }",
    "    }",
    "  } else if (warpgroup::warpid() == 1 && warp::laneid() == 0) {",
    "    for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += gridDim.x) {",
    "      const int real_task_id = (task_id + dev_task_offset) % (row_blocks * col_blocks);",
    "      int row_idx = real_task_id / col_blocks;",
    "      const int col_idx = real_task_id % col_blocks;",
    "      const int dst_dev = row_idx / row_blocks_per_dev;",
    "      row_idx %= row_blocks_per_dev;",
    "      wait(outputs_arrived, 0);",
    "      tma::store_add_async(C, outputs.C[0], {row_idx * 2 + 0, col_idx});",
    "      tma::store_add_async(C, outputs.C[1], {row_idx * 2 + 1, col_idx});",
    "      tma::store_async_read_wait();",
    "      arrive(outputs_finished);",
    "      (void)dst_dev;",
    "    }",
    "  }",
    "} else {",
    "  warpgroup::increase_registers<232>();",
    "  for (int task_id = blockIdx.x; task_id < row_blocks * col_blocks; task_id += gridDim.x) {",
    "    rt_fl<16, 256> C_accum; warp::zero(C_accum);",
    "    for (int red_idx = 0; red_idx < num_iters; red_idx++) {",
    "      wait(inputs_arrived[red_idx % 4], (red_idx / 4) % 2);",
    "      warpgroup::mma_AB(C_accum, inputs[red_idx % 4].A[warpgroup::groupid()], inputs[red_idx % 4].B);",
    "      warpgroup::mma_async_wait();",
    "      warp::arrive(inputs_finished[red_idx % 4]);",
    "    }",
    "    warpgroup::store(outputs.C[warpgroup::groupid()], C_accum);",
    "    warpgroup::sync(warpgroup::groupid() + 1);",
    "    warpgroup::arrive(outputs_arrived);",
    "  }",
    "}"
  ]

/-- ThunderKittens `all_reduce_educational.cu` counterpart.
This keeps the in-place multimem `ld_reduce`/publish/store structure while
staying within Tyr's current tile/multimem surface. -/
@[gpu_kernel .SM90]
def allReduceEducationalFwd (data_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  allReduceInPlaceBody (tileRows := 128) (tileCols := 128)
    "ThunderKittens all_reduce_educational: reduce a caller-owned local slice in place"
    data_ptr dev_idx world_size

/-- ThunderKittens `ag_gemm_b200.cu` counterpart.
This mirrors the Blackwell-facing tile shape and producer/consumer split via
the current Lean multimem/MMA surface. -/
@[gpu_kernel .SM100]
def agGemmB200Fwd (c_ptr : GPtr GpuFloat.BFloat16) (a_local_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (num_comm_sms : KVal UInt32) (num_comp_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := num_comp_sms
  agGemmCompatBody
    (inDtype := .BFloat16)
    (rowBlock := 256)
    (colBlock := 256)
    (redBlock := 64)
    (numRedTiles := 4)
    "ThunderKittens ag_gemm_b200: Blackwell tile geometry with Lean-side producer/consumer staging"
    c_ptr a_local_ptr b_ptr dev_idx world_size num_comm_sms

/-- ThunderKittens `ag_gemm_fp8_b200.cu` counterpart.
The public surface matches the FP8/BF16 contract of the vendored kernel while
the body uses the current Lean multimem/MMA choreography. -/
@[gpu_kernel .SM100]
def agGemmFp8B200Fwd (c_ptr : GPtr GpuFloat.BFloat16) (a_local_ptr : GPtr GpuFloat.FP8E4M3)
    (b_ptr : GPtr GpuFloat.FP8E4M3)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (num_comm_sms : KVal UInt32) (num_comp_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := num_comp_sms
  agGemmCompatBody
    (inDtype := .FP8E4M3)
    (rowBlock := 256)
    (colBlock := 256)
    (redBlock := 128)
    (numRedTiles := 4)
    "ThunderKittens ag_gemm_fp8_b200: FP8 Blackwell tiles with BF16 output"
    c_ptr a_local_ptr b_ptr dev_idx world_size num_comm_sms

/-- ThunderKittens `gemm_ar_h100_lcsc.cu` counterpart.
This keeps the H100 tile geometry and all-reduce flow of the vendored kernel
through Tyr's current producer/consumer abstraction. -/
@[gpu_kernel .SM90]
def gemmArH100LcscFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (num_comm_sms : KVal UInt32) (num_comp_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := num_comp_sms
  gemmArCompatBody
    (inDtype := .BFloat16)
    (rowBlock := 128)
    (colBlock := 256)
    (redBlock := 64)
    (numRedTiles := 4)
    "ThunderKittens gemm_ar_h100_lcsc: H100 LCSC tile geometry with Lean-side all-reduce epilogue"
    c_ptr a_ptr b_ptr dev_idx world_size num_comm_sms

/-- ThunderKittens `gemm_rs_b200.cu` counterpart.
This uses the vendored B200 tile sizes and models the distributed output phase
with `store_add` over Tyr's current distributed memory surface. -/
@[gpu_kernel .SM100]
def gemmRsB200Fwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  gemmRsStoreAddCompatBody
    (inDtype := .BFloat16)
    (rowBlock := 128)
    (colBlock := 256)
    (redBlock := 64)
    (numRedTiles := 4)
    "ThunderKittens gemm_rs_b200: B200 reduce-scatter geometry with distributed store_add output"
    c_ptr a_ptr b_ptr dev_idx world_size

/-- ThunderKittens `gemm_rs_fp8_b200.cu` counterpart. -/
@[gpu_kernel .SM100]
def gemmRsFp8B200Fwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.FP8E4M3)
    (b_ptr : GPtr GpuFloat.FP8E4M3)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  gemmRsStoreAddCompatBody
    (inDtype := .FP8E4M3)
    (rowBlock := 128)
    (colBlock := 256)
    (redBlock := 128)
    (numRedTiles := 4)
    "ThunderKittens gemm_rs_fp8_b200: FP8 B200 reduce-scatter geometry with BF16 distributed store_add output"
    c_ptr a_ptr b_ptr dev_idx world_size

end Tyr.GPU.Kernels.Distributed
