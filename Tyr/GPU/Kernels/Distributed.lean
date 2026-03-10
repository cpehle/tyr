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

/-! ## Concrete Collectives -/

/-- ThunderKittens-style all-gather: stage a local shard, publish via multimem,
then write the gathered view through the caller-provided output layout. -/
@[gpu_kernel .SM90]
def allGatherFwd (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let tileRows : Nat := 128
  let tileCols : Nat := 128
  let coord ← blockCoord2D

  let localShard : RT GpuFloat.BFloat16 tileRows tileCols ← allocRT .BFloat16 tileRows tileCols
  let gathered : RT GpuFloat.BFloat16 tileRows tileCols ← allocRT .BFloat16 tileRows tileCols

  let inputShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols
  let multicastShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols
  let outputShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols

  comment "ThunderKittens all_gather: load local shard -> multimem publish -> barrier -> store gathered view"
  Support.asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * 2)
  load localShard inputShared
  multimemStore multicastShared localShard
  Support.barrierAllDevices "all_gather publish complete" 0

  load gathered multicastShared
  store outputShared gathered
  storeGlobal output_ptr outputShared coord
  Support.barrierAllDevices "all_gather epilogue" 1

/-- ThunderKittens-style all-reduce: multimem reduce, republish the reduced tile,
then store through the local output view. -/
@[gpu_kernel .SM90]
def allReduceFwd (output_ptr : GPtr GpuFloat.Float32) (input_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let tileRows : Nat := 128
  let tileCols : Nat := 128
  let coord ← blockCoord2D

  let reduced : RT GpuFloat.Float32 tileRows tileCols ← allocRT .Float32 tileRows tileCols

  let inputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let publishShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let outputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols

  comment "ThunderKittens all_reduce: ld_reduce over multicast source, then publish the reduced result"
  Support.asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * 4)
  multimemLoadReduce reduced inputShared .Sum
  multimemStore publishShared reduced
  Support.barrierAllDevices "all_reduce publish complete" 0

  store outputShared reduced
  storeGlobal output_ptr outputShared coord
  Support.barrierAllDevices "all_reduce epilogue" 1

/-- ThunderKittens-style reduce-scatter: reduce a multicast source and store the
caller-selected shard view. -/
@[gpu_kernel .SM90]
def reduceScatterFwd (output_ptr : GPtr GpuFloat.Float32) (input_ptr : GPtr GpuFloat.Float32)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let tileRows : Nat := 128
  let tileCols : Nat := 128
  let coord ← blockCoord2D

  let reduced : RT GpuFloat.Float32 tileRows tileCols ← allocRT .Float32 tileRows tileCols
  let inputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols
  let outputShared : ST GpuFloat.Float32 tileRows tileCols ← allocST .Float32 tileRows tileCols

  comment "ThunderKittens reduce_scatter: reduce from multimem view, store only the local shard"
  Support.asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * 4)
  multimemLoadReduce reduced inputShared .Sum
  Support.barrierAllDevices "reduce_scatter reduction complete" 0

  store outputShared reduced
  storeGlobal output_ptr outputShared coord
  Support.barrierAllDevices "reduce_scatter epilogue" 1

/-- Concrete all-to-all surface for head-parallel -> sequence-parallel transport.
The caller supplies already-partitioned input/output views; this kernel owns the
TMA load, barrier, and store choreography. -/
@[gpu_kernel .SM90]
def allToAllHeadsToSeq (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "ThunderKittens all_to_all<scatter=head, gather=seq>"
  Support.allToAllTile "heads->seq shard" output_ptr input_ptr coord
  Support.barrierAllDevices "all_to_all heads->seq epilogue" 1

/-- Concrete all-to-all surface for sequence-parallel -> head-parallel transport. -/
@[gpu_kernel .SM90]
def allToAllSeqToHeads (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "ThunderKittens all_to_all<scatter=seq, gather=head>"
  Support.allToAllTile "seq->heads shard" output_ptr input_ptr coord
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
  let _ := (dev_idx, world_size, num_comm_sms)
  let rowBlock : Nat := 128
  let colBlock : Nat := 128
  let redBlock : Nat := 64
  let numRedTiles : Nat := 4
  let coord ← blockCoord2D

  let aStage : ST GpuFloat.BFloat16 rowBlock redBlock ← allocST .BFloat16 rowBlock redBlock
  let aGatherStage : ST GpuFloat.BFloat16 rowBlock redBlock ← allocST .BFloat16 rowBlock redBlock
  let bStage : ST GpuFloat.BFloat16 redBlock colBlock .Col ← allocST .BFloat16 redBlock colBlock .Col
  let cStage : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment "ThunderKittens ag_gemm_h100: comm producer all-gathers A, consumer performs MMA"
  ifWarpGroup 0 do
    for redIdx in krange 0 numRedTiles do
      expectBytes semA (rowBlock * redBlock * 2)
      loadGlobalAsync aStage a_local_ptr (coord.withCol redIdx.id) semA.id
      waitSemaphore semA
      let aLocal : RT GpuFloat.BFloat16 rowBlock redBlock ← allocRT .BFloat16 rowBlock redBlock
      load aLocal aStage
      multimemStore aGatherStage aLocal
      Support.barrierAllDevices "ag_gemm all-gather row shard ready" 0

      expectBytes semB (redBlock * colBlock * 2)
      loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
      waitSemaphore semB
      namedBarrierArrive 2 128

  ifWarpGroup 1 do
    let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
    for _redIdx in krange 0 numRedTiles do
      namedBarrierSync 2 128
      let aGathered : RT GpuFloat.BFloat16 rowBlock redBlock ← allocRT .BFloat16 rowBlock redBlock
      let bTile : RT GpuFloat.BFloat16 redBlock colBlock .Col ← allocRT .BFloat16 redBlock colBlock .Col
      load aGathered aGatherStage
      load bTile bStage
      mma cAcc aGathered bTile cAcc
      sync

    let out : RT GpuFloat.BFloat16 rowBlock colBlock ← allocRT .BFloat16 rowBlock colBlock
    convert out cAcc
    store cStage out
    storeGlobal c_ptr cStage coord

/-- GEMM + AllReduce.
Consumer warpgroups compute local partial `C`; producer warpgroup reduces the
tile with multimem and stores the replicated result. -/
@[gpu_kernel .SM90]
def gemmArFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) (num_comm_sms : KVal UInt32)
    : KernelM Unit := do
  let _ := (dev_idx, world_size, num_comm_sms)
  let rowBlock : Nat := 128
  let colBlock : Nat := 128
  let redBlock : Nat := 64
  let numRedTiles : Nat := 4
  let coord ← blockCoord2D

  let aStage : ST GpuFloat.BFloat16 rowBlock redBlock ← allocST .BFloat16 rowBlock redBlock
  let bStage : ST GpuFloat.BFloat16 redBlock colBlock .Col ← allocST .BFloat16 redBlock colBlock .Col
  let partialShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let reducedShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment "ThunderKittens gemm_ar_h100: compute tile locally, then in-network all-reduce"
  ifWarpGroup 1 do
    let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
    for redIdx in krange 0 numRedTiles do
      expectBytes semA (rowBlock * redBlock * 2)
      loadGlobalAsync aStage a_ptr (coord.withCol redIdx.id) semA.id
      waitSemaphore semA
      expectBytes semB (redBlock * colBlock * 2)
      loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
      waitSemaphore semB

      let aTile : RT GpuFloat.BFloat16 rowBlock redBlock ← allocRT .BFloat16 rowBlock redBlock
      let bTile : RT GpuFloat.BFloat16 redBlock colBlock .Col ← allocRT .BFloat16 redBlock colBlock .Col
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

/-- GEMM + ReduceScatter.
Consumer warpgroups compute the local tile; producer warpgroup performs the
reduce-scatter write into the caller-provided sharded output view. -/
@[gpu_kernel .SM90]
def gemmRsFwd (c_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.BFloat16)
    (b_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let rowBlock : Nat := 128
  let colBlock : Nat := 128
  let redBlock : Nat := 64
  let numRedTiles : Nat := 4
  let coord ← blockCoord2D

  let aStage : ST GpuFloat.BFloat16 rowBlock redBlock ← allocST .BFloat16 rowBlock redBlock
  let bStage : ST GpuFloat.BFloat16 redBlock colBlock .Col ← allocST .BFloat16 redBlock colBlock .Col
  let partialShared : ST GpuFloat.Float32 rowBlock colBlock ← allocST .Float32 rowBlock colBlock
  let outShared : ST GpuFloat.BFloat16 rowBlock colBlock ← allocST .BFloat16 rowBlock colBlock

  let semA ← allocSemaphore
  let semB ← allocSemaphore
  initSemaphore semA 1
  initSemaphore semB 1

  comment "ThunderKittens gemm_rs_h100: compute local MMA, reduce-scatter to the sharded output view"
  let cAcc : RT GpuFloat.Float32 rowBlock colBlock ← zeroRT .Float32 rowBlock colBlock
  for redIdx in krange 0 numRedTiles do
    expectBytes semA (rowBlock * redBlock * 2)
    loadGlobalAsync aStage a_ptr (coord.withCol redIdx.id) semA.id
    waitSemaphore semA
    expectBytes semB (redBlock * colBlock * 2)
    loadGlobalAsync bStage b_ptr (coord.withRow redIdx.id) semB.id
    waitSemaphore semB

    let aTile : RT GpuFloat.BFloat16 rowBlock redBlock ← allocRT .BFloat16 rowBlock redBlock
    let bTile : RT GpuFloat.BFloat16 redBlock colBlock .Col ← allocRT .BFloat16 redBlock colBlock .Col
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

end Tyr.GPU.Kernels.Distributed
