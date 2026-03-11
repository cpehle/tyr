import Tyr.GPU.Kernels.Prelude

/-!
# Tyr.GPU.Kernels.Support

Shared helpers for the concrete kernel catalog.

This module exists to keep the per-family kernel files focused on the kernel
phase structure rather than repeating the same TMA load, cross-device barrier,
and small vector helper definitions.
-/

namespace Tyr.GPU.Kernels.Support

open Tyr.GPU
open Tyr.GPU.Codegen

private def axisSelectUInt32
    (axis : KVal UInt32)
    (axis0 axis1 axis2 axis3 : KVal UInt32)
    (name : String) : KernelM (KVal UInt32) := do
  let zero ← constIntVal 0 s!"{name}_axis0"
  let one ← constIntVal 1 s!"{name}_axis1"
  let two ← constIntVal 2 s!"{name}_axis2"
  let is0 ← scalarEq axis zero s!"{name}_is0"
  let is1 ← scalarEq axis one s!"{name}_is1"
  let is2 ← scalarEq axis two s!"{name}_is2"
  let tail23 ← scalarSelect is2 axis2 axis3 s!"{name}_tail23"
  let tail123 ← scalarSelect is1 axis1 tail23 s!"{name}_tail123"
  scalarSelect is0 axis0 tail123 s!"{name}_selected"

private def axisPredicate012
    (axis : KVal UInt32)
    (name : String) : KernelM (KVal Bool × KVal Bool × KVal Bool) := do
  let zero ← constIntVal 0 s!"{name}_axis0"
  let one ← constIntVal 1 s!"{name}_axis1"
  let two ← constIntVal 2 s!"{name}_axis2"
  let is0 ← scalarEq axis zero s!"{name}_is0"
  let is1 ← scalarEq axis one s!"{name}_is1"
  let is2 ← scalarEq axis two s!"{name}_is2"
  pure (is0, is1, is2)

private def keepUnlessAxis3
    (is0 is1 is2 : KVal Bool)
    (keep : KVal UInt32)
    (replace : KVal UInt32)
    (name : String) : KernelM (KVal UInt32) := do
  let after2 ← scalarSelect is2 keep replace s!"{name}_after2"
  let after1 ← scalarSelect is1 keep after2 s!"{name}_after1"
  scalarSelect is0 keep after1 s!"{name}_after0"

/-- Async global-to-shared tile load with an explicit byte-count contract. -/
def asyncTileLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout) (src : GPtr dtype) (coord : RTileCoord)
    (bytes : Nat) : KernelM Unit := do
  let sem ← allocSemaphore
  initSemaphore sem 1
  expectBytes sem bytes
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

/-- Cross-device barrier helper used by the multi-device kernel families. -/
def barrierAllDevices (label : String) (barrierId : Nat) : KernelM Unit := do
  comment s!"Cross-device barrier: {label}"
  arriveAndWait barrierId

/-- Small vector max helper until the DSL grows a first-class `maxVec` op. -/
def maxVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Max dst.id a.id b.id)

/-- Split a row-major register tile into left/right column halves. -/
def splitCols {dtype : GpuFloat} {rows leftCols rightCols : Nat}
    (src : RT dtype rows (leftCols + rightCols) .Row)
    : KernelM (RT dtype rows leftCols .Row × RT dtype rows rightCols .Row) := do
  let left : RT dtype rows leftCols ← allocRT dtype rows leftCols
  let right : RT dtype rows rightCols ← allocRT dtype rows rightCols
  sliceCols left src 0 leftCols
  sliceCols right src leftCols rightCols
  pure (left, right)

/-- Split a row-major register tile into top/bottom row halves. -/
def splitRows {dtype : GpuFloat} {topRows bottomRows cols : Nat}
    (src : RT dtype (topRows + bottomRows) cols .Row)
    : KernelM (RT dtype topRows cols .Row × RT dtype bottomRows cols .Row) := do
  let top : RT dtype topRows cols ← allocRT dtype topRows cols
  let bottom : RT dtype bottomRows cols ← allocRT dtype bottomRows cols
  sliceRows top src 0 topRows
  sliceRows bottom src topRows bottomRows
  pure (top, bottom)

/-- Build the decay vectors used by the ThunderKittens linear-attention kernel. -/
def buildLinearAttnDecayVectors
    (slope : KVal Float32)
    : KernelM (RV GpuFloat.Float32 64 × RV GpuFloat.Float32 64 × RV GpuFloat.Float32 64) := do
  let idx : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let slopeVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let qDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let chunkScalar ← constFloatVal 64.0 "chunk"
  let chunkVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let kDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let negSlope ← scalarNeg slope "neg_slope"
  let blockNeg ← scalarMulVal negSlope chunkScalar "block_neg"
  let blockDecay ← scalarExp blockNeg "block_decay"
  let stateDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  iotaVec idx
  fillVecScalar slopeVec slope
  mulVec qDecay idx slopeVec
  scalarMulVec qDecay qDecay (-1.0)
  expVec qDecay qDecay

  fillVecScalar chunkVec chunkScalar
  subVec kDecay chunkVec idx
  mulVec kDecay kDecay slopeVec
  scalarMulVec kDecay kDecay (-1.0)
  expVec kDecay kDecay

  fillVecScalar stateDecay blockDecay
  pure (qDecay, kDecay, stateDecay)

/-- Apply the causal exponential mask used by the ThunderKittens linear-attention
kernel to a `64x64` local score tile. -/
def applyLinearAttnDecayMask
    (dst : RT GpuFloat.Float32 64 64)
    (src : RT GpuFloat.Float32 64 64)
    (qDecay : RV GpuFloat.Float32 64)
    (kDecay : RV GpuFloat.Float32 64) : KernelM Unit := do
  copy dst src
  makeCausal dst dst (some 0.0)
  mulCol dst dst qDecay
  mulRow dst dst kDecay

/-- Generic source-shaped all-to-all exchange tile used by the Ulysses and
distributed transport surfaces. `scatter_axis` and `gather_axis` follow the
ThunderKittens numbering over `(batch, depth, row_blocks, col_blocks)`. -/
def allToAllTile (label : String) (output_ptr : GPtr GpuFloat.BFloat16)
    (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (scatter_axis : KVal UInt32) (gather_axis : KVal UInt32) : KernelM Unit := do
  let _ := world_size
  comment s!"All-to-all transport for {label}"
  let rowTile ← constIntVal 16 "all_to_all_row_tile"
  let colTile ← constIntVal 128 "all_to_all_col_tile"
  let taskIdx : KVal UInt32 := ⟨← getBlockIdxX, "task_idx"⟩

  let inputBatch ← layoutBatch input_ptr "input_batch"
  let inputDepth ← layoutDepth input_ptr "input_depth"
  let inputRows ← layoutRows input_ptr "input_rows"
  let inputCols ← layoutCols input_ptr "input_cols"
  let inputRowBlocks ← scalarDivVal inputRows rowTile "input_row_blocks"
  let inputColBlocks ← scalarDivVal inputCols colTile "input_col_blocks"
  let tilesPerDepth ← scalarMulVal inputRowBlocks inputColBlocks "input_tiles_per_depth"
  let tilesPerBatch ← scalarMulVal inputDepth tilesPerDepth "input_tiles_per_batch"

  let batchIdx ← scalarDivVal taskIdx tilesPerBatch "batch_idx"
  let taskAfterBatch ← scalarMod taskIdx tilesPerBatch "task_after_batch"
  let depthIdx ← scalarDivVal taskAfterBatch tilesPerDepth "depth_idx"
  let taskAfterDepth ← scalarMod taskAfterBatch tilesPerDepth "task_after_depth"
  let rowBlockIdx ← scalarDivVal taskAfterDepth inputColBlocks "row_block_idx"
  let colBlockIdx ← scalarMod taskAfterDepth inputColBlocks "col_block_idx"

  let inputCoord := makeRTileCoord batchIdx.id depthIdx.id rowBlockIdx.id colBlockIdx.id
  let tile : ST GpuFloat.BFloat16 16 128 ← allocST .BFloat16 16 128
  asyncTileLoad tile input_ptr inputCoord (16 * 128 * GpuFloat.bytes .BFloat16)

  let outputBatch ← layoutBatch output_ptr "output_batch"
  let outputDepth ← layoutDepth output_ptr "output_depth"
  let outputRows ← layoutRows output_ptr "output_rows"
  let outputCols ← layoutCols output_ptr "output_cols"
  let outputRowBlocks ← scalarDivVal outputRows rowTile "output_row_blocks"
  let outputColBlocks ← scalarDivVal outputCols colTile "output_col_blocks"

  let scatterDim ← axisSelectUInt32
    scatter_axis batchIdx depthIdx rowBlockIdx colBlockIdx "scatter_dim"
  let scatterChunk ← axisSelectUInt32
    scatter_axis outputBatch outputDepth outputRowBlocks outputColBlocks "scatter_chunk"
  let localScatterDim ← scalarMod scatterDim scatterChunk "local_scatter_dim"

  let (scatterIs0, scatterIs1, scatterIs2) ← axisPredicate012 scatter_axis "scatter"
  let localBatch ← scalarSelect scatterIs0 localScatterDim batchIdx "local_batch"
  let localDepth ← scalarSelect scatterIs1 localScatterDim depthIdx "local_depth"
  let localRow ← scalarSelect scatterIs2 localScatterDim rowBlockIdx "local_row_block"
  let localCol ← keepUnlessAxis3
    scatterIs0 scatterIs1 scatterIs2 colBlockIdx localScatterDim "local_col_block"

  let batchOffset ← scalarMulVal inputBatch dev_idx "batch_gather_offset"
  let depthOffset ← scalarMulVal inputDepth dev_idx "depth_gather_offset"
  let rowOffset ← scalarMulVal inputRowBlocks dev_idx "row_gather_offset"
  let colOffset ← scalarMulVal inputColBlocks dev_idx "col_gather_offset"

  let batchGathered ← scalarAddVal localBatch batchOffset "batch_gathered"
  let depthGathered ← scalarAddVal localDepth depthOffset "depth_gathered"
  let rowGathered ← scalarAddVal localRow rowOffset "row_gathered"
  let colGathered ← scalarAddVal localCol colOffset "col_gathered"

  let (gatherIs0, gatherIs1, gatherIs2) ← axisPredicate012 gather_axis "gather"
  let outputBatchIdx ← scalarSelect gatherIs0 batchGathered localBatch "output_batch_idx"
  let outputDepthIdx ← scalarSelect gatherIs1 depthGathered localDepth "output_depth_idx"
  let outputRowIdx ← scalarSelect gatherIs2 rowGathered localRow "output_row_idx"
  let outputColIdx ← keepUnlessAxis3
    gatherIs0 gatherIs1 gatherIs2 localCol colGathered "output_col_idx"

  let outputCoord := makeRTileCoord
    outputBatchIdx.id outputDepthIdx.id outputRowIdx.id outputColIdx.id
  storeGlobalAsync output_ptr tile outputCoord

end Tyr.GPU.Kernels.Support
