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

private def rawLines (lines : Array String) : KernelM Unit :=
  emitRaw (String.intercalate "\n" lines.toList)

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
    (dev_idx : KVal UInt32) (_world_size : KVal UInt32)
    (scatter_axis : KVal UInt32) (gather_axis : KVal UInt32) : KernelM Unit := do
  comment s!"All-to-all transport for {label}"
  rawLines #[
    "using shared_tile = st_bf<16, 128>;",
    s!"auto &output = {output_ptr.id.toIdent};",
    s!"auto &input = {input_ptr.id.toIdent};",
    s!"const int dev_idx = {dev_idx.id.toIdent};",
    s!"const int scatter_axis = {scatter_axis.id.toIdent};",
    s!"const int gather_axis = {gather_axis.id.toIdent};",
    "extern __shared__ int __shm[];",
    "tma_swizzle_allocator allocator((int*)&__shm[0]);",
    "shared_tile &tile = allocator.allocate<shared_tile>();",
    "__shared__ semaphore arrived;",
    "init_semaphore(arrived, 0, 1);",
    "int task_idx = blockIdx.x;",
    "int batch_idx = task_idx / (input.depth() * (input.rows() / 16) * (input.cols() / 128));",
    "task_idx %= (input.depth() * (input.rows() / 16) * (input.cols() / 128));",
    "int depth_idx = task_idx / ((input.rows() / 16) * (input.cols() / 128));",
    "task_idx %= ((input.rows() / 16) * (input.cols() / 128));",
    "int row_block_idx = task_idx / (input.cols() / 128);",
    "int col_block_idx = task_idx % (input.cols() / 128);",
    "tma::expect_bytes(arrived, sizeof(tile));",
    "tma::load_async(tile, input, {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);",
    "int dst_dev_idx = 0;",
    "if (scatter_axis == 0) { dst_dev_idx = batch_idx / output.batch(); batch_idx %= output.batch(); }",
    "else if (scatter_axis == 1) { dst_dev_idx = depth_idx / output.depth(); depth_idx %= output.depth(); }",
    "else if (scatter_axis == 2) { dst_dev_idx = row_block_idx / (output.rows() / 16); row_block_idx %= (output.rows() / 16); }",
    "else { dst_dev_idx = col_block_idx / (output.cols() / 128); col_block_idx %= (output.cols() / 128); }",
    "if (gather_axis == 0) batch_idx += input.batch() * dev_idx;",
    "else if (gather_axis == 1) depth_idx += input.depth() * dev_idx;",
    "else if (gather_axis == 2) row_block_idx += (input.rows() / 16) * dev_idx;",
    "else col_block_idx += (input.cols() / 128) * dev_idx;",
    "wait(arrived, 0);",
    "tma::store_async(output, tile, {batch_idx, depth_idx, row_block_idx, col_block_idx});"
  ]

end Tyr.GPU.Kernels.Support
