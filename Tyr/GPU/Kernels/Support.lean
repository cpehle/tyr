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

/-- Generic 16x128 all-to-all exchange tile used by the Ulysses and distributed
transport surfaces. -/
def allToAllTile (label : String) (output_ptr : GPtr GpuFloat.BFloat16)
    (input_ptr : GPtr GpuFloat.BFloat16) (coord : RTileCoord)
    (storeAdd : Bool := false) : KernelM Unit := do
  let tileRows : Nat := 16
  let tileCols : Nat := 128
  let shard : RT GpuFloat.BFloat16 tileRows tileCols ← allocRT .BFloat16 tileRows tileCols
  let inputShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols
  let exchangeShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols

  comment s!"All-to-all transport for {label}"
  asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * 2)
  load shard inputShared
  multimemStore exchangeShared shard
  barrierAllDevices s!"{label} exchange complete" 0
  if storeAdd then
    storeGlobalAdd output_ptr exchangeShared coord
  else
    storeGlobalAsync output_ptr exchangeShared coord
  sync

end Tyr.GPU.Kernels.Support
