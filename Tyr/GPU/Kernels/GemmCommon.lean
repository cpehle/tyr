import Tyr.GPU.Kernels.Prelude

/-!
  Tyr/GPU/Kernels/GemmCommon.lean

  Internal helpers shared across the GEMM-family kernel leaves.

  These utilities intentionally model the tiled, CTA-local view that the Lean
  GPU DSL can represent today: register/shared-memory tiles, FP32
  accumulation, and explicit row/column scale application. More detailed
  ThunderKittens features like TMEM fragments, packed NVFP4 storage, or e8m0
  scale tiles remain the responsibility of the compatibility layers that call
  into these helpers.
-/

namespace Tyr.GPU.Kernels.GemmCommon

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Generic CTA-local tiled accumulator used by the GEMM leaf modules. -/
def tiledAccumulator {tileM tileK tileN kBlocks : Nat} {inDtype : GpuFloat}
    (banner : String)
    (sourceNote : String)
    (aPtr : GPtr inDtype)
    (bPtr : GPtr inDtype)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 tileM tileN × RTileCoord) := do
  let _ := (m, n, k)
  comment banner
  comment sourceNote
  comment s!"Tile shape: A {tileM}x{tileK}, B {tileN}x{tileK}, C {tileM}x{tileN}"

  let coord ← blockCoord2D

  let a : RT inDtype tileM tileK ← allocRT inDtype tileM tileK
  let b : RT inDtype tileN tileK ← allocRT inDtype tileN tileK
  let accum : RT GpuFloat.Float32 tileM tileN ← zeroRT .Float32 tileM tileN

  let aShared : ST inDtype tileM tileK ← allocST inDtype tileM tileK
  let bShared : ST inDtype tileN tileK ← allocST inDtype tileN tileK

  for kBlk in krange 0 kBlocks do
    let aCoord := coord.withCol kBlk.id
    let bCoord := (coord.withRow coord.c).withCol kBlk.id
    loadGlobal aShared aPtr aCoord
    loadGlobal bShared bPtr bCoord
    sync
    load a aShared
    load b bShared
    mmaT accum a b accum
    sync

  pure (accum, coord)

/-- Convert an FP32 accumulator tile into the requested output dtype and store
it back to global memory. -/
def storeConvertedTile {outDtype : GpuFloat} {tileM tileN : Nat}
    (outPtr : GPtr outDtype)
    (coord : RTileCoord)
    (src : RT GpuFloat.Float32 tileM tileN)
    : KernelM Unit := do
  let out : RT outDtype tileM tileN ← allocRT outDtype tileM tileN
  let outShared : ST outDtype tileM tileN ← allocST outDtype tileM tileN
  convert out src
  store outShared out
  storeGlobal outPtr outShared coord

/-- Store an FP32 tile to global memory without an intermediate type
conversion. -/
def storeFloat32Tile {tileM tileN : Nat}
    (outPtr : GPtr GpuFloat.Float32)
    (coord : RTileCoord)
    (src : RT GpuFloat.Float32 tileM tileN)
    : KernelM Unit := do
  let outShared : ST GpuFloat.Float32 tileM tileN ← allocST .Float32 tileM tileN
  store outShared src
  storeGlobal outPtr outShared coord

/-- Load row and column scale vectors that line up with the current output tile
coordinate. -/
def loadRowColScaleVectors {tileM tileN : Nat}
    (scaleRowPtr : GPtr GpuFloat.Float32)
    (scaleColPtr : GPtr GpuFloat.Float32)
    (coord : RTileCoord)
    : KernelM (RV GpuFloat.Float32 tileM × RV GpuFloat.Float32 tileN) := do
  let scaleRowShared : SV GpuFloat.Float32 tileM ← allocSV .Float32 tileM
  let scaleColShared : SV GpuFloat.Float32 tileN ← allocSV .Float32 tileN
  let scaleRow : RV GpuFloat.Float32 tileM ← allocRV .Float32 tileM
  let scaleCol : RV GpuFloat.Float32 tileN ← allocRV .Float32 tileN
  loadVecGlobalRow scaleRowShared scaleRowPtr coord
  loadVecGlobalCol scaleColShared scaleColPtr coord
  loadVec scaleRow scaleRowShared
  loadVec scaleCol scaleColShared
  pure (scaleRow, scaleCol)

/-- Load row and column scale vectors from arbitrary floating dtypes and convert
them to Float32 for epilogue math. -/
def loadRowColScaleVectorsAsFloat32
    {tileM tileN : Nat}
    {scaleRowDtype scaleColDtype : GpuFloat}
    (scaleRowPtr : GPtr scaleRowDtype)
    (scaleColPtr : GPtr scaleColDtype)
    (coord : RTileCoord)
    : KernelM (RV GpuFloat.Float32 tileM × RV GpuFloat.Float32 tileN) := do
  let scaleRowShared : SV scaleRowDtype tileM ← allocSV scaleRowDtype tileM
  let scaleColShared : SV scaleColDtype tileN ← allocSV scaleColDtype tileN
  let scaleRowNative : RV scaleRowDtype tileM ← allocRV scaleRowDtype tileM
  let scaleColNative : RV scaleColDtype tileN ← allocRV scaleColDtype tileN
  let scaleRow : RV GpuFloat.Float32 tileM ← allocRV .Float32 tileM
  let scaleCol : RV GpuFloat.Float32 tileN ← allocRV .Float32 tileN
  loadVecGlobalRow scaleRowShared scaleRowPtr coord
  loadVecGlobalCol scaleColShared scaleColPtr coord
  loadVec scaleRowNative scaleRowShared
  loadVec scaleColNative scaleColShared
  convertVec scaleRow scaleRowNative
  convertVec scaleCol scaleColNative
  pure (scaleRow, scaleCol)

/-- Apply per-row and per-column scales to an FP32 accumulator tile. -/
def applyRowColScales {tileM tileN : Nat}
    (accum : RT GpuFloat.Float32 tileM tileN)
    (scaleRow : RV GpuFloat.Float32 tileM)
    (scaleCol : RV GpuFloat.Float32 tileN)
    : KernelM (RT GpuFloat.Float32 tileM tileN) := do
  let scaled : RT GpuFloat.Float32 tileM tileN ← allocRT .Float32 tileM tileN
  mulCol scaled accum scaleRow
  mulRow scaled scaled scaleCol
  pure scaled

/-- Apply one runtime Float32 scalar to an FP32 accumulator tile by splatting it
across the output columns. -/
def applyGlobalScalar {tileM tileN : Nat}
    (accum : RT GpuFloat.Float32 tileM tileN)
    (scalar : KVal Float32)
    : KernelM (RT GpuFloat.Float32 tileM tileN) := do
  let onesScalar ← constFloatVal 1.0 "one"
  let scaleRow : RV GpuFloat.Float32 tileM ← allocRV .Float32 tileM
  let scaleCol : RV GpuFloat.Float32 tileN ← allocRV .Float32 tileN
  fillVecScalar scaleRow onesScalar
  fillVecScalar scaleCol scalar
  applyRowColScales accum scaleRow scaleCol

end Tyr.GPU.Kernels.GemmCommon
