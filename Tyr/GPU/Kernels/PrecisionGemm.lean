/-
  Tyr/GPU/Kernels/PrecisionGemm.lean

  H100-oriented FP8 GEMM kernels.

  This module now serves two roles:

  - `tkH100Fp8E4M3GemmFwd` and `tkH100Fp8ScaledGemmFwd` are the canonical
    ThunderKittens-shaped H100 FP8 surfaces, following
    `kernels/gemm/fp8_h100/*`.
  - The older mixed-precision and fused-epilogue kernels remain as
    compatibility conveniences built on the same H100 tiled mainloop; they are
    not separate ThunderKittens source ports.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.PrecisionGemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev fp8TileM : Nat := 64
private abbrev fp8TileK : Nat := 128
private abbrev fp8TileN : Nat := 256
private abbrev fp8KBlocks : Nat := 4

private def h100Fp8Accumulator {inDtype : GpuFloat}
    (banner : String)
    (aPtr : GPtr inDtype)
    (bPtr : GPtr inDtype)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 fp8TileM fp8TileN × RTileCoord) := do
  let _ := (m, n, k)
  comment banner
  comment "ThunderKittens fp8_h100 tile shape: A 64x128, B 256x128, C 64x256"

  let coord ← blockCoord2D

  let a : RT inDtype fp8TileM fp8TileK ← allocRT inDtype fp8TileM fp8TileK
  let b : RT inDtype fp8TileN fp8TileK ← allocRT inDtype fp8TileN fp8TileK
  let accum : RT GpuFloat.Float32 fp8TileM fp8TileN ← zeroRT .Float32 fp8TileM fp8TileN

  let aShared : ST inDtype fp8TileM fp8TileK ← allocST inDtype fp8TileM fp8TileK
  let bShared : ST inDtype fp8TileN fp8TileK ← allocST inDtype fp8TileN fp8TileK

  for kBlk in krange 0 fp8KBlocks do
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

private def storeConvertedTile {outDtype : GpuFloat}
    (outPtr : GPtr outDtype)
    (coord : RTileCoord)
    (src : RT GpuFloat.Float32 fp8TileM fp8TileN)
    : KernelM Unit := do
  let out : RT outDtype fp8TileM fp8TileN ← allocRT outDtype fp8TileM fp8TileN
  let outShared : ST outDtype fp8TileM fp8TileN ← allocST outDtype fp8TileM fp8TileN
  convert out src
  store outShared out
  storeGlobal outPtr outShared coord

private def storeFloat32Tile
    (outPtr : GPtr GpuFloat.Float32)
    (coord : RTileCoord)
    (src : RT GpuFloat.Float32 fp8TileM fp8TileN)
    : KernelM Unit := do
  let outShared : ST GpuFloat.Float32 fp8TileM fp8TileN ← allocST .Float32 fp8TileM fp8TileN
  store outShared src
  storeGlobal outPtr outShared coord

private def loadScaleVectors
    (scaleAPtr : GPtr GpuFloat.Float32)
    (scaleBPtr : GPtr GpuFloat.Float32)
    (coord : RTileCoord)
    : KernelM (RV GpuFloat.Float32 fp8TileM × RV GpuFloat.Float32 fp8TileN) := do
  let scaleAShared : SV GpuFloat.Float32 fp8TileM ← allocSV .Float32 fp8TileM
  let scaleBShared : SV GpuFloat.Float32 fp8TileN ← allocSV .Float32 fp8TileN
  let scaleA : RV GpuFloat.Float32 fp8TileM ← allocRV .Float32 fp8TileM
  let scaleB : RV GpuFloat.Float32 fp8TileN ← allocRV .Float32 fp8TileN
  loadVecGlobalRow scaleAShared scaleAPtr coord
  loadVecGlobalCol scaleBShared scaleBPtr coord
  loadVec scaleA scaleAShared
  loadVec scaleB scaleBShared
  pure (scaleA, scaleB)

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
  storeConvertedTile cPtr coord accum

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
  let (scaleA, scaleB) ← loadScaleVectors scaleAPtr scaleBPtr coord
  let scaled : RT GpuFloat.Float32 fp8TileM fp8TileN ← allocRT .Float32 fp8TileM fp8TileN
  mulCol scaled accum scaleA
  mulRow scaled scaled scaleB
  storeFloat32Tile cPtr coord scaled

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
  storeConvertedTile cPtr coord accum

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
  storeConvertedTile cPtr coord accum

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

  storeConvertedTile cPtr coord accum

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

  storeConvertedTile cPtr coord scaled

end Tyr.GPU.Kernels.PrecisionGemm
