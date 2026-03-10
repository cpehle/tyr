/-
  Tyr/GPU/Kernels/PrecisionGemm.lean

  FP8-family GEMM kernels.

  This module now serves two roles:

  - `tkH100Fp8E4M3GemmFwd` and `tkH100Fp8ScaledGemmFwd` are the canonical
    ThunderKittens-shaped H100 FP8 surfaces, following
    `kernels/gemm/fp8_h100/*`.
  - `tkB200Fp8E4M3Gemm1CtaCompatFwd`, `tkB200Fp8E4M3Gemm2CtaCompatFwd`, and
    `tkB200MxFp8GemmCompatFwd` are Blackwell-oriented compatibility surfaces
    for the vendored `fp8_b200/*` and `mxfp8_b200/*` kernels.
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

private def b200Fp8CompatAccumulator {inDtype : GpuFloat}
    (banner : String)
    (sourceNote : String)
    (aPtr : GPtr inDtype)
    (bPtr : GPtr inDtype)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 b200TileM b200TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := b200TileM)
    (tileK := b200TileK)
    (tileN := b200TileN)
    (kBlocks := b200KBlocks)
    banner
    sourceNote
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

/-! ## Blackwell Compatibility Surfaces -/

/-- Blackwell-compatible surface corresponding to `fp8_b200_gemm_1cta.cu`.

The vendored kernel is a single-CTA Blackwell kernel with TMEM-backed output
staging. The current Lean DSL does not model TMEM explicitly, so this surface
keeps the source-aligned 128x256x128 tile geometry while remaining explicit
about its compatibility status. -/
@[gpu_kernel .SM100]
def tkB200Fp8E4M3Gemm1CtaCompatFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200Fp8CompatAccumulator
    "=== B200 FP8 GEMM (1 CTA compatibility) ==="
    "ThunderKittens fp8_b200_gemm_1cta shape, flattened to the CTA-local tiled operations available in the Lean DSL"
    aPtr bPtr m n k
  GemmCommon.storeConvertedTile dPtr coord accum

/-- Blackwell-compatible surface corresponding to `fp8_b200_gemm_2cta.cu`.

The source kernel splits the B tile and output handoff across a two-CTA
cluster. This compatibility surface preserves the same logical output tile and
input dtypes, but collapses the cluster choreography into one CTA-local
register/shared-memory view. -/
@[gpu_kernel .SM100]
def tkB200Fp8E4M3Gemm2CtaCompatFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200Fp8CompatAccumulator
    "=== B200 FP8 GEMM (2 CTA compatibility) ==="
    "ThunderKittens fp8_b200_gemm_2cta tile family, represented as a single CTA-local compatibility surface because cluster/TMEM exchange is not modeled directly"
    aPtr bPtr m n k
  GemmCommon.storeConvertedTile dPtr coord accum

/-- Blackwell-compatible MXFP8 surface corresponding to
`mxfp8_b200_gemm.cu`.

The source kernel uses packed `fp8e8m0` scale tiles. Since the current Lean
DSL only models `FP8E4M3` and `FP8E5M2`, this surface takes explicit Float32
row/column scale vectors as compatibility stand-ins and applies them after FP32
accumulation. -/
@[gpu_kernel .SM100]
def tkB200MxFp8GemmCompatFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (scaleAPtr : GPtr GpuFloat.Float32)
    (scaleBPtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200Fp8CompatAccumulator
    "=== B200 MXFP8 GEMM (compatibility) ==="
    "ThunderKittens mxfp8_b200 tile family with row/column Float32 scale proxies standing in for packed e8m0 scale tiles"
    aPtr bPtr m n k
  let (scaleA, scaleB) ← GemmCommon.loadRowColScaleVectors
    (tileM := b200TileM)
    (tileN := b200TileN)
    scaleAPtr scaleBPtr coord
  let scaled ← GemmCommon.applyRowColScales accum scaleA scaleB
  GemmCommon.storeConvertedTile dPtr coord scaled

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
