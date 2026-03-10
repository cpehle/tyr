/-
  Tyr/GPU/Kernels/NvFp4Gemm.lean

  Blackwell-oriented NVFP4 GEMM compatibility surfaces.

  The ThunderKittens source for `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`
  depends on packed `fp4e2m1` storage, tensor-memory fragments, and cluster/TMA
  choreography that the current Lean DSL cannot represent directly. This module
  therefore makes that status explicit:

  - `tkB200NvFp4GemmCompatFwd` is the canonical B200/NVFP4 control-flow surface
    in Tyr today, but it is compatibility-only because `GpuFloat` does not yet
    model packed NVFP4 values.
  - The helper kernels below are also compatibility kernels and say so in their
    names; none of them should be read as native FP8 GEMMs.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.NvFp4Gemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev ctaTileM : Nat := 128
private abbrev quantTileK : Nat := 256
private abbrev ctaTileN : Nat := 256
private abbrev quantScaleRows : Nat := 4
private abbrev quantKBlocks : Nat := 4

private def b200NvFp4CompatAccumulator
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (aScalePtr : GPtr GpuFloat.Float16)
    (bScalePtr : GPtr GpuFloat.Float16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 ctaTileM ctaTileN × RTileCoord) := do
  let _ := (m, n, k)
  comment "=== B200 NVFP4 compatibility GEMM ==="
  comment "CTA-local view of the ThunderKittens producer/consumer NVFP4 pipeline"
  comment "Packed fp4e2m1 tiles are represented with FP8E4M3 proxy storage until the DSL grows native NVFP4 types"

  let coord ← blockCoord2D

  let a : RT GpuFloat.FP8E4M3 ctaTileM quantTileK ← allocRT .FP8E4M3 ctaTileM quantTileK
  let b : RT GpuFloat.FP8E4M3 ctaTileN quantTileK ← allocRT .FP8E4M3 ctaTileN quantTileK
  let accum : RT GpuFloat.Float32 ctaTileM ctaTileN ← zeroRT .Float32 ctaTileM ctaTileN

  let aScale : RT GpuFloat.Float16 quantScaleRows quantTileK ← allocRT .Float16 quantScaleRows quantTileK
  let bScale : RT GpuFloat.Float16 quantScaleRows quantTileK ← allocRT .Float16 quantScaleRows quantTileK

  let aShared : ST GpuFloat.FP8E4M3 ctaTileM quantTileK ← allocST .FP8E4M3 ctaTileM quantTileK
  let bShared : ST GpuFloat.FP8E4M3 ctaTileN quantTileK ← allocST .FP8E4M3 ctaTileN quantTileK
  let aScaleShared : ST GpuFloat.Float16 quantScaleRows quantTileK ← allocST .Float16 quantScaleRows quantTileK
  let bScaleShared : ST GpuFloat.Float16 quantScaleRows quantTileK ← allocST .Float16 quantScaleRows quantTileK

  for kBlk in krange 0 quantKBlocks do
    let aCoord := coord.withCol kBlk.id
    let bCoord := (coord.withRow coord.c).withCol kBlk.id
    loadGlobal aShared aPtr aCoord
    loadGlobal bShared bPtr bCoord
    loadGlobal aScaleShared aScalePtr aCoord
    loadGlobal bScaleShared bScalePtr bCoord
    sync
    load a aShared
    load b bShared
    load aScale aScaleShared
    load bScale bScaleShared
    let _ := (aScale, bScale)
    comment "Per-block NVFP4 scales are explicitly staged here; in the real kernel they feed the tensor-memory dequant path"
    mmaT accum a b accum
    sync

  pure (accum, coord)

private def storeCompatOutput
    (outPtr : GPtr GpuFloat.BFloat16)
    (coord : RTileCoord)
    (src : RT GpuFloat.Float32 ctaTileM ctaTileN)
    : KernelM Unit := do
  let out : RT GpuFloat.BFloat16 ctaTileM ctaTileN ← allocRT .BFloat16 ctaTileM ctaTileN
  let outShared : ST GpuFloat.BFloat16 ctaTileM ctaTileN ← allocST .BFloat16 ctaTileM ctaTileN
  convert out src
  store outShared out
  storeGlobal outPtr outShared coord

/-- Canonical B200 NVFP4 control-flow surface for the current DSL.

This keeps the ThunderKittens block geometry and explicit scale staging, while
remaining compatibility-only because the Lean type system still lacks packed
`fp4e2m1` storage and tensor-memory fragments. -/
@[gpu_kernel .SM100]
def tkB200NvFp4GemmCompatFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (bPtr : GPtr GpuFloat.FP8E4M3)
    (aScalePtr : GPtr GpuFloat.Float16)
    (bScalePtr : GPtr GpuFloat.Float16)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200NvFp4CompatAccumulator aPtr bPtr aScalePtr bScalePtr m n k
  comment "Consumer epilogue: materialize the CTA-local BF16 tile"
  storeCompatOutput dPtr coord accum

/-- Compatibility quantizer that produces the FP8 proxy storage plus the
row-wise scale vector needed by the NVFP4 compatibility kernels. -/
@[gpu_kernel .SM100]
def quantizeToFp4Compat
    (xPtr : GPtr GpuFloat.BFloat16)
    (scalePtr : GPtr GpuFloat.Float32)
    (xQPtr : GPtr GpuFloat.FP8E4M3)
    (m : KVal UInt64)
    (n : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n)
  comment "=== Compatibility quantization to NVFP4 proxy storage ==="
  comment "Computes a row-wise scale, normalizes by the NVFP4 max magnitude (6.0), then stores FP8 proxy values"

  let coord ← blockCoord2D

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let absX : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let normalized : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let xQ : RT GpuFloat.FP8E4M3 64 64 ← allocRT .FP8E4M3 64 64

  let absMax : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let scale : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let safeScale : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let xQShared : ST GpuFloat.FP8E4M3 64 64 ← allocST .FP8E4M3 64 64
  let scaleShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  loadGlobal xShared xPtr coord
  sync
  load x xShared
  convert xF x
  abs absX xF
  rowMax absMax absX
  scalarMulVec scale absMax 0.16666666666666666
  scalarAddVec safeScale scale 1.0e-6
  divCol normalized xF safeScale
  convert xQ normalized

  store xQShared xQ
  storeGlobal xQPtr xQShared coord
  storeVec scaleShared safeScale
  storeVecGlobalRow scalePtr scaleShared coord

/-- Compatibility-only mixed GEMM where activations stay in FP8 and the weight
matrix uses the NVFP4 proxy storage plus a per-column dequant vector. -/
@[gpu_kernel .SM100]
def mixedFp4Fp8CompatGemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (wPtr : GPtr GpuFloat.FP8E4M3)
    (wScalePtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, k)
  comment "=== FP8 by NVFP4-proxy weight GEMM (compatibility) ==="

  let coord ← blockCoord2D

  let a : RT GpuFloat.FP8E4M3 64 128 ← allocRT .FP8E4M3 64 128
  let w : RT GpuFloat.FP8E4M3 256 128 ← allocRT .FP8E4M3 256 128
  let accum : RT GpuFloat.Float32 64 256 ← zeroRT .Float32 64 256

  let wScale : RV GpuFloat.Float32 256 ← allocRV .Float32 256

  let aShared : ST GpuFloat.FP8E4M3 64 128 ← allocST .FP8E4M3 64 128
  let wShared : ST GpuFloat.FP8E4M3 256 128 ← allocST .FP8E4M3 256 128
  let wScaleShared : SV GpuFloat.Float32 256 ← allocSV .Float32 256

  loadVecGlobalCol wScaleShared wScalePtr coord
  loadVec wScale wScaleShared

  for kBlk in krange 0 4 do
    let aCoord := coord.withCol kBlk.id
    let wCoord := (coord.withRow coord.c).withCol kBlk.id
    loadGlobal aShared aPtr aCoord
    loadGlobal wShared wPtr wCoord
    sync
    load a aShared
    load w wShared
    mmaT accum a w accum
    sync

  let dequantized : RT GpuFloat.Float32 64 256 ← allocRT .Float32 64 256
  mulRow dequantized accum wScale
  let out : RT GpuFloat.BFloat16 64 256 ← allocRT .BFloat16 64 256
  let outShared : ST GpuFloat.BFloat16 64 256 ← allocST .BFloat16 64 256
  convert out dequantized
  store outShared out
  storeGlobal dPtr outShared coord

/-- Backwards-compatible short name for the canonical B200 compatibility
surface. -/
abbrev nvfp4GemmFwd := tkB200NvFp4GemmCompatFwd

/-- Backwards-compatible short name for the NVFP4 compatibility quantizer. -/
abbrev quantizeToFp4 := quantizeToFp4Compat

/-- Backwards-compatible short name for the mixed compatibility surface. -/
abbrev mixedFp4Fp8GemmFwd := mixedFp4Fp8CompatGemmFwd

end Tyr.GPU.Kernels.NvFp4Gemm
