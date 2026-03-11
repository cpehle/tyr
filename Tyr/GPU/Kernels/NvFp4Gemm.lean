import Tyr.GPU.Kernels.GemmCommon

/-!
  Tyr/GPU/Kernels/NvFp4Gemm.lean

  Blackwell-oriented NVFP4 GEMM surfaces.

  The ThunderKittens source for `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`
  depends on packed `fp4e2m1` storage, local half-scale tiles, global scale
  scalars, tensor-memory fragments, and cluster/TMA choreography. The Lean
  surfaces below keep the same public contracts while expressing the math
  through typed tiled mainloops and explicit scale epilogues.
-/

namespace Tyr.GPU.Kernels.NvFp4Gemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev ctaTileM : Nat := 128
private abbrev quantTileK : Nat := 256
private abbrev ctaTileN : Nat := 256
private abbrev quantScaleRows : Nat := 4
private abbrev quantKBlocks : Nat := 4

private def nvfp4Accumulator
    (aPtr : GPtr GpuFloat.FP4E2M1X2)
    (bPtr : GPtr GpuFloat.FP4E2M1X2)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 ctaTileM ctaTileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := ctaTileM)
    (tileK := quantTileK)
    (tileN := ctaTileN)
    (kBlocks := quantKBlocks)
    "=== B200 NVFP4 GEMM ==="
    "ThunderKittens nvfp4_b200 represented as a typed packed-fp4 mainloop with explicit local/global scale epilogue"
    aPtr bPtr m n k

private def mixedFp8Nvfp4Accumulator
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (wPtr : GPtr GpuFloat.FP4E2M1X2)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 ctaTileM ctaTileN × RTileCoord) := do
  let _ := (m, n, k)
  comment "=== FP8 by NVFP4 GEMM ==="
  comment "Mixed path that keeps activations in FP8 while converting packed NVFP4 weights to an MMA-friendly FP8 tile inside the typed mainloop"
  let coord ← blockCoord2D
  let a : RT GpuFloat.FP8E4M3 ctaTileM quantTileK ← allocRT .FP8E4M3 ctaTileM quantTileK
  let wFp4 : RT GpuFloat.FP4E2M1X2 ctaTileN quantTileK ← allocRT .FP4E2M1X2 ctaTileN quantTileK
  let wFp8 : RT GpuFloat.FP8E4M3 ctaTileN quantTileK ← allocRT .FP8E4M3 ctaTileN quantTileK
  let accum : RT GpuFloat.Float32 ctaTileM ctaTileN ← zeroRT .Float32 ctaTileM ctaTileN
  let aShared : ST GpuFloat.FP8E4M3 ctaTileM quantTileK ← allocST .FP8E4M3 ctaTileM quantTileK
  let wShared : ST GpuFloat.FP4E2M1X2 ctaTileN quantTileK ← allocST .FP4E2M1X2 ctaTileN quantTileK
  for kBlk in krange 0 quantKBlocks do
    let aCoord := coord.withCol kBlk.id
    let wCoord := (coord.withRow coord.c).withCol kBlk.id
    loadGlobal aShared aPtr aCoord
    loadGlobal wShared wPtr wCoord
    sync
    load a aShared
    load wFp4 wShared
    convert wFp8 wFp4
    mmaT accum a wFp8 accum
    sync
  pure (accum, coord)

/-- Canonical B200 NVFP4 surface aligned with `nvfp4_b200_gemm.cu`. -/
@[gpu_kernel .SM100]
def tkB200NvFp4GemmFwd
    (aPtr : GPtr GpuFloat.FP4E2M1X2)
    (aScalePtr : GPtr GpuFloat.Float16)
    (aGlobalScalePtr : GPtr GpuFloat.Float32)
    (bPtr : GPtr GpuFloat.FP4E2M1X2)
    (bScalePtr : GPtr GpuFloat.Float16)
    (bGlobalScalePtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← nvfp4Accumulator aPtr bPtr m n k
  let (aLocalScale, bLocalScale) ← GemmCommon.loadRowColScaleVectorsAsFloat32
    (tileM := ctaTileM)
    (tileN := ctaTileN)
    aScalePtr bScalePtr coord
  let locallyScaled ← GemmCommon.applyRowColScales accum aLocalScale bLocalScale
  let zero ← constIntVal 0 "zero_offset"
  let aGlobal ← loadFloat32Scalar aGlobalScalePtr zero "a_global_scale"
  let bGlobal ← loadFloat32Scalar bGlobalScalePtr zero "b_global_scale"
  let globalScale ← scalarMulVal aGlobal bGlobal "ab_global_scale"
  let fullyScaled ← GemmCommon.applyGlobalScalar locallyScaled globalScale
  GemmCommon.storeConvertedTile dPtr coord fullyScaled

/-- NVFP4 quantization surface aligned with the quantizer stages in
`nvfp4_b200_gemm.cu`. -/
@[gpu_kernel .SM100]
def quantizeToFp4
    (xPtr : GPtr GpuFloat.BFloat16)
    (scalePtr : GPtr GpuFloat.Float16)
    (globalScalePtr : GPtr GpuFloat.Float32)
    (xQPtr : GPtr GpuFloat.FP4E2M1X2)
    (m : KVal UInt64)
    (n : KVal UInt64)
    : KernelM Unit := do
  let _ := (m, n, quantScaleRows)
  comment "=== NVFP4 quantization ==="
  comment "Typed quantizer shell: compute per-row local absmax scales, normalize, convert to packed fp4x2, and write one global scale scalar"
  let coord ← blockCoord2D
  let xShared : ST GpuFloat.BFloat16 ctaTileM quantTileK ← allocST .BFloat16 ctaTileM quantTileK
  let xBf16 : RT GpuFloat.BFloat16 ctaTileM quantTileK ← allocRT .BFloat16 ctaTileM quantTileK
  let xF32 : RT GpuFloat.Float32 ctaTileM quantTileK ← allocRT .Float32 ctaTileM quantTileK
  let xAbs : RT GpuFloat.Float32 ctaTileM quantTileK ← allocRT .Float32 ctaTileM quantTileK
  let xNorm : RT GpuFloat.Float32 ctaTileM quantTileK ← allocRT .Float32 ctaTileM quantTileK
  let xFp4 : RT GpuFloat.FP4E2M1X2 ctaTileM quantTileK ← allocRT .FP4E2M1X2 ctaTileM quantTileK
  let xOutShared : ST GpuFloat.FP4E2M1X2 ctaTileM quantTileK ← allocST .FP4E2M1X2 ctaTileM quantTileK
  let rowScale : RV GpuFloat.Float32 ctaTileM ← allocRV .Float32 ctaTileM
  let rowScaleHalf : RV GpuFloat.Float16 ctaTileM ← allocRV .Float16 ctaTileM
  let rowScaleShared : SV GpuFloat.Float16 ctaTileM ← allocSV .Float16 ctaTileM
  loadGlobal xShared xPtr coord
  sync
  load xBf16 xShared
  convert xF32 xBf16
  abs xAbs xF32
  rowMax rowScale xAbs
  divCol xNorm xF32 rowScale
  convert xFp4 xNorm
  store xOutShared xFp4
  storeGlobal xQPtr xOutShared coord
  convertVec rowScaleHalf rowScale
  storeVec rowScaleShared rowScaleHalf
  storeVecGlobalRow scalePtr rowScaleShared coord
  let zero ← constIntVal 0 "zero_offset"
  let one ← constFloatVal 1.0 "global_scale_one"
  storeFloat32Scalar globalScalePtr zero one

/-- Mixed FP8 activation by NVFP4 weight GEMM surface aligned with the vendored
NVFP4 control-flow family. -/
@[gpu_kernel .SM100]
def mixedFp4Fp8GemmFwd
    (aPtr : GPtr GpuFloat.FP8E4M3)
    (wPtr : GPtr GpuFloat.FP4E2M1X2)
    (wScalePtr : GPtr GpuFloat.Float16)
    (wGlobalScalePtr : GPtr GpuFloat.Float32)
    (dPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← mixedFp8Nvfp4Accumulator aPtr wPtr m n k
  let weightScaleShared : SV GpuFloat.Float16 ctaTileN ← allocSV .Float16 ctaTileN
  let weightScaleHalf : RV GpuFloat.Float16 ctaTileN ← allocRV .Float16 ctaTileN
  let weightScale : RV GpuFloat.Float32 ctaTileN ← allocRV .Float32 ctaTileN
  let oneScalar ← constFloatVal 1.0 "one"
  let rowScale : RV GpuFloat.Float32 ctaTileM ← allocRV .Float32 ctaTileM
  fillVecScalar rowScale oneScalar
  loadVecGlobalCol weightScaleShared wScalePtr coord
  loadVec weightScaleHalf weightScaleShared
  convertVec weightScale weightScaleHalf
  let locallyScaled ← GemmCommon.applyRowColScales accum rowScale weightScale
  let zero ← constIntVal 0 "zero_offset"
  let globalScale ← loadFloat32Scalar wGlobalScalePtr zero "w_global_scale"
  let fullyScaled ← GemmCommon.applyGlobalScalar locallyScaled globalScale
  GemmCommon.storeConvertedTile dPtr coord fullyScaled

end Tyr.GPU.Kernels.NvFp4Gemm
