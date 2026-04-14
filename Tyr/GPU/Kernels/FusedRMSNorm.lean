/-
  Tyr/GPU/Kernels/FusedRMSNorm.lean

  Fused residual + RMSNorm kernels.

  This is the RMSNorm sibling of `Tyr.GPU.Kernels.FusedLayerNorm`: same
  64x1024 tile shape, same residual output surface, but RMS normalization with
  a learnable weight and no bias or mean subtraction.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev tileDim : Nat := 64
private abbrev hiddenTileCount : Nat := 16
private abbrev hiddenDim : Nat := tileDim * hiddenTileCount
private def invHiddenDim : Float := 0.0009765625 -- 1 / 1024
private def rmsNormEps : Float := 1.0e-6

private def fusedRMSNormResidual1024Bf16Body
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "Fused residual + RMSNorm on 64x1024 tiles"

  let coord ← blockCoord2D

  let x ← allocRT .BFloat16 tileDim tileDim
  let residual ← allocRT .BFloat16 tileDim tileDim
  let outResid ← allocRT .BFloat16 tileDim tileDim
  let outResidF ← allocRT .Float32 tileDim tileDim
  let sq ← allocRT .Float32 tileDim tileDim

  let tileShared ← allocST .BFloat16 tileDim tileDim

  let sqSum ← zeroRV .Float32 tileDim
  let tmpSqSum ← allocRV .Float32 tileDim
  let meanSquare ← allocRV .Float32 tileDim
  let invRms ← allocRV .Float32 tileDim

  let weightShared ← allocSV .BFloat16 tileDim
  let weightRV ← allocRV .BFloat16 tileDim
  let weightF ← allocRV .Float32 tileDim

  comment s!"Pass 1: out_resid and RMS statistics across full hidden dimension ({hiddenDim})"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    convert outResidF outResid
    mul sq outResidF outResidF
    rowSum tmpSqSum sq
    addVec sqSum sqSum tmpSqSum

  scalarMulVec meanSquare sqSum invHiddenDim
  scalarAddVec meanSquare meanSquare rmsNormEps
  rsqrtVec invRms meanSquare

  comment "Pass 2: normalize and apply RMSNorm weight"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared
    convert outResidF outResid

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared
    convertVec weightF weightRV

    mulCol outResidF outResidF invRms
    mulRow outResidF outResidF weightF

    convert outResid outResidF
    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

private def fusedRMSNormResidual1024F32Body
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "Float32 fused residual + RMSNorm on 64x1024 tiles"

  let coord ← blockCoord2D

  let x ← allocRT .Float32 tileDim tileDim
  let residual ← allocRT .Float32 tileDim tileDim
  let outResid ← allocRT .Float32 tileDim tileDim
  let sq ← allocRT .Float32 tileDim tileDim

  let tileShared ← allocST .Float32 tileDim tileDim

  let sqSum ← zeroRV .Float32 tileDim
  let tmpSqSum ← allocRV .Float32 tileDim
  let meanSquare ← allocRV .Float32 tileDim
  let invRms ← allocRV .Float32 tileDim

  let weightShared ← allocSV .Float32 tileDim
  let weightRV ← allocRV .Float32 tileDim

  comment s!"Pass 1: out_resid and RMS statistics across full hidden dimension ({hiddenDim})"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    mul sq outResid outResid
    rowSum tmpSqSum sq
    addVec sqSum sqSum tmpSqSum

  scalarMulVec meanSquare sqSum invHiddenDim
  scalarAddVec meanSquare meanSquare rmsNormEps
  rsqrtVec invRms meanSquare

  comment "Pass 2: normalize and apply RMSNorm weight"
  for hiddenIdx in krange 0 hiddenTileCount do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared

    mulCol outResid outResid invRms
    mulRow outResid outResid weightRV

    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

/-- Fused residual + RMSNorm kernel for `d_model = 1024` (`16 x 64` hidden tiles). -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  fusedRMSNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the canonical bf16 fused RMSNorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024Blackwell
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  fusedRMSNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Float32 variant of `tkFusedRMSNormResidual1024`. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024F32
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  fusedRMSNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the f32 fused RMSNorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedRMSNormResidual1024F32Blackwell
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  fusedRMSNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr out_ptr out_resid_ptr

end Tyr.GPU.Kernels
