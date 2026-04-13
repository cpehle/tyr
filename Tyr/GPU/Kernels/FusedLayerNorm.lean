/-
  Tyr/GPU/Kernels/FusedLayerNorm.lean

  Fused LayerNorm kernels.

  `tkFusedLayerNormResidual1024` is the canonical ThunderKittens-aligned port of
  `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu`.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def fusedLayerNormResidual1024Bf16Body
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  comment "ThunderKittens-style layernorm + residual on 64x1024 tiles"

  let hiddenTiles : Nat := 16
  let invHidden : Float := 0.0009765625 -- 1 / 1024
  let eps : Float := 1.0e-5

  let coord ← blockCoord2D

  let x ← allocRT .BFloat16 64 64
  let residual ← allocRT .BFloat16 64 64
  let outResid ← allocRT .BFloat16 64 64
  let outResidF ← allocRT .Float32 64 64
  let centered ← allocRT .Float32 64 64
  let sq ← allocRT .Float32 64 64

  let tileShared ← allocST .BFloat16 64 64

  let sum ← zeroRV .Float32 64
  let varSum ← zeroRV .Float32 64
  let tmpSum ← allocRV .Float32 64
  let tmpVarSum ← allocRV .Float32 64
  let mean ← allocRV .Float32 64
  let var ← allocRV .Float32 64
  let invStd ← allocRV .Float32 64

  let weightShared ← allocSV .BFloat16 64
  let biasShared ← allocSV .BFloat16 64
  let weightRV ← allocRV .BFloat16 64
  let biasRV ← allocRV .BFloat16 64
  let weightF ← allocRV .Float32 64
  let biasF ← allocRV .Float32 64

  comment "Pass 1: out_resid and row statistics across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
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
    rowSum tmpSum outResidF
    addVec sum sum tmpSum

  scalarMulVec mean sum invHidden

  comment "Pass 2: centered variance across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared
    convert outResidF outResid

    subCol centered outResidF mean
    mul sq centered centered
    rowSum tmpVarSum sq
    addVec varSum varSum tmpVarSum

  scalarMulVec var varSum invHidden
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 3: normalize + affine (weight, bias)"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared
    convert outResidF outResid

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared
    convertVec weightF weightRV

    loadVecGlobalCol biasShared bias_ptr tileCoord
    loadVec biasRV biasShared
    convertVec biasF biasRV

    subCol centered outResidF mean
    mulCol centered centered invStd
    mulRow centered centered weightF
    addRow centered centered biasF

    convert outResid centered
    store tileShared outResid
    sync
    storeGlobal out_ptr tileShared tileCoord

private def fusedLayerNormResidual1024F32Body
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "Float32 fused layernorm + residual on 64x1024 tiles"

  let hiddenTiles : Nat := 16
  let invHidden : Float := 0.0009765625 -- 1 / 1024
  let eps : Float := 1.0e-5

  let coord ← blockCoord2D

  let x ← allocRT .Float32 64 64
  let residual ← allocRT .Float32 64 64
  let outResid ← allocRT .Float32 64 64
  let centered ← allocRT .Float32 64 64
  let sq ← allocRT .Float32 64 64

  let tileShared ← allocST .Float32 64 64

  let sum ← zeroRV .Float32 64
  let varSum ← zeroRV .Float32 64
  let tmpSum ← allocRV .Float32 64
  let tmpVarSum ← allocRV .Float32 64
  let mean ← allocRV .Float32 64
  let var ← allocRV .Float32 64
  let invStd ← allocRV .Float32 64

  let weightShared ← allocSV .Float32 64
  let biasShared ← allocSV .Float32 64
  let weightRV ← allocRV .Float32 64
  let biasRV ← allocRV .Float32 64

  comment "Pass 1: out_resid and row statistics across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
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

    rowSum tmpSum outResid
    addVec sum sum tmpSum

  scalarMulVec mean sum invHidden

  comment "Pass 2: centered variance across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    subCol centered outResid mean
    mul sq centered centered
    rowSum tmpVarSum sq
    addVec varSum varSum tmpVarSum

  scalarMulVec var varSum invHidden
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 3: normalize + affine (weight, bias)"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared

    loadVecGlobalCol biasShared bias_ptr tileCoord
    loadVec biasRV biasShared

    subCol centered outResid mean
    mulCol centered centered invStd
    mulRow centered centered weightRV
    addRow centered centered biasRV

    store tileShared centered
    sync
    storeGlobal out_ptr tileShared tileCoord

/-- ThunderKittens-aligned fused residual + layernorm kernel for `d_model = 1024`.

This is the canonical porting surface for `kernels/layernorm/layernorm.cu`.
It keeps the same two-pass structure as the ThunderKittens kernel:

- pass 1 computes `out_resid = x + residual` and full-row statistics
- pass 2 reloads `out_resid`, applies normalization, then affine parameters

The older kernels below remain useful as DSL sketches, but this kernel should be
treated as the authoritative fused-layernorm implementation in Tyr. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  fusedLayerNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the canonical bf16 fused layernorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024Blackwell
    (x_ptr : GPtr GpuFloat.BFloat16)
    (residual_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16)
    (out_resid_ptr : GPtr GpuFloat.BFloat16) : KernelM Unit := do
  setFamily .Blackwell
  fusedLayerNormResidual1024Bf16Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Float32 variant of `tkFusedLayerNormResidual1024`.

This keeps the same two-pass structure and launch geometry, but operates on
`f32` inputs and outputs directly instead of silently quantizing at the host
boundary. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024F32
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  fusedLayerNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

/-- Blackwell-family variant of the f32 fused layernorm kernel. -/
@[gpu_kernel .SM90]
def tkFusedLayerNormResidual1024F32Blackwell
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  fusedLayerNormResidual1024F32Body
    x_ptr residual_ptr weight_ptr bias_ptr out_ptr out_resid_ptr

end Tyr.GPU.Kernels
