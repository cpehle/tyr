/-
  Tyr/GPU/Kernels/FusedLayerNorm.lean

  Fused LayerNorm kernels.

  This module now serves two roles:

  - `tkFusedLayerNormResidual1024` is the canonical ThunderKittens-aligned port
    of `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu`.
  - `Tyr.GPU.Kernels.FusedLayerNorm.*` keeps the older sketch kernels around for
    IR experimentation, but those should not be treated as the primary port.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

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

  -- Reuse one shared tile buffer to keep shared memory pressure bounded.
  let tileShared ← allocST .BFloat16 64 64

  let sum ← zeroRV .Float32 64
  let sumSq ← zeroRV .Float32 64
  let tmpSum ← allocRV .Float32 64
  let tmpSqSum ← allocRV .Float32 64
  let mean ← allocRV .Float32 64
  let meanSq ← allocRV .Float32 64
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
    mul sq outResidF outResidF
    rowSum tmpSqSum sq
    addVec sumSq sumSq tmpSqSum

  scalarMulVec mean sum invHidden
  scalarMulVec var sumSq invHidden
  mulVec meanSq mean mean
  subVec var var meanSq
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 2: normalize + affine (weight, bias)"
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

end Tyr.GPU.Kernels

namespace Tyr.GPU.Kernels.FusedLayerNorm

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Fused LayerNorm

These kernels are retained as sketch-level DSL examples. They fuse the
following operations:
1. Dropout on input x
2. Residual connection: x_resid = residual + dropout(x)
3. LayerNorm: o = (x_resid - mean) / std * weight + bias

For the ThunderKittens-aligned implementation, use
`Tyr.GPU.Kernels.tkFusedLayerNormResidual1024` above.

The sketches reduce memory bandwidth by:
- Computing statistics in registers
- Avoiding intermediate tensor writes
- Pipelining loads with computation
-/

/-- Sketch fused LayerNorm with dropout and residual.

This is useful for IR experimentation, but it is not the canonical
ThunderKittens port. -/
@[gpu_kernel .SM90]
def fusedLayerNormFwd : KernelM Unit := do
  comment "=== Fused LayerNorm Forward ==="
  comment "dropout(x) + residual -> normalize -> scale + bias"

  -- Working vectors (d_model = 1024 typical)
  -- Process as 16 vectors of 64 elements each
  let x : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64
  let residual : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64
  let xResid : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Statistics (computed per token)
  let sum : RV GpuFloat.Float32 1 ← zeroRV .Float32 1
  let sumSq : RV GpuFloat.Float32 1 ← zeroRV .Float32 1

  -- Normalization parameters
  let weight : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64
  let bias : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64

  -- Output
  let out : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64

  -- Shared memory
  let xShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let residualShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let weightShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let biasShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let outShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64
  let outResidShared : SV GpuFloat.BFloat16 64 ← allocSV .BFloat16 64

  comment "Load normalization parameters (long-resident)"
  loadVec weight weightShared
  loadVec bias biasShared

  comment "Process tokens"
  for _tokenIdx in krange 0 64 do  -- batch * seq_len tokens
    comment "Load input and residual"
    loadVec x xShared
    loadVec residual residualShared

    comment "Step 1: Dropout (simplified - would need random state)"
    -- In practice: multiply by dropout mask and scale by 1/(1-p)
    -- dropout_mask(x, keep_prob)

    comment "Step 2: Residual connection"
    let xF : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let residualF : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    convertVec xF x
    convertVec residualF residual
    addVec xResid xF residualF

    comment "Store x_resid for use in backward pass"
    let xResidBf : RV GpuFloat.BFloat16 64 ← allocRV .BFloat16 64
    convertVec xResidBf xResid
    storeVec outResidShared xResidBf

    comment "Step 3: Compute mean"
    -- sum = reduce_sum(x_resid)
    -- mean = sum / d_model
    -- Simplified: just use the first element as proxy
    zeroVec sum

    comment "Step 4: Subtract mean"
    -- x_centered = x_resid - mean
    -- Simplified: assuming mean is broadcast

    comment "Step 5: Compute variance"
    -- var = reduce_sum(x_centered^2) / d_model
    let xSq : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    mulVec xSq xResid xResid
    zeroVec sumSq

    comment "Step 6: Compute inverse std"
    -- inv_std = rsqrt(var + eps)
    -- Simplified

    comment "Step 7: Normalize"
    -- x_norm = x_centered * inv_std
    -- Simplified: just use xResid

    comment "Step 8: Scale and shift"
    let weightF : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let biasF : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    convertVec weightF weight
    convertVec biasF bias
    mulVec xResid xResid weightF
    addVec xResid xResid biasF

    comment "Store output"
    convertVec out xResid
    storeVec outShared out

    sync

-- Verify auto-generated kernel

/-! ## Tiled LayerNorm

Alternative implementation using tiles for larger hidden dimensions.
More efficient when d_model >> 64.
-/

/-- Sketch tiled fused LayerNorm.

This remains a compact tiled example, not the canonical ThunderKittens port. -/
@[gpu_kernel .SM90]
def fusedLayerNormTiledFwd : KernelM Unit := do
  comment "=== Fused LayerNorm (Tiled) ==="

  -- Input tile (batch of tokens × hidden dim slice)
  let x : RT GpuFloat.BFloat16 16 64 ← allocRT .BFloat16 16 64
  let residual : RT GpuFloat.BFloat16 16 64 ← allocRT .BFloat16 16 64
  let xResid : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64

  -- Statistics per row (token)
  let mean : RV GpuFloat.Float32 16 ← allocRV .Float32 16
  let var : RV GpuFloat.Float32 16 ← allocRV .Float32 16
  let invStd : RV GpuFloat.Float32 16 ← allocRV .Float32 16

  -- Normalization parameters (broadcast across rows)
  let weight : RT GpuFloat.BFloat16 16 64 ← allocRT .BFloat16 16 64
  let bias : RT GpuFloat.BFloat16 16 64 ← allocRT .BFloat16 16 64

  -- Output
  let out : RT GpuFloat.BFloat16 16 64 ← allocRT .BFloat16 16 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 16 64 ← allocST .BFloat16 16 64
  let residualShared : ST GpuFloat.BFloat16 16 64 ← allocST .BFloat16 16 64
  let weightShared : ST GpuFloat.BFloat16 16 64 ← allocST .BFloat16 16 64
  let biasShared : ST GpuFloat.BFloat16 16 64 ← allocST .BFloat16 16 64
  let outShared : ST GpuFloat.BFloat16 16 64 ← allocST .BFloat16 16 64

  comment "Load parameters"
  load weight weightShared
  load bias biasShared

  comment "Process tiles"
  for _tileIdx in krange 0 16 do
    comment "Load input tile"
    load x xShared
    load residual residualShared

    comment "Residual connection"
    let xF : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    let residualF : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    convert xF x
    convert residualF residual
    add xResid xF residualF

    comment "Compute row-wise mean"
    rowSum mean xResid
    -- Scale by 1/d_model (simplified)

    comment "Subtract mean"
    let xCentered : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    subCol xCentered xResid mean

    comment "Compute variance"
    let xSq : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    mul xSq xCentered xCentered
    rowSum var xSq
    -- Scale by 1/d_model and add epsilon

    comment "Compute inverse std"
    rsqrtVec invStd var

    comment "Normalize"
    mulCol xResid xCentered invStd

    comment "Scale and shift"
    let weightF : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    let biasF : RT GpuFloat.Float32 16 64 ← allocRT .Float32 16 64
    convert weightF weight
    convert biasF bias
    mul xResid xResid weightF
    add xResid xResid biasF

    comment "Store output"
    convert out xResid
    store outShared out

    sync

-- Verify auto-generated kernel

-- Print generated kernels

end Tyr.GPU.Kernels.FusedLayerNorm
