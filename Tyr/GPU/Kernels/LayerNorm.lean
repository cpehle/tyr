/-
  Tyr/GPU/Kernels/LayerNorm.lean

  LayerNorm and RMSNorm kernels (Forward and Backward).
  Using native Lean4 GPU DSL with compile-time dimension checking.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Forward Kernels -/

/-- LayerNorm forward using tile-based computation

Uses 64x64 tiles where each row is a token and columns span hidden dim.
This is more efficient than vector-based for larger hidden dimensions.
-/
@[gpu_kernel .SM90]
def layerNormTiledNew (x_ptr : GPtr GpuFloat.BFloat16) (weight_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16) (out_ptr : GPtr GpuFloat.BFloat16)
    (batch_size : KVal UInt64) (hidden_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numTiles : Nat := 16
  comment "=== LayerNorm Forward (Tiled) ==="

  comment "Compute tile coordinates from block index"
  let coord ← blockCoord2D

  comment "Register tiles for input/output"
  let x : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let xf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let temp : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  comment "Statistics vectors (one per row/token)"
  let mean : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let var : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let invStd : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  comment "Normalization parameters"
  let weight : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let bias : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "Shared memory"
  let xShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let outShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let weightShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let biasShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load normalization parameters (long-resident)"
  loadGlobal weightShared weight_ptr coord
  loadGlobal biasShared bias_ptr coord
  sync
  load weight weightShared
  load bias biasShared

  comment "Process tiles of tokens"
  for tileIdx in krange 0 numTiles do
    comment "Load input tile from global memory"
    loadGlobal xShared x_ptr (coord.withRow tileIdx.id)
    sync
    load x xShared

    comment "Convert to float32 for precision"
    convert xf x

    comment "Step 1: Compute row-wise mean"
    rowSum mean xf
    -- Note: Would need scalar division by tileSize

    comment "Step 2: Subtract mean from each row"
    subCol temp xf mean

    comment "Step 3: Compute variance = mean((x - mean)^2)"
    mul xf temp temp
    rowSum var xf
    -- Note: Would need scalar division + eps

    comment "Step 4: Compute inverse std = 1/sqrt(var + eps)"
    -- rsqrt invStd var  -- Would need vec rsqrt

    comment "Step 5: Normalize: (x - mean) * invStd"
    mulCol temp temp invStd

    comment "Step 6: Scale and shift"
    let weightF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    let biasF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    convert weightF weight
    convert biasF bias
    mul temp temp weightF
    add temp temp biasF

    comment "Convert back to bf16 and store to global"
    convert x temp
    store outShared x
    storeGlobal out_ptr outShared (coord.withRow tileIdx.id)

    sync

/-- RMSNorm forward (simpler variant without mean subtraction) -/
@[gpu_kernel .SM90]
def rmsNormTiledNew (x_ptr : GPtr GpuFloat.BFloat16) (weight_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16) (hidden_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numTiles : Nat := 16
  comment "=== RMSNorm Forward (Tiled) ==="

  comment "Compute tile coordinates"
  let coord ← blockCoord2D

  let x : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let xf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let temp : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  let rmsSq : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let invRms : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  let weight : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  let xShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let outShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let weightShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load weight (long-resident)"
  loadGlobal weightShared weight_ptr coord
  sync
  load weight weightShared

  comment "Process tiles"
  for tileIdx in krange 0 numTiles do
    comment "Load input from global"
    loadGlobal xShared x_ptr (coord.withRow tileIdx.id)
    sync
    load x xShared

    comment "Convert to float32"
    convert xf x

    comment "Compute sum of squares per row"
    mul temp xf xf
    rowSum rmsSq temp

    comment "Compute inverse RMS (would need vec rsqrt)"
    -- rsqrtVec invRms rmsSq

    comment "Normalize: x * invRms"
    mulCol temp xf invRms

    comment "Scale by weight"
    let weightF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    convert weightF weight
    mul temp temp weightF

    comment "Convert back and store to global"
    convert x temp
    store outShared x
    storeGlobal out_ptr outShared (coord.withRow tileIdx.id)

    sync

/-! ## Backward Kernels -/

/-- LayerNorm backward kernel

Computes:
- dx: Gradient w.r.t input x
- dweight: Gradient w.r.t weight (gamma)
- dbias: Gradient w.r.t bias (beta)

Assumptions:
- 64x64 tiles
- hidden_dim = 64 (or handled by tiling)
-/
@[gpu_kernel .SM90]
def layerNormBwdTiled (dO_ptr : GPtr GpuFloat.BFloat16) (x_ptr : GPtr GpuFloat.BFloat16)
    (weight_ptr : GPtr GpuFloat.BFloat16) (mean_ptr : GPtr GpuFloat.Float32)
    (inv_std_ptr : GPtr GpuFloat.Float32)
    (dx_ptr : GPtr GpuFloat.BFloat16) (dweight_ptr : GPtr GpuFloat.Float32)
    (dbias_ptr : GPtr GpuFloat.Float32)
    (batch_size : KVal UInt64) (hidden_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numTiles : Nat := 16
  let hiddenDimFloat : Float := 64.0 -- Approximation for H
  comment "=== LayerNorm Backward ==="

  comment "Compute tile coordinates"
  let coord ← blockCoord2D

  comment "Register tiles"
  let dO : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let x : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let weight : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  -- Float32 conversions for precision
  let dOf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let xf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let weightf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  -- Intermediate terms
  let xHat : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let term1 : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let term2 : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dx : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dxBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  -- Vectors (per row/token)
  let mean : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let invStd : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let sumTerm1 : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let sumTerm2 : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  -- Accumulators for parameters (per column/feature)
  -- These accumulate across the batch (rows)
  let dGammaAccum : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dBetaAccum : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize

  -- Shared memory
  let dOShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let xShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let weightShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let meanShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let invStdShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dxShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load weight (gamma) from global"
  loadGlobal weightShared weight_ptr coord
  sync
  load weight weightShared
  convert weightf weight

  comment "Loop over batch tiles"
  for tileIdx in krange 0 numTiles do
    comment "Load batch inputs from global"
    loadGlobal dOShared dO_ptr (coord.withRow tileIdx.id)
    loadGlobal xShared x_ptr (coord.withRow tileIdx.id)
    sync

    load dO dOShared
    load x xShared
    loadVec mean meanShared
    loadVec invStd invStdShared

    convert dOf dO
    convert xf x

    comment "Compute x_hat = (x - mean) * inv_std"
    subCol xHat xf mean
    mulCol xHat xHat invStd

    comment "Compute term1 = dO * weight (dL/dx_hat)"
    mul term1 dOf weightf

    comment "Accumulate gradients for gamma and beta"
    -- dBeta += sum_rows(dO)
    -- dGamma += sum_rows(dO * x_hat)

    add dBetaAccum dBetaAccum dOf

    mul term2 dOf xHat
    add dGammaAccum dGammaAccum term2

    comment "Compute dx"
    -- dx = inv_std * (term1 - mean(term1) - x_hat * mean(term1 * x_hat))
    -- sum_term1 = rowSum(term1)
    rowSum sumTerm1 term1

    -- term2 reused: term1 * x_hat
    mul term2 term1 xHat
    rowSum sumTerm2 term2

    -- term1 -= sum_term1 / H
    -- term1 -= x_hat * sum_term2 / H
    -- Note: Need scalarMulVec to divide sums by H (not yet implemented)
    -- scalarMulVec sumTerm1 sumTerm1 (1.0 / hiddenDimFloat)
    -- scalarMulVec sumTerm2 sumTerm2 (1.0 / hiddenDimFloat)

    subCol term1 term1 sumTerm1 -- term1 - mean(term1)

    mulCol term2 xHat sumTerm2 -- x_hat * mean(...)
    sub term1 term1 term2      -- ... - ...

    mulCol dx term1 invStd

    comment "Store dx to global"
    convert dxBf16 dx
    store dxShared dxBf16
    storeGlobal dx_ptr dxShared (coord.withRow tileIdx.id)

    sync

  comment "Reduce dGamma and dBeta accumulators across rows"
  -- `dGammaAccum` has sum of gradients for each element in the block (if we didn't reduce).
  -- Now we reduce it to a single row vector.
  let dGammaFinal : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let dBetaFinal : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  
  colSum dGammaFinal dGammaAccum
  colSum dBetaFinal dBetaAccum

  comment "Store parameter gradients (atomic add needed)"
  let dGammaSV : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dBetaSV : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  
  storeVec dGammaSV dGammaFinal
  storeVec dBetaSV dBetaFinal
  
  storeVecGlobalAddCol dweight_ptr dGammaSV coord
  storeVecGlobalAddCol dbias_ptr dBetaSV coord

-- Verify auto-generated kernel and launch definitions
#check layerNormTiledNew.kernel
#check layerNormTiledNew.launch
#check rmsNormTiledNew.kernel
#check rmsNormTiledNew.launch
#check layerNormBwdTiled.kernel
#check layerNormBwdTiled.launch

-- Generate C++ code
#eval IO.println "=== LayerNorm Tiled ===" *> IO.println (generateKernel layerNormTiledNew.kernel)
#eval IO.println "\n=== RMSNorm Tiled ===" *> IO.println (generateKernel rmsNormTiledNew.kernel)
#eval IO.println "\n=== LayerNorm Backward ===" *> IO.println (generateKernel layerNormBwdTiled.kernel)

end Tyr.GPU.Kernels
