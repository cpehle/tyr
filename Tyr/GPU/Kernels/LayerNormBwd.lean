/-
  Tyr/GPU/Kernels/LayerNormBwd.lean

  LayerNorm backward kernel using native Lean4 GPU DSL.
  Computes gradients for input, weight, and bias.
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
  let hiddenDimFloat : Float := 64.0 -- Approximation for H
  comment "=== LayerNorm Backward ==="

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
  
  -- Global gradient accumulators (to store at end)
  let dGammaShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dBetaShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize

  comment "Load weight (gamma) - broadcasted to rows implicitly if loaded as tile?"
  -- Assuming input format of weight is (1, H) or (H).
  -- We need to broadcast it to (N, H).
  -- If we load it into ST with (64, 64), we might need `broadcast` op.
  -- For now assuming weight is pre-broadcasted or we use a broadcast load.
  load weight weightShared
  convert weightf weight

  comment "Loop over batch tiles"
  for batchIdx in krange 0 16 do
    comment "Load batch inputs"
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
    -- We use `add` because dGammaAccum is a tile.
    -- We want to sum columns? No, sum ROWS to get (1, H).
    -- But we don't have a `colSum` to vector.
    -- We have `colSum` to... row vector?
    -- Actually `dGammaAccum` is (64, 64). We want to add `dO` to it?
    -- If we treat `dGammaAccum` as a buffer where we accumulate `dO` and later reduce?
    -- Or we reduce `dO` now.
    -- Let's just accumulate `dO` into `dBetaAccum` and `term1` into `dGammaAccum`? 
    -- No `term1 = dO * weight`. dGamma needs `dO * x_hat`.
    -- `mul term2 dOf xHat` -> `dO * x_hat`.
    -- Then accumulate `term2` into `dGammaAccum`.
    -- Finally we will reduce `dGammaAccum` across rows.
    
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
    -- We need to divide sums by H
    scalarMul sumTerm1 sumTerm1 (1.0 / hiddenDimFloat)
    scalarMul sumTerm2 sumTerm2 (1.0 / hiddenDimFloat)

    subCol term1 term1 sumTerm1 -- term1 - mean(term1)
    
    mulCol term2 xHat sumTerm2 -- x_hat * mean(...)
    sub term1 term1 term2      -- ... - ...

    mulCol dx term1 invStd

    comment "Store dx"
    convert dxBf16 dx
    store dxShared dxBf16
    
    sync

  comment "Reduce dGamma and dBeta accumulators across rows"
  -- `dGammaAccum` has sum of gradients for each element in the block (if we didn't reduce).
  -- Now we reduce it to a single row vector.
  let dGammaFinal : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let dBetaFinal : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  
  colSum dGammaFinal dGammaAccum
  colSum dBetaFinal dBetaAccum

  comment "Store parameter gradients (atomic add needed)"
  -- We don't have storeVecAdd for global memory directly exposed in DSL yet?
  -- Using storeAdd via ST.
  -- But ST is 2D. We need to store RV.
  -- Expand RV to ST? Or just store to first row of ST?
  -- Assuming we can store RV to SV.
  let dGammaSV : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dBetaSV : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  
  storeVec dGammaSV dGammaFinal
  storeVec dBetaSV dBetaFinal
  
  -- Then TMA/Global store from SV?
  -- `storeAdd` works on VarId.
  -- `storeAdd dweight_ptr dGammaSV`? No, `storeAdd` takes `dst src`.
  -- `dweight_ptr` is GPtr.
  -- We likely need `tmaStore` or similar.
  -- Or just `store` if we are the only block?
  -- If parallel over batch, we need atomic.
  -- `storeAdd` in `KStmt` is usually `store_add(dst, src)`.
  -- If `dst` is `GPtr`? `KStmt` supports `storeAdd dst src`.
  -- But `dst` must be a `VarId`. `GPtr` wraps `VarId`.
  -- Let's try `storeAdd` with GPtr's ID.
  
  -- But `dweight_ptr` points to global memory. `dGammaSV` is shared.
  -- ThunderKittens `store_add` usually stores from Reg/Shared to Global.
  -- Let's use `tma::store_add` if available or `store_add`.
  
  -- Re-check `IR.lean` `storeAdd` semantics.
  -- `| storeAdd (dst src : VarId)`
  -- `EmitNew` says: `store_add({dst}, {src})`.
  -- If `dst` is global pointer, it works.
  
  -- Wait, `dGammaSV` is SV (shared vector).
  -- `store_add` might expect ST (shared tile) or RT.
  -- ThunderKittens usually handles tiles.
  -- If we have a vector, maybe we need to treat it as a 1-row tile.
  
  storeAddShared (GPtr.varId dweight_ptr) (SV.varId dGammaSV)
  storeAddShared (GPtr.varId dbias_ptr) (SV.varId dBetaSV)

/-- Helper for storeAdd with explicit VarIds -/
def storeAddShared (dst : VarId) (src : VarId) : KernelM Unit :=
  emit (.storeAdd dst src)

-- Generate C++ code
#eval IO.println "=== LayerNorm Backward ===" *> IO.println (generateKernel layerNormBwdTiled.kernel)

end Tyr.GPU.Kernels
