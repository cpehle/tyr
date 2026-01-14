/-
  Tyr/GPU/Kernels/MambaNew.lean

  Mamba2 (Selective State Space Model) using native Lean4 GPU DSL.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Mamba2 forward kernel

Selective State Space Model with:
1. Cumulative decay tracking via prefix sum
2. Local attention with position-dependent decay
3. Cross-position state accumulation
-/
@[gpu_kernel .SM90]
def mamba2FwdNew (q_ptr : GPtr GpuFloat.BFloat16) (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16) (a_ptr : GPtr GpuFloat.Float32)
    (o_ptr : GPtr GpuFloat.BFloat16) (batch_size : KVal UInt64)
    (num_heads : KVal UInt64) (seq_len : KVal UInt64) : KernelM Unit := do
  let seqTile : Nat := 64
  let headDim : Nat := 64
  comment "=== Mamba2 Forward ==="

  comment "Register tiles for Q, K, V"
  let qReg : RT GpuFloat.BFloat16 16 headDim ← allocRT .BFloat16 16 headDim
  let kReg : RT GpuFloat.BFloat16 16 headDim ← allocRT .BFloat16 16 headDim
  let vReg : RT GpuFloat.BFloat16 16 headDim .Col ← allocRT .BFloat16 16 headDim .Col

  comment "Attention scores and decay matrix"
  let attBlock : RT GpuFloat.Float32 16 16 ← allocRT .Float32 16 16
  let localDecay : RT GpuFloat.Float32 16 16 ← allocRT .Float32 16 16

  comment "Output accumulator"
  let oReg : RT GpuFloat.Float32 16 headDim ← zeroRT .Float32 16 headDim

  comment "Decay factor vectors"
  let aVec : RV GpuFloat.Float32 16 ← allocRV .Float32 16
  let aCumsum : RV GpuFloat.Float32 16 ← allocRV .Float32 16

  comment "KV state for cross-position accumulation"
  let kvState : RT GpuFloat.Float32 headDim headDim ← zeroRT .Float32 headDim headDim

  comment "Shared memory"
  let qShared : ST GpuFloat.BFloat16 16 headDim ← allocST .BFloat16 16 headDim
  let kShared : ST GpuFloat.BFloat16 16 headDim ← allocST .BFloat16 16 headDim
  let vShared : ST GpuFloat.BFloat16 16 headDim .Col ← allocST .BFloat16 16 headDim .Col

  comment "Main sequence loop"
  forLoop 0 16 do
    comment "Load Q, K, V for this chunk"
    load qReg qShared
    load kReg kShared
    load vReg vShared

    comment "Compute cumulative sum of decay factors"
    cumsumRow localDecay localDecay  -- Simplified

    comment "Build decay matrix: exp(cumsum_i - cumsum_j)"
    exp localDecay localDecay

    comment "Apply causal mask"
    makeCausal localDecay localDecay (some 0.0)

    comment "Local attention: att = Q @ K^T"
    mmaT attBlock qReg kReg attBlock

    comment "Weight by decay"
    mul attBlock attBlock localDecay

    comment "Output: O = att @ V"
    let attBf16 : RT GpuFloat.BFloat16 16 16 ← allocRT .BFloat16 16 16
    convert attBf16 attBlock
    let vReg16 : RT GpuFloat.BFloat16 16 headDim .Col ← allocRT .BFloat16 16 headDim .Col
    mma oReg attBf16 vReg16 oReg

    comment "Reset for next chunk"
    zero oReg

    sync

/-- Simplified Mamba SSM recurrence for understanding -/
@[gpu_kernel .SM90]
def mambaSimpleNew (x_ptr : GPtr GpuFloat.BFloat16) (A_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.BFloat16) (seq_len : KVal UInt64) : KernelM Unit := do
  let headDim : Nat := 64
  comment "=== Mamba SSM: y = Cx + Du, x' = Ax + Bu ==="

  comment "State vector h"
  let h : RV GpuFloat.Float32 headDim ← allocRV .Float32 headDim

  comment "Input/output"
  let u : RV GpuFloat.Float32 1 ← allocRV .Float32 1
  let y : RV GpuFloat.Float32 1 ← allocRV .Float32 1

  comment "SSM parameters"
  let aBar : RV GpuFloat.Float32 headDim ← allocRV .Float32 headDim
  let bBar : RV GpuFloat.Float32 headDim ← allocRV .Float32 headDim
  let cVec : RV GpuFloat.Float32 headDim ← allocRV .Float32 headDim

  comment "Sequential recurrence"
  forLoop 0 128 do
    comment "State update: h = A*h + B*u"
    -- Simplified: would need vector-vector ops

    comment "Output: y = C @ h"
    -- dot product

    sync

-- Verify auto-generated kernel and launch definitions
#check mamba2FwdNew.kernel
#check mamba2FwdNew.launch
#check mambaSimpleNew.kernel
#check mambaSimpleNew.launch

-- Generate C++ code
#eval IO.println "=== Mamba2 ===" *> IO.println (generateKernel mamba2FwdNew.kernel)
#eval IO.println "\n=== Mamba Simple ===" *> IO.println (generateKernel mambaSimpleNew.kernel)

end Tyr.GPU.Kernels
