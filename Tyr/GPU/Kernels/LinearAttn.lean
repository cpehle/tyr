/-
  Tyr/GPU/Kernels/LinearAttn.lean

  Linear Attention kernel implementation.
  Based on ThunderKittens patterns.

  Linear attention replaces softmax(Q @ K^T) with φ(Q) @ φ(K)^T
  where φ is a feature map (often identity or ReLU).

  This allows O(N) complexity instead of O(N^2) by computing:
  - S = φ(K)^T @ V  (state accumulation)
  - O = φ(Q) @ S    (output computation)
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

namespace Tyr.GPU.Kernels.LinearAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Linear Attention

Linear attention uses the associativity of matrix multiplication:
  softmax(Q @ K^T) @ V ≈ φ(Q) @ (φ(K)^T @ V)

Where φ can be:
- Identity (simplest)
- ReLU (RetNet style)
- ELU + 1 (Linear Transformer)
- Learned feature map

The state S = φ(K)^T @ V can be accumulated across chunks.
-/

/-- Linear attention forward with state accumulation -/
@[gpu_kernel .SM90]
def linearAttnFwd : KernelM Unit := do
  comment "=== Linear Attention Forward ==="

  -- Q, K, V tiles
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Feature-mapped versions (after φ)
  let phiQ : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let phiK : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- State: S = φ(K)^T @ V, shape [head_dim x head_dim]
  let state : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Output accumulator
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Normalization (for numerical stability)
  let zVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let zState : RV GpuFloat.Float32 64 ← zeroRV .Float32 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let stateShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  comment "Initialize state from previous chunk (if any)"
  load state stateShared

  comment "Process sequence chunks"
  forLoop 0 16 do
    comment "Load Q, K, V"
    load q qShared
    load k kShared
    load v vShared

    comment "Apply feature map φ (using ReLU)"
    convert phiQ q
    convert phiK k
    relu phiQ phiQ
    relu phiK phiK

    comment "Update state: S += φ(K)^T @ V"
    let phiKBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiKBf phiK
    let phiKT : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout phiKT phiKBf
    mma state phiKBf v state

    comment "Compute output: O = φ(Q) @ S"
    let phiQBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiQBf phiQ
    let stateBfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert stateBfRow state
    let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout stateBf stateBfRow
    mma o phiQBf stateBf o

    comment "Update normalization: z += φ(K)^T @ 1"
    rowSum zVec phiK
    addVec zState zState zVec

    sync

  comment "Store final state for next chunk"
  store stateShared state

/-- Build linear attention forward kernel -/
def linearAttnFwdKernel : Kernel :=
  buildKernelM "linear_attn_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "state", dtype := .Float32, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false }
  ] linearAttnFwd

/-! ## Causal Linear Attention

For causal (autoregressive) linear attention, we need to maintain
the running state and compute outputs incrementally.
-/

/-- Causal linear attention with chunk-wise state -/
@[gpu_kernel .SM90]
def causalLinearAttnFwd : KernelM Unit := do
  comment "=== Causal Linear Attention Forward ==="

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let phiQ : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let phiK : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let vF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Running state and output
  let state : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Intra-chunk attention (quadratic within chunk)
  let intraAtt : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  forLoop 0 16 do
    load q qShared
    load k kShared
    load v vShared

    comment "Apply feature map"
    convert phiQ q
    convert phiK k
    relu phiQ phiQ
    relu phiK phiK

    comment "Inter-chunk: O_inter = φ(Q) @ state_prev"
    let phiQBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiQBf phiQ
    let stateBfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert stateBfRow state
    let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout stateBf stateBfRow
    mma o phiQBf stateBf o

    comment "Intra-chunk: causal attention within chunk"
    mmaT intraAtt phiQ phiK intraAtt
    makeCausal intraAtt intraAtt (some 0.0)

    comment "O_intra = intra_att @ V"
    let intraAttBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert intraAttBf intraAtt
    mma o intraAttBf v o

    comment "Update state: state += φ(K)^T @ V"
    let phiKBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiKBf phiK
    mma state phiKBf v state

    sync

def causalLinearAttnFwdKernel : Kernel :=
  buildKernelM "causal_linear_attn_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "state", dtype := .Float32, isPointer := true }
  ] causalLinearAttnFwd

-- Print generated kernels
#eval IO.println "=== Linear Attention ===" *> IO.println (generateKernel linearAttnFwdKernel)
#eval IO.println "\n=== Causal Linear Attention ===" *> IO.println (generateKernel causalLinearAttnFwdKernel)

end Tyr.GPU.Kernels.LinearAttn
