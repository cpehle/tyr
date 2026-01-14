/-
  Tyr/GPU/Kernels/Hedgehog.lean

  Hedgehog hybrid attention kernel implementation.
  Based on ThunderKittens patterns.

  Key features:
  - Hybrid of linear attention + sliding window attention
  - State accumulation for linear attention component
  - Feature map application (ReLU or learned)
  - exp2 for efficient softmax
  - Block-wise processing with state carry-over
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

namespace Tyr.GPU.Kernels.Hedgehog

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Hedgehog Hybrid Attention

Hedgehog combines:
1. **Linear Attention**: O(N) complexity using φ(Q) @ (φ(K)^T @ V)
2. **Sliding Window**: Local attention within a window for fine-grained patterns

The output is: O = α * O_linear + (1-α) * O_window
where α can be learned or fixed.
-/

/-- Hedgehog forward pass - hybrid linear + sliding window attention -/
@[gpu_kernel .SM90]
def hedgehogFwd : KernelM Unit := do
  comment "=== Hedgehog Hybrid Attention Forward ==="

  -- Q, K, V tiles
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Feature-mapped versions (after φ)
  let phiQ : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let phiK : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- State for linear attention: S = φ(K)^T @ V
  let state : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Normalization state for linear attention
  let zState : RV GpuFloat.Float32 64 ← zeroRV .Float32 64

  -- Sliding window attention scores
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Softmax tracking
  let maxVec : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← zeroRV .Float32 64
  let prevMax : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Outputs
  let oLinear : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let oWindow : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let oFinal : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let stateShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load previous state (if any)"
  load state stateShared

  comment "Process sequence chunks"
  forLoop 0 16 do
    comment "Load Q, K, V for this chunk"
    load q qShared
    load k kShared
    load v vShared

    comment "=== Linear Attention Component ==="

    comment "Apply feature map φ (ReLU for efficiency)"
    convert phiQ q
    convert phiK k
    relu phiQ phiQ
    relu phiK phiK

    comment "O_linear = φ(Q) @ state"
    let phiQBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiQBf phiQ
    let stateBfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert stateBfRow state
    let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout stateBf stateBfRow
    mma oLinear phiQBf stateBf oLinear

    comment "Update state: state += φ(K)^T @ V"
    let phiKBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert phiKBf phiK
    mma state phiKBf v state

    comment "Update normalization: z += row_sum(φ(K))"
    let kSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum kSum phiK
    addVec zState zState kSum

    comment "=== Sliding Window Attention Component ==="

    comment "Compute attention scores: att = Q @ K^T"
    let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    convert qF q
    convert kF k
    mmaT att qF kF (← zeroRT .Float32 64 64)

    comment "Apply causal mask for sliding window"
    makeCausal att att (some (-1.0e10))

    comment "Online softmax with exp2 for efficiency"
    -- Save previous max for rescaling
    copyVec prevMax maxVec

    -- Update max: max = max(max, row_max(att))
    rowMaxAccum maxVec att maxVec

    -- Compute exp2(att - max)
    subCol att att maxVec
    exp2 att att

    -- Rescale previous sum and output
    let scale : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    subVec scale prevMax maxVec
    expVec scale scale
    mulVec sumVec sumVec scale
    mulCol oWindow oWindow scale

    -- Update sum
    let rowS : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum rowS att
    addVec sumVec sumVec rowS

    comment "O_window += att @ V"
    let attBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert attBf att
    mma oWindow attBf v oWindow

    comment "=== Combine Linear and Window Attention ==="

    comment "Normalize window attention: O_window = O_window / sum"
    divCol oWindow oWindow sumVec

    comment "Normalize linear attention: O_linear = O_linear / z"
    divCol oLinear oLinear zState

    comment "Combine: O = 0.5 * O_linear + 0.5 * O_window (equal weighting)"
    scalarMul oLinear oLinear 0.5
    scalarMul oWindow oWindow 0.5
    add oFinal oLinear oWindow

    comment "Store output"
    convert out oFinal
    store outShared out

    sync

  comment "Store final state for next chunk"
  store stateShared state

/-- Build Hedgehog forward kernel -/
def hedgehogFwdKernel : Kernel :=
  buildKernelM "hedgehog_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "state", dtype := .Float32, isPointer := true },
    { name := "z_state", dtype := .Float32, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false },
    { name := "head_dim", dtype := .Float32, isPointer := false }
  ] hedgehogFwd

/-! ## Hedgehog with Learned Mixing

Variant with learnable mixing coefficients α.
-/

/-- Hedgehog with learned mixing weights -/
@[gpu_kernel .SM90]
def hedgehogLearnedFwd : KernelM Unit := do
  comment "=== Hedgehog with Learned Mixing ==="

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Mixing weights (per-position or per-head)
  let alpha : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let phiQ : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let phiK : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let state : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let oLinear : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let oWindow : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let oFinal : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let alphaShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load mixing weights"
  loadVec alpha alphaShared

  forLoop 0 16 do
    load q qShared
    load k kShared
    load v vShared

    comment "Linear attention"
    convert phiQ q
    convert phiK k
    relu phiQ phiQ
    relu phiK phiK
    let phiQBf ← allocRT .BFloat16 64 64
    convert phiQBf phiQ
    let stateBfRow ← allocRT .BFloat16 64 64
    convert stateBfRow state
    let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout stateBf stateBfRow
    mma oLinear phiQBf stateBf oLinear
    let phiKBf ← allocRT .BFloat16 64 64
    convert phiKBf phiK
    mma state phiKBf v state

    comment "Window attention"
    let qF ← allocRT .Float32 64 64
    let kF ← allocRT .Float32 64 64
    convert qF q
    convert kF k
    mmaT att qF kF (← zeroRT .Float32 64 64)
    makeCausal att att (some (-1.0e10))
    exp att att
    let rowS ← allocRV .Float32 64
    rowSum rowS att
    divCol att att rowS
    let attBf ← allocRT .BFloat16 64 64
    convert attBf att
    mma oWindow attBf v oWindow

    comment "Learned mixing: O = α * O_linear + (1-α) * O_window"
    mulCol oLinear oLinear alpha
    -- Compute (1 - alpha) * O_window
    let oneMinusAlpha ← allocRV .Float32 64
    scalarMul oneMinusAlpha alpha (-1.0)  -- -alpha
    scalarAdd oneMinusAlpha oneMinusAlpha 1.0  -- 1 - alpha
    mulCol oWindow oWindow oneMinusAlpha
    add oFinal oLinear oWindow

    convert out oFinal
    store outShared out
    sync

def hedgehogLearnedFwdKernel : Kernel :=
  buildKernelM "hedgehog_learned_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "alpha", dtype := .Float32, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "state", dtype := .Float32, isPointer := true }
  ] hedgehogLearnedFwd

-- Print generated kernels
#eval IO.println "=== Hedgehog ===" *> IO.println (generateKernel hedgehogFwdKernel)
#eval IO.println "\n=== Hedgehog Learned ===" *> IO.println (generateKernel hedgehogLearnedFwdKernel)

end Tyr.GPU.Kernels.Hedgehog
