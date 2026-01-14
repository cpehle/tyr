/-
  Tyr/GPU/Codegen/Examples.lean

  Example kernels demonstrating the native Lean4 GPU DSL.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU.Codegen.Examples

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Simple Matrix Multiply Example -/

/-- Simple matrix multiply: C = A @ B -/
def simpleMatmul : KernelM Unit := do
  comment "Simple 64x64 matrix multiply"

  -- Allocate register tiles (bf16 inputs, f32 accumulator)
  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Shared memory for loading
  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  comment "Load A and B from shared memory"
  load a aShared
  load b bShared

  comment "Compute C = A @ B (bf16 inputs, f32 accumulator)"
  mma c a b c

  sync

/-- Build the simple matmul kernel -/
def simpleMatmulKernel : Kernel :=
  buildKernelM "simple_matmul" .SM90 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .Float32, isPointer := true }
  ] simpleMatmul

/-! ## FlashAttention-style Example -/

/-- FlashAttention forward pass (simplified) -/
def flashAttnFwd : KernelM Unit := do
  setArch .SM90
  comment "=== FlashAttention Forward ==="

  comment "Register tiles for Q, K, V (bf16)"
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  comment "Attention scores and output accumulators (f32)"
  let s : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  comment "Online softmax tracking"
  let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  comment "Shared memory"
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  comment "Load Q (long-resident)"
  load q qShared

  comment "=== Main KV loop ==="
  forLoop 0 4 do
    comment "Load K and V"
    load k kShared
    load v vShared

    comment "Attention scores: S = Q @ K^T (bf16 inputs, f32 output)"
    mmaT s q k s

    comment "Apply causal mask"
    makeCausal s s (some (-1e10))

    comment "Online softmax: track row max"
    rowMaxAccum rowMax s rowMax

    comment "Subtract max and exponentiate"
    subCol s s rowMax
    exp s s

    comment "Track row sum"
    rowSumAccum rowSum s rowSum

    comment "Convert S to bf16 for V multiply"
    let sBf16 : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert sBf16 s

    comment "Accumulate output: O += softmax(S) @ V"
    mma o sBf16 v o

    sync

  comment "Final normalization: O = O / rowSum"
  divCol o o rowSum

/-- Build the FlashAttention kernel -/
def flashAttnKernel : Kernel :=
  buildKernelM "flash_attn_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false },
    { name := "head_dim", dtype := .Float32, isPointer := false }
  ] flashAttnFwd

/-! ## LayerNorm Example -/

/-- LayerNorm forward pass -/
def layerNormFwd : KernelM Unit := do
  comment "=== LayerNorm Forward ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Statistics vectors"
  let mean : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let var : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  comment "Working tile (float32 for precision)"
  let xf : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  comment "Load input"
  load x xShared

  comment "Convert to float32"
  convert xf x

  comment "Compute mean"
  rowSum mean xf

  comment "Subtract mean"
  subCol xf xf mean

  comment "Compute variance"
  let xfSq : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  mul xfSq xf xf
  rowSum var xfSq

  comment "Normalize: x = (x - mean) / sqrt(var + eps)"
  -- In real code would compute rsqrt and multiply

  sync

/-- Build the LayerNorm kernel -/
def layerNormKernel : Kernel :=
  buildKernelM "layer_norm_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "gamma", dtype := .BFloat16, isPointer := true },
    { name := "beta", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true }
  ] layerNormFwd

/-! ## Code Generation Tests -/

-- Generate C++ for simple matmul
#eval IO.println "=== Simple Matmul ===" *> IO.println (generateKernel simpleMatmulKernel)

-- Generate C++ for FlashAttention
#eval IO.println "\n=== FlashAttention ===" *> IO.println (generateKernel flashAttnKernel)

-- Generate C++ for LayerNorm
#eval IO.println "\n=== LayerNorm ===" *> IO.println (generateKernel layerNormKernel)

end Tyr.GPU.Codegen.Examples
