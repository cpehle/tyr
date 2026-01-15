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
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Codegen.Examples

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Simple Matrix Multiply Example -/

/-- Simple matrix multiply: C = A @ B -/
@[gpu_kernel .SM90]
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

-- Verify auto-generated kernel
#check simpleMatmul.kernel
#check simpleMatmul.launch

/-! ## FlashAttention-style Example -/

/-- FlashAttention forward pass (simplified) -/
@[gpu_kernel .SM90]
def flashAttnFwdExample : KernelM Unit := do
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
  for kvBlkIdx in krange 0 4 do
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

-- Verify auto-generated kernel
#check flashAttnFwdExample.kernel
#check flashAttnFwdExample.launch

/-! ## LayerNorm Example -/

/-- LayerNorm forward pass -/
@[gpu_kernel .SM90]
def layerNormFwdExample : KernelM Unit := do
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

-- Verify auto-generated kernel
#check layerNormFwdExample.kernel
#check layerNormFwdExample.launch

/-! ## Code Generation Tests -/

-- Generate C++ for simple matmul
#eval IO.println "=== Simple Matmul ===" *> IO.println (generateKernel simpleMatmul.kernel)

-- Generate C++ for FlashAttention
#eval IO.println "\n=== FlashAttention ===" *> IO.println (generateKernel flashAttnFwdExample.kernel)

-- Generate C++ for LayerNorm
#eval IO.println "\n=== LayerNorm ===" *> IO.println (generateKernel layerNormFwdExample.kernel)

end Tyr.GPU.Codegen.Examples
