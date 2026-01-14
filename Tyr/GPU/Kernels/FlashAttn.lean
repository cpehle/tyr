/-
  Tyr/GPU/Kernels/FlashAttnNew.lean

  FlashAttention kernel using native Lean4 GPU DSL.
  Type-safe with compile-time dimension checking.
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

/-- FlashAttention forward kernel (simplified single-block version)

This demonstrates the native Lean4 DSL where:
- Variables are Lean identifiers, not strings
- Dimensions are checked at compile time
- Standard do-notation is used
-/
@[gpu_kernel .SM90]
def flashAttnFwdNew (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64) : KernelM Unit := do
  comment "=== FlashAttention Forward ==="

  comment "Declare register tiles"
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64  -- Row-major, transposed in mmaT
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  comment "Row-wise tracking for online softmax"
  let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  comment "Declare shared tiles for loading from global"
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  comment "Load Q from shared to register (long-resident)"
  load q qShared

  comment "Main loop over K, V blocks"
  forLoop 0 4 do
    comment "Load K, V from shared"
    load k kShared
    load v vShared

    comment "S = Q × K^T (bf16 inputs, f32 accumulator)"
    mmaT s q k s

    comment "Apply causal mask"
    makeCausal s s (some (-1e10))

    comment "Online softmax: update row_max"
    rowMaxAccum rowMax s rowMax

    comment "Subtract max and exponentiate"
    subCol s s rowMax
    exp s s

    comment "Convert to bf16 for V multiply"
    convert p s

    comment "Update row_sum"
    rowSumAccum rowSum s rowSum

    comment "Accumulate O = O + P × V"
    mma o p v o

    comment "Synchronize before next iteration"
    sync

  comment "Final normalization: O = O / row_sum"
  divCol o o rowSum

/-- Simple GEMM kernel for testing -/
@[gpu_kernel .SM90]
def simpleGemmNew (A_ptr : GPtr GpuFloat.BFloat16) (B_ptr : GPtr GpuFloat.BFloat16)
    (C_ptr : GPtr GpuFloat.Float32) (M : KVal UInt64) (N : KVal UInt64) (K : KVal UInt64)
    : KernelM Unit := do
  comment "=== Simple GEMM ==="

  comment "Declare tiles"
  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  comment "Main GEMM loop"
  forLoop 0 8 do
    load a aShared
    load b bShared
    mma c a b c
    sync

-- Verify auto-generated kernel and launch definitions
#check flashAttnFwdNew.kernel
#check flashAttnFwdNew.launch
#check simpleGemmNew.kernel
#check simpleGemmNew.launch

-- Generate C++ code for FlashAttention
#eval IO.println "=== FlashAttention ===" *> IO.println (generateKernel flashAttnFwdNew.kernel)

-- Generate C++ code for simple GEMM
#eval IO.println "\n=== Simple GEMM ===" *> IO.println (generateKernel simpleGemmNew.kernel)

end Tyr.GPU.Kernels
