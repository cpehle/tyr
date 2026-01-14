/-
  Tyr/GPU/Kernels/ActivationsNew.lean

  Activation kernels (GELU, SwiGLU, GeGLU) using native Lean4 GPU DSL.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- GELU activation with bias fusion -/
def geluFwdNew : KernelM Unit := do
  comment "=== GELU with Bias ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let bias : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let biasShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load bias (long-resident)"
  load bias biasShared

  forLoop 0 16 do
    comment "Load input"
    load x xShared

    comment "Add bias"
    add x x bias

    comment "Apply GELU"
    gelu x x

    comment "Store output"
    store xShared x
    sync

def geluFwdKernel : Kernel :=
  buildKernelM "gelu_fwd" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "bias_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] geluFwdNew

/-- SwiGLU activation: x * sigmoid(gate) -/
def swiGluFwdNew : KernelM Unit := do
  comment "=== SwiGLU ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let gate : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let gateShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  forLoop 0 16 do
    comment "Load x and gate"
    load x xShared
    load gate gateShared

    comment "Apply sigmoid to gate"
    sigmoid gate gate

    comment "x * sigmoid(gate)"
    mul x x gate

    comment "Store output"
    store xShared x
    sync

def swiGluFwdKernel : Kernel :=
  buildKernelM "swiglu_fwd" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "gate_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] swiGluFwdNew

/-- GeGLU activation: x * GELU(gate) -/
def geGluFwdNew : KernelM Unit := do
  comment "=== GeGLU ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let gate : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let gateShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  forLoop 0 16 do
    load x xShared
    load gate gateShared

    comment "Apply GELU to gate"
    gelu gate gate

    comment "x * GELU(gate)"
    mul x x gate

    store xShared x
    sync

def geGluFwdKernel : Kernel :=
  buildKernelM "geglu_fwd" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "gate_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] geGluFwdNew

-- Generate C++ code
#eval IO.println "=== GELU ===" *> IO.println (generateKernel geluFwdKernel)
#eval IO.println "\n=== SwiGLU ===" *> IO.println (generateKernel swiGluFwdKernel)
#eval IO.println "\n=== GeGLU ===" *> IO.println (generateKernel geGluFwdKernel)

end Tyr.GPU.Kernels
