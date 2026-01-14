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
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- GELU activation with bias fusion -/
@[gpu_kernel .SM90]
def geluFwd (x_ptr : GPtr GpuFloat.BFloat16) (bias_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16) (size : KVal UInt64) : KernelM Unit := do
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

/-- SwiGLU activation: x * sigmoid(gate) -/
@[gpu_kernel .SM90]
def swiGluFwd (x_ptr : GPtr GpuFloat.BFloat16) (gate_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16) (size : KVal UInt64) : KernelM Unit := do
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

/-- GeGLU activation: x * GELU(gate) -/
@[gpu_kernel .SM90]
def geGluFwd (x_ptr : GPtr GpuFloat.BFloat16) (gate_ptr : GPtr GpuFloat.BFloat16)
    (out_ptr : GPtr GpuFloat.BFloat16) (size : KVal UInt64) : KernelM Unit := do
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

-- Verify auto-generated kernel and launch definitions
#check geluFwd.kernel
#check geluFwd.launch
#check swiGluFwd.kernel
#check swiGluFwd.launch
#check geGluFwd.kernel
#check geGluFwd.launch

-- Generate C++ code
#eval IO.println "=== GELU ===" *> IO.println (generateKernel geluFwd.kernel)
#eval IO.println "\n=== SwiGLU ===" *> IO.println (generateKernel swiGluFwd.kernel)
#eval IO.println "\n=== GeGLU ===" *> IO.println (generateKernel geGluFwd.kernel)

end Tyr.GPU.Kernels
