/-
  Tyr/GPU/Kernels/Flux.lean

  Flux activation kernels (fused linear + GELU/Gate).
  Based on ThunderKittens patterns.

  Key features:
  - Producer-consumer pattern with TMA loads
  - Fused linear layer + activation
  - Hardware-accelerated fast_tanh for GELU
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

namespace Tyr.GPU.Kernels.Flux

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Flux GELU Kernel

Fused linear + GELU activation using producer-consumer pattern.
GELU formula: f * 0.5 * (1 + fast_tanh(f * 0.79788456 * (1 + f² * 0.044715)))
-/

/-- Flux GELU forward pass - fused linear + GELU -/
@[gpu_kernel .SM90]
def fluxGeluFwd : KernelM Unit := do
  comment "=== Flux GELU Forward (Fused Linear + GELU) ==="

  -- Input tile
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Weight tile (col-major for tensor cores)
  let w : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Accumulator (float32 for precision)
  let acc : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Working tiles for GELU computation
  let f : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let f2 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let tanh_arg : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Producer-consumer loop"
  forLoop 0 8 do
    comment "Load input and weights"
    load x xShared
    load w wShared

    comment "Linear: acc = x @ w"
    mma acc x w acc

    comment "Convert to float32 for GELU computation"
    convert f acc

    comment "GELU: f * 0.5 * (1 + fast_tanh(f * 0.79788456 * (1 + f² * 0.044715)))"
    -- f² = f * f
    mul f2 f f
    -- f² * 0.044715
    scalarMul f2 f2 0.044715
    -- 1 + f² * 0.044715
    scalarAdd f2 f2 1.0
    -- f * 0.79788456 * (1 + f² * 0.044715)
    scalarMul tanh_arg f 0.79788456
    mul tanh_arg tanh_arg f2
    -- fast_tanh(...)
    fastTanh tanh_arg tanh_arg
    -- 1 + fast_tanh(...)
    scalarAdd tanh_arg tanh_arg 1.0
    -- f * 0.5 * (1 + fast_tanh(...))
    scalarMul f f 0.5
    mul f f tanh_arg

    comment "Convert back to bf16 and store"
    convert out f
    store outShared out

    sync

/-- Build Flux GELU forward kernel -/
def fluxGeluFwdKernel : Kernel :=
  buildKernelM "flux_gelu_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "w", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "M", dtype := .Float32, isPointer := false },
    { name := "N", dtype := .Float32, isPointer := false },
    { name := "K", dtype := .Float32, isPointer := false }
  ] fluxGeluFwd

/-! ## Flux Gate Kernel (SwiGLU-style)

Gate multiplication with residual addition.
-/

/-- Flux Gate forward pass - gating mechanism -/
@[gpu_kernel .SM90]
def fluxGateFwd : KernelM Unit := do
  comment "=== Flux Gate Forward (SwiGLU-style) ==="

  -- Main activation tile
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Gate tile
  let gate : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Residual input
  let residual : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Working tiles (float32 for precision)
  let acc : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let gateF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let residualF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let gateShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let residualShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Process sequence"
  forLoop 0 16 do
    comment "Load inputs"
    load x xShared
    load gate gateShared
    load residual residualShared

    comment "Convert to float32"
    convert acc x
    convert gateF gate
    convert residualF residual

    comment "Gate multiplication: acc = acc * gate"
    mul acc acc gateF

    comment "Residual addition: acc = acc + residual"
    add acc acc residualF

    comment "Convert back and store"
    convert out acc
    store outShared out

    sync

/-- Build Flux Gate forward kernel -/
def fluxGateFwdKernel : Kernel :=
  buildKernelM "flux_gate_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "gate", dtype := .BFloat16, isPointer := true },
    { name := "residual", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] fluxGateFwd

/-! ## Flux Linear + SiLU (GLU variant)

Linear transformation followed by SiLU gating.
-/

/-- Flux Linear + SiLU forward pass -/
@[gpu_kernel .SM90]
def fluxSiluFwd : KernelM Unit := do
  comment "=== Flux Linear + SiLU ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let wUp : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let wGate : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let up : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let gateVal : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wUpShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let wGateShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  forLoop 0 8 do
    comment "Load inputs"
    load x xShared
    load wUp wUpShared
    load wGate wGateShared

    comment "Compute up projection: up = x @ wUp"
    mma up x wUp up

    comment "Compute gate projection: gate = x @ wGate"
    mma gateVal x wGate gateVal

    comment "Apply SiLU to gate: gate = gate * sigmoid(gate)"
    silu gateVal gateVal

    comment "Multiply: out = up * silu(gate)"
    mul up up gateVal

    comment "Store output"
    convert out up
    store outShared out

    sync

def fluxSiluFwdKernel : Kernel :=
  buildKernelM "flux_silu_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "w_up", dtype := .BFloat16, isPointer := true },
    { name := "w_gate", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "M", dtype := .Float32, isPointer := false },
    { name := "N", dtype := .Float32, isPointer := false },
    { name := "K", dtype := .Float32, isPointer := false }
  ] fluxSiluFwd

-- Print generated kernels
#eval IO.println "=== Flux GELU ===" *> IO.println (generateKernel fluxGeluFwdKernel)
#eval IO.println "\n=== Flux Gate ===" *> IO.println (generateKernel fluxGateFwdKernel)
#eval IO.println "\n=== Flux SiLU ===" *> IO.println (generateKernel fluxSiluFwdKernel)

end Tyr.GPU.Kernels.Flux
