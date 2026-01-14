/-
  Tyr/GPU/Kernels/RotaryNew.lean

  Rotary Positional Embedding (RoPE) using native Lean4 GPU DSL.
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

/-- Rotary Positional Embedding forward

Applies rotation to Q and K tensors:
  x1' = x1 * cos(θ) - x2 * sin(θ)
  x2' = x1 * sin(θ) + x2 * cos(θ)
-/
def rotaryFwdNew (headDim : Nat := 64) : KernelM Unit := do
  comment "=== Rotary Position Embedding ==="

  comment "Register tiles for Q and K (each row is a token)"
  let q : RT GpuFloat.BFloat16 16 headDim ← allocRT .BFloat16 16 headDim
  let k : RT GpuFloat.BFloat16 16 headDim ← allocRT .BFloat16 16 headDim

  comment "First and second halves for rotation"
  let q1 : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let q2 : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let k1 : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let k2 : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)

  comment "Temporary tiles for rotated values"
  let q1Rot : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let q2Rot : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let k1Rot : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let k2Rot : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)

  comment "Sin/cos tables (precomputed per position, bf16 for compute)"
  let sinTable : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)
  let cosTable : RT GpuFloat.BFloat16 16 (headDim/2) ← allocRT .BFloat16 16 (headDim/2)

  comment "Shared memory"
  let qShared : ST GpuFloat.BFloat16 16 headDim ← allocST .BFloat16 16 headDim
  let kShared : ST GpuFloat.BFloat16 16 headDim ← allocST .BFloat16 16 headDim

  comment "Process tokens in blocks"
  forLoop 0 64 do
    comment "Load Q and K tiles"
    load q qShared
    load k kShared

    comment "Split into halves (would use subtile in real code)"
    -- Note: slicing operations would go here

    comment "Apply rotation to Q: q1' = q1*cos - q2*sin"
    mul q1Rot q1 cosTable
    mul q2Rot q2 sinTable
    sub q1Rot q1Rot q2Rot

    comment "q2' = q1*sin + q2*cos"
    mul q1 q1 sinTable
    mul q2 q2 cosTable
    add q2Rot q1 q2

    comment "Apply rotation to K similarly"
    mul k1Rot k1 cosTable
    mul k2Rot k2 sinTable
    sub k1Rot k1Rot k2Rot

    mul k1 k1 sinTable
    mul k2 k2 cosTable
    add k2Rot k1 k2

    comment "Store rotated Q and K"
    store qShared q
    store kShared k

    sync

def rotaryFwdKernel : Kernel :=
  buildKernelM "rotary_fwd" .SM90 #[
    { name := "q_ptr", dtype := .BFloat16, isPointer := true },
    { name := "k_ptr", dtype := .BFloat16, isPointer := true },
    { name := "sin_ptr", dtype := .Float32, isPointer := true },
    { name := "cos_ptr", dtype := .Float32, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false },
    { name := "num_heads", dtype := .Float32, isPointer := false }
  ] (rotaryFwdNew 64)

-- Generate C++ code
#eval IO.println "=== Rotary ===" *> IO.println (generateKernel rotaryFwdKernel)

end Tyr.GPU.Kernels
