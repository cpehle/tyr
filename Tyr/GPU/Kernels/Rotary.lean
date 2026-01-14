/-
  Tyr/GPU/Kernels/Rotary.lean

  Rotary Position Embedding (RoPE) kernel implementation.
  Based on ThunderKittens patterns.

  RoPE applies a rotation to query and key vectors based on position:
  - Split x into x1 (first half) and x2 (second half)
  - Apply 2D rotation: [cos -sin; sin cos]
  - x1_out = x1*cos - x2*sin
  - x2_out = x1*sin + x2*cos
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

namespace Tyr.GPU.Kernels.Rotary

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Rotary Position Embedding

RoPE encodes position information by rotating vectors in 2D subspaces.
For each position i and dimension pair (2k, 2k+1):
  x'[2k]   = x[2k]*cos(θ_k*i) - x[2k+1]*sin(θ_k*i)
  x'[2k+1] = x[2k]*sin(θ_k*i) + x[2k+1]*cos(θ_k*i)

Where θ_k = 10000^(-2k/d) are the frequency bases.
-/

/-- Rotary embedding forward pass -/
@[gpu_kernel .SM90]
def rotaryFwd : KernelM Unit := do
  comment "=== Rotary Position Embedding Forward ==="

  -- Input tile: 64 x 64 (batch of positions x head_dim)
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Split halves (each 64 x 32)
  let x1 : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32
  let x2 : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32

  -- Precomputed sin/cos for these positions (64 x 32)
  let sinT : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32
  let cosT : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32

  -- Temporaries for rotation
  let temp1 : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32
  let temp2 : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32
  let negX2 : RT GpuFloat.Float32 64 32 ← allocRT .Float32 64 32

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let sinShared : ST GpuFloat.Float32 64 32 ← allocST .Float32 64 32
  let cosShared : ST GpuFloat.Float32 64 32 ← allocST .Float32 64 32
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load precomputed sin/cos (long-resident)"
  load sinT sinShared
  load cosT cosShared

  comment "Process sequence positions"
  forLoop 0 16 do
    comment "Load input"
    load x xShared
    convert xF x

    comment "Compute x1 * cos"
    mul temp1 x1 cosT

    comment "Compute -x2 * sin for first half"
    neg negX2 x2
    mul negX2 negX2 sinT

    comment "x1_out = x1*cos - x2*sin"
    add temp1 temp1 negX2

    comment "Compute x1 * sin for second half"
    mul negX2 x1 sinT

    comment "x2_out = x2*cos + x1*sin"
    mul temp2 x2 cosT
    add temp2 temp2 negX2

    comment "Store result"
    convert out x
    store outShared out
    sync

/-- Build rotary forward kernel -/
def rotaryFwdKernel : Kernel :=
  buildKernelM "rotary_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "sin", dtype := .Float32, isPointer := true },
    { name := "cos", dtype := .Float32, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false },
    { name := "head_dim", dtype := .Float32, isPointer := false }
  ] rotaryFwd

-- Print generated kernel
#eval IO.println "=== Rotary Forward ===" *> IO.println (generateKernel rotaryFwdKernel)

end Tyr.GPU.Kernels.Rotary
