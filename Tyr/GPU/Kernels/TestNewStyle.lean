/-
  Tyr/GPU/Kernels/TestNewStyle.lean

  Test file for the new @[gpu_kernel] attribute style.
  Demonstrates typed kernel parameters with automatic extraction.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.FFI

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## New Style: Using @[gpu_kernel] Attribute

The new style eliminates manual string-based parameter names and isPointer booleans.
Parameters are typed Lean values usable in the kernel body.
-/

/-- Simple vector add kernel demonstrating new style.

    Parameters are typed:
    - `a`, `b`, `c` : Global memory pointers (GPtr BFloat16)
    - `size` : Scalar value (KVal UInt64)

    The @[gpu_kernel] attribute:
    1. Extracts parameter info from the function signature
    2. Generates a companion `vecAdd.kernel` definition
    3. Registers the kernel for FFI generation
-/
@[gpu_kernel .SM90]
def vecAdd (a : GPtr GpuFloat.BFloat16) (b : GPtr GpuFloat.BFloat16)
           (c : GPtr GpuFloat.BFloat16) (size : KVal UInt64) : KernelM Unit := do
  comment "=== Vector Add Kernel (New Style) ==="
  comment s!"Parameters: a={a.name}, b={b.name}, c={c.name}, size={size.name}"

  -- Allocate tiles
  let tileA : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let tileB : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let tileC : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory tiles
  let sharedA : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let sharedB : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let sharedC : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  -- Load using TMA (with typed parameters)
  tmaLoad sharedA a size
  tmaLoad sharedB b size

  -- Load from shared to registers
  load tileA sharedA
  load tileB sharedB

  -- Compute c = a + b
  add tileC tileA tileB

  -- Store result
  store sharedC tileC
  tmaStore c sharedC size

  sync

/-- LayerNorm kernel in new style -/
@[gpu_kernel .SM90]
def layerNormNew (x : GPtr GpuFloat.BFloat16) (weight : GPtr GpuFloat.BFloat16)
                 (bias : GPtr GpuFloat.BFloat16) (out : GPtr GpuFloat.BFloat16)
                 (batchSize : KVal UInt64) (hiddenDim : KVal UInt64)
                 : KernelM Unit := do
  comment "=== LayerNorm Forward (New Style) ==="

  -- Register tiles
  let xTile : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xf : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let temp : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Statistics vectors
  let mean : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let var : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  -- Load input using TMA with typed parameters
  tmaLoad xShared x batchSize

  -- Load to registers and convert
  load xTile xShared
  convert xf xTile

  -- Compute mean
  rowSum mean xf

  -- Subtract mean
  subCol temp xf mean

  -- Compute variance
  mul xf temp temp
  rowSum var xf

  -- Normalize and output
  convert xTile temp
  store outShared xTile
  tmaStore out outShared batchSize

  sync

/-! ## Verification

The attribute auto-generates:
- vecAdd.kernel : Kernel
- layerNormNew.kernel : Kernel

We can verify the types are correct.
-/

-- Verify the companion definitions exist and have the correct type
#check vecAdd.kernel
#check layerNormNew.kernel

-- Test code generation from the attribute-generated kernels
#eval IO.println "=== vecAdd.kernel (auto-generated) ==="
#eval IO.println (generateKernel vecAdd.kernel)

#eval IO.println "\n=== layerNormNew.kernel (auto-generated) ==="
#eval IO.println (generateKernel layerNormNew.kernel)

end Tyr.GPU.Kernels
