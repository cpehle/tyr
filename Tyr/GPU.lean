import Tyr.GPU.Types
import Tyr.GPU.Capabilities
import Tyr.GPU.Tile
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Kernels.Examples

/-!
# Tyr.GPU

`Tyr.GPU` is the umbrella module for Tyr's tile-oriented GPU kernel DSL.
It provides a typed path from Lean declarations to generated CUDA/C++ kernel code.

## DSL Introduction

The GPU DSL in Tyr is split into layers:

1. **Type layer** (`Tyr.GPU.Types`, `Tyr.GPU.Tile`, `Tyr.GPU.Capabilities`):
   architecture, dtype, layout, and capability concepts.
2. **Kernel IR layer** (`Tyr.GPU.Codegen.IR`, `Monad`, `Ops`, `Loop`, `GlobalLayout`):
   a typed builder API that emits `KStmt` instructions.
3. **Emission layer** (`Tyr.GPU.Codegen.EmitNew`):
   lowers IR to CUDA/C++ source.

The key idea is that shape/layout compatibility is checked in Lean types, while
the final generated code remains close to hand-written GPU kernels.

## Quick Workflow

1. Define parameterized kernels with explicit inputs (`GPtr`, `KVal`).
2. Compute runtime tile coordinates (`blockCoord2D`, loop indices).
3. Move data via global/shared/register tiles (`loadGlobal`, `load`, `storeGlobal`).
4. Build/emit backend code through `@[gpu_kernel]` + `generateKernel`.

Preferred example shape:

```lean
@[gpu_kernel .SM90]
def addKernel
    (xPtr : GPtr GpuFloat.Float32)
    (yPtr : GPtr GpuFloat.Float32)
    (outPtr : GPtr GpuFloat.Float32)
    (n : KVal UInt64) : KernelM Unit := do
  let _ := n
  let coord ← blockCoord2D
  let xS : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let yS : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let oS : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  loadGlobal xS xPtr coord
  loadGlobal yS yPtr coord
  sync
  let x : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let y : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  load x xS
  load y yS
  add x x y
  store oS x
  storeGlobal outPtr oS coord
```

## Canonical Example Declarations

Use `Tyr.GPU.Kernels.Examples` as the documentation entrypoint for concrete
`@[gpu_kernel]` declarations:

- `Tyr.GPU.Kernels.Examples.simpleGemm`
- `Tyr.GPU.Kernels.Examples.flashAttnFwd`
- `Tyr.GPU.Kernels.Examples.layerNorm`
- `Tyr.GPU.Kernels.Examples.ampereGemm`
- `Tyr.GPU.Kernels.Examples.blackwellGemm`

## On Input-Free Kernels

Input-free kernels still exist in the codebase, but mostly as:

- IR/lowering smoke tests,
- focused examples of one operation family,
- architecture-dispatch micro examples.

They are useful for compiler development and diagnostics, but they are not the
best reference for real model integration. For end-to-end usage, prefer kernels
with explicit pointer/scalar inputs and global-memory I/O.

## Major Components

- Architecture/tile primitives (`GpuArch`, layouts, register/shared tile types).
- Capability modeling (`GpuCapabilities`, `RequiresTMA`, `RequiresWGMMA`).
- Kernel IR/AST/codegen interfaces (`Kernel`, `KStmt`, builders, emitters).
- Global-memory and loop helpers for kernel assembly and indexing.

## Scope

This is the primary entrypoint for kernel authoring and GPU code generation in Tyr.
It does not execute training loops by itself; model/runtime layers consume emitted kernels.
Current code generation is designed around a tile-kernel backend, but this API is
intended to stay backend-facing rather than backend-branded.
-/

namespace Tyr

-- Re-export main types from GPU submodules
export GPU (GpuFloat TileLoc TileLayout GpuArch SwizzleMode Scope)
export GPU (GpuCapabilities RequiresTMA RequiresWGMMA)
export GPU (Tile RegisterTile SharedTile RT ST RV SV)
export GPU.Codegen (RegTile SmemTile RegVec SmemVec GpuPtr KernelVal)
export GPU.Codegen (MMATranspose ReduceOp ReduceAxis UnaryOp BinaryOp
                    BroadcastAxis MaskOp TernaryOp SemaphoreOp)
export GPU.Codegen (KStmt Kernel KParam KernelM buildKernelM generateKernel)
-- GlobalLayout types for memory I/O
export GPU.Codegen (GlobalLayout GL GL2 GL3 GL4 TileCoord RTileCoord GPtr)
-- Global memory operations
export GPU.Codegen (loadGlobal storeGlobal loadGlobalAsync storeGlobalAsync storeGlobalAdd)
export GPU.Codegen (blockCoord2D getBlockIdxX getBlockIdxY getBlockIdxZ)
-- Loop types for standard for syntax
export GPU.Codegen (KIdx KRange krange)

end Tyr
