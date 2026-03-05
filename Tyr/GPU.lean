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

1. Allocate typed tiles/vectors (`RT`, `ST`, `RV`, `SV`).
2. Compose operations in `KernelM` (`load`, `mma`, reductions, async I/O).
3. Build a `Kernel` with `buildKernelM`.
4. Emit backend code with `generateKernel`.

Example sketch:

```lean
def toy : Kernel :=
  buildKernelM "toy" .SM90 #[] do
    let a ← allocRT .BFloat16 16 16 .Row
    let b ← allocRT .BFloat16 16 16 .Col
    let c ← zeroRT .Float32 16 16 .Row
    mma c a b c
```

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
