/-
  Tyr/GPU.lean

  ThunderKittens GPU kernel abstraction for Lean4.
  Provides type-safe tile operations and C++ code generation.
-/
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
