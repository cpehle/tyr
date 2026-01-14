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
import Tyr.GPU.Codegen.EmitNew

namespace Tyr

-- Re-export main types from GPU submodules
export GPU (GpuFloat TileLoc TileLayout GpuArch SwizzleMode Scope)
export GPU (GpuCapabilities RequiresTMA RequiresWGMMA)
export GPU (Tile RegisterTile SharedTile RT ST RV SV GL)
export GPU.Codegen (RegTile SmemTile RegVec SmemVec GpuPtr KernelVal)
export GPU.Codegen (MMATranspose ReduceOp ReduceAxis UnaryOp BinaryOp
                    BroadcastAxis MaskOp TernaryOp SemaphoreOp)
export GPU.Codegen (KStmt Kernel KParam KernelM buildKernelM generateKernel)

end Tyr
