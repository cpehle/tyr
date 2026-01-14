/-
  Tyr/GPU.lean

  ThunderKittens GPU kernel abstraction for Lean4.
  Provides type-safe tile operations and C++ code generation.
-/
import Tyr.GPU.Types
import Tyr.GPU.Capabilities
import Tyr.GPU.Tile
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr

-- Re-export main types from GPU submodules
export GPU (GpuFloat TileLoc TileLayout GpuArch SwizzleMode Scope)
export GPU (GpuCapabilities RequiresTMA RequiresWGMMA)
export GPU (Tile RegisterTile SharedTile RT ST RV SV GL)
export GPU.Codegen (KExpr KernelDef KernelParam MMATranspose ReduceOp ReduceAxis
                    UnaryOp BinaryOp BroadcastAxis MaskOp generateCpp generateExpr)

end Tyr
