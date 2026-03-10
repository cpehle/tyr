import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

/-!
# Tyr.GPU.Kernels.Prelude

Shared import bridge for concrete kernel modules.

This module also centralizes the common codegen import set shared by the concrete
kernel catalog. That keeps the per-kernel files focused on kernel logic plus any
truly special imports (`Macros`, `Constraints`, `Arch`, `FFI`, ...).
-/
