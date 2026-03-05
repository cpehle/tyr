import Std.Data.HashMap
import Lean.ToExpr

/-!
# Tyr.GPU.Codegen.Var

`Tyr.GPU.Codegen.Var` defines `VarId`, the stable identifier used across the
kernel IR.

Instead of string-based names, the DSL uses index-based IDs:

- easier alpha-renaming and deterministic emission,
- simpler maps/sets during optimization and codegen passes,
- direct conversion to backend symbols (`v0`, `v1`, ...).

This mirrors how Lean's own internal IR tracks local bindings.
-/

namespace Tyr.GPU.Codegen

/-- Variable identifier (index-based, like Lean4 IR) -/
structure VarId where
  idx : Nat
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

namespace VarId
  /-- Convert to C++ identifier string -/
  def toIdent (v : VarId) : String := s!"v{v.idx}"

  /-- Ordering for VarId -/
  instance : Ord VarId where
    compare a b := compare a.idx b.idx

  instance : ToString VarId where
    toString v := v.toIdent
end VarId

end Tyr.GPU.Codegen
