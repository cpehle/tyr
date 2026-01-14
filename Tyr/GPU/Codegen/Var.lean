/-
  Tyr/GPU/Codegen/Var.lean

  Variable identifier for GPU kernel code generation.
  Follows Lean4 IR convention of using index-based identifiers.
-/
import Std.Data.HashMap

namespace Tyr.GPU.Codegen

/-- Variable identifier (index-based, like Lean4 IR) -/
structure VarId where
  idx : Nat
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

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
