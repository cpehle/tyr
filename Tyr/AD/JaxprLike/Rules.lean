import Tyr.AD.JaxprLike.Core
import Tyr.AD.Sparse

/-!
# Tyr.AD.JaxprLike.Rules

Local Jacobian rule contracts for LeanJaxpr equations.
This is the narrow interface between normalized IR equations and elimination edges.
-/

namespace Tyr.AD.JaxprLike

/-- Sparse linear map used on local Jacobian edges (shared with elimination). -/
abbrev SparseLinearMap := Tyr.AD.Sparse.SparseLinearMap

/-- Local Jacobian edge: src -> dst carrying a linear map. -/
structure LocalJacEdge where
  src : JVarId
  dst : JVarId
  map : SparseLinearMap := {}
  deriving Repr, Inhabited

/-- Context passed to local Jacobian rules. -/
structure RuleContext where
  jaxpr : LeanJaxpr
  eqnIndex0 : Nat
  deriving Inhabited

inductive RuleError where
  | unsupportedOp (op : OpName)
  | malformedEqn (msg : String)
  | internal (msg : String)
  deriving Repr, Inhabited

abbrev LocalJacRule :=
  JEqn → RuleContext → Except RuleError (Array LocalJacEdge)

/-- Conservative default rule: one identity-like edge per input to first output. -/
def defaultPlaceholderRule : LocalJacRule := fun eqn _ctx =>
  match eqn.outvars[0]? with
  | none => .error (.malformedEqn s!"Equation `{eqn.op}` has no output variable.")
  | some outv =>
    .ok <| eqn.invars.map fun inv => { src := inv.id, dst := outv.id, map := Tyr.AD.Sparse.identityLike }

def ruleErrorToMessage (err : RuleError) : String :=
  match err with
  | .unsupportedOp op => s!"No local-Jacobian rule for op `{op}`."
  | .malformedEqn msg => s!"Malformed equation for local-Jacobian extraction: {msg}"
  | .internal msg => s!"Local-Jacobian internal error: {msg}"

end Tyr.AD.JaxprLike
