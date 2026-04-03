import Tyr.AD.JaxprLike.Core
import Tyr.AD.JaxprLike.KStmtNames
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

/-- Semantic registry key used for local-Jacobian rule lookup. -/
inductive RuleKey where
  | op (op : OpName)
  | transpose
  | convert
  | dotGeneral
  | controlFlow (variant : Lean.Name)
  deriving Repr, Inhabited

instance : BEq RuleKey where
  beq lhs rhs :=
    match lhs, rhs with
    | .op op₁, .op op₂ => op₁ == op₂
    | .transpose, .transpose => true
    | .convert, .convert => true
    | .dotGeneral, .dotGeneral => true
    | .controlFlow v₁, .controlFlow v₂ => v₁ == v₂
    | _, _ => false

instance : Hashable RuleKey where
  hash
    | .op op => mixHash 0 (hash op)
    | .transpose => mixHash 1 0
    | .convert => mixHash 2 0
    | .dotGeneral => mixHash 3 0
    | .controlFlow variant => mixHash 4 (hash variant)

abbrev LocalJacRule :=
  JEqn → RuleContext → Except RuleError (Array LocalJacEdge)

private def isTransposeLikeOpName (op : OpName) : Bool :=
  op == kstmtTransposeOpName ||
    op == transposeAliasOpName ||
    op == `jax.lax.transpose_p ||
    op == `Graphax.transpose

private def isConvertLikeOpName (op : OpName) : Bool :=
  op == kstmtConvertOpName ||
    op == convertElementTypeAliasOpName ||
    op == `jax.lax.convert_element_type_p ||
    op == `Graphax.convert_element_type

/-- Canonical semantic registry key for a registered normalized/source op name. -/
def ruleKeyOfRegisteredOp (op : OpName) : RuleKey :=
  if isTransposeLikeOpName op then
    .transpose
  else if isConvertLikeOpName op then
    .convert
  else if isDotGeneralOpName op then
    .dotGeneral
  else if isScanAliasOpName op then
    .controlFlow `scan
  else if isCondAliasOpName op then
    .controlFlow `cond
  else
    .op op

namespace JEqn

/-- Semantic registry key for dispatching local-Jacobian rules from a typed equation. -/
def ruleKey (eqn : JEqn) : RuleKey :=
  match eqn.typed.schema, eqn.typed.payload with
  | .transpose, _ => .transpose
  | .convert, _ => .convert
  | .dotGeneral, _ => .dotGeneral
  | .controlFlow, .controlFlow info => .controlFlow info.variant
  | .unary, .unary tag =>
      if isTransposeLikeOpName tag then
        .transpose
      else if isConvertLikeOpName tag then
        .convert
      else
        .op eqn.op
  | _, _ => .op eqn.op

end JEqn

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
