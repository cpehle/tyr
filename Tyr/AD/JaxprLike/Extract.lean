import Tyr.AD.JaxprLike.RuleRegistry

/-!
# Tyr.AD.JaxprLike.Extract

Execution of local-Jacobian rules over a normalized `LeanJaxpr`.
-/

namespace Tyr.AD.JaxprLike

/-- Rule-execution error with equation/source context. -/
structure RuleExecutionError where
  eqnIndex0 : Nat
  op : OpName
  sourceOp? : Option OpName := none
  source : SourceRef
  message : String
  deriving Repr, Inhabited

private def formatSourceRef (source : SourceRef) : String :=
  let declName :=
    if source.decl == .anonymous then "<unknown>" else toString source.decl
  let linePart :=
    match source.line? with
    | some line => s!":{line}"
    | none => ""
  let colPart :=
    match source.col? with
    | some col => s!":{col}"
    | none => ""
  s!"{declName}{linePart}{colPart}"

def ruleExecutionErrorToString (e : RuleExecutionError) : String :=
  let source := formatSourceRef e.source
  let sourceOpPart :=
    match e.sourceOp? with
    | some sourceOp =>
      if sourceOp == e.op then
        ""
      else
        s!" (source `{sourceOp}`)"
    | none => ""
  s!"{e.message}: op `{e.op}`{sourceOpPart} at eqn #{e.eqnIndex0}, source={source}"

/--
Run registered local-Jacobian rules for every equation in `jaxpr`.
Fails with aggregated, source-aware errors when any equation rule is missing or malformed.
-/
def extractLocalJacEdges
    (jaxpr : LeanJaxpr) :
    Lean.CoreM (Except (Array RuleExecutionError) (Array LocalJacEdge)) := do
  let mut edges : Array LocalJacEdge := #[]
  let mut errors : Array RuleExecutionError := #[]

  for h : i in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[i]
    let ctx : RuleContext := { jaxpr := jaxpr, eqnIndex0 := i }
    match (← runLocalJacRule eqn ctx) with
    | .ok localEdges =>
      edges := edges ++ localEdges
    | .error err =>
      errors := errors.push {
        eqnIndex0 := i
        op := eqn.op
        sourceOp? := eqn.params.findName? .sourceOp
        source := eqn.source
        message := ruleErrorToMessage err
      }

  if errors.isEmpty then
    return .ok edges
  else
    return .error errors

end Tyr.AD.JaxprLike
