import Tyr.AD.JaxprLike.RuleRegistry

/-!
# Tyr.AD.JaxprLike.RuleCheck

Strict coverage checking for local Jacobian rule availability.
No fallback behavior is modeled here: missing rules are hard errors.
-/

namespace Tyr.AD.JaxprLike

structure CoverageError where
  op : OpName
  eqnIndex0 : Nat
  source : SourceRef
  outvarIds : Array Nat := #[]
  params : OpParams := #[]
  message : String
  deriving Repr, Inhabited

private def coverageFormatSourceRef (source : SourceRef) : String :=
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

private def coverageOutvarsToString (outvarIds : Array Nat) : String :=
  let pieces := outvarIds.toList.map (fun id => s!"x_{id}")
  "[" ++ String.intercalate ", " pieces ++ "]"

private def coverageMetadataParts (e : CoverageError) : Array String := Id.run do
  let mut parts : Array String := #[]

  match e.params.findName? .loweringKind with
  | some lowering =>
    parts := parts.push s!"loweringKind={lowering}"
  | none =>
    match e.params.findName? .kind with
    | some kind =>
      parts := parts.push s!"kind={kind}"
    | none => pure ()

  match e.params.findNat? .fnbodyOutVarIdx with
  | some outIdx =>
    parts := parts.push s!"fnbodyOutVarIdx={outIdx}"
  | none => pure ()

  match e.params.findNat? .stmtIdx0 with
  | some idx0 =>
    parts := parts.push s!"stmtIdx0={idx0}"
  | none =>
    match e.source.line? with
    | some line =>
      if line > 0 then
        parts := parts.push s!"stmtIdx0~{line - 1}"
      else
        pure ()
    | none => pure ()

  match e.params.findNat? .stmtIdx1 with
  | some idx1 =>
    parts := parts.push s!"stmtIdx1={idx1}"
  | none =>
    match e.source.line? with
    | some line =>
      if line > 0 then
        parts := parts.push s!"stmtIdx1~{line}"
      else
        pure ()
    | none => pure ()

  match e.params.findName? .opTag with
  | some opParam =>
    parts := parts.push s!"opParam={opParam}"
  | none => pure ()

  match e.params.findName? .sourceOp with
  | some sourceOp =>
    parts := parts.push s!"sourceOp={sourceOp}"
  | none => pure ()

  return parts

def coverageErrorToString (e : CoverageError) : String :=
  let source := coverageFormatSourceRef e.source
  let outvars := coverageOutvarsToString e.outvarIds
  let metadataParts := coverageMetadataParts e
  let metaPart :=
    if metadataParts.isEmpty then
      ""
    else
      s!"; meta={String.intercalate ", " metadataParts.toList}"
  s!"{e.message}: op `{e.op}` at eqn #{e.eqnIndex0}, outvars={outvars}, source={source}{metaPart}"

/-- Gather missing local-Jacobian rules for all equations in the jaxpr. -/
def checkRuleCoverage (jaxpr : LeanJaxpr) : Lean.CoreM (Array CoverageError) := do
  let mut errors : Array CoverageError := #[]
  for h : i in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[i]
    if (← getLocalJacRule? eqn.op).isNone then
      errors := errors.push {
        op := eqn.op
        eqnIndex0 := i
        source := eqn.source
        outvarIds := eqn.outvars.map (·.id)
        params := eqn.params
        message := "Missing local-Jacobian rule"
      }
  return errors

/-- Fail when any equation lacks a registered local-Jacobian rule. -/
def requireFullRuleCoverage (jaxpr : LeanJaxpr) : Lean.CoreM (Except (Array CoverageError) Unit) := do
  let errors ← checkRuleCoverage jaxpr
  if errors.isEmpty then
    return .ok ()
  else
    return .error errors

end Tyr.AD.JaxprLike
