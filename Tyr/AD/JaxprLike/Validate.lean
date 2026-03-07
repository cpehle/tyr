import Tyr.AD.JaxprLike.VertexOrder

/-!
# Tyr.AD.JaxprLike.Validate

Validation helpers for LeanJaxpr-like IR invariants.
The checks in this file are intentionally conservative and correctness-first.
-/

namespace Tyr.AD.JaxprLike

/-- True when `xs` contains no duplicates. -/
def hasNoDuplicates (xs : Array Nat) : Bool := Id.run do
  let mut seen : Std.HashSet Nat := {}
  for x in xs do
    if seen.contains x then
      return false
    seen := seen.insert x
  return true

/--
Collect variable IDs introduced at declaration sites.
This intentionally excludes usage sites (`eqn.invars`, `jaxpr.outvars`) where repeated IDs are expected.
-/
def collectDeclaredVarIds (jaxpr : LeanJaxpr) : Array Nat :=
  let eqnOutIds := jaxpr.eqns.foldl (init := #[]) fun acc eqn =>
    acc ++ eqn.outvars.map (·.id)
  jaxpr.constvars.map (·.id) ++
    jaxpr.invars.map (·.id) ++
    eqnOutIds

/-- Ensure declaration-site variable IDs are globally unique. -/
def validateUniqueVarIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectDeclaredVarIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique declared variable IDs detected."

/-- Ensure each equation produces at least one output variable. -/
def validateEqnOutvarsNonEmpty (jaxpr : LeanJaxpr) : Except String Unit :=
  match jaxpr.eqns.findIdx? (fun eqn => eqn.outvars.isEmpty) with
  | some idx => .error s!"LeanJaxpr validation failed: equation {idx} has no outputs."
  | none => .ok ()

private def initAvailableVarIds (jaxpr : LeanJaxpr) : Std.HashSet Nat := Id.run do
  let mut available : Std.HashSet Nat := {}
  for v in jaxpr.constvars do
    available := available.insert v.id
  for v in jaxpr.invars do
    available := available.insert v.id
  return available

/--
Ensure every equation input is available at its use site:
- declared in `constvars`
- declared in `invars`
- or produced by an earlier equation.
-/
def validateEqnInputsTopological (jaxpr : LeanJaxpr) : Except String Unit := Id.run do
  let mut available := initAvailableVarIds jaxpr
  for hEqn : eqnIdx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[eqnIdx0]
    let vertexId := vertexIdOfEqnIdx0 eqnIdx0
    for hIn : inIdx0 in [:eqn.invars.size] do
      let invar := eqn.invars[inIdx0]
      if !available.contains invar.id then
        return .error
          s!"LeanJaxpr validation failed: equation {eqnIdx0} (vertex {vertexId}) input {inIdx0} references unavailable variable ID {invar.id}. Expected declaration in constvars/invars or production by an earlier equation."
    for outvar in eqn.outvars do
      available := available.insert outvar.id
  return .ok ()

/-- Ensure each jaxpr output references a declared or produced variable ID. -/
def validateOutvarsAvailable (jaxpr : LeanJaxpr) : Except String Unit :=
  let available := Id.run do
    let mut available := initAvailableVarIds jaxpr
    for eqn in jaxpr.eqns do
      for outvar in eqn.outvars do
        available := available.insert outvar.id
    return available
  match jaxpr.outvars.findIdx? (fun outvar => !available.contains outvar.id) with
  | some outIdx0 =>
    let outvar := jaxpr.outvars[outIdx0]!
    .error
      s!"LeanJaxpr validation failed: output {outIdx0} references unavailable variable ID {outvar.id}. Outputs must be declared in constvars/invars or produced by an equation."
  | none =>
    .ok ()

/-- Aggregate validation pass used before elimination planning. -/
def validate (jaxpr : LeanJaxpr) : Except (Array String) Unit :=
  let errors := Id.run do
    let mut es : Array String := #[]
    if let .error msg := validateUniqueVarIds jaxpr then
      es := es.push msg
    if let .error msg := validateEqnOutvarsNonEmpty jaxpr then
      es := es.push msg
    if let .error msg := validateEqnInputsTopological jaxpr then
      es := es.push msg
    if let .error msg := validateOutvarsAvailable jaxpr then
      es := es.push msg
    return es
  if errors.isEmpty then .ok () else .error errors

end Tyr.AD.JaxprLike
