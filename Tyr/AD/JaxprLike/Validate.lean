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

/-- Collect effective op IDs, using deterministic fallback numbering when absent. -/
def collectEffectiveOpIds (jaxpr : LeanJaxpr) : Array OpId :=
  (Array.range jaxpr.eqns.size).map (jaxpr.effectiveOpIdAt ·)

/-- Ensure declaration-site variable IDs are globally unique. -/
def validateUniqueVarIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectDeclaredVarIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique declared variable IDs detected."

/-- Ensure effective op IDs are globally unique after fallback numbering. -/
def validateUniqueOpIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectEffectiveOpIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique effective op IDs detected."

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

private def hasExplicitPartitions (jaxpr : LeanJaxpr) : Bool :=
  !(jaxpr.partitions.inputs.isEmpty &&
    jaxpr.partitions.outputs.isEmpty &&
    jaxpr.partitions.eliminable.isEmpty)

private def collectAvailableValueIds (jaxpr : LeanJaxpr) : Std.HashSet Nat := Id.run do
  let mut available := initAvailableVarIds jaxpr
  for eqn in jaxpr.eqns do
    for outvar in eqn.outvars do
      available := available.insert outvar.id
  return available

/-- Validate explicit partition metadata when it is present on the graph. -/
def validateExplicitPartitions (jaxpr : LeanJaxpr) : Except String Unit := do
  if !hasExplicitPartitions jaxpr then
    pure ()
  let raw := jaxpr.partitions
  if !hasNoDuplicates raw.inputs then
    throw "LeanJaxpr validation failed: explicit input partition contains duplicate value IDs."
  if !hasNoDuplicates raw.outputs then
    throw "LeanJaxpr validation failed: explicit output partition contains duplicate value IDs."
  if !hasNoDuplicates raw.eliminable then
    throw "LeanJaxpr validation failed: explicit eliminable partition contains duplicate value IDs."
  let available := collectAvailableValueIds jaxpr
  let boundary :=
    (raw.inputs ++ raw.outputs).foldl (init := ({} : Std.HashSet Nat)) fun acc v => acc.insert v
  match raw.eliminable.find? (fun v => boundary.contains v) with
  | some bad =>
    throw s!"LeanJaxpr validation failed: explicit eliminable partition references boundary value ID {bad}."
  | none => pure ()
  let mut allPartitionIds := raw.inputs ++ raw.outputs ++ raw.eliminable
  match allPartitionIds.find? (fun v => !available.contains v) with
  | some bad =>
    throw s!"LeanJaxpr validation failed: explicit partition references unavailable value ID {bad}."
  | none => pure ()

/-- Validate the deterministic action surface derived for AlphaGrad/Graphax use. -/
def validateActionTable (jaxpr : LeanJaxpr) : Except String Unit := do
  let bindings := jaxpr.actionTable.bindings
  let validOpIds :=
    (collectEffectiveOpIds jaxpr).foldl (init := ({} : Std.HashSet OpId)) fun acc id => acc.insert id
  let eliminable :=
    jaxpr.eliminableGraphVertices.foldl (init := ({} : Std.HashSet Nat)) fun acc v => acc.insert v
  let boundary :=
    (jaxpr.inputVertices ++ jaxpr.outputVertices).foldl (init := ({} : Std.HashSet Nat)) fun acc v =>
      acc.insert v
  let mut expectedAction : Nat := 0
  for binding in bindings do
    if binding.action0 != expectedAction then
      throw s!"LeanJaxpr validation failed: action table expected action slot {expectedAction}, got {binding.action0}."
    if binding.vertex1 != binding.action0 + 1 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} must map to vertex {binding.action0 + 1}, got {binding.vertex1}."
    match binding.producerOpId? with
    | some opId =>
      if !validOpIds.contains opId then
        throw s!"LeanJaxpr validation failed: action slot {binding.action0} references unknown producer op ID {opId}."
    | none => pure ()
    if binding.isEliminable && !eliminable.contains binding.vertex1 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} marks non-eliminable vertex {binding.vertex1} eliminable."
    if binding.isBoundary && !boundary.contains binding.vertex1 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} marks non-boundary vertex {binding.vertex1} as boundary."
    expectedAction := expectedAction + 1

/-- Aggregate validation pass used before elimination planning. -/
def validate (jaxpr : LeanJaxpr) : Except (Array String) Unit :=
  let errors := Id.run do
    let mut es : Array String := #[]
    if let .error msg := validateUniqueVarIds jaxpr then
      es := es.push msg
    if let .error msg := validateUniqueOpIds jaxpr then
      es := es.push msg
    if let .error msg := validateEqnOutvarsNonEmpty jaxpr then
      es := es.push msg
    if let .error msg := validateEqnInputsTopological jaxpr then
      es := es.push msg
    if let .error msg := validateOutvarsAvailable jaxpr then
      es := es.push msg
    if let .error msg := validateExplicitPartitions jaxpr then
      es := es.push msg
    if let .error msg := validateActionTable jaxpr then
      es := es.push msg
    return es
  if errors.isEmpty then .ok () else .error errors

end Tyr.AD.JaxprLike
