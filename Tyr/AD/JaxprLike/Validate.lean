import Tyr.AD.JaxprLike.TypedOps
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

/-- Collect normalized op IDs. -/
def collectEffectiveOpIds (jaxpr : LeanJaxpr) : Array OpId :=
  jaxpr.eqns.map (·.id)

/-- Collect explicit higher-order region IDs. -/
def collectRegionIds (jaxpr : LeanJaxpr) : Array RegionId :=
  jaxpr.regions.map (·.id)

/-- Ensure declaration-site variable IDs are globally unique. -/
def validateUniqueVarIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectDeclaredVarIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique declared variable IDs detected."

/-- Ensure normalized op IDs are globally unique. -/
def validateUniqueOpIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectEffectiveOpIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique op IDs detected."

/-- Ensure region IDs are globally unique. -/
def validateUniqueRegionIds (jaxpr : LeanJaxpr) : Except String Unit :=
  if hasNoDuplicates (collectRegionIds jaxpr) then
    .ok ()
  else
    .error "LeanJaxpr validation failed: non-unique region IDs detected."

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

private def collectAvailableValueIds (jaxpr : LeanJaxpr) : Std.HashSet Nat := Id.run do
  let mut available := initAvailableVarIds jaxpr
  for eqn in jaxpr.eqns do
    for outvar in eqn.outvars do
      available := available.insert outvar.id
  return available

private def typedPayloadCompatible (typed : TypedOp) : Bool :=
  match typed.schema, typed.payload with
  | .generic, .none => true
  | .nullary, .nullary _ => true
  | .unary, .unary _ => true
  | .binary, .binary _ => true
  | .ternary, .ternary _ => true
  | .nary, .nary _ _ => true
  | .reduce, .reduce _ _ => true
  | .reduceAccum, .reduce _ _ => true
  | .broadcast, .broadcast _ => true
  | .binaryBroadcast, .binaryBroadcast _ _ => true
  | .transpose, .none => true
  | .swapLayout, .none => true
  | .convert, .none => true
  | .sliceRows, .sliceRows _ _ => true
  | .sliceCols, .sliceCols _ _ => true
  | .concatCols, .none => true
  | .outer, .none => true
  | .dotGeneral, .dotGeneral _ _ _ _ _ => true
  | .mma, .variant _ => true
  | .cumsum, .broadcast _ => true
  | .cumprod, .broadcast _ => true
  | .controlFlow, .controlFlow _ => true
  | _, _ => false

private def validateTypedEqnSemantics (eqnIdx0 : Nat) (eqn : JEqn) : Except String Unit := do
  match typedOpForNormalizedOp? eqn.op eqn.params eqn.invars.size eqn.outvars.size with
  | some expected =>
    if eqn.typed != expected then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` has typed semantics {reprStr eqn.typed}, expected {reprStr expected} from normalized op metadata."
  | none =>
    pure ()

private def validateTypedEqnSchema (eqnIdx0 : Nat) (eqn : JEqn) : Except String Unit := do
  if !typedPayloadCompatible eqn.typed then
    throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` has inconsistent typed schema/payload {reprStr eqn.typed}."
  let requireCounts (expectedInputs expectedOutputs : Nat) : Except String Unit := do
    if eqn.invars.size != expectedInputs then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed as {reprStr eqn.typed.schema} expects {expectedInputs} inputs, got {eqn.invars.size}."
    if eqn.outvars.size != expectedOutputs then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed as {reprStr eqn.typed.schema} expects {expectedOutputs} outputs, got {eqn.outvars.size}."
  match eqn.typed.schema with
  | .generic =>
    throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` still uses the generic typed schema. Construct normalized equations with explicit typed semantics."
  | .nullary =>
    requireCounts 0 1
  | .nary =>
    match eqn.typed.payload with
    | .nary _ arity =>
      requireCounts arity 1
    | _ =>
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` n-ary schema is missing arity payload."
  | .unary
  | .reduce
  | .broadcast
  | .transpose
  | .swapLayout
  | .convert
  | .sliceRows
  | .sliceCols
  | .cumsum
  | .cumprod =>
    requireCounts 1 1
  | .reduceAccum =>
    requireCounts 2 1
  | .binary
  | .binaryBroadcast
  | .concatCols
  | .outer
  | .dotGeneral =>
    requireCounts 2 1
  | .ternary
  | .mma =>
    requireCounts 3 1
  | .controlFlow =>
    match eqn.typed.payload with
    | .controlFlow info =>
      let expectedInputs :=
        info.predicateCount + info.dataInputCount + info.carryInputCount
      if eqn.invars.size != expectedInputs then
        throw s!"LeanJaxpr validation failed: equation {eqnIdx0} control-flow op `{eqn.op}` expects {expectedInputs} inputs from typed metadata, got {eqn.invars.size}."
      if eqn.outvars.isEmpty then
        throw s!"LeanJaxpr validation failed: equation {eqnIdx0} control-flow op `{eqn.op}` must produce at least one output."
      if info.variant == `scan && eqn.outvars.size < info.carryOutputCount then
        throw s!"LeanJaxpr validation failed: equation {eqnIdx0} scan op `{eqn.op}` declares {info.carryOutputCount} carry outputs, got only {eqn.outvars.size} outputs."
      if info.variant == `cond && info.predicateCount = 0 then
        throw s!"LeanJaxpr validation failed: equation {eqnIdx0} cond op `{eqn.op}` must declare at least one predicate input."
    | _ =>
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` control-flow schema is missing control-flow payload."

private def validateRegions (jaxpr : LeanJaxpr) : Except String Unit := do
  let regionIdSet :=
    (collectRegionIds jaxpr).foldl (init := ({} : Std.HashSet RegionId)) fun acc id =>
      acc.insert id
  for eqnIdx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[eqnIdx0]!
    match eqn.typed.payload with
    | .controlFlow info =>
      if !hasNoDuplicates info.regionIds then
        throw s!"LeanJaxpr validation failed: equation {eqnIdx0} control-flow op `{eqn.op}` references duplicate region IDs."
      for regionId in info.regionIds do
        if !regionIdSet.contains regionId then
          throw s!"LeanJaxpr validation failed: equation {eqnIdx0} control-flow op `{eqn.op}` references unknown region ID {regionId}."
      if info.variant == `scan && !(info.regionIds.isEmpty || info.regionIds.size = 1) then
        throw s!"LeanJaxpr validation failed: scan op `{eqn.op}` expects at most one body region, got {info.regionIds.size}."
      if info.variant == `cond && !(info.regionIds.isEmpty || info.regionIds.size = 2) then
        throw s!"LeanJaxpr validation failed: cond op `{eqn.op}` expects exactly two branch regions when regions are present, got {info.regionIds.size}."
      for regionId in info.regionIds do
        match jaxpr.regionById? regionId with
        | none =>
          throw s!"LeanJaxpr validation failed: missing region {regionId} for equation {eqnIdx0}."
        | some region =>
          let expectedInputs :=
            if info.variant == `cond then
              info.dataInputCount
            else
              info.carryInputCount + info.dataInputCount
          if region.invars.size != expectedInputs then
            throw s!"LeanJaxpr validation failed: region {regionId} (`{region.role}`) expects {expectedInputs} region inputs from control-flow metadata, got {region.invars.size}."
          if region.outvars.size != eqn.outvars.size then
            throw s!"LeanJaxpr validation failed: region {regionId} (`{region.role}`) expects {eqn.outvars.size} outputs to match op `{eqn.op}`, got {region.outvars.size}."
    | _ => pure ()

private def validateLegacyParamsAgreeWithTypedEqn (eqnIdx0 : Nat) (eqn : JEqn) : Except String Unit := do
  match eqn.typed.payload with
  | .dotGeneral variant lhsContract rhsContract lhsBatch rhsBatch =>
    if (eqn.params.findName? .variant).isSome && (eqn.params.findName? .variant != some variant) then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed dot_general variant disagrees with params."
    if (eqn.params.findNats? .lhsContract).isSome && (eqn.params.findNats? .lhsContract != some lhsContract) then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed lhsContract disagrees with params."
    if (eqn.params.findNats? .rhsContract).isSome && (eqn.params.findNats? .rhsContract != some rhsContract) then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed rhsContract disagrees with params."
    if (eqn.params.findNats? .lhsBatch).isSome && (eqn.params.findNats? .lhsBatch != some lhsBatch) then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed lhsBatch disagrees with params."
    if (eqn.params.findNats? .rhsBatch).isSome && (eqn.params.findNats? .rhsBatch != some rhsBatch) then
      throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed rhsBatch disagrees with params."
  | .controlFlow info =>
    let checkNat (key : OpParamKey) (expected : Nat) (label : String) : Except String Unit := do
      match eqn.params.findNat? key with
      | some actual =>
        if actual != expected then
          throw s!"LeanJaxpr validation failed: equation {eqnIdx0} op `{eqn.op}` typed {label}={expected} disagrees with params {label}={actual}."
      | none => pure ()
    checkNat .controlStaticArgCount info.staticArgCount "controlStaticArgCount"
    checkNat .condPredicateCount info.predicateCount "condPredicateCount"
    checkNat .condDataInputCount info.dataInputCount "condDataInputCount"
    checkNat .scanDataInputCount info.dataInputCount "scanDataInputCount"
    checkNat .scanCarryInputCount info.carryInputCount "scanCarryInputCount"
    checkNat .scanCarryOutputCount info.carryOutputCount "scanCarryOutputCount"
  | _ =>
    pure ()

private def validateValueRoleMetadata (jaxpr : LeanJaxpr) : Except String Unit := do
  let checkVar (site : String) (v : JVar) : Except String Unit := do
    let expected :=
      if jaxpr.outvars.any (fun outv => outv.id = v.id) then
        ValueRole.output
      else if jaxpr.constvars.any (fun constv => constv.id = v.id) then
        ValueRole.const
      else if jaxpr.invars.any (fun inv => inv.id = v.id) then
        ValueRole.input
      else
        ValueRole.intermediate
    match v.metaInfo.role? with
    | none =>
      throw s!"LeanJaxpr validation failed: {site} variable {v.id} is missing normalized ValueRole metadata."
    | some actual =>
      if actual != expected then
        throw s!"LeanJaxpr validation failed: {site} variable {v.id} has role {reprStr actual}, expected {reprStr expected}."
  for v in jaxpr.constvars do
    checkVar "const" v
  for v in jaxpr.invars do
    checkVar "input" v
  for v in jaxpr.outvars do
    checkVar "output" v
  for eqnIdx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[eqnIdx0]!
    for v in eqn.invars do
      checkVar s!"eqn {eqnIdx0} input" v
    for v in eqn.outvars do
      checkVar s!"eqn {eqnIdx0} output" v

def validateTypedEqns (jaxpr : LeanJaxpr) : Except String Unit := do
  for eqnIdx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[eqnIdx0]!
    validateTypedEqnSemantics eqnIdx0 eqn
    validateTypedEqnSchema eqnIdx0 eqn
    validateLegacyParamsAgreeWithTypedEqn eqnIdx0 eqn

/-- Normalized graphs must carry the materialized partition metadata explicitly. -/
def validateStoredPartitions (jaxpr : LeanJaxpr) : Except String Unit :=
  if jaxpr.partitions == LeanJaxpr.inferVertexPartitions jaxpr then
    .ok ()
  else
    .error "LeanJaxpr validation failed: stored graph partitions are missing or stale; construct a normalized LeanJaxpr."

/-- Normalized graphs must carry the materialized action table explicitly. -/
def validateStoredActionTable (jaxpr : LeanJaxpr) : Except String Unit :=
  if jaxpr.actions == LeanJaxpr.inferActionTable jaxpr then
    .ok ()
  else
    .error "LeanJaxpr validation failed: stored action table is missing or stale; construct a normalized LeanJaxpr."

/-- Validate explicit partition metadata when it is present on the graph. -/
def validateExplicitPartitions (jaxpr : LeanJaxpr) : Except String Unit := do
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
  if bindings.size != jaxpr.eliminableGraphVertices.size then
    throw s!"LeanJaxpr validation failed: action table size {bindings.size} does not match eliminable partition size {jaxpr.eliminableGraphVertices.size}."
  let mut expectedAction : Nat := 0
  for binding in bindings do
    if binding.action0 != expectedAction then
      throw s!"LeanJaxpr validation failed: action table expected action slot {expectedAction}, got {binding.action0}."
    if binding.vertex1 = 0 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} references non-positive vertex ID 0."
    if !eliminable.contains binding.vertex1 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} references non-eliminable vertex {binding.vertex1}."
    match binding.producerOpId? with
    | some opId =>
      if !validOpIds.contains opId then
        throw s!"LeanJaxpr validation failed: action slot {binding.action0} references unknown producer op ID {opId}."
    | none => pure ()
    if !binding.isEliminable then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} must be marked eliminable."
    if binding.isBoundary then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} may not mark eliminable vertex {binding.vertex1} as boundary."
    if boundary.contains binding.vertex1 then
      throw s!"LeanJaxpr validation failed: action slot {binding.action0} references boundary vertex {binding.vertex1}."
    expectedAction := expectedAction + 1

/-- Aggregate validation pass used before elimination planning. -/
def validate (jaxpr : LeanJaxpr) : Except (Array String) Unit :=
  let errors := Id.run do
    let mut es : Array String := #[]
    if let .error msg := validateUniqueVarIds jaxpr then
      es := es.push msg
    if let .error msg := validateUniqueOpIds jaxpr then
      es := es.push msg
    if let .error msg := validateUniqueRegionIds jaxpr then
      es := es.push msg
    if let .error msg := validateEqnOutvarsNonEmpty jaxpr then
      es := es.push msg
    if let .error msg := validateEqnInputsTopological jaxpr then
      es := es.push msg
    if let .error msg := validateOutvarsAvailable jaxpr then
      es := es.push msg
    if let .error msg := validateTypedEqns jaxpr then
      es := es.push msg
    if let .error msg := validateRegions jaxpr then
      es := es.push msg
    if let .error msg := validateValueRoleMetadata jaxpr then
      es := es.push msg
    if let .error msg := validateStoredPartitions jaxpr then
      es := es.push msg
    if let .error msg := validateExplicitPartitions jaxpr then
      es := es.push msg
    if let .error msg := validateStoredActionTable jaxpr then
      es := es.push msg
    if let .error msg := validateActionTable jaxpr then
      es := es.push msg
    return es
  if errors.isEmpty then .ok () else .error errors

end Tyr.AD.JaxprLike
