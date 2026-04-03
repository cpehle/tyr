import Std.Data.HashSet
import Lean.Compiler.IR.Basic

namespace Tyr.AD

/-- Shared 0-based action-space index used across normalized graphs and elimination search. -/
abbrev ActionId0 := Nat

end Tyr.AD

/-!
# Tyr.AD.JaxprLike.Core

Core data structures for a Jaxpr-like IR layer in Tyr.
This layer is intentionally lightweight and independent from elimination execution,
but now carries enough structured metadata to serve as a shared contract for
Graphax/AlphaGrad-oriented elimination planning.
-/

namespace Tyr.AD.JaxprLike

open Lean
open Lean.IR

/-- Stable ID for variables in the LeanJaxpr-like representation. -/
abbrev JVarId := Nat

/-- Stable ID for normalized equations/ops. -/
abbrev OpId := Nat

/-- Primitive/op identifier, mirroring Graphax's equation-level primitive naming. -/
abbrev OpName := Name

/-- Typed key space for equation-level op metadata. -/
inductive OpParamKey where
  | loweringKind
  | kind
  | opTag
  | axis
  | stmtIdx0
  | stmtIdx1
  | fnbodyOutVarIdx
  | startRow
  | numRows
  | startCol
  | numCols
  | lhsContract
  | rhsContract
  | lhsBatch
  | rhsBatch
  | padLow
  | padHigh
  | padInterior
  | variant
  | sourceOp
  | controlStaticArgCount
  | condPredicateCount
  | condDataInputCount
  | scanCarryInputCount
  | scanDataInputCount
  | scanCarryOutputCount
  | custom (name : String)
  deriving Repr, BEq, Inhabited, DecidableEq, Hashable

def OpParamKey.toString : OpParamKey → String
  | .loweringKind => "loweringKind"
  | .kind => "kind"
  | .opTag => "op"
  | .axis => "axis"
  | .stmtIdx0 => "stmtIdx0"
  | .stmtIdx1 => "stmtIdx1"
  | .fnbodyOutVarIdx => "fnbodyOutVarIdx"
  | .startRow => "startRow"
  | .numRows => "numRows"
  | .startCol => "startCol"
  | .numCols => "numCols"
  | .lhsContract => "lhsContract"
  | .rhsContract => "rhsContract"
  | .lhsBatch => "lhsBatch"
  | .rhsBatch => "rhsBatch"
  | .padLow => "padLow"
  | .padHigh => "padHigh"
  | .padInterior => "padInterior"
  | .variant => "variant"
  | .sourceOp => "sourceOp"
  | .controlStaticArgCount => "controlStaticArgCount"
  | .condPredicateCount => "condPredicateCount"
  | .condDataInputCount => "condDataInputCount"
  | .scanCarryInputCount => "scanCarryInputCount"
  | .scanDataInputCount => "scanDataInputCount"
  | .scanCarryOutputCount => "scanCarryOutputCount"
  | .custom name => name

instance : ToString OpParamKey := ⟨OpParamKey.toString⟩

/-- Typed value space for equation-level op metadata. -/
inductive OpParamValue where
  | nat (value : Nat)
  | name (value : Name)
  | nats (value : Array Nat)
  deriving Repr, BEq, Inhabited, DecidableEq

private def renderNatArray (xs : Array Nat) : String :=
  "[" ++ String.intercalate ", " (xs.toList.map (fun x => toString x)) ++ "]"

def OpParamValue.toString : OpParamValue → String
  | .nat value => s!"{value}"
  | .name value => s!"{value}"
  | .nats value => renderNatArray value

instance : ToString OpParamValue := ⟨OpParamValue.toString⟩

/-- Typed op metadata entry. -/
structure OpParam where
  key : OpParamKey
  value : OpParamValue
  deriving Repr, BEq, Inhabited

namespace OpParam

def mkNat (key : OpParamKey) (value : Nat) : OpParam :=
  { key := key, value := .nat value }

def mkName (key : OpParamKey) (value : Name) : OpParam :=
  { key := key, value := .name value }

def mkNats (key : OpParamKey) (value : Array Nat) : OpParam :=
  { key := key, value := .nats value }

end OpParam

/-- Typed parameter bag used by lowered equations. -/
abbrev OpParams := Array OpParam

namespace OpParams

def findValue? (params : OpParams) (key : OpParamKey) : Option OpParamValue := Id.run do
  for p in params do
    if p.key == key then
      return some p.value
  return none

def containsKey (params : OpParams) (key : OpParamKey) : Bool :=
  params.any (fun p => p.key == key)

def findNat? (params : OpParams) (key : OpParamKey) : Option Nat := do
  let value ← findValue? params key
  match value with
  | .nat n => some n
  | .name _ => none
  | .nats _ => none

def findName? (params : OpParams) (key : OpParamKey) : Option Name := do
  let value ← findValue? params key
  match value with
  | .name n => some n
  | .nat _ => none
  | .nats _ => none

def findNats? (params : OpParams) (key : OpParamKey) : Option (Array Nat) := do
  let value ← findValue? params key
  match value with
  | .nats ns => some ns
  | .nat _ => none
  | .name _ => none

/--
Merge two parameter bags while preferring entries from `overrides` when a key is
present in both collections. The left-to-right order of surviving entries is
preserved.
-/
def mergePreferRight (base overrides : OpParams) : OpParams :=
  base.filter (fun p => !(overrides.containsKey p.key)) ++ overrides

end OpParams

/-- AD participation marker carried on normalized variables. -/
inductive DiffParticipation where
  | diff
  | static
  | frozen
  deriving Repr, BEq, Inhabited, DecidableEq

/-- Semantic role of a value within the normalized graph boundary. -/
inductive ValueRole where
  | const
  | input
  | output
  | intermediate
  | parameter
  deriving Repr, BEq, Inhabited, DecidableEq

/-- AD-relevant metadata that survives normalization. -/
structure VarMeta where
  participation : DiffParticipation := .diff
  shape : Option (Array Nat) := none
  dtype : Option String := none
  sharding : Option String := none
  aliasGroup? : Option Nat := none
  role? : Option ValueRole := none
  deriving Repr, Inhabited

/--
Optional metadata hints for `FnBody -> LeanJaxpr` lowering.

- `varMetaByIrVar`: extra `VarMeta` keyed by Lean IR binder/parameter `VarId.idx`
- `eqnParamsByOutIrVar`: extra equation params keyed by the output binder
  `VarId.idx` of the corresponding `FnBody.vdecl`
-/
structure FnBodyLoweringHints where
  varMetaByIrVar : Std.HashMap Nat VarMeta := {}
  eqnParamsByOutIrVar : Std.HashMap Nat OpParams := {}
  deriving Repr, Inhabited

namespace FnBodyLoweringHints

private def mergeHashMapPreferRight
    {α β : Type}
    [BEq α] [Hashable α]
    (base overrides : Std.HashMap α β) :
    Std.HashMap α β := Id.run do
  let mut out := base
  for (k, v) in overrides.toList do
    out := out.insert k v
  return out

/--
Merge two hint packs while preferring metadata from `overrides` on key
collisions. This lets traced frontends register baseline metadata and still
allow more specific call-site overrides later.
-/
def mergePreferRight
    (base overrides : FnBodyLoweringHints) :
    FnBodyLoweringHints :=
  { varMetaByIrVar := mergeHashMapPreferRight base.varMetaByIrVar overrides.varMetaByIrVar
    eqnParamsByOutIrVar :=
      mergeHashMapPreferRight base.eqnParamsByOutIrVar overrides.eqnParamsByOutIrVar }

end FnBodyLoweringHints

/-- Source location metadata for diagnostics and coverage errors. -/
structure SourceRef where
  decl : Name := .anonymous
  line? : Option Nat := none
  col? : Option Nat := none
  deriving Repr, Inhabited

/-- Schema family for typed normalized ops. -/
inductive OpSchema where
  | generic
  | nullary
  | unary
  | binary
  | ternary
  | nary
  | reduce
  | reduceAccum
  | broadcast
  | binaryBroadcast
  | transpose
  | swapLayout
  | convert
  | sliceRows
  | sliceCols
  | concatCols
  | outer
  | dotGeneral
  | mma
  | cumsum
  | cumprod
  | controlFlow
  deriving Repr, BEq, Inhabited, DecidableEq

/-- Structured payload for higher-order control ops. -/
structure ControlFlowInfo where
  variant : Name
  staticArgCount : Nat := 0
  predicateCount : Nat := 0
  dataInputCount : Nat := 0
  carryInputCount : Nat := 0
  carryOutputCount : Nat := 0
  deriving Repr, BEq, Inhabited

/-- Typed normalized op payload used instead of ad hoc metadata where possible. -/
inductive OpPayload where
  | none
  | nullary (tag : Name)
  | unary (tag : Name)
  | binary (tag : Name)
  | ternary (tag : Name)
  | nary (tag : Name) (arity : Nat)
  | reduce (tag axis : Name)
  | broadcast (axis : Name)
  | binaryBroadcast (tag axis : Name)
  | sliceRows (startRow numRows : Nat)
  | sliceCols (startCol numCols : Nat)
  | dotGeneral
      (variant : Name)
      (lhsContract rhsContract lhsBatch rhsBatch : Array Nat)
  | variant (name : Name)
  | controlFlow (info : ControlFlowInfo)
  deriving Repr, BEq, Inhabited

/-- Typed op descriptor shared by graph construction, elimination, and policies. -/
structure TypedOp where
  schema : OpSchema := .generic
  payload : OpPayload := .none
  deriving Repr, BEq, Inhabited

namespace TypedOp

def generic : TypedOp := {}

def nullary (tag : Name) : TypedOp :=
  { schema := .nullary, payload := .nullary tag }

def unary (tag : Name) : TypedOp :=
  { schema := .unary, payload := .unary tag }

def binary (tag : Name) : TypedOp :=
  { schema := .binary, payload := .binary tag }

def ternary (tag : Name) : TypedOp :=
  { schema := .ternary, payload := .ternary tag }

def nary (tag : Name) (arity : Nat) : TypedOp :=
  { schema := .nary, payload := .nary tag arity }

def reduce (tag axis : Name) : TypedOp :=
  { schema := .reduce, payload := .reduce tag axis }

def reduceAccum (tag axis : Name) : TypedOp :=
  { schema := .reduceAccum, payload := .reduce tag axis }

def broadcast (axis : Name) : TypedOp :=
  { schema := .broadcast, payload := .broadcast axis }

def binaryBroadcast (tag axis : Name) : TypedOp :=
  { schema := .binaryBroadcast, payload := .binaryBroadcast tag axis }

def transpose : TypedOp :=
  { schema := .transpose }

def swapLayout : TypedOp :=
  { schema := .swapLayout }

def convert : TypedOp :=
  { schema := .convert }

def sliceRows (startRow numRows : Nat) : TypedOp :=
  { schema := .sliceRows, payload := .sliceRows startRow numRows }

def sliceCols (startCol numCols : Nat) : TypedOp :=
  { schema := .sliceCols, payload := .sliceCols startCol numCols }

def concatCols : TypedOp :=
  { schema := .concatCols }

def outer : TypedOp :=
  { schema := .outer }

def dotGeneral
    (variant : Name)
    (lhsContract rhsContract lhsBatch rhsBatch : Array Nat) : TypedOp :=
  {
    schema := .dotGeneral
    payload := .dotGeneral variant lhsContract rhsContract lhsBatch rhsBatch
  }

def mma (variant : Name) : TypedOp :=
  { schema := .mma, payload := .variant variant }

def cumsum (axis : Name) : TypedOp :=
  { schema := .cumsum, payload := .broadcast axis }

def cumprod (axis : Name) : TypedOp :=
  { schema := .cumprod, payload := .broadcast axis }

def controlFlow (info : ControlFlowInfo) : TypedOp :=
  { schema := .controlFlow, payload := .controlFlow info }

end TypedOp

/-- Explicit graph partitions used by Graphax-style elimination helpers. -/
structure VertexPartitions where
  inputs : Array JVarId := #[]
  outputs : Array JVarId := #[]
  eliminable : Array JVarId := #[]
  deriving Repr, Inhabited, BEq

/-- One action slot in the fixed AlphaGrad/Graphax action surface. -/
structure ActionBinding where
  action0 : ActionId0
  vertex1 : Nat
  valueId? : Option JVarId := none
  producerOpId? : Option OpId := none
  role? : Option ValueRole := none
  isBoundary : Bool := false
  isEliminable : Bool := false
  deriving Repr, Inhabited, BEq

/-- Deterministic action table derived from the explicit eliminable graph partition. -/
structure ActionTable where
  bindings : Array ActionBinding := #[]
  deriving Repr, Inhabited, BEq

namespace ActionTable

def isEmpty (table : ActionTable) : Bool :=
  table.bindings.isEmpty

def vertices1 (table : ActionTable) : Array Nat :=
  table.bindings.map (·.vertex1)

end ActionTable

/-- Variable in LeanJaxpr-like IR. -/
structure JVar where
  id : JVarId
  ty : IRType := .object
  metaInfo : VarMeta := {}
  deriving Repr, Inhabited

/-- Equation in LeanJaxpr-like IR. -/
structure JEqn where
  id : OpId
  op : OpName
  invars : Array JVar
  outvars : Array JVar
  params : OpParams := #[]
  typed : TypedOp
  source : SourceRef := {}
  deriving Repr, Inhabited

namespace JEqn

/-- Normalized/canonical op name used for rule lookup and lowering. -/
def normalizedOpName (eqn : JEqn) : OpName :=
  eqn.op

/-- Source/frontend op name when preserved by lowering, otherwise the normalized op. -/
def sourceOpName (eqn : JEqn) : OpName :=
  (eqn.params.findName? .sourceOp).getD eqn.op

/-- Typed structured op for the normalized equation. -/
def typedOp (eqn : JEqn) : TypedOp :=
  eqn.typed

private def opNameString (op : OpName) : String :=
  toString op

private def opNameContains (op : OpName) (needle : String) : Bool :=
  (opNameString op).contains needle

private def looksLikeDotGeneral (op : OpName) : Bool :=
  opNameContains op "dot_general"

private def looksLikeTranspose (op : OpName) : Bool :=
  opNameContains op "transpose"

private def looksLikeConvert (op : OpName) : Bool :=
  opNameContains op "convert"

private def looksLikeCond (op : OpName) : Bool :=
  opNameContains op "cond"

private def looksLikeScan (op : OpName) : Bool :=
  opNameContains op "scan"

private def inferredControlFlowInfo? (eqn : JEqn) : Option ControlFlowInfo :=
  if looksLikeCond eqn.op || (eqn.params.findNat? .condPredicateCount).isSome then
    some {
      variant := `cond
      staticArgCount := (eqn.params.findNat? .controlStaticArgCount).getD 0
      predicateCount := (eqn.params.findNat? .condPredicateCount).getD 0
      dataInputCount := (eqn.params.findNat? .condDataInputCount).getD 0
    }
  else if looksLikeScan eqn.op || (eqn.params.findNat? .scanCarryInputCount).isSome then
    some {
      variant := `scan
      staticArgCount := (eqn.params.findNat? .controlStaticArgCount).getD 0
      dataInputCount := (eqn.params.findNat? .scanDataInputCount).getD 0
      carryInputCount := (eqn.params.findNat? .scanCarryInputCount).getD 0
      carryOutputCount := (eqn.params.findNat? .scanCarryOutputCount).getD 0
    }
  else
    none

/--
Infer a structured typed op for hand-authored/manual equations that were built
without an explicit typed payload. Production lowerers should still populate
`typed` directly instead of relying on this normalization pass.
-/
def inferTypedOp (eqn : JEqn) : TypedOp :=
  if looksLikeDotGeneral eqn.op then
    TypedOp.dotGeneral
      ((eqn.params.findName? .variant).getD `generic)
      ((eqn.params.findNats? .lhsContract).getD #[])
      ((eqn.params.findNats? .rhsContract).getD #[])
      ((eqn.params.findNats? .lhsBatch).getD #[])
      ((eqn.params.findNats? .rhsBatch).getD #[])
  else
    match inferredControlFlowInfo? eqn with
    | some info =>
      TypedOp.controlFlow info
    | none =>
      if looksLikeTranspose eqn.op then
        TypedOp.transpose
      else if looksLikeConvert eqn.op then
        TypedOp.convert
      else
        match eqn.invars.size with
        | 0 => TypedOp.nullary eqn.op
        | 1 => TypedOp.unary eqn.op
        | 2 => TypedOp.binary eqn.op
        | 3 => TypedOp.ternary eqn.op
        | arity => TypedOp.nary eqn.op arity

/--
Normalized manual equation helper for tests/fixtures. This computes the typed
payload eagerly instead of relying on a later post-normalization pass.
-/
def ofNormalizedOp
    (id : OpId)
    (op : OpName)
    (invars outvars : Array JVar)
    (params : OpParams := #[])
    (source : SourceRef := {}) :
    JEqn :=
  let eqn : JEqn := {
    id := id
    op := op
    invars := invars
    outvars := outvars
    params := params
    typed := TypedOp.generic
    source := source
  }
  { eqn with typed := eqn.inferTypedOp }

/-- Primary output value ID when the equation has at least one output. -/
def primaryOutId? (eqn : JEqn) : Option JVarId :=
  eqn.outvars[0]?.map (·.id)

end JEqn

/-- Jaxpr-like normalized IR for elimination-based AD. -/
structure LeanJaxpr where
  constvars : Array JVar := #[]
  invars : Array JVar := #[]
  eqns : Array JEqn := #[]
  outvars : Array JVar := #[]
  partitions : VertexPartitions := {}
  actions : ActionTable := {}
  deriving Repr, Inhabited

namespace LeanJaxpr

/--
Populate anonymous equation source declarations with the owning declaration
name. Frontends that emit `LeanJaxpr` directly can omit per-equation `decl`
fields and still get stable diagnostics from `buildFromDecl`.
-/
def withDefaultSourceDecl
    (jaxpr : LeanJaxpr)
    (declName : Name) :
    LeanJaxpr :=
  { jaxpr with
      eqns := jaxpr.eqns.map fun eqn =>
        if eqn.source.decl == .anonymous then
          { eqn with source := { eqn.source with decl := declName } }
        else
          eqn }

private def dedupPreserveOrder (xs : Array JVarId) : Array JVarId := Id.run do
  let mut seen : Std.HashSet JVarId := {}
  let mut out : Array JVarId := #[]
  for x in xs do
    if !seen.contains x then
      seen := seen.insert x
      out := out.push x
  return out

/-- Graphax-style vertex numbering: equation index -> 1-based vertex ID. -/
def eqnVertexId1 (eqnIdx0 : Nat) : Nat :=
  eqnIdx0 + 1

/-- Inverse of `eqnVertexId1` with domain check for 1-based IDs. -/
def vertexToEqnIdx0? (vertexId1 : Nat) : Option Nat :=
  if vertexId1 = 0 then none else some (vertexId1 - 1)

/-- Default eliminable vertex set for a fully eliminable equation sequence. -/
def eliminableVertices1 (jaxpr : LeanJaxpr) : Array Nat :=
  (Array.range jaxpr.eqns.size).map eqnVertexId1

/-- Input-like graph vertices (`constvars ++ invars`) in declaration order. -/
private def derivedInputVertices (jaxpr : LeanJaxpr) : Array JVarId :=
  dedupPreserveOrder <| (jaxpr.constvars ++ jaxpr.invars).map (·.id)

/-- Output boundary graph vertices in declaration order. -/
private def derivedOutputVertices (jaxpr : LeanJaxpr) : Array JVarId :=
  dedupPreserveOrder <| jaxpr.outvars.map (·.id)

/--
Eliminable graph vertices in equation-topological order.
This tracks produced variables that are not final outputs.
-/
private def derivedEliminableGraphVertices (jaxpr : LeanJaxpr) : Array JVarId := Id.run do
  let outputs : Std.HashSet JVarId :=
    (derivedOutputVertices jaxpr).foldl (init := {}) fun acc v => acc.insert v
  let mut out : Array JVarId := #[]
  for eqn in jaxpr.eqns do
    for outvar in eqn.outvars do
      if !outputs.contains outvar.id then
        out := out.push outvar.id
  return out

/-- Infer graph partitions from boundary declarations and produced values. -/
def inferVertexPartitions (jaxpr : LeanJaxpr) : VertexPartitions :=
  {
    inputs := derivedInputVertices jaxpr
    outputs := derivedOutputVertices jaxpr
    eliminable := derivedEliminableGraphVertices jaxpr
  }

/-- Graph partitions are explicit normalized metadata, not inferred on access. -/
def vertexPartitions (jaxpr : LeanJaxpr) : VertexPartitions :=
  jaxpr.partitions

/-- Input-like graph vertices (`constvars ++ invars`) in declaration order. -/
def inputVertices (jaxpr : LeanJaxpr) : Array JVarId :=
  jaxpr.vertexPartitions.inputs

/-- Output boundary graph vertices in declaration order. -/
def outputVertices (jaxpr : LeanJaxpr) : Array JVarId :=
  jaxpr.vertexPartitions.outputs

/--
Eliminable graph vertices in equation-topological order.
This tracks produced variables that are not final outputs.
-/
def eliminableGraphVertices (jaxpr : LeanJaxpr) : Array JVarId :=
  jaxpr.vertexPartitions.eliminable

private def valueRoleOfId (jaxpr : LeanJaxpr) (id : JVarId) : ValueRole :=
  if jaxpr.outvars.any (fun v => v.id = id) then
    .output
  else if jaxpr.constvars.any (fun v => v.id = id) then
    .const
  else if jaxpr.invars.any (fun v => v.id = id) then
    .input
  else
    .intermediate

private def withRoleIfMissing (role : ValueRole) (v : JVar) : JVar :=
  if v.metaInfo.role?.isSome then
    v
  else
    { v with metaInfo := { v.metaInfo with role? := some role } }

/-- Populate missing `ValueRole` annotations from graph boundaries and outputs. -/
def withInferredValueRoles (jaxpr : LeanJaxpr) : LeanJaxpr :=
  let annotate := fun v => withRoleIfMissing (valueRoleOfId jaxpr v.id) v
  {
    jaxpr with
    constvars := jaxpr.constvars.map annotate
    invars := jaxpr.invars.map annotate
    outvars := jaxpr.outvars.map annotate
    eqns := jaxpr.eqns.map fun eqn =>
      {
        eqn with
        invars := eqn.invars.map annotate
        outvars := eqn.outvars.map annotate
      }
  }

/--
Upgrade generic hand-authored equations to typed normalized equations during
normalization. This keeps manual tests and fixtures aligned with the stricter
shared IR contract without weakening runtime rule dispatch.
-/
def withInferredTypedEqns (jaxpr : LeanJaxpr) : LeanJaxpr :=
  {
    jaxpr with
    eqns := jaxpr.eqns.map fun eqn =>
      if eqn.typed.schema == .generic then
        { eqn with typed := eqn.inferTypedOp }
      else
        eqn
  }

private def allPresentPositiveVertices (jaxpr : LeanJaxpr) : Std.HashSet Nat := Id.run do
  let mut present : Std.HashSet Nat := {}
  let remember := fun (acc : Std.HashSet Nat) (id : Nat) =>
    if 0 < id then acc.insert id else acc
  for v in jaxpr.constvars ++ jaxpr.invars ++ jaxpr.outvars do
    present := remember present v.id
  for eqn in jaxpr.eqns do
    for v in eqn.invars ++ eqn.outvars do
      present := remember present v.id
  return present

private def producerOpIdsByValueId (jaxpr : LeanJaxpr) : Std.HashMap JVarId OpId := Id.run do
  let mut out : Std.HashMap JVarId OpId := {}
  for eqn in jaxpr.eqns do
    let opId := eqn.id
    for outvar in eqn.outvars do
      out := out.insert outvar.id opId
  return out

/-- Infer the fixed action surface from the explicit eliminable partition. -/
def inferActionTable (jaxpr : LeanJaxpr) : ActionTable :=
  let parts := jaxpr.inferVertexPartitions
  let present := allPresentPositiveVertices jaxpr
  let producers := producerOpIdsByValueId jaxpr
  {
    bindings := parts.eliminable.mapIdx fun action0 vertex1 =>
      {
        action0 := action0
        vertex1 := vertex1
        valueId? := if present.contains vertex1 then some vertex1 else none
        producerOpId? := producers.get? vertex1
        role? := some (valueRoleOfId jaxpr vertex1)
        isBoundary := false
        isEliminable := true
      }
  }

/-- Action table is explicit normalized metadata, not inferred on access. -/
def actionTable (jaxpr : LeanJaxpr) : ActionTable :=
  jaxpr.actions

/-- Construct a normalized LeanJaxpr directly from boundary variables and equations. -/
def mkNormalized
    (constvars : Array JVar := #[])
    (invars : Array JVar := #[])
    (eqns : Array JEqn := #[])
    (outvars : Array JVar := #[]) :
    LeanJaxpr :=
  let jaxpr : LeanJaxpr := {
    constvars := constvars
    invars := invars
    eqns := eqns
    outvars := outvars
  }
  let jaxpr := jaxpr.withInferredTypedEqns
  let jaxpr := jaxpr.withInferredValueRoles
  { jaxpr with
    partitions := jaxpr.inferVertexPartitions
    actions := jaxpr.inferActionTable }

end LeanJaxpr

end Tyr.AD.JaxprLike
