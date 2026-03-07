import Lean.Compiler.IR.Basic

/-!
# Tyr.AD.JaxprLike.Core

Core data structures for a Jaxpr-like IR layer in Tyr.
This layer is intentionally lightweight and independent from elimination execution.
-/

namespace Tyr.AD.JaxprLike

open Lean
open Lean.IR

/-- Stable ID for variables in the LeanJaxpr-like representation. -/
abbrev JVarId := Nat

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
  | variant
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
  | .variant => "variant"
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

end OpParams

/-- AD participation marker carried on normalized variables. -/
inductive DiffParticipation where
  | diff
  | static
  | frozen
  deriving Repr, BEq, Inhabited, DecidableEq

/-- AD-relevant metadata that survives normalization. -/
structure VarMeta where
  participation : DiffParticipation := .diff
  shape : Option (Array Nat) := none
  dtype : Option String := none
  sharding : Option String := none
  aliasGroup? : Option Nat := none
  deriving Repr, Inhabited

/-- Source location metadata for diagnostics and coverage errors. -/
structure SourceRef where
  decl : Name := .anonymous
  line? : Option Nat := none
  col? : Option Nat := none
  deriving Repr, Inhabited

/-- Variable in LeanJaxpr-like IR. -/
structure JVar where
  id : JVarId
  ty : IRType := .object
  metaInfo : VarMeta := {}
  deriving Repr, Inhabited

/-- Equation in LeanJaxpr-like IR. -/
structure JEqn where
  op : OpName
  invars : Array JVar
  outvars : Array JVar
  params : OpParams := #[]
  source : SourceRef := {}
  deriving Repr, Inhabited

/-- Jaxpr-like normalized IR for elimination-based AD. -/
structure LeanJaxpr where
  constvars : Array JVar := #[]
  invars : Array JVar := #[]
  eqns : Array JEqn := #[]
  outvars : Array JVar := #[]
  deriving Repr, Inhabited

/-- Graphax-style vertex numbering: equation index -> 1-based vertex ID. -/
def eqnVertexId1 (eqnIdx0 : Nat) : Nat :=
  eqnIdx0 + 1

/-- Inverse of `eqnVertexId1` with domain check for 1-based IDs. -/
def vertexToEqnIdx0? (vertexId1 : Nat) : Option Nat :=
  if vertexId1 = 0 then none else some (vertexId1 - 1)

/-- Default eliminable vertex set for a fully eliminable equation sequence. -/
def eliminableVertices1 (jaxpr : LeanJaxpr) : Array Nat :=
  (Array.range jaxpr.eqns.size).map eqnVertexId1

end Tyr.AD.JaxprLike
