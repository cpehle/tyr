import Std.Data.HashMap
import Std.Data.HashSet
import Tyr.AD.JaxprLike.Rules

/-!
# Tyr.AD.Elim.Graph

Elimination graph over local Jacobian edges.
-/

namespace Tyr.AD.Elim

open Tyr.AD.JaxprLike

abbrev AdjRow := Std.HashMap JVarId SparseLinearMap
abbrev AdjMap := Std.HashMap JVarId AdjRow

/--
Producer-side semantic metadata for a graph value vertex.
This lets AlphaGrad consume the normalized op contract carried by `LeanJaxpr`,
not just anonymous edge topology.
-/
structure VertexProducer where
  opId : OpId
  op : OpName
  typed : TypedOp := {}
  source : SourceRef := {}
  role? : Option ValueRole := none
  deriving Repr, Inhabited, BEq

/-- Bidirectional adjacency for elimination over local Jacobian edges. -/
structure ElimGraph where
  /-- Forward adjacency: `src -> (dst -> map)`. -/
  forward : AdjMap := {}
  /-- Backward adjacency: `dst -> (src -> map)`. -/
  backward : AdjMap := {}
  /-- Non-eliminable input-like boundary vertices. -/
  inputs : Array JVarId := #[]
  /-- Non-eliminable output boundary vertices. -/
  outputs : Array JVarId := #[]
  /-- Eliminable vertices in deterministic forward order. -/
  eliminable : Array JVarId := #[]
  /-- Fixed action surface (`ActionId0 -> VertexId1`) used by AlphaGrad-style search. -/
  actionVertices : Array JVarId := #[]
  /-- Producer-side semantics keyed by produced value vertex. -/
  producers : Std.HashMap JVarId VertexProducer := {}
  deriving Repr, Inhabited

private def dedupPreserveOrder (xs : Array JVarId) : Array JVarId := Id.run do
  let mut seen : Std.HashSet JVarId := {}
  let mut out : Array JVarId := #[]
  for x in xs do
    if !seen.contains x then
      seen := seen.insert x
      out := out.push x
  return out

private def sortedPairs (row : AdjRow) : Array (JVarId × SparseLinearMap) :=
  (row.toList.mergeSort (fun a b => a.1 < b.1)).toArray

private def rowInsert (adj : AdjMap) (src dst : JVarId) (map : SparseLinearMap) : AdjMap :=
  let row := adj.getD src {}
  adj.insert src (row.insert dst map)

private def rowErase (adj : AdjMap) (src dst : JVarId) : AdjMap :=
  match adj.get? src with
  | none => adj
  | some row =>
    let row' := row.erase dst
    if row'.isEmpty then adj.erase src else adj.insert src row'

/-- Lookup edge map on the forward view. -/
def findEdge? (g : ElimGraph) (src dst : JVarId) : Option SparseLinearMap := do
  let row ← g.forward.get? src
  row.get? dst

/-- Deterministic outgoing neighbors sorted by destination ID. -/
def outNeighbors (g : ElimGraph) (src : JVarId) : Array (JVarId × SparseLinearMap) :=
  match g.forward.get? src with
  | none => #[]
  | some row => sortedPairs row

/-- Deterministic incoming neighbors sorted by source ID. -/
def inNeighbors (g : ElimGraph) (dst : JVarId) : Array (JVarId × SparseLinearMap) :=
  match g.backward.get? dst with
  | none => #[]
  | some row => sortedPairs row

/-- Producer-side semantic metadata for a value vertex, when known. -/
def producerInfo? (g : ElimGraph) (vertex : JVarId) : Option VertexProducer :=
  g.producers.get? vertex

/-- Insert or replace `src -> dst`. Keeps forward/backward adjacency in sync. -/
def insertEdge (g : ElimGraph) (src dst : JVarId) (map : SparseLinearMap) : ElimGraph :=
  { g with
    forward := rowInsert g.forward src dst map
    backward := rowInsert g.backward dst src map
  }

/-- Erase `src -> dst` from both adjacency views. -/
def eraseEdge (g : ElimGraph) (src dst : JVarId) : ElimGraph :=
  { g with
    forward := rowErase g.forward src dst
    backward := rowErase g.backward dst src
  }

/-- Remove all outgoing edges from `src`. -/
def eraseOutgoingEdges (g : ElimGraph) (src : JVarId) : ElimGraph :=
  match g.forward.get? src with
  | none => g
  | some row =>
    let backward := row.toList.foldl (init := g.backward) fun acc pair =>
      rowErase acc pair.1 src
    { g with
      forward := g.forward.erase src
      backward := backward
    }

/-- Remove all incoming edges into `dst`. -/
def eraseIncomingEdges (g : ElimGraph) (dst : JVarId) : ElimGraph :=
  match g.backward.get? dst with
  | none => g
  | some row =>
    let forward := row.toList.foldl (init := g.forward) fun acc pair =>
      rowErase acc pair.1 dst
    { g with
      forward := forward
      backward := g.backward.erase dst
    }

/-- Remove all edges incident to vertex `v`. -/
def eraseVertexEdges (g : ElimGraph) (v : JVarId) : ElimGraph :=
  let g' := eraseOutgoingEdges g v
  let g'' := eraseIncomingEdges g' v
  {
    g'' with
    inputs := g''.inputs.filter (· != v)
    outputs := g''.outputs.filter (· != v)
    eliminable := g''.eliminable.filter (· != v)
    producers := g''.producers.erase v
  }

/-- Vertex is present iff it has at least one incoming or outgoing edge. -/
def hasVertex (g : ElimGraph) (v : JVarId) : Bool :=
  (g.forward.get? v).isSome ||
    (g.backward.get? v).isSome ||
    g.inputs.contains v ||
    g.outputs.contains v ||
    g.eliminable.contains v

/-- True iff `v` is marked as a non-eliminable boundary vertex. -/
def isBoundaryVertex (g : ElimGraph) (v : JVarId) : Bool :=
  g.inputs.contains v || g.outputs.contains v

/-- True iff `v` is explicitly marked eliminable. -/
def isEliminableVertex (g : ElimGraph) (v : JVarId) : Bool :=
  g.eliminable.contains v

/-- Deterministic Graphax-style forward order over explicit eliminable vertices. -/
def forwardEliminationOrder (g : ElimGraph) : Array JVarId :=
  g.eliminable

/-- Deterministic Graphax-style reverse order over explicit eliminable vertices. -/
def reverseEliminationOrder (g : ElimGraph) : Array JVarId :=
  g.eliminable.reverse

/-- Deterministic vertex listing from union of adjacency keys. -/
def vertices (g : ElimGraph) : Array JVarId := Id.run do
  let mut seen : Std.HashSet JVarId := {}
  let mut out : Array JVarId := #[]
  for (v, _) in g.forward.toList do
    if !seen.contains v then
      seen := seen.insert v
      out := out.push v
  for (v, _) in g.backward.toList do
    if !seen.contains v then
      seen := seen.insert v
      out := out.push v
  for v in g.inputs ++ g.outputs ++ g.eliminable do
    if !seen.contains v then
      seen := seen.insert v
      out := out.push v
  return (out.toList.mergeSort (· < ·)).toArray

private def firstDuplicateVertex? (xs : Array JVarId) : Option JVarId := Id.run do
  let mut seen : Std.HashSet JVarId := {}
  for x in xs do
    if seen.contains x then
      return some x
    seen := seen.insert x
  return none

private def producerBindingsOfJaxpr (jaxpr : LeanJaxpr) : Array (JVarId × VertexProducer) := Id.run do
  let mut out : Array (JVarId × VertexProducer) := #[]
  for eqn in jaxpr.eqns do
    for outvar in eqn.outvars do
      out := out.push (
        outvar.id,
        {
          opId := eqn.id
          op := eqn.op
          typed := eqn.typedOp
          source := eqn.source
          role? := outvar.metaInfo.role?
        }
      )
  return out

/-- Attach producer-side semantic metadata extracted from a normalized `LeanJaxpr`. -/
def withJaxprProducers
    (g : ElimGraph)
    (jaxpr : LeanJaxpr) :
    Except String ElimGraph := Id.run do
  let mut producers := g.producers
  for binding in producerBindingsOfJaxpr jaxpr do
    let vertex := binding.1
    let info := binding.2
    match producers.get? vertex with
    | some existing =>
      if existing != info then
        return .error s!"Conflicting producer metadata attached to vertex {vertex}."
    | none =>
      producers := producers.insert vertex info
  return .ok { g with producers := producers }

/--
Attach explicit graph partitions.
`inputs` and `outputs` may overlap, but `eliminable` must stay disjoint from both.
-/
def withPartitions
    (g : ElimGraph)
    (inputs outputs eliminable : Array JVarId) :
    Except String ElimGraph := do
  if let some dup := firstDuplicateVertex? inputs then
    throw s!"Input partition contains duplicate vertex {dup}."
  if let some dup := firstDuplicateVertex? outputs then
    throw s!"Output partition contains duplicate vertex {dup}."
  if let some dup := firstDuplicateVertex? eliminable then
    throw s!"Eliminable partition contains duplicate vertex {dup}."
  let boundary : Std.HashSet JVarId :=
    (inputs ++ outputs).foldl (init := {}) fun acc v => acc.insert v
  match eliminable.find? (fun v => boundary.contains v) with
  | some bad =>
    throw s!"Eliminable partition references boundary vertex {bad}; eliminable vertices must exclude declared inputs/outputs."
  | none =>
    let g' := {
      g with
      inputs := dedupPreserveOrder inputs
      outputs := dedupPreserveOrder outputs
      eliminable := dedupPreserveOrder eliminable
    }
    pure {
      g' with
      -- Explicit partitions define a new graph surface; default the action
      -- table to the explicit eliminable set unless a later `withActionVertices`
      -- call overrides it.
      actionVertices := g'.eliminable
    }

/-- Attach an explicit action-space lookup table (`ActionId0 -> VertexId1`). -/
def withActionVertices
    (g : ElimGraph)
    (actionVertices : Array JVarId) :
    Except String ElimGraph := do
  if let some dup := firstDuplicateVertex? actionVertices then
    throw s!"Action-space vertex table contains duplicate vertex {dup}."
  match actionVertices.find? (fun v => v = 0) with
  | some _ =>
    throw "Action-space vertex table may only contain positive vertex IDs."
  | none =>
    pure { g with actionVertices := dedupPreserveOrder actionVertices }

/-- Build elimination graph from local Jacobian edges. -/
def ofLocalJacEdges (edges : Array LocalJacEdge) : ElimGraph :=
  let base :=
    edges.foldl (init := ({} : ElimGraph)) fun g e =>
      insertEdge g e.src e.dst e.map
  let inferred := vertices base
  let g := { base with eliminable := inferred }
  { g with actionVertices := g.eliminable }

/-- Build elimination graph from local Jacobian edges plus explicit partitions. -/
def ofLocalJacEdgesWithPartitions
    (edges : Array LocalJacEdge)
    (inputs outputs eliminable : Array JVarId) :
    Except String ElimGraph :=
  withPartitions (ofLocalJacEdges edges) inputs outputs eliminable

/-- Build elimination graph from local Jacobian edges plus explicit partitions and action surface. -/
def ofLocalJacEdgesWithPartitionsAndActions
    (edges : Array LocalJacEdge)
    (inputs outputs eliminable actionVertices : Array JVarId) :
    Except String ElimGraph := do
  let g ← ofLocalJacEdgesWithPartitions edges inputs outputs eliminable
  withActionVertices g actionVertices

/--
Build an elimination graph from local Jacobian edges while preserving the
normalized producer semantics carried by a source `LeanJaxpr`.
-/
def ofLocalJacEdgesWithJaxpr
    (edges : Array LocalJacEdge)
    (jaxpr : LeanJaxpr) :
    Except String ElimGraph := do
  let g ← ofLocalJacEdgesWithPartitionsAndActions
    edges
    jaxpr.vertexPartitions.inputs
    jaxpr.vertexPartitions.outputs
    jaxpr.vertexPartitions.eliminable
    jaxpr.actionTable.vertices1
  withJaxprProducers g jaxpr

end Tyr.AD.Elim
