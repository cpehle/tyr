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

/-- Bidirectional adjacency for elimination over local Jacobian edges. -/
structure ElimGraph where
  /-- Forward adjacency: `src -> (dst -> map)`. -/
  forward : AdjMap := {}
  /-- Backward adjacency: `dst -> (src -> map)`. -/
  backward : AdjMap := {}
  deriving Repr, Inhabited

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

/-- Insert or replace `src -> dst`. Keeps forward/backward adjacency in sync. -/
def insertEdge (g : ElimGraph) (src dst : JVarId) (map : SparseLinearMap) : ElimGraph :=
  {
    forward := rowInsert g.forward src dst map
    backward := rowInsert g.backward dst src map
  }

/-- Erase `src -> dst` from both adjacency views. -/
def eraseEdge (g : ElimGraph) (src dst : JVarId) : ElimGraph :=
  {
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
    {
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
    {
      forward := forward
      backward := g.backward.erase dst
    }

/-- Remove all edges incident to vertex `v`. -/
def eraseVertexEdges (g : ElimGraph) (v : JVarId) : ElimGraph :=
  let g' := eraseOutgoingEdges g v
  eraseIncomingEdges g' v

/-- Vertex is present iff it has at least one incoming or outgoing edge. -/
def hasVertex (g : ElimGraph) (v : JVarId) : Bool :=
  (g.forward.get? v).isSome || (g.backward.get? v).isSome

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
  return (out.toList.mergeSort (· < ·)).toArray

/-- Build elimination graph from local Jacobian edges. -/
def ofLocalJacEdges (edges : Array LocalJacEdge) : ElimGraph :=
  edges.foldl (init := ({} : ElimGraph)) fun g e =>
    insertEdge g e.src e.dst e.map

end Tyr.AD.Elim
