import Std.Data.HashSet
import Tyr.AD.Elim.Graph
import Tyr.AD.Sparse

/-!
# Tyr.AD.Elim.Eliminate

Vertex elimination skeleton over `LocalJacEdge` graphs.
-/

namespace Tyr.AD.Elim

open Tyr.AD.JaxprLike

/-- Checked sparse composition as `outMap ∘ inMap`. -/
def composeSparseLinearMap
    (inMap outMap : SparseLinearMap) :
    Except String SparseLinearMap :=
  Tyr.AD.Sparse.compose inMap outMap

/-- Checked sparse additive merge. -/
def addSparseLinearMap
    (lhs rhs : SparseLinearMap) :
    Except String SparseLinearMap :=
  Tyr.AD.Sparse.add lhs rhs

structure ElimStepStats where
  vertex : JVarId
  incomingEdges : Nat := 0
  outgoingEdges : Nat := 0
  composedPairs : Nat := 0
  insertedEdges : Nat := 0
  updatedEdges : Nat := 0
  deriving Repr, Inhabited

structure ElimRunResult where
  graph : ElimGraph
  steps : Array ElimStepStats := #[]
  deriving Repr, Inhabited

/--
Eliminate vertex `v` by composing every `i -> v` with every `v -> o`,
then remove all edges incident to `v`.
-/
def eliminateVertex
    (g : ElimGraph)
    (v : JVarId) :
    Except String (ElimGraph × ElimStepStats) := do
  let incoming := inNeighbors g v
  let outgoing := outNeighbors g v

  let mut work := g
  let mut inserted := 0
  let mut updated := 0

  for inPair in incoming do
    let i := inPair.1
    let mapIn := inPair.2
    for outPair in outgoing do
      let o := outPair.1
      let mapOut := outPair.2
      let composed ← composeSparseLinearMap mapIn mapOut
      match findEdge? work i o with
      | some existing =>
        let merged ← addSparseLinearMap existing composed
        work := insertEdge work i o merged
        updated := updated + 1
      | none =>
        work := insertEdge work i o composed
        inserted := inserted + 1

  let pruned := eraseVertexEdges work v
  let stats : ElimStepStats :=
    {
      vertex := v
      incomingEdges := incoming.size
      outgoingEdges := outgoing.size
      composedPairs := incoming.size * outgoing.size
      insertedEdges := inserted
      updatedEdges := updated
    }
  pure (pruned, stats)

/-- Basic validation: each vertex appears once and exists in the input graph. -/
def validateEliminationOrder (g : ElimGraph) (order : Array JVarId) : Except String Unit := Id.run do
  let mut seen : Std.HashSet JVarId := {}
  for v in order do
    if seen.contains v then
      return .error s!"Elimination order contains duplicate vertex {v}."
    if !(hasVertex g v) then
      return .error s!"Elimination order references unknown vertex {v}."
    seen := seen.insert v
  return .ok ()

/-- Run elimination in the given order after basic order validation. -/
def runElimination (g : ElimGraph) (order : Array JVarId) : Except String ElimRunResult := do
  validateEliminationOrder g order
  let mut graph := g
  let mut steps : Array ElimStepStats := #[]
  for v in order do
    let (graph', stats) ← eliminateVertex graph v
    graph := graph'
    steps := steps.push stats
  return { graph := graph, steps := steps }

/-- Convenience entrypoint from local Jacobian edges. -/
def runEliminationOnEdges
    (edges : Array LocalJacEdge)
    (order : Array JVarId) :
    Except String ElimRunResult :=
  runElimination (ofLocalJacEdges edges) order

end Tyr.AD.Elim
