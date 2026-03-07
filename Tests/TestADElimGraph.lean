import LeanTest
import Tyr.AD.Elim

namespace Tests.ADElimGraph

open LeanTest
open Tyr.AD.Elim
open Tyr.AD.JaxprLike

private def mkEdge (src dst : Nat) (repr : String) : LocalJacEdge :=
  { src := src, dst := dst, map := { repr := Tyr.AD.Sparse.SparseMapTag.namedStr repr } }

private def mkShapedEdge
    (src dst : Nat)
    (repr : String)
    (inDim outDim : Nat) :
    LocalJacEdge :=
  {
    src := src
    dst := dst
    map := {
      repr := Tyr.AD.Sparse.SparseMapTag.namedStr repr,
      inDim? := some inDim,
      outDim? := some outDim
    }
  }

@[test]
def testGraphBuildAndNeighborOrdering : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkEdge 4 2 "m42",
    mkEdge 1 2 "m12",
    mkEdge 2 3 "m23"
  ]
  let g := ofLocalJacEdges edges
  LeanTest.assertTrue (hasVertex g 2) "Vertex 2 should exist in graph"
  LeanTest.assertEqual (inNeighbors g 2 |>.map (fun p => p.1)) #[1, 4]
    "Incoming neighbors should be deterministic and sorted"
  LeanTest.assertEqual (outNeighbors g 2 |>.map (fun p => p.1)) #[3]
    "Outgoing neighbors should be deterministic and sorted"

@[test]
def testEliminateVertexSkeleton : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkEdge 1 2 "m12",
    mkEdge 4 2 "m42",
    mkEdge 2 3 "m23"
  ]
  let g := ofLocalJacEdges edges
  match eliminateVertex g 2 with
  | .error msg =>
    LeanTest.fail s!"eliminateVertex should succeed, got: {msg}"
  | .ok (g', stats) =>
    LeanTest.assertEqual stats.incomingEdges 2 "Expected 2 incoming edges into eliminated vertex"
    LeanTest.assertEqual stats.outgoingEdges 1 "Expected 1 outgoing edge from eliminated vertex"
    LeanTest.assertEqual stats.composedPairs 2 "Expected pairwise composition across incoming/outgoing sets"
    LeanTest.assertEqual stats.insertedEdges 2 "Expected two new bridge edges"
    LeanTest.assertEqual stats.updatedEdges 0 "Expected no updates without pre-existing bridge edge"

    LeanTest.assertTrue (!(hasVertex g' 2)) "Eliminated vertex should have no incident edges"
    LeanTest.assertEqual (outNeighbors g' 1 |>.map (fun p => p.1)) #[3]
      "Expected bridge edge 1 -> 3 after elimination"
    LeanTest.assertEqual (outNeighbors g' 4 |>.map (fun p => p.1)) #[3]
      "Expected bridge edge 4 -> 3 after elimination"

@[test]
def testRunEliminationOrderValidation : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkEdge 1 2 "m12",
    mkEdge 2 3 "m23"
  ]
  let g := ofLocalJacEdges edges

  match runElimination g #[2, 2] with
  | .ok _ => LeanTest.fail "Duplicate vertex in order should fail"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "duplicate vertex 2")
      s!"Unexpected duplicate-order error: {msg}"

  match runElimination g #[9] with
  | .ok _ => LeanTest.fail "Unknown vertex in order should fail"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "unknown vertex 9")
      s!"Unexpected unknown-vertex error: {msg}"

@[test]
def testRunEliminationOnEdges : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkEdge 1 2 "m12",
    mkEdge 2 3 "m23"
  ]
  match runEliminationOnEdges edges #[2] with
  | .error msg => LeanTest.fail s!"runEliminationOnEdges should succeed, got: {msg}"
  | .ok res =>
    LeanTest.assertEqual res.steps.size 1 "Expected one elimination step"
    LeanTest.assertTrue (!(hasVertex res.graph 2)) "Eliminated vertex should be removed from incident edges"

@[test]
def testPartitionedGraphForwardReverseAndCompleteValidation : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkEdge 1 2 "m12",
    mkEdge 2 4 "m24"
  ]
  match ofLocalJacEdgesWithPartitions edges #[1] #[4] #[2, 3] with
  | .error msg =>
    LeanTest.fail s!"Partitioned graph build should succeed, got: {msg}"
  | .ok g =>
    LeanTest.assertTrue (hasVertex g 3)
      "Partition-only eliminable vertices should still be visible in the graph domain"
    LeanTest.assertTrue (isBoundaryVertex g 1)
      "Input partition vertex should be treated as a boundary vertex"
    LeanTest.assertTrue (!(isBoundaryVertex g 2))
      "Eliminable interior vertex should not be treated as a boundary vertex"
    LeanTest.assertEqual (forwardEliminationOrder g) #[2, 3]
      "Forward elimination order should preserve explicit eliminable order"
    LeanTest.assertEqual (reverseEliminationOrder g) #[3, 2]
      "Reverse elimination order should reverse the explicit eliminable order"

    match validateCompleteEliminationOrder g #[2] with
    | .ok () =>
      LeanTest.fail "Complete elimination validation should reject missing eliminable vertices"
    | .error msg =>
      LeanTest.assertTrue (msg.contains "Missing eliminable vertices: #[3]")
        s!"Unexpected missing-vertex diagnostic: {msg}"

    match runForwardElimination g with
    | .error msg =>
      LeanTest.fail s!"runForwardElimination should succeed, got: {msg}"
    | .ok res =>
      LeanTest.assertEqual res.steps.size 2
        "Forward elimination should visit every eliminable vertex exactly once"
      LeanTest.assertTrue (!(hasVertex res.graph 2))
        "Eliminated interior vertex should be removed from the final graph"

@[test]
def testEliminateVertexRejectsSparseDimMismatch : IO Unit := do
  let edges : Array LocalJacEdge := #[
    mkShapedEdge 1 2 "m12" 1 2,
    mkShapedEdge 2 3 "m23" 3 1
  ]
  let g := ofLocalJacEdges edges
  match eliminateVertex g 2 with
  | .ok _ =>
    LeanTest.fail "eliminateVertex should fail on incompatible sparse map dimensions"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "dimension mismatch")
      s!"Expected sparse compose mismatch diagnostic, got: {msg}"

end Tests.ADElimGraph
