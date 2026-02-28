import LeanTest
import Tyr.MctxDag

open torch.mctxdag

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

private def chooseLeastVisitedAction [BEq K] [Hashable K]
    (tree : DagTree S K E) (nodeIndex : Nat) : Nat :=
  let visits := tree.childrenVisits.getD nodeIndex #[]
  if visits.isEmpty then
    0
  else
    let init : Nat × Nat := (0, visits.getD 0 0)
    let (best, _) := (List.range visits.size).foldl (init := init) fun (acc : Nat × Nat) a =>
      let c := visits.getD a acc.2
      if c < acc.2 then (a, c) else acc
    best

@[test]
def testMctxDagTranspositionReuse : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.0, 0.0]
    value := 0.0
    embedding := 1
  }

  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action _ =>
    let nextEmb : UInt64 := if action = 0 then 777 else 777
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := 0.0 }, nextEmb)

  let rootFn : RootActionSelectionFn UInt64 UInt64 Unit := fun _ tree nodeIndex =>
    chooseLeastVisitedAction tree nodeIndex
  let interiorFn : InteriorActionSelectionFn UInt64 UInt64 Unit := fun _ tree nodeIndex _ =>
    chooseLeastVisitedAction tree nodeIndex

  let tree0 := instantiateDagTreeFromRoot root root.embedding 4 #[false, false] ()
  let tree := searchWithDag
    (params := ())
    (rngKey := 0)
    (tree := tree0)
    (recurrentFn := recurrent)
    (keyFn := id)
    (rootActionSelectionFn := rootFn)
    (interiorActionSelectionFn := interiorFn)
    (numSimulations := 2)

  LeanTest.assertEqual tree.numAllocated 2
    "Two root actions reaching same key should share one child node"

  let c0 := (tree.childrenIndex.getD ROOT_INDEX #[]).getD 0 UNVISITED
  let c1 := (tree.childrenIndex.getD ROOT_INDEX #[]).getD 1 UNVISITED
  LeanTest.assertTrue (c0 = c1 && c0 = Int.ofNat 1)
    s!"Expected both root actions to point to shared node 1, got ({c0}, {c1})"

@[test]
def testMctxDagAlphaZeroPersistentSubtree : IO Unit := do
  let root1 : RootFnOutput UInt64 := {
    priorLogits := #[0.2, -0.3, 0.9, 0.1]
    value := -0.2
    embedding := 0
  }
  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    let nextEmb := emb * 131 + UInt64.ofNat (action + 1)
    ({ reward := 0.05, discount := 0.95, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.1 }, nextEmb)

  let out1 := alphazeroPolicyDag
    (params := ())
    (rngKey := 11)
    (root := root1)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 6)
    (searchTree := none)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let carried := getSubtree out1.searchTree out1.action
  let before := carried.nodeVisits.getD ROOT_INDEX 0

  let root2 : RootFnOutput UInt64 := {
    priorLogits := root1.priorLogits
    value := 0.7
    embedding := 0
  }

  let out2 := alphazeroPolicyDag
    (params := ())
    (rngKey := 19)
    (root := root2)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 3)
    (searchTree := some carried)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let after := out2.searchTree.nodeVisits.getD ROOT_INDEX 0
  LeanTest.assertEqual after (before + 3)
    "Persistent subtree should be reused and extended by new simulations"
  LeanTest.assertTrue (approx (out2.searchTree.rawValues.getD ROOT_INDEX 0.0) root2.value 1e-6)
    "Continuing search should refresh root raw value from new root output"

@[test]
def testMctxDagResetSearchTree : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.0, 1.0]
    value := 0.0
    embedding := 12
  }

  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    ({ reward := 0.0, discount := 0.0, priorLogits := #[0.0, 0.0], value := 0.0 }, emb + UInt64.ofNat (action + 1))

  let out := muzeroPolicyDag
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrent)
    (keyFn := id)
    (numSimulations := 2)
    (dirichletFraction := 0.0)

  let reset := resetSearchTree out.searchTree
  LeanTest.assertEqual reset.numAllocated 0 "Reset DAG tree should clear allocated nodes"
  LeanTest.assertTrue (reset.keyToNode.isEmpty) "Reset DAG tree should clear transposition table"
  LeanTest.assertTrue ((List.range reset.nodeVisits.size).all fun i => reset.nodeVisits.getD i 1 = 0)
    "Reset DAG tree should clear node visits"
