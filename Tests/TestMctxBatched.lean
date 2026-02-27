import LeanTest
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

@[test]
def testMuZeroPolicyBatchedBandit : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[-1.0, 0.0, 2.0, 3.0],
      #[3.0, 1.0, 0.0, -1.0]
    ]
    value := #[0.0, 0.0]
    embedding := #[(), ()]
  }

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    ({
      reward := Array.replicate b 0.0
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 4 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let invalidActions : Array (Array Bool) := #[
    #[false, false, false, true],
    #[true, false, false, false]
  ]

  let out := muzeroPolicyBatched
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 1)
    (invalidActions := some invalidActions)
    (dirichletFraction := 0.0)

  LeanTest.assertEqual out.action.size 2 "Expected 2 batched actions"
  LeanTest.assertEqual (out.action.getD 0 0) 2 "Batch 0 should pick action 2"
  LeanTest.assertEqual (out.action.getD 1 0) 1 "Batch 1 should pick action 1"

@[test]
def testSearchBatchedSummaryShapes : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[0.0, 1.0, 2.0],
      #[2.0, 1.0, 0.0],
      #[0.5, 0.2, 0.1]
    ]
    value := #[0.0, 0.0, 0.0]
    embedding := #[(), (), ()]
  }

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    ({
      reward := Array.replicate b 0.0
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 3 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let rootFn : RootActionSelectionFn Unit Unit := fun _ tree nodeIndex =>
    muzeroActionSelection tree nodeIndex 0 qtransformByParentAndSiblings
  let interiorFn : InteriorActionSelectionFn Unit Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransformByParentAndSiblings

  let tree := searchBatched
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (rootActionSelectionFn := rootFn)
    (interiorActionSelectionFn := interiorFn)
    (numSimulations := 2)

  let summary := tree.summary
  LeanTest.assertEqual summary.value.size 3 "Summary should contain one value per batch element"
  LeanTest.assertEqual summary.visitCounts.size 3 "Visit counts should be batched"
  LeanTest.assertEqual (summary.visitCounts.getD 0 #[]).size 3 "Each row should have num_actions counts"

@[test]
def testGumbelPolicyBatchedMasking : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[0.0, -1.0, 2.0, 3.0],
      #[1.0, 2.0, 0.5, -3.0]
    ]
    value := #[-1.0, -1.0]
    embedding := #[(), ()]
  }

  let rewards : Array (Array Float) := #[
    #[20.0, 3.0, -1.0, 10.0],
    #[1.0, 4.0, 2.0, -5.0]
  ]

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    let reward := (List.range b).toArray.map fun i =>
      let a := actions.getD i 0
      (rewards.getD i #[]).getD a 0.0
    ({
      reward := reward
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 4 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let invalidActions : Array (Array Bool) := #[
    #[true, false, false, true],
    #[true, false, false, true]
  ]

  let out := gumbelMuZeroPolicyBatched
    (params := ())
    (rngKey := 7)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 12)
    (invalidActions := some invalidActions)

  let a0 := out.action.getD 0 0
  let a1 := out.action.getD 1 0
  LeanTest.assertTrue (a0 = 1 || a0 = 2) s!"Batch0 action should be valid, got {a0}"
  LeanTest.assertTrue (a1 = 1 || a1 = 2) s!"Batch1 action should be valid, got {a1}"

  let weights0 := out.actionWeights.getD 0 #[]
  let sum0 := weights0.foldl (init := 0.0) (· + ·)
  LeanTest.assertTrue (approx sum0 1.0 1e-5) s!"Action weights should sum to 1, got {sum0}"
