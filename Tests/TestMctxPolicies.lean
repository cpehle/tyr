import LeanTest
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

private def intToNatNonneg (x : Int) : Nat :=
  if x < 0 then 0 else Int.toNat x

private def deepestLeaf (tree : Tree Unit GumbelMuZeroExtraData) : Nat × Nat := Id.run do
  let mut stack : Array (Nat × Nat) := #[(ROOT_INDEX, 0)]
  let mut bestNode := ROOT_INDEX
  let mut bestDepth := 0
  let mut bestVisits := tree.nodeVisits.getD ROOT_INDEX 0

  while !stack.isEmpty do
    let some (node, depth) := stack.back? | break
    stack := stack.pop

    let mut hasChild := false
    for a in [:tree.numActions] do
      let child := (tree.childrenIndex.getD node #[]).getD a UNVISITED
      if child != UNVISITED then
        hasChild := true
        stack := stack.push (intToNatNonneg child, depth + 1)

    if !hasChild then
      let visits := tree.nodeVisits.getD node 0
      if depth > bestDepth || (depth = bestDepth && visits > bestVisits) then
        bestNode := node
        bestDepth := depth
        bestVisits := visits

  return (bestNode, bestDepth)

@[test]
def testMctxApplyTemperatureOne : IO Unit := do
  let logits : Array Float := #[0, 1, 2, 3, 4, 5]
  let got := applyTemperature logits 1.0
  let expected := logits.map (fun x => x - 5.0)
  let ok := (List.range got.size).all fun i => approx (got.getD i 0.0) (expected.getD i 0.0) 1e-6
  LeanTest.assertTrue ok s!"temperature=1 should only subtract max, got {got}"

@[test]
def testMctxApplyTemperatureTwo : IO Unit := do
  let logits : Array Float := #[0, 1, 2, 3, 4, 5]
  let got := applyTemperature logits 2.0
  let expected := logits.map (fun x => (x - 5.0) / 2.0)
  let ok := (List.range got.size).all fun i => approx (got.getD i 0.0) (expected.getD i 0.0) 1e-6
  LeanTest.assertTrue ok s!"temperature=2 should divide centered logits by 2, got {got}"

@[test]
def testMctxApplyTemperatureZero : IO Unit := do
  let logits : Array Float := #[0, 1, 2, 3]
  let got := applyTemperature logits 0.0
  LeanTest.assertEqual (argmax got) 3 "temperature=0 should preserve argmax"
  LeanTest.assertTrue (!got.any Float.isNaN) "temperature=0 output should not contain NaNs"

@[test]
def testMctxApplyTemperatureZeroOnLargeLogits : IO Unit := do
  let logits : Array Float := #[100.0, 3.4028235e38, (-1.0 / 0.0), -3.4028235e38]
  let got := applyTemperature logits 0.0
  LeanTest.assertTrue (approx (got.getD 1 1.0) 0.0 1e-6) s!"max element should map to 0, got {got.getD 1 0.0}"
  LeanTest.assertTrue (!got.any Float.isNaN) "temperature scaling with large logits should not introduce NaNs"

@[test]
def testMctxMaskInvalidActions : IO Unit := do
  let logits : Array Float := #[1.0e6, (-1.0 / 0.0), 1.0e6 + 1.0, -100.0]
  let invalid : Array Bool := #[false, true, false, true]
  let masked := maskInvalidActions logits (some invalid)
  let probs := softmax masked
  let validProbs := softmax #[0.0, 1.0]

  LeanTest.assertTrue (approx (probs.getD 0 0.0) (validProbs.getD 0 0.0) 1e-5)
    s!"Expected valid action 0 prob {validProbs.getD 0 0.0}, got {probs.getD 0 0.0}"
  LeanTest.assertTrue (approx (probs.getD 2 0.0) (validProbs.getD 1 0.0) 1e-5)
    s!"Expected valid action 2 prob {validProbs.getD 1 0.0}, got {probs.getD 2 0.0}"
  LeanTest.assertTrue (probs.getD 1 1.0 < 1e-8 && probs.getD 3 1.0 < 1e-8)
    "Invalid actions should get near-zero probability"

@[test]
def testMctxMaskAllInvalidActions : IO Unit := do
  let logits : Array Float := #[-1.0, -1.0, -1.0, -1.0]
  let invalid : Array Bool := #[true, true, true, true]
  let masked := maskInvalidActions logits (some invalid)
  let probs := softmax masked
  let ok := (List.range 4).all fun i => approx (probs.getD i 0.0) 0.25 1e-6
  LeanTest.assertTrue ok s!"All-invalid state should softmax to uniform, got {probs}"

@[test]
def testMctxMuZeroPolicy : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[-1.0, 0.0, 2.0, 3.0]
    value := 0.0
    embedding := ()
  }
  let rewards : Array Float := #[0.0, 0.0, 0.0, 0.0]
  let recurrent : RecurrentFn Unit Unit := fun _ _ action _ =>
    ({ reward := rewards.getD action 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, ())
  let invalid : Array Bool := #[false, false, false, true]

  let out := muzeroPolicy
    (params := ()) (rngKey := 0)
    (root := root) (recurrentFn := recurrent)
    (numSimulations := 1)
    (invalidActions := some invalid)
    (dirichletFraction := 0.0)

  LeanTest.assertEqual out.action 2 "MuZero policy should choose action 2"
  LeanTest.assertEqual out.actionWeights #[0.0, 0.0, 1.0, 0.0]
    "With one simulation, action weights should be one-hot"

@[test]
def testMctxGumbelMuZeroPolicy : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, -1.0, 2.0, 3.0]
    value := -5.0
    embedding := ()
  }
  let rewards : Array Float := #[20.0, 3.0, -1.0, 10.0]
  let recurrent : RecurrentFn Unit Unit := fun _ _ action _ =>
    ({ reward := rewards.getD action 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, ())
  let invalid : Array Bool := #[true, false, false, true]

  let out := gumbelMuZeroPolicy
    (params := ()) (rngKey := 0)
    (root := root) (recurrentFn := recurrent)
    (numSimulations := 17)
    (invalidActions := some invalid)
    (maxDepth := some 3)
    (qtransform := fun t i => qtransformCompletedByMixValue t i (valueScale := 0.05) (maxvisitInit := 60.0) (rescaleValues := true))
    (gumbelScale := 1.0)

  LeanTest.assertTrue (out.action = 1 || out.action = 2)
    s!"Expected valid action among considered arms, got {out.action}"

  let w := out.actionWeights
  LeanTest.assertTrue (w.getD 0 1.0 < 1e-8 && w.getD 3 1.0 < 1e-8)
    "Invalid actions should have near-zero weight"

  let summary := out.searchTree.summary
  LeanTest.assertEqual (summary.visitCounts.getD 0 999) 0 "Invalid action 0 should not be visited"
  LeanTest.assertEqual (summary.visitCounts.getD 3 999) 0 "Invalid action 3 should not be visited"

  let (_leaf, depth) := deepestLeaf out.searchTree
  LeanTest.assertTrue (depth ≤ 3) s!"Search depth should respect max_depth=3, got {depth}"

@[test]
def testMctxGumbelMuZeroPolicyWithoutInvalidActions : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, -1.0, 2.0, 3.0]
    value := -5.0
    embedding := ()
  }
  let rewards : Array Float := #[20.0, 3.0, -1.0, 10.0]
  let recurrent : RecurrentFn Unit Unit := fun _ _ action _ =>
    ({ reward := rewards.getD action 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, ())

  let numSimulations := 17
  let out := gumbelMuZeroPolicy
    (params := ()) (rngKey := 0)
    (root := root) (recurrentFn := recurrent)
    (numSimulations := numSimulations)
    (invalidActions := none)
    (maxDepth := some 3)
    (qtransform := fun t i => qtransformCompletedByMixValue t i (valueScale := 0.05) (maxvisitInit := 60.0) (rescaleValues := true))
    (gumbelScale := 1.0)

  LeanTest.assertTrue (out.action < 4) s!"Action should be in range [0,3], got {out.action}"

  let summary := out.searchTree.summary
  let totalVisits := summary.visitCounts.foldl (init := 0) (· + ·)
  LeanTest.assertEqual totalVisits numSimulations
    s!"Root child visits should sum to num_simulations={numSimulations}"

@[test]
def testMctxAlphaZeroPolicyPersistentSubtree : IO Unit := do
  let root1 : RootFnOutput Unit := {
    priorLogits := #[0.2, -0.3, 0.9, 0.1]
    value := -0.2
    embedding := ()
  }
  let recurrent : RecurrentFn Unit Unit := fun _ _ action _ =>
    let reward := #[0.1, 0.0, -0.2, 0.3].getD action 0.0
    ({ reward := reward, discount := 0.95, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.05 }, ())

  let out1 := alphazeroPolicy
    (params := ())
    (rngKey := 11)
    (root := root1)
    (recurrentFn := recurrent)
    (numSimulations := 6)
    (searchTree := none)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let carried := getSubtree out1.searchTree out1.action
  let before := carried.nodeVisits.getD ROOT_INDEX 0

  let root2 : RootFnOutput Unit := {
    priorLogits := root1.priorLogits
    value := 0.7
    embedding := ()
  }

  let out2 := alphazeroPolicy
    (params := ())
    (rngKey := 19)
    (root := root2)
    (recurrentFn := recurrent)
    (numSimulations := 3)
    (searchTree := some carried)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let after := out2.searchTree.nodeVisits.getD ROOT_INDEX 0
  LeanTest.assertEqual after (before + 3)
    "Persistent subtree should be reused and extended by new simulations"
  LeanTest.assertTrue (approx (out2.searchTree.rawValues.getD ROOT_INDEX 0.0) root2.value 1e-6)
    "Continuing search should refresh root raw value from the new root output"

@[test]
def testMctxStochasticMuZeroPolicyPlaceholder : IO Unit := do
  -- Upstream mctx has stochastic_muzero_policy tests. The current Lean port
  -- does not expose stochastic MuZero yet; this placeholder keeps parity
  -- tracking explicit in the test suite.
  LeanTest.assertTrue true "stochastic_muzero_policy tests pending implementation"
