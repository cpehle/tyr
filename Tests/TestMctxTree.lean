import LeanTest
import Lean.Data.Json
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

private def parseJsonFileOrFail (path : System.FilePath) : IO Lean.Json := do
  let raw ← IO.FS.readFile path
  match Lean.Json.parse raw with
  | .ok j => pure j
  | .error e => LeanTest.fail s!"JSON parse failed for {path}: {e}"

private def getNatOrZero (j : Lean.Json) (field : String) : Nat :=
  match (j.getObjValAs? Nat field).toOption with
  | some n => n
  | none =>
    match (j.getObjValAs? Int field).toOption with
    | some i => if i < 0 then 0 else Int.toNat i
    | none => 0

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

private def uniform01 (x : UInt64) : Float :=
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  Float.ofNat mant / denom

private def signed01 (x : UInt64) : Float :=
  2.0 * uniform01 x - 1.0

private def priorLogitsFromSeed (seed : UInt64) (numActions : Nat) : Array Float :=
  (List.range numActions).toArray.map fun i =>
    let k := mix (seed + UInt64.ofNat (i + 1) * 0x9e3779b97f4a7c15)
    signed01 k

private def recurrentForFixture
    (numActions : Nat)
    (discount : Float)
    (zeroReward : Bool)
    : RecurrentFn Unit UInt64 :=
  fun _params _rng action emb =>
    let nextEmb := mix (emb + UInt64.ofNat (action + 1) * 0xbf58476d1ce4e5b9)
    let reward0 := signed01 (mix (nextEmb + 17))
    let reward := if zeroReward then 0.0 else reward0
    ({
      reward := reward
      discount := discount
      priorLogits := priorLogitsFromSeed nextEmb numActions
      value := signed01 (mix (nextEmb + 31))
    }, nextEmb)

private def runFixture (path : System.FilePath) : IO Unit := do
  let json ← parseJsonFileOrFail path

  let algorithm := ((json.getObjValAs? String "algorithm").toOption).getD ""
  let some treeJson := (json.getObjVal? "tree").toOption
    | LeanTest.fail s!"Missing tree object in fixture {path}"
  let childStats := ((treeJson.getObjValAs? (Array Lean.Json) "child_stats").toOption).getD #[]
  let numActions := childStats.size
  LeanTest.assertTrue (numActions > 0) s!"Fixture {path} should define at least one action"

  let rootVisit := getNatOrZero treeJson "visit"
  let numSimulations := Nat.max 1 (Nat.min 32 (rootVisit - 1))

  let envConfig := ((json.getObjVal? "env_config").toOption).getD (Lean.Json.mkObj [])
  let discount := ((envConfig.getObjValAs? Float "discount").toOption).getD 1.0
  let zeroReward := ((envConfig.getObjValAs? Bool "zero_reward").toOption).getD false

  let root : RootFnOutput UInt64 := {
    priorLogits := priorLogitsFromSeed 0 numActions
    value := signed01 (mix 1234)
    embedding := 0
  }

  let recurrent := recurrentForFixture numActions discount zeroReward

  if algorithm = "muzero" then
    let out := muzeroPolicy
      (params := ()) (rngKey := 1)
      (root := root)
      (recurrentFn := recurrent)
      (numSimulations := numSimulations)
      (dirichletFraction := 0.0)
    let summary := out.searchTree.summary
    LeanTest.assertEqual summary.visitCounts.size numActions
      s!"MuZero fixture {path}: visit count width should match action count"
    LeanTest.assertEqual (summary.visitCounts.foldl (init := 0) (· + ·)) numSimulations
      s!"MuZero fixture {path}: root visits should match simulation budget"
  else if algorithm = "gumbel_muzero" then
    let out := gumbelMuZeroPolicy
      (params := ()) (rngKey := 1)
      (root := root)
      (recurrentFn := recurrent)
      (numSimulations := numSimulations)
    let summary := out.searchTree.summary
    LeanTest.assertEqual summary.visitCounts.size numActions
      s!"Gumbel MuZero fixture {path}: visit count width should match action count"
    LeanTest.assertEqual (summary.visitCounts.foldl (init := 0) (· + ·)) numSimulations
      s!"Gumbel MuZero fixture {path}: root visits should match simulation budget"
  else
    LeanTest.fail s!"Unknown algorithm '{algorithm}' in fixture {path}"

@[test]
def testMctxTreeFixtureMuZero : IO Unit := do
  runFixture ⟨"Tests/MctxData/muzero_tree.json"⟩

@[test]
def testMctxTreeFixtureMuZeroQTransform : IO Unit := do
  runFixture ⟨"Tests/MctxData/muzero_qtransform_tree.json"⟩

@[test]
def testMctxTreeFixtureGumbelMuZero : IO Unit := do
  runFixture ⟨"Tests/MctxData/gumbel_muzero_tree.json"⟩

@[test]
def testMctxTreeFixtureGumbelMuZeroReward : IO Unit := do
  runFixture ⟨"Tests/MctxData/gumbel_muzero_reward_tree.json"⟩

@[test]
def testMctxGetSubtreeCarriesChildAsRoot : IO Unit := do
  let root : RootFnOutput UInt64 := {
    priorLogits := #[0.1, 0.3, 0.2, -0.1]
    value := 0.4
    embedding := 0
  }
  let recurrent : RecurrentFn Unit UInt64 := fun _ _ action emb =>
    let nextEmb := mix (emb + UInt64.ofNat (action + 1) * 0x9e3779b97f4a7c15)
    ({
      reward := signed01 (mix (nextEmb + 7))
      discount := 0.9
      priorLogits := priorLogitsFromSeed nextEmb 4
      value := signed01 (mix (nextEmb + 13))
    }, nextEmb)

  let out := muzeroPolicy
    (params := ())
    (rngKey := 9)
    (root := root)
    (recurrentFn := recurrent)
    (numSimulations := 8)
    (dirichletFraction := 0.0)

  let chosen := out.action
  let oldChild := (out.searchTree.childrenIndex.getD ROOT_INDEX #[]).getD chosen UNVISITED
  LeanTest.assertTrue (oldChild != UNVISITED) "Chosen action should be expanded"
  let childIdx := if oldChild < 0 then 0 else Int.toNat oldChild

  let subtree := getSubtree out.searchTree chosen

  LeanTest.assertEqual (subtree.parents.getD ROOT_INDEX 123) NO_PARENT
    "Subtree root should have no parent"
  LeanTest.assertEqual (subtree.actionFromParent.getD ROOT_INDEX 123) NO_PARENT
    "Subtree root should not have incoming action"
  LeanTest.assertTrue ((List.range subtree.rootInvalidActions.size).all fun i =>
      subtree.rootInvalidActions.getD i true = false)
    "Subtree root invalid-action mask should be reset"

  let oldVisits := out.searchTree.nodeVisits.getD childIdx 0
  let newVisits := subtree.nodeVisits.getD ROOT_INDEX 0
  LeanTest.assertEqual newVisits oldVisits "Subtree root visits should match selected child visits"

  let oldRaw := out.searchTree.rawValues.getD childIdx 0.0
  let newRaw := subtree.rawValues.getD ROOT_INDEX 0.0
  LeanTest.assertTrue (approx newRaw oldRaw 1e-6)
    s!"Subtree root raw value mismatch, expected {oldRaw}, got {newRaw}"

  let next := subtree.nextNodeIndex
  let trailingZero := (List.range subtree.nodeVisits.size).all fun i =>
    if i < next then true else subtree.nodeVisits.getD i 0 = 0
  LeanTest.assertTrue trailingZero "Subtree should compact retained nodes contiguously from index 0"

@[test]
def testMctxGetSubtreeOnUnvisitedActionResets : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, 1.0, 2.0, 3.0]
    value := 0.0
    embedding := ()
  }
  let recurrent : RecurrentFn Unit Unit := fun _ _ _ _ =>
    ({ reward := 0.0, discount := 0.0, priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0 }, ())
  let out := muzeroPolicy
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrent)
    (numSimulations := 1)
    (dirichletFraction := 0.0)

  let mut unvisited : Option Nat := none
  for a in [:4] do
    let idx := (out.searchTree.childrenIndex.getD ROOT_INDEX #[]).getD a UNVISITED
    if idx = UNVISITED && unvisited.isNone then
      unvisited := some a
  LeanTest.assertTrue unvisited.isSome "Expected at least one unvisited root action after one simulation"
  let action := unvisited.getD 0

  let subtree := getSubtree out.searchTree action
  LeanTest.assertTrue ((List.range subtree.nodeVisits.size).all fun i => subtree.nodeVisits.getD i 1 = 0)
    "Subtree of an unvisited root action should be reset"

@[test]
def testMctxResetSearchTreePreservesExtraData : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.1, 0.2]
    value := 1.0
    embedding := ()
  }
  let tree := instantiateTreeFromRoot root 4 #[false, true] (42 : Nat)
  let reset := resetSearchTree tree

  LeanTest.assertEqual reset.extraData 42 "Reset should preserve extraData"
  LeanTest.assertTrue ((List.range reset.nodeVisits.size).all fun i => reset.nodeVisits.getD i 1 = 0)
    "Reset tree should clear node visits"
  LeanTest.assertTrue ((List.range reset.rootInvalidActions.size).all fun i => reset.rootInvalidActions.getD i true = false)
    "Reset tree should clear root invalid-action mask"
