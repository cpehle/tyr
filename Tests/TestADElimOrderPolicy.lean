import LeanTest
import Tyr.AD.Elim

namespace Tests.ADElimOrderPolicy

open LeanTest
open Tyr.AD
open Tyr.AD.Elim
open Tyr.AD.JaxprLike

private def approx (a b : Float) (tol : Float := 1e-9) : Bool :=
  Float.abs (a - b) < tol

private def expectErrorEq (res : Except String α) (expected : String) : IO Unit := do
  match res with
  | .ok _ => LeanTest.fail s!"Expected error: {expected}"
  | .error msg => LeanTest.assertEqual msg expected s!"Unexpected error: {msg}"

@[test]
def testActionVertexAdapterRoundtrip : IO Unit := do
  let actionVertices : Array VertexId1 := #[1, 3, 6]

  expectErrorEq
    (actionToVertexInSpace? actionVertices 5)
    "Invalid ActionId0 5. Expected action ID in [0, 2] for action-space size 3."

  let inSpaceActions : Array ActionId0 := #[0, 1, 2]
  match actionsToVerticesInSpace? actionVertices inSpaceActions with
  | .error msg =>
    LeanTest.fail s!"actionsToVerticesInSpace? should succeed, got error: {msg}"
  | .ok vertices1 =>
    LeanTest.assertEqual vertices1 actionVertices
      "Explicit action surface should define the action->vertex mapping"
    match verticesToActionsInSpace? actionVertices vertices1 with
    | .error msg =>
      LeanTest.fail s!"verticesToActionsInSpace? should succeed, got error: {msg}"
    | .ok roundtrip =>
      LeanTest.assertEqual roundtrip inSpaceActions
        "Vertex->action should invert the explicit action surface"

  match vertexToActionInSpace? actionVertices 3 with
  | .error msg =>
    LeanTest.fail s!"vertexToActionInSpace? should succeed for configured vertex, got error: {msg}"
  | .ok action =>
    LeanTest.assertEqual action 1
      "Vertex 3 should map to its position in the explicit action surface"

@[test]
def testActionVertexAdapterRangeFailures : IO Unit := do
  expectErrorEq
    (actionToVertexInSpace? #[2, 4, 7] 3)
    "Invalid ActionId0 3. Expected action ID in [0, 2] for action-space size 3."

  expectErrorEq
    (vertexToActionInSpace? #[2, 4, 7] 0)
    "VertexId1 0 is not present in the configured action-space vertex set."

  expectErrorEq
    (verticesToActionsInSpace? #[2, 4, 7] #[2, 5])
    "VertexId1 5 is not present in the configured action-space vertex set."

@[test]
def testExplicitPolicyValidationErrors : IO Unit := do
  expectErrorEq
    (normalizeOrderPolicyShape 3 5 (.explicitVertex #[1, 2, 2]))
    "Custom order contains duplicate vertex IDs."

  expectErrorEq
    (normalizeOrderPolicyShape 3 5 (.explicitVertex #[1, 0, 3]))
    "Invalid VertexId1 0. Expected vertex ID in [1, 5]."

  expectErrorEq
    (normalizeOrderPolicyShape 3 5 (.explicitVertex #[1, 2]))
    "Custom order length 2 does not match expected eliminable count 3."

@[test]
def testActionFeasibleWithEliminationAndConstraints : IO Unit := do
  let actionVertices : Array VertexId1 := #[1, 2, 3, 4]
  let isEliminated : VertexId1 → Bool := fun v => (v == 2) || (v == 4)
  let constraintFeasible : VertexId1 → Bool := fun v => v != 3

  LeanTest.assertTrue
    (actionFeasibleInSpace actionVertices isEliminated constraintFeasible 0)
    "Action 0 (vertex 1) should be feasible when not eliminated and constraint-feasible"

  LeanTest.assertTrue
    (!(actionFeasibleInSpace actionVertices isEliminated constraintFeasible 1))
    "Action 1 (vertex 2) should be infeasible when already eliminated"

  LeanTest.assertTrue
    (!(actionFeasibleInSpace actionVertices isEliminated constraintFeasible 2))
    "Action 2 (vertex 3) should be infeasible when constraint predicate rejects it"

  LeanTest.assertTrue
    (!(actionFeasibleInSpace actionVertices isEliminated constraintFeasible 4))
    "Out-of-range actions should be infeasible"

private def sampleGraph : ElimGraph :=
  match ofLocalJacEdgesWithPartitions
      #[{ src := 1, dst := 2 }, { src := 2, dst := 4 }]
      #[1]
      #[4]
      #[2, 3] with
  | .ok g => g
  | .error msg => panic! msg

@[test]
def testNormalizeOrderPolicyAgainstGraphForwardReverse : IO Unit := do
  match normalizeOrderPolicyAgainstGraph sampleGraph .forward with
  | .error msg =>
    LeanTest.fail s!"Forward policy should normalize against graph, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? (some #[2, 3])
      "Forward policy should resolve to the graph's eliminable forward order"

  match normalizeOrderPolicyAgainstGraph sampleGraph .reverse with
  | .error msg =>
    LeanTest.fail s!"Reverse policy should normalize against graph, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? (some #[3, 2])
      "Reverse policy should resolve to the graph's eliminable reverse order"

@[test]
def testNormalizeOrderPolicyAgainstGraphExplicitAndAlphaGradValidation : IO Unit := do
  match normalizeOrderPolicyAgainstGraph sampleGraph (.explicitVertex #[2, 3]) with
  | .error msg =>
    LeanTest.fail s!"Explicit eliminable order should normalize, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? (some #[2, 3])
      "Explicit order should be preserved after graph-aware normalization"

  expectErrorEq
    (normalizeOrderPolicyAgainstGraph sampleGraph (.explicitVertex #[1, 2]))
    "Custom order references non-eliminable vertex 1."

  match normalizeOrderPolicyAgainstGraph sampleGraph (.alphaGradAction #[0, 1]) with
  | .error msg =>
    LeanTest.fail s!"AlphaGrad actions should normalize against graph eliminables, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? (some #[2, 3])
      "AlphaGrad action order should convert to the corresponding eliminable vertex order"

  expectErrorEq
    (normalizeOrderPolicyAgainstGraph sampleGraph (.alphaGradAction #[0, 2]))
    "Invalid ActionId0 2. Expected action ID in [0, 1]."

@[test]
def testNormalizeOrderPolicyAgainstGraphHeuristicAliasesAndUnresolved : IO Unit := do
  match normalizeOrderPolicyAgainstGraph sampleGraph (.heuristic "fwd") with
  | .error msg =>
    LeanTest.fail s!"Heuristic alias `fwd` should normalize, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? (some #[2, 3])
      "Heuristic alias `fwd` should resolve to forward eliminable order"

  match normalizeOrderPolicyAgainstGraph sampleGraph (.heuristic "markowitz") with
  | .error msg =>
    LeanTest.fail s!"Unknown heuristic names should remain representable, got: {msg}"
  | .ok normalized =>
    LeanTest.assertEqual normalized.baseOrder1? none
      "Unimplemented heuristics should remain unresolved until a scheduler is wired"

end Tests.ADElimOrderPolicy
