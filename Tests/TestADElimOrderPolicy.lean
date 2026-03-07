import LeanTest
import Tyr.AD.Elim

namespace Tests.ADElimOrderPolicy

open LeanTest
open Tyr.AD.Elim

private def approx (a b : Float) (tol : Float := 1e-9) : Bool :=
  Float.abs (a - b) < tol

private def expectErrorEq (res : Except String α) (expected : String) : IO Unit := do
  match res with
  | .ok _ => LeanTest.fail s!"Expected error: {expected}"
  | .error msg => LeanTest.assertEqual msg expected s!"Unexpected error: {msg}"

@[test]
def testActionVertexAdapterRoundtrip : IO Unit := do
  let numVertices := 6
  let actions0 : Array ActionId0 := #[0, 2, 5]

  match actionsToVertices? numVertices actions0 with
  | .error msg =>
    LeanTest.fail s!"actionsToVertices? should succeed, got error: {msg}"
  | .ok vertices1 =>
    LeanTest.assertEqual vertices1 #[1, 3, 6] "Action->vertex mapping should shift IDs by +1"
    match verticesToActions? numVertices vertices1 with
    | .error msg =>
      LeanTest.fail s!"verticesToActions? should succeed, got error: {msg}"
    | .ok roundtrip =>
      LeanTest.assertEqual roundtrip actions0 "Vertex->action should invert action->vertex mapping"

  match vertexToAction? numVertices 4 with
  | .error msg =>
    LeanTest.fail s!"vertexToAction? should succeed for in-range vertex, got error: {msg}"
  | .ok action =>
    LeanTest.assertEqual action 3 "vertexToAction? should map 1-based vertex 4 to action 3"

  LeanTest.assertEqual (actionToVertex 3) 4 "actionToVertex should map 0-based action 3 to vertex 4"

@[test]
def testActionVertexAdapterRangeFailures : IO Unit := do
  expectErrorEq
    (actionToVertex? 6 6)
    "Invalid ActionId0 6. Expected action ID in [0, 5]."

  expectErrorEq
    (vertexToAction? 6 0)
    "Invalid VertexId1 0. Expected vertex ID in [1, 6]."

  expectErrorEq
    (verticesToActions? 6 #[1, 7])
    "Invalid VertexId1 7. Expected vertex ID in [1, 6]."

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
def testNormalizeAlphaGradOrderSuccess : IO Unit := do
  let constraints : ConstraintSpec := {
    hardPrecedence := #[(1, 3)]
    softPrecedence := #[(2, 5, 0.25)]
    groups := #[#[1, 5]]
    commHints := #[{ pattern := .allGather, bytes := 128, collectiveCount := 2 }]
  }

  match normalizeAlphaGradOrder 3 5 #[0, 2, 4] (some constraints) with
  | .error msg =>
    LeanTest.fail s!"normalizeAlphaGradOrder should succeed, got error: {msg}"
  | .ok (order1, outConstraints) =>
    LeanTest.assertEqual order1 #[1, 3, 5]
      "normalizeAlphaGradOrder should convert ActionId0 values to VertexId1 values"

    LeanTest.assertEqual outConstraints.hardPrecedence.size 1
      "Hard precedence constraints should be preserved"
    LeanTest.assertEqual (outConstraints.hardPrecedence.getD 0 (0, 0)) (1, 3)
      "Hard precedence payload should roundtrip"

    LeanTest.assertEqual outConstraints.softPrecedence.size 1
      "Soft precedence constraints should be preserved"
    let soft0 := outConstraints.softPrecedence.getD 0 (0, 0, 0.0)
    LeanTest.assertEqual soft0.1 2 "Soft precedence source should roundtrip"
    LeanTest.assertEqual soft0.2.1 5 "Soft precedence target should roundtrip"
    LeanTest.assertTrue (approx soft0.2.2 0.25)
      s!"Soft precedence weight should roundtrip, got {soft0.2.2}"

    LeanTest.assertEqual outConstraints.groups.size 1 "Constraint groups should be preserved"
    LeanTest.assertEqual (outConstraints.groups.getD 0 #[]) #[1, 5] "Group payload should roundtrip"

    LeanTest.assertEqual outConstraints.commHints.size 1 "Comm hints should be preserved"
    let hint0 := outConstraints.commHints.getD 0 { pattern := .allReduce, bytes := 0, collectiveCount := 0 }
    LeanTest.assertTrue (hint0.pattern == .allGather) "Comm hint pattern should roundtrip"
    LeanTest.assertEqual hint0.bytes 128 "Comm hint bytes should roundtrip"
    LeanTest.assertEqual hint0.collectiveCount 2 "Comm hint collectiveCount should roundtrip"

@[test]
def testNormalizeAlphaGradOrderFailure : IO Unit := do
  expectErrorEq
    (normalizeAlphaGradOrder 3 5 #[0, 2] none)
    "Action sequence length 2 does not match expected eliminable count 3."

  expectErrorEq
    (normalizeAlphaGradOrder 2 5 #[1, 5] none)
    "Invalid ActionId0 5. Expected action ID in [0, 4]."

@[test]
def testActionFeasibleWithEliminationAndConstraints : IO Unit := do
  let isEliminated : VertexId1 → Bool := fun v => (v == 2) || (v == 4)
  let constraintFeasible : VertexId1 → Bool := fun v => v != 3

  LeanTest.assertTrue
    (actionFeasible 4 isEliminated constraintFeasible 0)
    "Action 0 (vertex 1) should be feasible when not eliminated and constraint-feasible"

  LeanTest.assertTrue
    (!(actionFeasible 4 isEliminated constraintFeasible 1))
    "Action 1 (vertex 2) should be infeasible when already eliminated"

  LeanTest.assertTrue
    (!(actionFeasible 4 isEliminated constraintFeasible 2))
    "Action 2 (vertex 3) should be infeasible when constraint predicate rejects it"

  LeanTest.assertTrue
    (!(actionFeasible 4 isEliminated constraintFeasible 4))
    "Out-of-range actions should be infeasible"

end Tests.ADElimOrderPolicy
