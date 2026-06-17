import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqEventSkeletonBridge

open LeanTest
open torch
open torch.DiffEq
open torch.DiffEq.EventSkeletonBridge

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test]
def testAcceptedSegmentsFromSolveLoop : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    RK4.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let controller : ConstantStepSize := {}
  let solverState := solver.init term 0.0 0.2 (1.0 : Float) ()
  let controllerState :=
    StepSizeController.init controller term 0.0 0.2 (1.0 : Float) ()
      (some 0.1) solver.func (solver.errorOrder term)
  let loop0 :=
    initialSolveLoopState 0.0 0.2 (1.0 : Float)
      solverState controllerState false #[]
  let loop :=
    runSolveLoop term solver controller 0.2 () (some 8) #[]
      (solver.errorOrder term) loop0

  LeanTest.assertTrue (loop.result == Result.successful)
    "Expected successful solve loop"
  LeanTest.assertEqual loop.accepted 2
    "Constant-step RK4 solve over [0, 0.2] with dt=0.1 should accept two steps"

  let segments := acceptedSegmentsFromSolveLoopState loop
  LeanTest.assertEqual segments.size 2
    "Bridge should expose one skeleton segment per accepted attempt"
  LeanTest.assertEqual segments[0]!.attemptIndex 0
    "First segment should preserve the first solve-loop attempt index"
  LeanTest.assertTrue (approx segments[0]!.tStart 0.0 1.0e-12)
    s!"Expected first segment to start at 0.0, got {segments[0]!.tStart}"
  LeanTest.assertTrue (approx segments[0]!.tAfter 0.1 1.0e-12)
    s!"Expected first segment to end at 0.1, got {segments[0]!.tAfter}"
  LeanTest.assertTrue (approx segments[1]!.tStart 0.1 1.0e-12)
    s!"Expected second segment to start at 0.1, got {segments[1]!.tStart}"
  LeanTest.assertTrue (approx segments[1]!.tAfter 0.2 1.0e-12)
    s!"Expected second segment to end at 0.2, got {segments[1]!.tAfter}"

  let summary := summaryFromSolveLoopState loop
  LeanTest.assertEqual summary.segmentCount 2
    "Summary should count accepted skeleton segments"
  LeanTest.assertEqual summary.localizedSegments 0
    "No event localization should be reported for this plain ODE solve"

  let g := graphFromSolveLoopState loop
  LeanTest.assertEqual g.vertices.size 2
    "Solve-loop skeleton graph should contain two interval vertices"
  LeanTest.assertTrue (g.containsMoveKind .intervalAdjoint)
    "Solve-loop skeleton graph should contain interval-adjoint moves"
  LeanTest.assertTrue (g.containsMoveKind .checkpointBoundary)
    "Solve-loop skeleton graph should contain checkpoint-boundary moves"

@[test]
def testAcceptedSegmentsSkipRejectedAttempts : IO Unit := do
  let state : StepSizeState Unit := { dt := 0.1, state := () }
  let accepted : SolveLoopAttempt Float Unit := {
    tStart := 0.0
    yStart := 1.0
    tAttempt := 0.1
    yAttempt := 0.9
    tAfter := 0.1
    yAfter := 0.9
    controllerStateBefore := state
    controllerStateAfter := state
    accepted := true
    stepResult := Result.successful
    decisionResult := Result.successful
    madeJumpBefore := false
    madeJumpAfter := false
  }
  let rejected : SolveLoopAttempt Float Unit := {
    accepted with
    tStart := 0.1
    tAttempt := 0.2
    tAfter := 0.1
    accepted := false
  }
  let segments := acceptedSegmentsFromAttempts #[accepted, rejected, { accepted with tStart := 0.1, tAttempt := 0.2, tAfter := 0.2 }]
  LeanTest.assertEqual segments.size 2
    "Bridge should skip rejected attempts"
  LeanTest.assertEqual segments[0]!.attemptIndex 0
    "First accepted segment should retain attempt index 0"
  LeanTest.assertEqual segments[1]!.attemptIndex 2
    "Second accepted segment should retain attempt index 2 after a rejection"

end Tests.DiffEqEventSkeletonBridge
