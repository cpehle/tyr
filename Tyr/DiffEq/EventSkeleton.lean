import Tyr.DiffEq.Integrate
import Tyr.EventSkeleton.Interval

/-!
# Tyr.DiffEq.EventSkeleton

Bridge from the existing DiffEq solve loop to the separate event-skeleton
representation.  The bridge is intentionally read-only: it projects accepted
solve attempts into skeleton interval segments without changing integration.
-/

namespace torch.DiffEq.EventSkeletonBridge

open Tyr.EventSkeleton

def segmentOfAcceptedAttempt?
    {Y ControllerState : Type}
    (segmentId attemptIndex : Nat)
    (attempt : SolveLoopAttempt Y ControllerState) :
    Option AcceptedStepSegment :=
  if attempt.accepted then
    some {
      id := segmentId
      attemptIndex := attemptIndex
      tStart := attempt.tStart
      tAttempt := attempt.tAttempt
      tAfter := attempt.tAfter
      madeJumpBefore := attempt.madeJumpBefore
      madeJumpAfter := attempt.madeJumpAfter
      label := s!"attempt:{attemptIndex}"
    }
  else
    none

def acceptedSegmentsFromAttempts
    {Y ControllerState : Type}
    (attempts : Array (SolveLoopAttempt Y ControllerState)) :
    Array AcceptedStepSegment := Id.run do
  let mut segments : Array AcceptedStepSegment := #[]
  for i in [:attempts.size] do
    match attempts[i]? with
    | none => pure ()
    | some attempt =>
        match segmentOfAcceptedAttempt? segments.size i attempt with
        | some seg => segments := segments.push seg
        | none => pure ()
  return segments

def acceptedSegmentsFromSolveLoopState
    {Y SolverState ControllerState : Type}
    (loop : SolveLoopState Y SolverState ControllerState) :
    Array AcceptedStepSegment :=
  acceptedSegmentsFromAttempts loop.attempts

structure SolveSkeletonSummary where
  attempted : Nat
  accepted : Nat
  rejected : Nat
  segmentCount : Nat
  localizedSegments : Nat
  jumpFlagCrossings : Nat
  deriving Repr, Inhabited

def summaryFromSegments
    (attempted accepted rejected : Nat)
    (segments : Array AcceptedStepSegment) :
    SolveSkeletonSummary :=
  {
    attempted := attempted
    accepted := accepted
    rejected := rejected
    segmentCount := segments.size
    localizedSegments := segments.foldl
      (fun acc seg => if seg.localizedByEvent then acc + 1 else acc) 0
    jumpFlagCrossings := segments.foldl
      (fun acc seg => if seg.crossedJumpFlag then acc + 1 else acc) 0
  }

def summaryFromSolveLoopState
    {Y SolverState ControllerState : Type}
    (loop : SolveLoopState Y SolverState ControllerState) :
    SolveSkeletonSummary :=
  summaryFromSegments loop.attempted loop.accepted loop.rejected
    (acceptedSegmentsFromSolveLoopState loop)

def graphFromSolveLoopState
    {Y SolverState ControllerState : Type}
    (loop : SolveLoopState Y SolverState ControllerState)
    (checkpoint : Bool := true) :
    SkeletonGraph :=
  graphFromAcceptedSegments (acceptedSegmentsFromSolveLoopState loop) checkpoint

end torch.DiffEq.EventSkeletonBridge
