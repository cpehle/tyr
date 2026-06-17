import Tyr.EventSkeleton.Core

/-!
# Tyr.EventSkeleton.Interval

Interval and checkpoint planning records for event-skeleton differentiation.
-/

namespace Tyr.EventSkeleton

/--
A continuous interval in the event skeleton, normally projected from an
accepted DiffEq solve attempt.
-/
structure AcceptedStepSegment where
  id : SegmentId
  attemptIndex : Nat
  tStart : Float
  tAttempt : Float
  tAfter : Float
  madeJumpBefore : Bool := false
  madeJumpAfter : Bool := false
  label : String := ""
  deriving Repr, Inhabited

namespace AcceptedStepSegment

def duration (seg : AcceptedStepSegment) : Float :=
  seg.tAfter - seg.tStart

/--
True when the integrator attempted to step to `tAttempt` but the accepted
segment ended earlier, usually because an event root was localized.
-/
def localizedByEvent (seg : AcceptedStepSegment) : Bool :=
  seg.tAfter != seg.tAttempt

def crossedJumpFlag (seg : AcceptedStepSegment) : Bool :=
  seg.madeJumpBefore != seg.madeJumpAfter

def intervalVertex (seg : AcceptedStepSegment) : SkeletonVertex :=
  {
    id := seg.id
    kind := .interval
    label := if seg.label.isEmpty then s!"segment:{seg.id}" else seg.label
  }

end AcceptedStepSegment

def mkIntervalAdjointMove (seg : AcceptedStepSegment) : SkeletonMove :=
  {
    kind := .intervalAdjoint
    targets := #[seg.id]
    cost := { work := Float.abs seg.duration }
    label := s!"interval-adjoint:{seg.id}"
  }

def mkCheckpointBoundaryMove (seg : AcceptedStepSegment) : SkeletonMove :=
  {
    kind := .checkpointBoundary
    targets := #[seg.id]
    cost := { memory := 1.0 }
    label := s!"checkpoint-boundary:{seg.id}"
  }

def mkRematerializeSegmentMove (seg : AcceptedStepSegment) : SkeletonMove :=
  {
    kind := .rematerializeSegment
    targets := #[seg.id]
    cost := { work := Float.abs seg.duration }
    label := s!"rematerialize-segment:{seg.id}"
  }

def mkFreezeControlMove (seg : AcceptedStepSegment) : SkeletonMove :=
  {
    kind := .freezeControl
    targets := #[seg.id]
    label := s!"freeze-control:{seg.id}"
  }

/--
Default deterministic interval plan: keep the segment as an explicit interval
adjoint boundary, and optionally store a checkpoint boundary.
-/
def planForAcceptedSegment
    (seg : AcceptedStepSegment)
    (checkpoint : Bool := true) : Array SkeletonMove := Id.run do
  let mut moves := #[mkIntervalAdjointMove seg]
  if checkpoint then
    moves := moves.push (mkCheckpointBoundaryMove seg)
  return moves

def graphFromAcceptedSegments
    (segments : Array AcceptedStepSegment)
    (checkpoint : Bool := true) : SkeletonGraph := Id.run do
  let mut g : SkeletonGraph := {}
  for seg in segments do
    g := g.addVertex seg.intervalVertex
    for move in planForAcceptedSegment seg checkpoint do
      g := g.addMove move
  return g

end Tyr.EventSkeleton
