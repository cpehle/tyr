import Tyr.AD.Elim.OrderPolicy

/-!
# Tyr.AD.Elim.CommModel

Communication cost abstractions for elimination scheduling.
-/

namespace Tyr.AD.Elim

structure CommCost where
  bytes : Nat := 0
  collectives : Nat := 0
  pointToPointMsgs : Nat := 0
  deriving Repr, Inhabited

def CommCost.add (a b : CommCost) : CommCost :=
  {
    bytes := a.bytes + b.bytes
    collectives := a.collectives + b.collectives
    pointToPointMsgs := a.pointToPointMsgs + b.pointToPointMsgs
  }

instance : HAdd CommCost CommCost CommCost where
  hAdd := CommCost.add

def fromHint (hint : CommHint) : CommCost :=
  match hint.pattern with
  | .allReduce | .allGather | .reduceScatter =>
    { bytes := hint.bytes, collectives := hint.collectiveCount }
  | .pointToPoint =>
    { bytes := hint.bytes, pointToPointMsgs := hint.collectiveCount }

def sumHints (hints : Array CommHint) : CommCost :=
  hints.foldl (init := {}) fun acc h => acc + fromHint h

end Tyr.AD.Elim
