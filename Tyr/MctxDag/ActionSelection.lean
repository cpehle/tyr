import Tyr.MctxDag.QTransforms

namespace torch.mctxdag

/-- Q-transform function type. -/
abbrev QTransform (S K E : Type) [BEq K] [Hashable K] := DagTree S K E → NodeIndex → Array Float

/-- Root action selector signature. -/
abbrev RootActionSelectionFn (S K E : Type) [BEq K] [Hashable K] :=
  UInt64 → DagTree S K E → NodeIndex → Action

/-- Interior action selector signature. -/
abbrev InteriorActionSelectionFn (S K E : Type) [BEq K] [Hashable K] :=
  UInt64 → DagTree S K E → NodeIndex → Depth → Action

/-- Switches between root and interior selection by depth. -/
def switchingActionSelectionWrapper
    [BEq K]
    [Hashable K]
    (rootFn : RootActionSelectionFn S K E)
    (interiorFn : InteriorActionSelectionFn S K E)
    : InteriorActionSelectionFn S K E :=
  fun rng tree nodeIndex depth =>
    if depth = 0 then rootFn rng tree nodeIndex else interiorFn rng tree nodeIndex depth

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

/-- PUCT action selection used by MuZero. -/
def muzeroActionSelection
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : NodeIndex)
    (depth : Depth)
    (qtransform : QTransform S K E := qtransformByParentAndSiblings)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    : Action :=
  let visitCounts := tree.childrenVisits.getD nodeIndex #[]
  let nodeVisit := Float.ofNat (tree.nodeVisits.getD nodeIndex 0)
  let pbC := pbCInit + Float.log ((nodeVisit + pbCBase + 1.0) / pbCBase)
  let priorProbs := softmax (tree.childrenPriorLogits.getD nodeIndex #[])
  let policyScore := (List.range priorProbs.size).toArray.map fun i =>
    Float.sqrt nodeVisit * pbC * priorProbs.getD i 0.0 /
      (Float.ofNat (visitCounts.getD i 0) + 1.0)
  let valueScore := qtransform tree nodeIndex
  let toArgmax := addArrays valueScore policyScore
  let invalid := if depth = 0 then some tree.rootInvalidActions else none
  maskedArgmax toArgmax invalid

end torch.mctxdag
