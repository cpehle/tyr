import Tyr.AD.Elim.CommModel
import Tyr.AD.Elim.Graph

/-!
# Tyr.AD.Elim.Cost

Cost model scaffolding for elimination scheduling and AlphaGrad objective wiring.
-/

namespace Tyr.AD.Elim

structure StepCost where
  flops : Nat := 0
  localBytes : Nat := 0
  comm : CommCost := {}
  deriving Repr, Inhabited

structure CostWeights where
  alphaFlops : Float := 1.0
  betaLocalBytes : Float := 0.0
  gammaCommBytes : Float := 0.0
  deltaCollectives : Float := 0.0
  epsilonP2PMsgs : Float := 0.0
  deriving Repr, Inhabited

def weightedScore (w : CostWeights) (c : StepCost) : Float :=
  w.alphaFlops * Float.ofNat c.flops +
    w.betaLocalBytes * Float.ofNat c.localBytes +
    w.gammaCommBytes * Float.ofNat c.comm.bytes +
    w.deltaCollectives * Float.ofNat c.comm.collectives +
    w.epsilonP2PMsgs * Float.ofNat c.comm.pointToPointMsgs

/-- Skeleton estimator that derives communication terms from order constraints hints. -/
def estimateCostFromConstraints (constraints : ConstraintSpec) : StepCost :=
  { comm := sumHints constraints.commHints }

private def groupIndexOf? (groups : Array (Array VertexId1)) (vertex : VertexId1) : Option Nat := Id.run do
  for h : i in [:groups.size] do
    if (groups[i]).contains vertex then
      return some i
  return none

private def crossesGroupBoundary
    (groups : Array (Array VertexId1))
    (a b : VertexId1) :
    Bool :=
  match groupIndexOf? groups a, groupIndexOf? groups b with
  | some ga, some gb => ga != gb
  | _, _ => false

private def edgePayloadBytes (map : Tyr.AD.Sparse.SparseLinearMap) : Nat :=
  (max 1 map.entries.size) * 16

/--
Estimate dynamic communication for one elimination step from graph cut structure:
- crossing incident edges into/out of the eliminated vertex
- crossing fill-in pairs induced by elimination (`in(v) × out(v)`).
-/
def estimateCommFromGraphStep
    (constraints : ConstraintSpec)
    (g : ElimGraph)
    (vertex : VertexId1) :
    CommCost := Id.run do
  if constraints.groups.isEmpty then
    return {}

  let incoming := inNeighbors g vertex
  let outgoing := outNeighbors g vertex

  let mut bytes : Nat := 0
  let mut p2p : Nat := 0
  let mut fillCross : Nat := 0

  for inPair in incoming do
    let src := inPair.1
    let map := inPair.2
    if crossesGroupBoundary constraints.groups src vertex then
      bytes := bytes + edgePayloadBytes map
      p2p := p2p + 1

  for outPair in outgoing do
    let dst := outPair.1
    let map := outPair.2
    if crossesGroupBoundary constraints.groups vertex dst then
      bytes := bytes + edgePayloadBytes map
      p2p := p2p + 1

  for inPair in incoming do
    let src := inPair.1
    for outPair in outgoing do
      let dst := outPair.1
      if crossesGroupBoundary constraints.groups src dst then
        fillCross := fillCross + 1

  let fillBytes := fillCross * 16
  let collectives := if fillCross >= 2 then 1 else 0
  return {
    bytes := bytes + fillBytes
    collectives := collectives
    pointToPointMsgs := p2p + fillCross
  }

/--
Combined per-step communication estimate:
- static hints declared in policy constraints
- dynamic cut/fill estimate from current elimination graph and chosen vertex.
-/
def estimateCommForEliminationStep
    (constraints : ConstraintSpec)
    (g : ElimGraph)
    (vertex : VertexId1) :
    CommCost :=
  sumHints constraints.commHints + estimateCommFromGraphStep constraints g vertex

/--
Cheap one-step surrogate cost from current graph structure without mutating it.
Used by non-terminal policy/value heuristics where exact elimination stats are unavailable.
-/
def estimateHeuristicStepCostFromGraph
    (constraints : ConstraintSpec)
    (g : ElimGraph)
    (vertex : VertexId1) :
    StepCost :=
  let inDeg := (inNeighbors g vertex).size
  let outDeg := (outNeighbors g vertex).size
  {
    flops := inDeg * outDeg
    localBytes := inDeg + outDeg + (inDeg * outDeg)
    comm := estimateCommForEliminationStep constraints g vertex
  }

/-- Positive objective penalty (lower is better) from weighted cost plus extra terms. -/
def objectivePenalty
    (w : CostWeights)
    (c : StepCost)
    (extraPenalty : Float := 0.0) :
    Float :=
  weightedScore w c + extraPenalty

/-- Convert objective penalty to reward contribution (higher reward is better). -/
def rewardFromPenalty (penalty : Float) : Float :=
  - penalty

/-- Convert minimization score to reward contribution (higher reward is better). -/
def rewardFromCost (w : CostWeights) (c : StepCost) : Float :=
  rewardFromPenalty (weightedScore w c)

end Tyr.AD.Elim
