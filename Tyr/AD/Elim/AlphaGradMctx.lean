import Tyr.AD.Elim.Eliminate
import Tyr.AD.Elim.ConstraintFeasibility
import Tyr.AD.Elim.AlphaGradAdapter
import Tyr.AD.Elim.Cost
import Tyr.Mctx
import Tyr.MctxDag

/-!
# Tyr.AD.Elim.AlphaGradMctx

AlphaGrad-style elimination environment and MCTS wrappers built on `Tyr.Mctx`.

Design invariants:
- Action IDs remain 0-based (`ActionId0`).
- Elimination vertices remain 1-based (`VertexId1`), crossed only via checked adapters.
- Constraint feasibility is enforced in action masking and transition checks.
- No fallback behavior: infeasible constrained states surface deterministic errors.
-/

namespace Tyr.AD.Elim

open torch.mctx

/--
Configurable action-space conventions used by AlphaGrad-style policies:
- `fullVertices`: action space is `[0, numVertices)`, with `vertex = action + 1`.
- `explicitVertices`: action space is an explicit vertex subset/table.
-/
inductive AlphaGradActionSpace where
  | fullVertices
  | explicitVertices (vertices1 : Array VertexId1)
  deriving Repr, Inhabited

/-- Environment and objective knobs for AlphaGrad-style elimination search. -/
structure AlphaGradMctxConfig where
  constraints : ConstraintSpec := {}
  costWeights : CostWeights := {}
  /-- Action-space convention for AlphaGrad compatibility paths. -/
  actionSpace : AlphaGradActionSpace := .fullVertices
  /-- Search discount used by recurrent dynamics. -/
  discount : Float := 1.0
  /-- Reward applied when an invalid/infeasible action is selected. -/
  invalidActionPenalty : Float := -1.0e6
  /-- Additional reward applied when a non-terminal state has no feasible action. -/
  infeasibleStatePenalty : Float := -1.0e4
  /-- Terminal success bonus when all vertices are eliminated. -/
  terminalBonus : Float := 0.0
  /-- Optional hard cap on episode length. -/
  maxEpisodeSteps : Option Nat := none
  deriving Repr, Inhabited

/-- MCTS search hyperparameters for AlphaGrad planning. -/
structure AlphaGradMctsConfig where
  numSimulations : Nat := 32
  maxDepth : Option Nat := none
  maxNumConsideredActions : Nat := 16
  gumbelScale : Float := 1.0
  /-- DAG AlphaZero capacity override (defaults to `numSimulations + 1`). -/
  dagMaxNodes : Option Nat := none
  /-- Root Dirichlet noise for DAG AlphaZero policy. -/
  dagDirichletFraction : Float := 0.0
  /-- Action sampling temperature for DAG AlphaZero policy. -/
  dagTemperature : Float := 1.0
  deriving Repr, Inhabited

/-- DAG-backed MCTS policy family for AlphaGrad search. -/
inductive AlphaGradDagMctsPolicy where
  | alphaZero
  | gumbelMuZero
  deriving Repr, Inhabited

/-- Mutable AlphaGrad elimination state carried as MCTS embedding. -/
structure AlphaGradState where
  graph : ElimGraph := {}
  numVertices : Nat := 0
  /-- Action-space lookup table (`ActionId0 -> VertexId1`). -/
  actionVertices : Array VertexId1 := #[]
  /-- Elimination mask indexed by `ActionId0`. -/
  eliminatedActions : Array Bool := #[]
  /-- Action-native trajectory storage (keeps action `0` unambiguous). -/
  actionTrace : Array ActionId0 := #[]
  /-- Vertex-space replay trace (`VertexId1 = actionVertices[action]`). -/
  vertexTrace : Array VertexId1 := #[]
  cumulativeReward : Float := 0.0
  violation? : Option String := none
  deriving Repr, Inhabited

/-- Result of one environment transition. -/
structure AlphaGradTransition where
  nextState : AlphaGradState
  reward : Float
  done : Bool
  deriving Repr, Inhabited

/-- Search-time decision package for one MCTS-guided elimination step. -/
structure AlphaGradDecision where
  action : ActionId0
  vertex : VertexId1
  actionWeights : Array Float
  reward : Float
  done : Bool
  state : AlphaGradState
  deriving Repr

/-- End-to-end episode output. -/
structure AlphaGradEpisodeResult where
  finalState : AlphaGradState
  actions0 : Array ActionId0
  order1 : Array VertexId1
  stepRewards : Array Float
  totalReward : Float
  deriving Repr

/-- Canonical sparse-entry key for DAG state hashing. -/
structure AlphaGradDagMapEntryKey where
  src : Nat
  dst : Nat
  weightBits : UInt64
  deriving Repr, Inhabited, BEq, Hashable

/-- Canonical sparse-map key for DAG state hashing. -/
structure AlphaGradDagMapKey where
  inDim? : Option Nat
  outDim? : Option Nat
  entries : Array AlphaGradDagMapEntryKey
  deriving Repr, Inhabited, BEq, Hashable

/-- Canonical DAG edge key for state transposition hashing. -/
structure AlphaGradDagEdgeKey where
  src : VertexId1
  dst : VertexId1
  map : AlphaGradDagMapKey
  deriving Repr, Inhabited, BEq, Hashable

/-- Canonical DAG state key used by `Tyr.MctxDag` transposition table. -/
structure AlphaGradDagKey where
  numVertices : Nat
  actionVertices : Array VertexId1
  stepCount : Nat
  eliminatedActions : Array Bool
  violation : Bool
  edges : Array AlphaGradDagEdgeKey
  deriving Repr, Inhabited, BEq, Hashable

/-- DAG search tree specialized to AlphaGrad state/key. -/
abbrev AlphaGradDagTree :=
  torch.mctxdag.DagTree AlphaGradState AlphaGradDagKey Unit

private def noDuplicateActions (actions0 : Array ActionId0) : Bool :=
  hasNoDuplicates actions0

private def noDuplicateVertices (vertices1 : Array VertexId1) : Bool :=
  hasNoDuplicates vertices1

/-- Resolve and validate the configured action-space vertex table. -/
def resolveActionVertices?
    (cfg : AlphaGradMctxConfig)
    (numVertices : Nat) :
    Except String (Array VertexId1) := do
  let actionVertices :=
    match cfg.actionSpace with
    | .fullVertices => defaultActionVertices numVertices
    | .explicitVertices vertices1 => vertices1
  if actionVertices.isEmpty then
    throw "AlphaGrad action-space vertex table must be non-empty."
  validateVertexIds numVertices actionVertices
  if !noDuplicateVertices actionVertices then
    throw "AlphaGrad action-space vertex table contains duplicate vertex IDs."
  pure actionVertices

/-- Initialize AlphaGrad state with strict shape/domain checks. -/
def initAlphaGradState?
    (graph : ElimGraph)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[])
    (actionVertices? : Option (Array VertexId1) := none) :
    Except String AlphaGradState := do
  if numVertices = 0 then
    throw "AlphaGrad state requires at least one eliminable vertex."
  let actionVertices := actionVertices?.getD (defaultActionVertices numVertices)
  if actionVertices.isEmpty then
    throw "AlphaGrad state requires at least one action-space vertex."
  validateVertexIds numVertices actionVertices
  if !noDuplicateVertices actionVertices then
    throw "AlphaGrad state action-space vertex table contains duplicate vertex IDs."
  validateActionIds actionVertices.size actionPrefix
  if !noDuplicateActions actionPrefix then
    throw "Action prefix contains duplicate action IDs."
  let mut eliminated := Array.replicate actionVertices.size false
  let mut vertices : Array VertexId1 := #[]
  for action in actionPrefix do
    let vertex ← actionToVertexInSpace? actionVertices action
    if action < eliminated.size then
      eliminated := eliminated.set! action true
    vertices := vertices.push vertex
  pure {
    graph := graph
    numVertices := numVertices
    actionVertices := actionVertices
    eliminatedActions := eliminated
    actionTrace := actionPrefix
    vertexTrace := vertices
    cumulativeReward := 0.0
    violation? := none
  }

/-- Convenience initializer from local Jacobian edges. -/
def initAlphaGradStateFromEdges?
    (edges : Array Tyr.AD.JaxprLike.LocalJacEdge)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[])
    (actionVertices? : Option (Array VertexId1) := none) :
    Except String AlphaGradState :=
  initAlphaGradState? (ofLocalJacEdges edges) numVertices actionPrefix actionVertices?

/-- Number of actions already applied. -/
def AlphaGradState.stepCount (s : AlphaGradState) : Nat :=
  s.actionTrace.size

/-- Number of actions in the configured action space. -/
def AlphaGradState.numActions (s : AlphaGradState) : Nat :=
  s.actionVertices.size

/-- Number of eliminated vertices. -/
def AlphaGradState.eliminatedCount (s : AlphaGradState) : Nat :=
  s.eliminatedActions.foldl (init := 0) fun acc b => if b then acc + 1 else acc

/-- Domain-checked state-local map from action to vertex. -/
def AlphaGradState.actionToVertex?
    (s : AlphaGradState)
    (action : ActionId0) :
    Except String VertexId1 :=
  actionToVertexInSpace? s.actionVertices action

/-- Domain-checked state-local map from vertex to action. -/
def AlphaGradState.vertexToAction?
    (s : AlphaGradState)
    (vertex : VertexId1) :
    Except String ActionId0 :=
  vertexToActionInSpace? s.actionVertices vertex

/-- Check if action has already been eliminated. -/
def AlphaGradState.isActionEliminated (s : AlphaGradState) (action : ActionId0) : Bool :=
  s.eliminatedActions.getD action false

/-- Check if vertex has already been eliminated (1-based domain). -/
def AlphaGradState.isVertexEliminated (s : AlphaGradState) (vertex : VertexId1) : Bool :=
  match s.vertexToAction? vertex with
  | .ok action => s.isActionEliminated action
  | .error _ => false

/-- Terminal if violation is present, all vertices are eliminated, or step cap reached. -/
def isTerminal (cfg : AlphaGradMctxConfig) (s : AlphaGradState) : Bool :=
  s.violation?.isSome ||
    s.eliminatedCount >= s.numActions ||
    match cfg.maxEpisodeSteps with
    | none => false
    | some maxSteps => s.stepCount >= maxSteps

/-- Hard-constraint feasibility in vertex space for a given state. -/
def vertexConstraintFeasible
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (candidate : VertexId1) :
    Bool :=
  Tyr.AD.Elim.constraintFeasible cfg.constraints (fun v => s.isVertexEliminated v) candidate

/-- Full action feasibility integrating elimination mask and hard constraints. -/
def actionConstraintFeasible
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (action : ActionId0) :
    Bool :=
  actionFeasibleInSpace s.actionVertices
    (fun v => s.isVertexEliminated v)
    (fun v => vertexConstraintFeasible cfg s v)
    action

/-- Invalid-action mask in action space (`true` means invalid). -/
def invalidActionMask
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Bool :=
  (Array.range s.numActions).map fun action => !(actionConstraintFeasible cfg s action)

/-- True iff at least one feasible action remains. -/
def hasFeasibleAction
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Bool :=
  !(invalidActionMask cfg s).all (fun b => b)

/-- Markowitz-like static score proxy from current graph state. -/
def markowitzScore (g : ElimGraph) (vertex : VertexId1) : Nat :=
  (inNeighbors g vertex).size * (outNeighbors g vertex).size

/-- Deterministic canonicalization of sparse-map entries for DAG keys. -/
private def canonicalMapEntries
  (m : Tyr.AD.Sparse.SparseLinearMap) :
    Array AlphaGradDagMapEntryKey :=
  (m.entries.toList.mergeSort (fun a b =>
      if a.src = b.src then
        if a.dst = b.dst then
          a.weight < b.weight
        else
          a.dst < b.dst
      else
        a.src < b.src)).toArray.map fun e =>
      { src := e.src, dst := e.dst, weightBits := Float.toBits e.weight }

/-- Canonical sparse-map key independent of textual repr strings. -/
private def canonicalMapKey
    (m : Tyr.AD.Sparse.SparseLinearMap) :
    AlphaGradDagMapKey :=
  {
    inDim? := m.inDim?
    outDim? := m.outDim?
    entries := canonicalMapEntries m
  }

/-- Deterministic flattened edge listing for DAG transposition keys. -/
def dagEdgeKeys (g : ElimGraph) : Array AlphaGradDagEdgeKey := Id.run do
  let mut out : Array AlphaGradDagEdgeKey := #[]
  for src in vertices g do
    for pair in outNeighbors g src do
      out := out.push { src := src, dst := pair.1, map := canonicalMapKey pair.2 }
  return out

/-- Canonical key used by `MctxDag` to merge transposition-equivalent states. -/
def dagStateKey (s : AlphaGradState) : AlphaGradDagKey :=
  {
    numVertices := s.numVertices
    actionVertices := s.actionVertices
    stepCount := s.stepCount
    eliminatedActions := s.eliminatedActions
    violation := s.violation?.isSome
    edges := dagEdgeKeys s.graph
  }

/-- Soft precedence penalty paid when selecting `candidate` before its preferred predecessors. -/
def softPrecedencePenalty
    (constraints : ConstraintSpec)
    (isEliminated : VertexId1 → Bool)
    (candidate : VertexId1) :
    Float :=
  constraints.softPrecedence.foldl (init := 0.0) fun acc edge =>
    let u := edge.1
    let v := edge.2.1
    let w := edge.2.2
    if candidate == v && !(isEliminated u) then
      acc + w
    else
      acc

/-- Remaining soft precedence penalty mass for unresolved vertices. -/
def unresolvedSoftPenalty
    (constraints : ConstraintSpec)
    (isEliminated : VertexId1 → Bool) :
    Float :=
  constraints.softPrecedence.foldl (init := 0.0) fun acc edge =>
    let u := edge.1
    let v := edge.2.1
    let w := edge.2.2
    if !(isEliminated u) && !(isEliminated v) then acc + w else acc

private def stepCostPenalty
    (cfg : AlphaGradMctxConfig)
    (g : ElimGraph)
    (vertex : VertexId1) :
    Float :=
  weightedScore cfg.costWeights (estimateHeuristicStepCostFromGraph cfg.constraints g vertex)

private def policyPenaltyForVertex
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (vertex : VertexId1) :
    Float :=
  let softPenalty := softPrecedencePenalty cfg.constraints (fun v => s.isVertexEliminated v) vertex
  stepCostPenalty cfg s.graph vertex + softPenalty

private def feasibleStepCostPenalties
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float :=
  (Array.range s.numActions).foldl (init := #[]) fun acc action =>
    if actionConstraintFeasible cfg s action then
      match s.actionToVertex? action with
      | .ok vertex => acc.push (stepCostPenalty cfg s.graph vertex)
      | .error _ => acc
    else
      acc

private def meanPenalty (penalties : Array Float) : Float :=
  if penalties.isEmpty then
    0.0
  else
    penalties.foldl (init := 0.0) (· + ·) / Float.ofNat penalties.size

/-- Heuristic prior logits for MCTS expansion (masked in action space). -/
def heuristicPriorLogits
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float :=
  if isTerminal cfg s then
    Array.replicate s.numActions (0.0 - 1.0e30)
  else
    (Array.range s.numActions).map fun action =>
      if actionConstraintFeasible cfg s action then
        match s.actionToVertex? action with
        | .ok vertex =>
          0.0 - (policyPenaltyForVertex cfg s vertex)
        | .error _ =>
          0.0 - 1.0e30
      else
        0.0 - 1.0e30

/-- Heuristic value estimate used for root/recurrent outputs. -/
def heuristicValue
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Float :=
  match s.violation? with
  | some _ => cfg.invalidActionPenalty
  | none =>
    let remaining := s.numActions - s.eliminatedCount
    let unresolvedSoft := unresolvedSoftPenalty cfg.constraints (fun v => s.isVertexEliminated v)
    let frontierCost := meanPenalty (feasibleStepCostPenalties cfg s)
    if remaining = 0 then
      cfg.terminalBonus
    else
      0.0 - (Float.ofNat remaining + unresolvedSoft + frontierCost)

private def baseStepCost
    (cfg : AlphaGradMctxConfig)
    (gBefore : ElimGraph)
    (vertex : VertexId1)
    (stats : ElimStepStats) :
    StepCost :=
  {
    flops := stats.composedPairs
    localBytes := stats.insertedEdges + stats.updatedEdges
    comm := estimateCommForEliminationStep cfg.constraints gBefore vertex
  }

/-- Apply one action with strict feasibility checks. -/
def applyAction
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (action : ActionId0) :
    AlphaGradState × Float :=
  match s.actionToVertex? action with
  | .error msg =>
    ({ s with violation? := some msg }, cfg.invalidActionPenalty)
  | .ok vertex =>
    if !(actionConstraintFeasible cfg s action) then
      let msg := s!"Infeasible action {action} (vertex {vertex}) under current elimination/constraint mask."
      ({ s with violation? := some msg }, cfg.invalidActionPenalty)
    else
      match eliminateVertex s.graph vertex with
      | .error err =>
        let msg := s!"Elimination failed for vertex {vertex}: {err}"
        ({ s with violation? := some msg }, cfg.invalidActionPenalty)
      | .ok (graph', stats) =>
        let eliminated :=
          if action < s.eliminatedActions.size then
            s.eliminatedActions.set! action true
          else
            s.eliminatedActions
        let cost := baseStepCost cfg s.graph vertex stats
        let softPenalty := softPrecedencePenalty cfg.constraints (fun v => s.isVertexEliminated v) vertex
        let reward := rewardFromPenalty (objectivePenalty cfg.costWeights cost softPenalty)
        let s' := {
          s with
          graph := graph'
          eliminatedActions := eliminated
          actionTrace := s.actionTrace.push action
          vertexTrace := s.vertexTrace.push vertex
          cumulativeReward := s.cumulativeReward + reward
        }
        (s', reward)

private def stampInfeasibleState
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (reward : Float) :
    AlphaGradState × Float :=
  if isTerminal cfg s || hasFeasibleAction cfg s then
    (s, reward)
  else
    let msg :=
      s!"No feasible actions remain after step {s.stepCount} with {s.numActions - s.eliminatedCount} actions left."
    let penalty := cfg.infeasibleStatePenalty
    ({ s with violation? := some msg, cumulativeReward := s.cumulativeReward + penalty }, reward + penalty)

/-- Full transition dynamics used by recurrent_fn and rollout execution. -/
def transition
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (action : ActionId0) :
    AlphaGradTransition :=
  let (s1, r1) := applyAction cfg s action
  let (s2, r2) := stampInfeasibleState cfg s1 r1
  let successTerminal := s2.violation?.isNone && s2.eliminatedCount = s2.numActions
  let (s3, r3) :=
    if successTerminal then
      ({ s2 with cumulativeReward := s2.cumulativeReward + cfg.terminalBonus }, r2 + cfg.terminalBonus)
    else
      (s2, r2)
  { nextState := s3, reward := r3, done := isTerminal cfg s3 }

/-- Validate `true = invalid` mask semantics for action-space masks. -/
private def validateInvalidMaskSemantics?
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (invalid : Array Bool)
    (ctx : String) :
    Except String Unit := do
  if invalid.size != s.numActions then
    throw s!"Invalid-action mask size {invalid.size} does not match action-space size {s.numActions} in {ctx}."
  for action in [:s.numActions] do
    let expectedInvalid := !(actionConstraintFeasible cfg s action)
    let gotInvalid := invalid.getD action true
    if gotInvalid != expectedInvalid then
      let vertexStr :=
        match s.actionToVertex? action with
        | .ok vertex => s!"vertex {vertex}"
        | .error _ => "unmapped-vertex"
      throw s!"Mask semantics mismatch in {ctx}: action {action} ({vertexStr}) expected invalid={expectedInvalid}, got invalid={gotInvalid}. (`true` must mean invalid)."
  pure ()

/-- Validate that logits are hard-masked for invalid actions. -/
private def validateLogitsRespectMask?
    (s : AlphaGradState)
    (invalid : Array Bool)
    (logits : Array Float)
    (ctx : String) :
    Except String Unit := do
  if logits.size != invalid.size then
    throw s!"Logit size {logits.size} does not match invalid-mask size {invalid.size} in {ctx}."
  for action in [:invalid.size] do
    let isInvalid := invalid.getD action true
    let logit := logits.getD action (0.0 - 1.0e30)
    if isInvalid && logit > -1.0e20 then
      let vertexStr :=
        match s.actionToVertex? action with
        | .ok vertex => s!"vertex {vertex}"
        | .error _ => "unmapped-vertex"
      throw s!"Masked action {action} ({vertexStr}) has logit {logit} in {ctx}; expected strongly negative masked logit."
  pure ()

/-- MCTS recurrent dynamics over AlphaGrad state embedding. -/
def recurrentFn : RecurrentFn AlphaGradMctxConfig AlphaGradState :=
  fun cfg _rng action state =>
    let t := transition cfg state action
    let invalid := invalidActionMask cfg t.nextState
    let priors := heuristicPriorLogits cfg t.nextState
    let value := heuristicValue cfg t.nextState
    let discount := if t.done then 0.0 else cfg.discount
    match validateInvalidMaskSemantics? cfg t.nextState invalid "recurrentFn", validateLogitsRespectMask? t.nextState invalid priors "recurrentFn" with
    | .ok (), .ok () =>
      ({ reward := t.reward, discount := discount, priorLogits := priors, value := value }, t.nextState)
    | .error msg, _ =>
      let sDiag := { t.nextState with violation? := some msg }
      ({ reward := t.reward, discount := 0.0, priorLogits := Array.replicate sDiag.numActions (0.0 - 1.0e30), value := cfg.invalidActionPenalty }, sDiag)
    | _, .error msg =>
      let sDiag := { t.nextState with violation? := some msg }
      ({ reward := t.reward, discount := 0.0, priorLogits := Array.replicate sDiag.numActions (0.0 - 1.0e30), value := cfg.invalidActionPenalty }, sDiag)

/-- Root output + explicit invalid-action mask for current state. -/
def rootFromState?
    (cfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Except String (RootFnOutput AlphaGradState × Array Bool) := do
  if s.eliminatedActions.size != s.numActions then
    throw s!"State eliminated-mask size {s.eliminatedActions.size} does not match action-space size {s.numActions}."
  let invalid := invalidActionMask cfg s
  validateInvalidMaskSemantics? cfg s invalid "rootFromState?"
  if !(isTerminal cfg s) && invalid.all (fun b => b) then
    throw s!"No feasible actions available at step {s.stepCount} with {s.numActions - s.eliminatedCount} actions left."
  let priors := heuristicPriorLogits cfg s
  validateLogitsRespectMask? s invalid priors "rootFromState?"
  let root : RootFnOutput AlphaGradState := {
    priorLogits := priors
    value := heuristicValue cfg s
    embedding := s
  }
  pure (root, invalid)

/-- Run one Gumbel-MuZero-guided AlphaGrad step. -/
def searchStep?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s : AlphaGradState) :
    Except String AlphaGradDecision := do
  if isTerminal envCfg s then
    throw "Cannot run searchStep? on terminal AlphaGrad state."

  let (root, invalid) ← rootFromState? envCfg s
  let depth := mctsCfg.maxDepth.getD (s.numActions + 1)

  let out := gumbelMuZeroPolicy
    (params := envCfg)
    (rngKey := rngKey)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := mctsCfg.numSimulations)
    (invalidActions := some invalid)
    (maxDepth := some depth)
    (maxNumConsideredActions := mctsCfg.maxNumConsideredActions)
    (gumbelScale := mctsCfg.gumbelScale)

  let vertex ← s.actionToVertex? out.action
  let t := transition envCfg s out.action

  pure {
    action := out.action
    vertex := vertex
    actionWeights := out.actionWeights
    reward := t.reward
    done := t.done
    state := t.nextState
  }

/--
Build a decision object from one selected action and resulting transition.
-/
private def mkDecisionFromAction
    (envCfg : AlphaGradMctxConfig)
    (s : AlphaGradState)
    (action : ActionId0)
    (weights : Array Float) :
    Except String AlphaGradDecision := do
  let vertex ← s.actionToVertex? action
  let t := transition envCfg s action
  pure {
    action := action
    vertex := vertex
    actionWeights := weights
    reward := t.reward
    done := t.done
    state := t.nextState
  }

/--
Run one DAG-backed search step with a selectable policy family.
- `alphaZero`: supports persistent DAG-tree carry-over.
- `gumbelMuZero`: runs DAG Gumbel policy and returns no carry-over tree.
-/
def searchStepDagWithPolicy?
    (policy : AlphaGradDagMctsPolicy)
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s : AlphaGradState)
    (searchTree : Option AlphaGradDagTree := none) :
    Except String (AlphaGradDecision × Option AlphaGradDagTree) := do
  if isTerminal envCfg s then
    throw "Cannot run searchStepDagWithPolicy? on terminal AlphaGrad state."

  let (root, invalid) ← rootFromState? envCfg s
  let depth := mctsCfg.maxDepth.getD (s.numActions + 1)
  match policy with
  | .alphaZero =>
    let out := torch.mctxdag.alphazeroPolicyDag
      (params := envCfg)
      (rngKey := rngKey)
      (root := root)
      (recurrentFn := recurrentFn)
      (keyFn := dagStateKey)
      (numSimulations := mctsCfg.numSimulations)
      (searchTree := searchTree)
      (maxNodes := mctsCfg.dagMaxNodes)
      (invalidActions := some invalid)
      (maxDepth := some depth)
      (dirichletFraction := mctsCfg.dagDirichletFraction)
      (temperature := mctsCfg.dagTemperature)
    let decision ← mkDecisionFromAction envCfg s out.action out.actionWeights
    pure (decision, some out.searchTree)
  | .gumbelMuZero =>
    let out := torch.mctxdag.gumbelMuZeroPolicyDag
      (params := envCfg)
      (rngKey := rngKey)
      (root := root)
      (recurrentFn := recurrentFn)
      (keyFn := dagStateKey)
      (numSimulations := mctsCfg.numSimulations)
      (invalidActions := some invalid)
      (maxDepth := some depth)
      (maxNumConsideredActions := mctsCfg.maxNumConsideredActions)
      (gumbelScale := mctsCfg.gumbelScale)
    let decision ← mkDecisionFromAction envCfg s out.action out.actionWeights
    pure (decision, none)

/--
Run one DAG-backed AlphaZero-style search step.
Returns both the selected decision and the updated DAG search tree so callers
can preserve/transplant it across environment steps.
-/
def searchStepDag?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s : AlphaGradState)
    (searchTree : Option AlphaGradDagTree := none) :
    Except String (AlphaGradDecision × AlphaGradDagTree) := do
  match (← searchStepDagWithPolicy? .alphaZero envCfg mctsCfg rngKey s searchTree) with
  | (decision, some tree) => pure (decision, tree)
  | (_, none) => throw "AlphaZero DAG policy must return a DAG search tree."

/-- Run one DAG-backed Gumbel MuZero search step (no persistent subtree input). -/
def searchStepDagGumbel?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s : AlphaGradState) :
    Except String AlphaGradDecision := do
  let (decision, _tree?) ← searchStepDagWithPolicy? .gumbelMuZero envCfg mctsCfg rngKey s
  pure decision

/-- Replay an action trace through strict AlphaGrad transition semantics. -/
def replayActions?
    (envCfg : AlphaGradMctxConfig)
    (s0 : AlphaGradState)
    (actions0 : Array ActionId0) :
    Except String AlphaGradState := do
  validateActionIds s0.numActions actions0
  if !noDuplicateActions actions0 then
    throw "Replay action sequence contains duplicate action IDs."

  let mut s := s0
  for action in actions0 do
    if isTerminal envCfg s then
      throw "Replay reached terminal state before consuming all actions."
    let t := transition envCfg s action
    s := t.nextState

  match s.violation? with
  | some msg => throw msg
  | none => pure s

/-- Plan a full elimination episode by repeatedly calling MCTS step selection. -/
def searchEpisode?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s0 : AlphaGradState) :
    Except String AlphaGradEpisodeResult :=
  Id.run do
    let maxSteps := envCfg.maxEpisodeSteps.getD s0.numActions
    let mut s := s0
    let mut rewards : Array Float := #[]
    let mut step : Nat := 0

    while step < maxSteps && !(isTerminal envCfg s) do
      let key := rngKey + UInt64.ofNat (step + 1)
      match searchStep? envCfg mctsCfg key s with
      | .error msg =>
        return .error msg
      | .ok decision =>
        s := decision.state
        rewards := rewards.push decision.reward
        step := step + 1

    if !(isTerminal envCfg s) then
      return .error s!"AlphaGrad search did not terminate within {maxSteps} steps."

    match s.violation? with
    | some msg => return .error msg
    | none =>
      return .ok {
        finalState := s
        actions0 := s.actionTrace
        order1 := s.vertexTrace
        stepRewards := rewards
        totalReward := s.cumulativeReward
      }

/-- Generic DAG-backed full episode planner for selectable policy families. -/
def searchEpisodeDagWithPolicy?
    (policy : AlphaGradDagMctsPolicy)
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s0 : AlphaGradState) :
    Except String AlphaGradEpisodeResult :=
  Id.run do
    let maxSteps := envCfg.maxEpisodeSteps.getD s0.numActions
    let mut s := s0
    let mut rewards : Array Float := #[]
    let mut step : Nat := 0
    let mut dagTree? : Option AlphaGradDagTree := none

    while step < maxSteps && !(isTerminal envCfg s) do
      let key := rngKey + UInt64.ofNat (step + 1)
      match searchStepDagWithPolicy? policy envCfg mctsCfg key s dagTree? with
      | .error msg =>
        return .error msg
      | .ok (decision, tree?) =>
        s := decision.state
        rewards := rewards.push decision.reward
        step := step + 1
        match policy, tree? with
        | .alphaZero, some tree =>
          dagTree? :=
            if decision.done then
              none
            else
              some (torch.mctxdag.getSubtree tree decision.action)
        | .alphaZero, none =>
          return .error "AlphaZero DAG policy did not return a search tree."
        | .gumbelMuZero, _ =>
          dagTree? := none

    if !(isTerminal envCfg s) then
      let tag :=
        match policy with
        | .alphaZero => "AlphaGrad DAG search"
        | .gumbelMuZero => "AlphaGrad DAG Gumbel search"
      return .error s!"{tag} did not terminate within {maxSteps} steps."

    match s.violation? with
    | some msg => return .error msg
    | none =>
      return .ok {
        finalState := s
        actions0 := s.actionTrace
        order1 := s.vertexTrace
        stepRewards := rewards
        totalReward := s.cumulativeReward
      }

/-- DAG-backed full episode planner with subtree carry-over between steps. -/
def searchEpisodeDag?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s0 : AlphaGradState) :
    Except String AlphaGradEpisodeResult :=
  searchEpisodeDagWithPolicy? .alphaZero envCfg mctsCfg rngKey s0

/-- DAG + Gumbel full episode planner (rebuilds DAG search each step). -/
def searchEpisodeDagGumbel?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (s0 : AlphaGradState) :
    Except String AlphaGradEpisodeResult :=
  searchEpisodeDagWithPolicy? .gumbelMuZero envCfg mctsCfg rngKey s0

/-- Convenience entrypoint from local Jacobian edges. -/
def searchEpisodeFromEdges?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (edges : Array Tyr.AD.JaxprLike.LocalJacEdge)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[]) :
    Except String AlphaGradEpisodeResult := do
  let actionVertices ← resolveActionVertices? envCfg numVertices
  let s0 ← initAlphaGradStateFromEdges? edges numVertices actionPrefix (some actionVertices)
  searchEpisode? envCfg mctsCfg rngKey s0

/-- Policy-selectable DAG-backed convenience entrypoint from local Jacobian edges. -/
def searchEpisodeDagWithPolicyFromEdges?
    (policy : AlphaGradDagMctsPolicy)
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (edges : Array Tyr.AD.JaxprLike.LocalJacEdge)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[]) :
    Except String AlphaGradEpisodeResult := do
  let actionVertices ← resolveActionVertices? envCfg numVertices
  let s0 ← initAlphaGradStateFromEdges? edges numVertices actionPrefix (some actionVertices)
  searchEpisodeDagWithPolicy? policy envCfg mctsCfg rngKey s0

/-- DAG-backed convenience entrypoint from local Jacobian edges. -/
def searchEpisodeDagFromEdges?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (edges : Array Tyr.AD.JaxprLike.LocalJacEdge)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[]) :
    Except String AlphaGradEpisodeResult := do
  searchEpisodeDagWithPolicyFromEdges? .alphaZero envCfg mctsCfg rngKey edges numVertices actionPrefix

/-- DAG + Gumbel convenience entrypoint from local Jacobian edges. -/
def searchEpisodeDagGumbelFromEdges?
    (envCfg : AlphaGradMctxConfig)
    (mctsCfg : AlphaGradMctsConfig)
    (rngKey : UInt64)
    (edges : Array Tyr.AD.JaxprLike.LocalJacEdge)
    (numVertices : Nat)
    (actionPrefix : Array ActionId0 := #[]) :
    Except String AlphaGradEpisodeResult := do
  searchEpisodeDagWithPolicyFromEdges? .gumbelMuZero envCfg mctsCfg rngKey edges numVertices actionPrefix

end Tyr.AD.Elim
