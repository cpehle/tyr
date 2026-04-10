import Examples.AlphaGradPort.Model
import Tyr.Optim
import Tyr.Mctx

/-!
  Examples/AlphaGradPort/PolicyTrain.lean

  Full network training for AlphaGrad elimination:
  - PPO actor-critic training
  - AlphaZero-style MCTS self-play training

  Both paths use real trainable neural parameters,
  autograd backprop, and `Optim.adamw` updates.
-/

namespace Examples.AlphaGradPort

open torch
open torch.mctx
open Tyr.AD
open Tyr.AD.Elim

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

private def uniform01 (x : UInt64) : Float :=
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  Float.ofNat mant / denom

private def sampleCategorical?
    (probs : Array Float)
    (seed : UInt64) :
    Except String (ActionId0 × UInt64) := do
  if probs.isEmpty then
    throw "Cannot sample from empty probability vector."

  let seed' := mix seed
  let u := uniform01 seed'
  let mut cdf := 0.0
  let mut chosen? : Option ActionId0 := none

  for i in [:probs.size] do
    let pRaw := probs.getD i 0.0
    let p := if pRaw < 0.0 then 0.0 else pRaw
    cdf := cdf + p
    if chosen?.isNone && u <= cdf then
      chosen? := some i

  let chosen :=
    match chosen? with
    | some a => a
    | none => Id.run do
      let mut best : ActionId0 := 0
      let mut bestP := probs.getD 0 0.0
      for i in [1:probs.size] do
        let p := probs.getD i 0.0
        if p > bestP then
          best := i
          bestP := p
      return best

  pure (chosen, mix (seed' + 0x9e3779b97f4a7c15))

private def taskVertexDim (task : TaskSpec) : UInt64 :=
  task.numVertices.toUInt64

private def taskTokenDim (task : TaskSpec) : UInt64 :=
  (observationTokenDim task.graph task.numVertices).toUInt64

inductive AlphaGradValueTransform where
  | id
  | log
  | default
  deriving Repr, Inhabited, BEq

private def floatSign (x : Float) : Float :=
  if x < 0.0 then
    -1.0
  else if x > 0.0 then
    1.0
  else
    0.0

private def symlog (x : Float) : Float :=
  floatSign x * Float.log (Float.abs x + 1.0)

private def symexp (x : Float) : Float :=
  floatSign x * (Float.exp (Float.abs x) - 1.0)

private def defaultValueTransform (x : Float) (eps : Float := 0.001) : Float :=
  floatSign x * (Float.sqrt (Float.abs x + 1.0) - 1.0) + eps * x

private def defaultInverseValueTransform (x : Float) (eps : Float := 0.001) : Float :=
  let numer := Float.sqrt (1.0 + 4.0 * eps * (Float.abs x + 1.0 + eps)) - 1.0
  let core := numer / (2.0 * eps)
  floatSign x * (core * core - 1.0)

private def applyValueTransform
    (kind : AlphaGradValueTransform)
    (x : Float) :
    Float :=
  match kind with
  | .id => x
  | .log => symlog x
  | .default => defaultValueTransform x

private def invertValueTransform
    (kind : AlphaGradValueTransform)
    (x : Float) :
    Float :=
  match kind with
  | .id => x
  | .log => symexp x
  | .default => defaultInverseValueTransform x

private structure PPORolloutStep where
  features : Array Float
  action : ActionId0
  reward : Float
  value : Float
  nextValue : Float
  oldLogProb : Float
  done : Bool
  deriving Repr

private def computeGAE
    (steps : Array PPORolloutStep)
    (gamma gaeLambda : Float) :
    Array Float × Array Float := Id.run do
  let n := steps.size
  let mut advantages := Array.replicate n 0.0
  let mut returns := Array.replicate n 0.0
  let mut gae := 0.0
  for k in [:n] do
    let i := n - 1 - k
    let st := steps.getD i {
      features := #[]
      action := (0 : ActionId0)
      reward := 0.0
      value := 0.0
      nextValue := 0.0
      oldLogProb := 0.0
      done := true
    }
    let nonTerminal := if st.done then 0.0 else 1.0
    let delta := st.reward + gamma * st.nextValue * nonTerminal - st.value
    gae := delta + gamma * gaeLambda * nonTerminal * gae
    advantages := advantages.set! i gae
    returns := returns.set! i (gae + st.value)
  return (advantages, returns)

private def meanArray (xs : Array Float) : Float :=
  if xs.isEmpty then
    0.0
  else
    xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size

private def stdArray (xs : Array Float) (eps : Float := 1e-8) : Float :=
  if xs.isEmpty then
    1.0
  else
    let m := meanArray xs
    let var :=
      xs.foldl (init := 0.0) fun acc x =>
        let d := x - m
        acc + d * d
    Float.sqrt (var / Float.ofNat xs.size + eps)

private def normalizeArray (xs : Array Float) : Array Float :=
  if xs.isEmpty then
    #[]
  else
    let m := meanArray xs
    let s := stdArray xs
    xs.map (fun x => (x - m) / s)

private def rolloutPPOEpisode
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : EvalNet vertexDim tokenDim)
    (seed : UInt64) :
    Except String (Array PPORolloutStep × Float × UInt64) := do
  if vertexDim != taskVertexDim task then
    throw s!"Network vertexDim={vertexDim} does not match task vertex domain={task.numVertices}."
  if tokenDim != taskTokenDim task then
    throw s!"Network tokenDim={tokenDim} does not match task observation width={taskTokenDim task}."

  let s0 ← initAlphaGradState? task.graph task.numVertices
  let mut s := s0
  let mut key := seed
  let mut steps : Array PPORolloutStep := #[]
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut iters := 0

  while iters < maxSteps && !(isTerminal task.envCfg s) do
    let features := exportObservationFlat task.envCfg s
    let (logits, value) := evalStatePolicyValue net task.envCfg s
    let probs := softmax logits
    let (action, key') ← sampleCategorical? probs key
    key := key'
    let t := transition task.envCfg s action
    let nextValue :=
      if t.done then
        0.0
      else
        (evalStatePolicyValue net task.envCfg t.nextState).2
    let oldLogProb := logSafe (probs.getD action 0.0)
    steps := steps.push {
      features := features
      action := action
      reward := t.reward
      value := value
      nextValue := nextValue
      oldLogProb := oldLogProb
      done := t.done
    }
    s := t.nextState
    iters := iters + 1

  if !(isTerminal task.envCfg s) then
    throw s!"PPO rollout did not terminate within {maxSteps} steps."

  pure (steps, s.cumulativeReward, key)

private def evalGreedyReward?
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : EvalNet vertexDim tokenDim) :
    Except String Float := do
  if vertexDim != taskVertexDim task then
    throw s!"Network vertexDim={vertexDim} does not match task vertex domain={task.numVertices}."
  if tokenDim != taskTokenDim task then
    throw s!"Network tokenDim={tokenDim} does not match task observation width={taskTokenDim task}."
  let s0 ← initAlphaGradState? task.graph task.numVertices
  let mut s := s0
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut iters := 0
  while iters < maxSteps && !(isTerminal task.envCfg s) do
    let (logits, _value) := evalStatePolicyValue net task.envCfg s
    let action := argmax logits
    let t := transition task.envCfg s action
    s := t.nextState
    iters := iters + 1
  pure s.cumulativeReward

structure PPOTrainConfig where
  epochs : Nat := 64
  episodesPerEpoch : Nat := 16
  updateEpochs : Nat := 4
  gamma : Float := 0.99
  gaeLambda : Float := 0.95
  clipEps : Float := 0.2
  valueCoef : Float := 0.5
  entropyCoef : Float := 0.01
  learningRate : Float := 3.0e-4
  weightDecay : Float := 1.0e-4
  seed : UInt64 := 250197
  logEvery : Nat := 4
  deriving Repr

structure PPOSummary where
  taskName : String
  epochs : Nat
  episodesPerEpoch : Nat
  initialGreedyReward : Float
  lastGreedyReward : Float
  bestGreedyReward : Float
  bestGreedyEpoch : Nat
  bestEpochAvgRolloutReward : Float
  deriving Repr

private def ppoUpdate
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (steps : Array PPORolloutStep)
    (cfg : PPOTrainConfig) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim × Optim.AdamWState (AlphaGradNet vertexDim tokenDim) × Float × Float × Float)) := do
  if steps.isEmpty then
    return .error "PPO update received an empty rollout batch."

  let rows := steps.map (·.features)
  let actions : Array Int64 := steps.map (fun s => Int64.ofNat s.action)
  let oldLogProbs := steps.map (·.oldLogProb)
  let (advantagesRaw, returnsRaw) := computeGAE steps cfg.gamma cfg.gaeLambda
  let advantages := normalizeArray advantagesRaw

  match obsRowsToTensor3d? rows vertexDim tokenDim with
  | .error msg =>
    return .error msg
  | .ok ⟨n, obs⟩ =>
    if n.toNat != steps.size then
      return .error s!"Obs batch size mismatch: n={n}, steps={steps.size}."
    if actions.size != steps.size || oldLogProbs.size != steps.size || advantages.size != steps.size || returnsRaw.size != steps.size then
      return .error "PPO target arrays are not aligned with rollout size."

    let actionDyn := data.fromInt64Array actions
    let actionT0 : T #[n, 1] := reshape actionDyn #[n, 1]
    let actionT : T #[n, 1] := data.toLong actionT0
    let attnMaskT ←
      match attentionMaskTensor? rows vertexDim tokenDim with
      | .error msg =>
        return .error msg
      | .ok ⟨nMask, mask⟩ =>
        if nMask != n then
          return .error s!"Attention mask batch mismatch: expected {n}, got {nMask}."
        pure mask
    let actionIndexT ←
      match actionIndexTensor n task.graph.actionVertices with
      | .error msg =>
        return .error msg
      | .ok ⟨actionDim, idxs⟩ =>
        if actionDim.toNat != task.numActions then
          return .error s!"Action-surface width mismatch: expected {task.numActions}, got {actionDim}."
        pure idxs

    let oldLogDyn := data.fromFloatArray oldLogProbs
    let oldLogT : T #[n, 1] := reshape oldLogDyn #[n, 1]
    let advDyn := data.fromFloatArray advantages
    let advT : T #[n, 1] := reshape advDyn #[n, 1]
    let retDyn := data.fromFloatArray returnsRaw
    let retT : T #[n, 1] := reshape retDyn #[n, 1]

    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0
    let mut entropyV := 0.0

    for _ in [:cfg.updateEpochs] do
      let working := TensorStruct.zeroGrads (TensorStruct.makeLeafParams netCur)
      let (vertexLogits, values) := AlphaGradNet.forward working obs (some attnMaskT)
      let logits : T #[n, task.numActions.toUInt64] := gather vertexLogits (1 : Int64) actionIndexT
      let logProbsAll := nn.log_softmax logits (-1)
      let selectedLogProbs : T #[n, 1] := gather logProbsAll (1 : Int64) actionT

      let ratio := nn.exp (selectedLogProbs - oldLogT)
      let surr1 := ratio * advT
      let clipped := clampFloat ratio (1.0 - cfg.clipEps) (1.0 + cfg.clipEps)
      let surr2 := clipped * advT
      let choose1 := lt surr1 surr2
      let minSurr := where_ choose1 surr1 surr2
      let policyLoss := mul_scalar (nn.meanAll minSurr) (-1.0)

      let valueErr := values - retT
      let valueLoss := nn.meanAll (valueErr * valueErr)

      let probs := nn.softmax logits (-1)
      let entropyTerms := probs * logProbsAll
      let entropyPer : T #[n] := nn.sumDim entropyTerms 1 false
      let entropy := mul_scalar (nn.meanAll entropyPer) (-1.0)

      let totalLoss :=
        policyLoss +
        mul_scalar valueLoss cfg.valueCoef -
        mul_scalar entropy cfg.entropyCoef

      let _ ← autograd.backwardLoss totalLoss
      let grads := TensorStruct.grads working
      let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
      let (netNext, stateNext) := Optim.step opt working grads stateCur

      netCur := netNext
      stateCur := stateNext
      policyLossV := nn.item policyLoss
      valueLossV := nn.item valueLoss
      entropyV := nn.item entropy

    return .ok (netCur, stateCur, policyLossV, valueLossV, entropyV)

def trainPPO
    (task : TaskSpec)
    (cfg : PPOTrainConfig := {}) :
    IO (Except String PPOSummary) := do
  let vertexDim := taskVertexDim task
  let tokenDim := taskTokenDim task
  let net0 ← AlphaGradNet.init vertexDim tokenDim
  let net0 := TensorStruct.makeLeafParams net0
  let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
  let optState0 := opt.init net0
  let behavior0Eval ← buildEvalNet (TensorStruct.detach net0)

  let initialGreedyReward ←
    match evalGreedyReward? task behavior0Eval with
    | .ok r => pure r
    | .error msg =>
      return .error s!"PPO initial greedy eval failed: {msg}"

  IO.println s!"[AlphaGradPPO] task={task.name} epochs={cfg.epochs} episodes/epoch={cfg.episodesPerEpoch} vertices={task.numVertices} actions={task.numActions} eliminable={task.numEliminableVertices} edges={task.edges.size}"
  IO.println s!"[AlphaGradPPO] initial_greedy_reward={initialGreedyReward}"

  let mut net := net0
  let mut optState := optState0
  let mut key := cfg.seed
  let mut bestGreedy := initialGreedyReward
  let mut bestGreedyEpoch : Nat := 0
  let mut bestAvgRollout := -1.0e30
  let mut lastGreedy := initialGreedyReward

  for epoch in [:cfg.epochs] do
    let behaviorEval ← buildEvalNet (TensorStruct.detach net)
    let mut allSteps : Array PPORolloutStep := #[]
    let mut rewardSum := 0.0
    let mut episodesDone : Nat := 0

    for _ in [:cfg.episodesPerEpoch] do
      match rolloutPPOEpisode task behaviorEval key with
      | .error msg =>
        return .error s!"PPO rollout failed at epoch {epoch + 1}: {msg}"
      | .ok (steps, totalReward, key') =>
        key := key'
        allSteps := allSteps ++ steps
        rewardSum := rewardSum + totalReward
        episodesDone := episodesDone + 1

    let avgRollout :=
      if episodesDone = 0 then 0.0 else rewardSum / Float.ofNat episodesDone
    if avgRollout > bestAvgRollout then
      bestAvgRollout := avgRollout

    match (← ppoUpdate task net optState allSteps cfg) with
    | .error msg =>
      return .error s!"PPO update failed at epoch {epoch + 1}: {msg}"
    | .ok (net', optState', policyLoss, valueLoss, entropy) =>
      net := net'
      optState := optState'
      let evalNet ← buildEvalNet (TensorStruct.detach net)
      match evalGreedyReward? task evalNet with
      | .error msg =>
        return .error s!"PPO greedy eval failed at epoch {epoch + 1}: {msg}"
      | .ok greedyReward =>
        lastGreedy := greedyReward
        if greedyReward > bestGreedy then
          bestGreedy := greedyReward
          bestGreedyEpoch := epoch + 1
        let shouldLog :=
          cfg.logEvery > 0 &&
          (((epoch + 1) % cfg.logEvery = 0) || (epoch + 1 = cfg.epochs))
        if shouldLog then
          IO.println s!"[AlphaGradPPO] epoch={epoch + 1}/{cfg.epochs} avg_rollout={avgRollout} greedy={greedyReward} best_greedy={bestGreedy} policy_loss={policyLoss} value_loss={valueLoss} entropy={entropy}"

  let summary : PPOSummary := {
    taskName := task.name
    epochs := cfg.epochs
    episodesPerEpoch := cfg.episodesPerEpoch
    initialGreedyReward := initialGreedyReward
    lastGreedyReward := lastGreedy
    bestGreedyReward := bestGreedy
    bestGreedyEpoch := bestGreedyEpoch
    bestEpochAvgRolloutReward := bestAvgRollout
  }
  IO.println s!"[AlphaGradPPO] summary={reprStr summary}"
  pure (.ok summary)

private structure AZSearchParams (vertexDim tokenDim : UInt64) where
  envCfg : AlphaGradMctxConfig
  net : EvalNet vertexDim tokenDim
  valueTransform : AlphaGradValueTransform

private def recurrentFromNet
    {vertexDim tokenDim : UInt64} :
    RecurrentFn (AZSearchParams vertexDim tokenDim) AlphaGradState :=
  fun p _rng action s =>
    let t := transition p.envCfg s action
    let (priors, rawValue) := evalStatePolicyValue p.net p.envCfg t.nextState
    let value := invertValueTransform p.valueTransform rawValue
    let discount := if t.done then 0.0 else p.envCfg.discount
    ({ reward := t.reward, discount := discount, priorLogits := priors, value := value }, t.nextState)

private structure AZSample where
  features : Array Float
  policyTarget : Array Float
  valueTarget : Float
  deriving Repr

structure AlphaZeroTrainConfig where
  epochs : Nat := 48
  episodesPerEpoch : Nat := 8
  updateEpochs : Nat := 8
  gamma : Float := 1.0
  valueWeight : Float := 10.0
  learningRate : Float := 3.0e-4
  weightDecay : Float := 1.0e-4
  numSimulations : Nat := 48
  maxDepth : Option Nat := none
  valueTransform : AlphaGradValueTransform := .log
  qtransformValueScale : Float := 0.01
  qtransformMaxvisitInit : Float := 25.0
  qtransformRescaleValues : Bool := true
  qtransformUseMixedValue : Bool := true
  seed : UInt64 := 991337
  logEvery : Nat := 4
  deriving Repr

structure AlphaZeroSummary where
  taskName : String
  epochs : Nat
  episodesPerEpoch : Nat
  bestEpochAvgReward : Float
  finalGreedyReward : Float
  deriving Repr

private def discountedReturns (rewards : Array Float) (gamma : Float) : Array Float := Id.run do
  let n := rewards.size
  let mut out := Array.replicate n 0.0
  let mut acc := 0.0
  for k in [:n] do
    let i := n - 1 - k
    acc := rewards.getD i 0.0 + gamma * acc
    out := out.set! i acc
  return out

private def rolloutAlphaZeroEpisode
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : EvalNet vertexDim tokenDim)
    (cfg : AlphaZeroTrainConfig)
    (seed : UInt64) :
    Except String (Array AZSample × Float × UInt64) := do
  if vertexDim != taskVertexDim task then
    throw s!"Network vertexDim={vertexDim} does not match task vertex domain={task.numVertices}."
  if tokenDim != taskTokenDim task then
    throw s!"Network tokenDim={tokenDim} does not match task observation width={taskTokenDim task}."
  let s0 ← initAlphaGradState? task.graph task.numVertices
  let searchParams : AZSearchParams vertexDim tokenDim := {
    envCfg := task.envCfg
    net := net
    valueTransform := cfg.valueTransform
  }

  let mut s := s0
  let mut key := seed
  let mut featRows : Array (Array Float) := #[]
  let mut policyTargets : Array (Array Float) := #[]
  let mut rewards : Array Float := #[]
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut steps : Nat := 0

  while steps < maxSteps && !(isTerminal task.envCfg s) do
    let features := exportObservationFlat task.envCfg s
    let (priors, rawValue) := evalStatePolicyValue net task.envCfg s
    let value := invertValueTransform cfg.valueTransform rawValue
    let invalid := invalidActionMask task.envCfg s
    if invalid.all (fun b => b) then
      throw s!"AlphaZero root has no feasible actions at step {s.stepCount}."

    let root : RootFnOutput AlphaGradState := {
      priorLogits := priors
      value := value
      embedding := s
    }
    let out := gumbelMuZeroPolicy
      (params := searchParams)
      (rngKey := key)
      (root := root)
      (recurrentFn := recurrentFromNet)
      (numSimulations := cfg.numSimulations)
      (invalidActions := some invalid)
      (maxDepth := cfg.maxDepth)
      (qtransform := fun tree nodeIndex =>
        qtransformCompletedByMixValue
          tree nodeIndex
          cfg.qtransformValueScale
          cfg.qtransformMaxvisitInit
          cfg.qtransformRescaleValues
          cfg.qtransformUseMixedValue)
      (maxNumConsideredActions := task.mctsCfg.maxNumConsideredActions)
      (gumbelScale := task.mctsCfg.gumbelScale)

    let t := transition task.envCfg s out.action
    featRows := featRows.push features
    policyTargets := policyTargets.push out.actionWeights
    rewards := rewards.push t.reward
    s := t.nextState
    key := mix (key + UInt64.ofNat (steps + 1))
    steps := steps + 1

  if !(isTerminal task.envCfg s) then
    throw s!"AlphaZero rollout did not terminate within {maxSteps} steps."

  let returns := discountedReturns rewards cfg.gamma
  let mut samples : Array AZSample := #[]
  for i in [:featRows.size] do
    samples := samples.push {
      features := featRows.getD i #[]
      policyTarget := policyTargets.getD i #[]
      valueTarget := returns.getD i 0.0
    }
  pure (samples, s.cumulativeReward, key)

private def alphaZeroUpdate
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (samples : Array AZSample)
    (cfg : AlphaZeroTrainConfig) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim × Optim.AdamWState (AlphaGradNet vertexDim tokenDim) × Float × Float)) := do
  if samples.isEmpty then
    return .error "AlphaZero update received an empty sample batch."

  let obsRows := samples.map (·.features)
  let policyRows := samples.map (·.policyTarget)
  let valueTargets := samples.map (·.valueTarget)

  match obsRowsToTensor3d? obsRows vertexDim tokenDim with
  | .error msg =>
    return .error msg
  | .ok ⟨n, obs⟩ =>
    let attnMaskT ←
      match attentionMaskTensor? obsRows vertexDim tokenDim with
      | .error msg =>
        return .error msg
      | .ok ⟨nMask, mask⟩ =>
        if nMask != n then
          return .error s!"Attention mask batch mismatch: expected {n}, got {nMask}."
        pure mask
    if policyRows.size != n.toNat then
      return .error s!"AlphaZero policy rows mismatch: expected {n.toNat}, got {policyRows.size}."
    let actionNat := task.numActions
    let mut flatPolicy : Array Float := #[]
    for i in [:policyRows.size] do
      let row := policyRows.getD i #[]
      if row.size != actionNat then
        return .error s!"AlphaZero policy row {i} width mismatch: expected {actionNat}, got {row.size}."
      for x in row do
        flatPolicy := flatPolicy.push x
    let policyDyn := data.fromFloatArray flatPolicy
    let policyT : T #[n, task.numActions.toUInt64] := reshape policyDyn #[n, task.numActions.toUInt64]
    let actionIndexT ←
      match actionIndexTensor n task.graph.actionVertices with
      | .error msg =>
        return .error msg
      | .ok ⟨actionDim, idxs⟩ =>
        if actionDim.toNat != task.numActions then
          return .error s!"Action-surface width mismatch: expected {task.numActions}, got {actionDim}."
        pure idxs

    if valueTargets.size != n.toNat then
      return .error s!"AlphaZero value targets mismatch: expected {n.toNat}, got {valueTargets.size}."
    let valueTargets := valueTargets.map (applyValueTransform cfg.valueTransform)
    let valueDyn := data.fromFloatArray valueTargets
    let valueT : T #[n, 1] := reshape valueDyn #[n, 1]

    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0

    for _ in [:cfg.updateEpochs] do
      let working := TensorStruct.zeroGrads (TensorStruct.makeLeafParams netCur)
      let (vertexLogits, values) := AlphaGradNet.forward working obs (some attnMaskT)
      let logits : T #[n, task.numActions.toUInt64] := gather vertexLogits (1 : Int64) actionIndexT
      let logProbs := nn.log_softmax logits (-1)
      let policyPer : T #[n] := nn.sumDim (policyT * logProbs) 1 false
      let policyLoss := mul_scalar (nn.meanAll policyPer) (-1.0)
      let valueErr := values - valueT
      let valueLoss := nn.meanAll (valueErr * valueErr)
      let totalLoss := policyLoss + mul_scalar valueLoss cfg.valueWeight

      let _ ← autograd.backwardLoss totalLoss
      let grads := TensorStruct.grads working
      let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
      let (netNext, stateNext) := Optim.step opt working grads stateCur
      netCur := netNext
      stateCur := stateNext
      policyLossV := nn.item policyLoss
      valueLossV := nn.item valueLoss

    return .ok (netCur, stateCur, policyLossV, valueLossV)

def trainAlphaZero
    (task : TaskSpec)
    (cfg : AlphaZeroTrainConfig := {}) :
    IO (Except String AlphaZeroSummary) := do
  let vertexDim := taskVertexDim task
  let tokenDim := taskTokenDim task
  let net0 ← AlphaGradNet.init vertexDim tokenDim
  let net0 := TensorStruct.makeLeafParams net0
  let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
  let optState0 := opt.init net0

  IO.println s!"[AlphaGradAZ] task={task.name} epochs={cfg.epochs} episodes/epoch={cfg.episodesPerEpoch} sims={cfg.numSimulations} vertices={task.numVertices} actions={task.numActions} eliminable={task.numEliminableVertices} edges={task.edges.size}"

  let mut net := net0
  let mut optState := optState0
  let mut key := cfg.seed
  let mut bestAvgReward := -1.0e30

  for epoch in [:cfg.epochs] do
    let behavior ← buildEvalNet (TensorStruct.detach net)
    let mut samples : Array AZSample := #[]
    let mut rewardSum := 0.0
    let mut episodesDone : Nat := 0

    for _ in [:cfg.episodesPerEpoch] do
      match rolloutAlphaZeroEpisode task behavior cfg key with
      | .error msg =>
        return .error s!"AlphaZero rollout failed at epoch {epoch + 1}: {msg}"
      | .ok (epSamples, totalReward, key') =>
        key := key'
        samples := samples ++ epSamples
        rewardSum := rewardSum + totalReward
        episodesDone := episodesDone + 1

    let avgReward :=
      if episodesDone = 0 then 0.0 else rewardSum / Float.ofNat episodesDone
    if avgReward > bestAvgReward then
      bestAvgReward := avgReward

    match (← alphaZeroUpdate task net optState samples cfg) with
    | .error msg =>
      return .error s!"AlphaZero update failed at epoch {epoch + 1}: {msg}"
    | .ok (net', optState', policyLoss, valueLoss) =>
      net := net'
      optState := optState'
      let shouldLog :=
        cfg.logEvery > 0 &&
          (((epoch + 1) % cfg.logEvery = 0) || (epoch + 1 = cfg.epochs))
      if shouldLog then
        IO.println s!"[AlphaGradAZ] epoch={epoch + 1}/{cfg.epochs} avg_reward={avgReward} best_avg={bestAvgReward} policy_loss={policyLoss} value_loss={valueLoss}"

  let evalNet ← buildEvalNet (TensorStruct.detach net)
  let finalGreedyReward ←
    match evalGreedyReward? task evalNet with
    | .ok r => pure r
    | .error msg =>
      return .error s!"AlphaZero final greedy eval failed: {msg}"

  let summary : AlphaZeroSummary := {
    taskName := task.name
    epochs := cfg.epochs
    episodesPerEpoch := cfg.episodesPerEpoch
    bestEpochAvgReward := bestAvgReward
    finalGreedyReward := finalGreedyReward
  }
  IO.println s!"[AlphaGradAZ] summary={reprStr summary}"
  pure (.ok summary)

inductive TrainMode where
  | ppo
  | alphazero
  deriving Repr, Inhabited

instance : ToString TrainMode where
  toString
    | .ppo => "ppo"
    | .alphazero => "alphazero"

private def parseMode? (s : String) : Option TrainMode :=
  match s.trimAscii.toString.toLower with
  | "ppo" => some .ppo
  | "az" => some .alphazero
  | "alphazero" => some .alphazero
  | _ => none

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" <| ([
    "Usage:",
    "  lake exe AlphaGradPolicyTrain",
    "  lake exe AlphaGradPolicyTrain <mode>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name> <epochs>",
    "  lake exe AlphaGradPolicyTrain <mode> <task-name> <epochs> <episodes-per-epoch>",
    "Modes: ppo, alphazero (or az)",
    s!"Tasks: {taskNamesCsv}"
  ] : List String)

private def runTrain
    (mode : TrainMode)
    (taskName : TaskName)
    (epochs? : Option Nat := none)
    (episodesPerEpoch? : Option Nat := none) :
    IO UInt32 := do
  match (← materializeTask taskName) with
  | .error msg =>
    IO.eprintln s!"[AlphaGradTrain] task materialization failed: {msg}"
    pure 1
  | .ok task =>
    match mode with
    | .ppo =>
      let cfg : PPOTrainConfig := {
        epochs := epochs?.getD 64
        episodesPerEpoch := episodesPerEpoch?.getD 16
      }
      match (← trainPPO task cfg) with
      | .error msg =>
        IO.eprintln s!"[AlphaGradTrain][ppo] failed: {msg}"
        pure 1
      | .ok _ =>
        pure 0
    | .alphazero =>
      let cfg : AlphaZeroTrainConfig := {
        epochs := epochs?.getD 48
        episodesPerEpoch := episodesPerEpoch?.getD 8
        numSimulations := task.mctsCfg.numSimulations
      }
      match (← trainAlphaZero task cfg) with
      | .error msg =>
        IO.eprintln s!"[AlphaGradTrain][alphazero] failed: {msg}"
        pure 1
      | .ok _ =>
        pure 0

def policyTrainMain (args : List String) : IO UInt32 := do
  match args with
  | [] =>
    runTrain .ppo .roeFlux1d
  | [modeArg] =>
    match parseMode? modeArg with
    | none =>
      IO.eprintln s!"Unknown mode: {modeArg}"
      IO.eprintln usage
      pure 1
    | some mode =>
      runTrain mode .roeFlux1d
  | [modeArg, taskArg] =>
    match parseMode? modeArg, parseTaskName? taskArg with
    | some mode, some taskName =>
      runTrain mode taskName
    | _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, taskArg, epochsArg] =>
    match parseMode? modeArg, parseTaskName? taskArg, parseNatArg? epochsArg with
    | some mode, some taskName, some epochs =>
      runTrain mode taskName (some epochs)
    | _, _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg} {epochsArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, taskArg, epochsArg, epPerEpochArg] =>
    match parseMode? modeArg, parseTaskName? taskArg, parseNatArg? epochsArg, parseNatArg? epPerEpochArg with
    | some mode, some taskName, some epochs, some episodesPerEpoch =>
      runTrain mode taskName (some epochs) (some episodesPerEpoch)
    | _, _, _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {taskArg} {epochsArg} {epPerEpochArg}"
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

end Examples.AlphaGradPort
