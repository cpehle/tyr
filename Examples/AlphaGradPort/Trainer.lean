import Examples.AlphaGradPort.PolicyTrain
import Examples.AlphaGradPort.Replay
import Tyr.Checkpoint
import Tyr.Mctx.BatchedM
import Tyr.Train.RunLedger
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace Examples.AlphaGradPort

open Lean
open torch
open torch.mctx
open Tyr.AD
open Tyr.AD.Elim
open torch.checkpoint
open torch.Train.RunLedger

instance : ToJson TaskName where
  toJson task := Json.str (toString task)

instance : FromJson TaskName where
  fromJson?
    | .str s =>
      match parseTaskName? s with
      | some task => pure task
      | none => throw s!"Unknown AlphaGrad task '{s}'."
    | _ => throw "Expected AlphaGrad task name as JSON string."

instance : ToJson TrainMode where
  toJson mode := Json.str (toString mode)

instance : FromJson TrainMode where
  fromJson?
    | .str "ppo" => pure .ppo
    | .str "alphazero" => pure .alphazero
    | .str "az" => pure .alphazero
    | .str s => throw s!"Unknown AlphaGrad mode '{s}'."
    | _ => throw "Expected AlphaGrad mode as JSON string."

instance : ToJson AlphaGradValueTransform where
  toJson
    | .id => Json.str "id"
    | .log => Json.str "log"
    | .default => Json.str "default"

instance : FromJson AlphaGradValueTransform where
  fromJson?
    | .str "id" => pure .id
    | .str "log" => pure .log
    | .str "default" => pure .default
    | .str s => throw s!"Unknown AlphaGrad value transform '{s}'."
    | _ => throw "Expected AlphaGrad value transform as JSON string."

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
    | none =>
      Id.run do
        let mut best : ActionId0 := 0
        let mut bestP := probs.getD 0 0.0
        for i in [1:probs.size] do
          let p := probs.getD i 0.0
          if p > bestP then
            best := i
            bestP := p
        return best
  pure (chosen, mix (seed' + 0x9e3779b97f4a7c15))

private def floatSign (x : Float) : Float :=
  if x < 0.0 then -1.0 else if x > 0.0 then 1.0 else 0.0

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

private def capsVertexDim (caps : ObservationCaps) : UInt64 :=
  caps.vertexCap.toUInt64

private def capsTokenDim (caps : ObservationCaps) : UInt64 :=
  caps.tokenDim.toUInt64

private def evalGreedyReward?
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (caps : ObservationCaps) :
    IO (Except String Float) := do
  if vertexDim != capsVertexDim caps then
    return .error s!"Network vertexDim={vertexDim} does not match observation cap={caps.vertexCap}."
  if tokenDim != capsTokenDim caps then
    return .error s!"Network tokenDim={tokenDim} does not match observation token width={caps.tokenDim}."
  match initAlphaGradState? task.graph task.numVertices with
  | .error msg => pure (.error msg)
  | .ok s0 =>
    let mut s := s0
    let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
    let mut iters := 0
    while iters < maxSteps && !(isTerminal task.envCfg s) do
      match (← evalStatesPolicyValueTensor? net task caps #[s]) with
      | .error msg => return .error msg
      | .ok out =>
        let logits := out.actionLogits.getD 0 #[]
        let action := argmax logits
        let t := transition task.envCfg s action
        s := t.nextState
        iters := iters + 1
    pure (.ok s.cumulativeReward)

structure AlphaGradTrainerConfig where
  mode : TrainMode := .alphazero
  task : TaskName := .perceptron
  epochs : Nat := 64
  episodesPerEpoch : Nat := 16
  numEnvs : Nat := 8
  replayCapacity : Nat := 4096
  sampleBatchSize : Nat := 256
  updateBatchesPerEpoch : Nat := 4
  ppoUpdateEpochs : Nat := 4
  alphaZeroUpdateEpochs : Nat := 8
  gamma : Float := 1.0
  gaeLambda : Float := 0.95
  clipEps : Float := 0.2
  ppoValueCoef : Float := 0.5
  entropyCoef : Float := 0.01
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
  evalEvery : Nat := 1
  checkpointEvery : Nat := 1
  runDir : String := ""
  seed : Nat := 250197
  resume : Bool := false
  overwrite : Bool := false
  deriving Repr, Inhabited, ToJson, FromJson

structure TrainerSnapshot where
  mode : TrainMode
  task : TaskName
  completedEpochs : Nat := 0
  globalEpisodes : Nat := 0
  globalSamples : Nat := 0
  nextSeed : Nat := 0
  bestEvalReward : Float := -1.0e30
  bestSearchReward : Float := -1.0e30
  lastTrainReward : Float := 0.0
  lastPolicyLoss : Float := 0.0
  lastValueLoss : Float := 0.0
  lastEntropy : Float := 0.0
  deriving Repr, Inhabited, ToJson, FromJson

private structure PendingAZStep where
  features : Array Float
  policyTarget : Array Float
  reward : Float
  deriving Inhabited

private structure PendingPPOStep where
  features : Array Float
  action : ActionId0
  reward : Float
  value : Float
  nextValue : Float
  oldLogProb : Float
  done : Bool
  deriving Inhabited

private def meanArray (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size

private def stdArray (xs : Array Float) (eps : Float := 1e-8) : Float :=
  if xs.isEmpty then
    1.0
  else
    let m := meanArray xs
    let var := xs.foldl (init := 0.0) fun acc x =>
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

private def discountedReturns (rewards : Array Float) (gamma : Float) : Array Float := Id.run do
  let n := rewards.size
  let mut out := Array.replicate n 0.0
  let mut acc := 0.0
  for k in [:n] do
    let i := n - 1 - k
    acc := rewards.getD i 0.0 + gamma * acc
    out := out.set! i acc
  return out

private def computeGAE
    (steps : Array PendingPPOStep)
    (gamma gaeLambda : Float) :
    Array Float × Array Float := Id.run do
  let n := steps.size
  let mut advantages := Array.replicate n 0.0
  let mut returns := Array.replicate n 0.0
  let mut gae := 0.0
  for k in [:n] do
    let i := n - 1 - k
    let st := steps.getD i default
    let nonTerminal := if st.done then 0.0 else 1.0
    let delta := st.reward + gamma * st.nextValue * nonTerminal - st.value
    gae := delta + gamma * gaeLambda * nonTerminal * gae
    advantages := advantages.set! i gae
    returns := returns.set! i (gae + st.value)
  return (advantages, returns)

private def evalStatesPolicyValueTensorIO
    {vertexDim tokenDim : UInt64}
    (net : AlphaGradNet vertexDim tokenDim)
    (task : TaskSpec)
    (caps : ObservationCaps)
    (states : Array AlphaGradState) :
    IO BatchedActionEval := do
  let evalOut ← evalStatesPolicyValueTensor? net task caps states
  match evalOut with
  | .error msg => throw <| IO.userError msg
  | .ok out => pure out

private structure BatchedAZSearchParams (vertexDim tokenDim : UInt64) where
  task : TaskSpec
  caps : ObservationCaps
  net : AlphaGradNet vertexDim tokenDim
  valueTransform : AlphaGradValueTransform

private def batchedRecurrentFromNetM
    {vertexDim tokenDim : UInt64} :
    BatchedRecurrentFnM IO (BatchedAZSearchParams vertexDim tokenDim) AlphaGradState :=
  fun p _rng actions states => do
    let transitions := (List.range states.size).toArray.map fun i =>
      transition p.task.envCfg (states.getD i default) (actions.getD i 0)
    let nextStates := transitions.map (·.nextState)
    let evalOut ← evalStatesPolicyValueTensorIO p.net p.task p.caps nextStates
    let values := evalOut.values.map (invertValueTransform p.valueTransform)
    let rewards := transitions.map (·.reward)
    let discounts := transitions.map (fun t => if t.done then 0.0 else p.task.envCfg.discount)
    pure ({
      reward := rewards
      discount := discounts
      priorLogits := evalOut.actionLogits
      value := values
    }, nextStates)

private def collectAlphaZeroEpisodeBatch
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (caps : ObservationCaps)
    (net : AlphaGradNet vertexDim tokenDim)
    (cfg : AlphaGradTrainerConfig)
    (batchSize : Nat)
    (seed : UInt64) :
    IO (Except String (Array ReplaySample × Array Float × UInt64)) := do
  if batchSize = 0 then
    return .error "AlphaGrad batch collector requires batchSize > 0."
  let s0 ←
    match initAlphaGradState? task.graph task.numVertices with
    | .error msg => return .error msg
    | .ok s0 => pure s0
  let searchParams : BatchedAZSearchParams vertexDim tokenDim := {
    task := task
    caps := caps
    net := net
    valueTransform := cfg.valueTransform
  }
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut states := Array.replicate batchSize s0
  let mut active : Array Nat := (List.range batchSize).toArray
  let mut pending : Array (Array PendingAZStep) := Array.replicate batchSize #[]
  let mut totals : Array Float := Array.replicate batchSize 0.0
  let mut key := seed
  let mut step := 0
  while !active.isEmpty && step < maxSteps do
    let activeStates := active.map (fun idx => states.getD idx s0)
    let evalOut ←
      try
        evalStatesPolicyValueTensorIO net task caps activeStates
      catch e =>
        return .error e.toString
    let invalids := activeStates.map (invalidActionMask task.envCfg)
    if invalids.any (fun row => row.all (fun b => b)) then
      return .error s!"AlphaZero batched root has no feasible actions at step {step}."
    let root : BatchedRootFnOutput AlphaGradState := {
      priorLogits := evalOut.actionLogits
      value := evalOut.values.map (invertValueTransform cfg.valueTransform)
      embedding := activeStates
    }
    let out ←
      try
        gumbelMuZeroPolicyBatchedM
          (params := searchParams)
          (rngKey := key)
          (root := root)
          (recurrentFn := batchedRecurrentFromNetM)
          (numSimulations := cfg.numSimulations)
          (invalidActions := some invalids)
          (maxDepth := some (cfg.maxDepth.getD task.numEliminableVertices))
          (qtransform := fun tree nodeIndex =>
            qtransformCompletedByMixValue
              tree nodeIndex
              cfg.qtransformValueScale
              cfg.qtransformMaxvisitInit
              cfg.qtransformRescaleValues
              cfg.qtransformUseMixedValue)
          (maxNumConsideredActions := task.mctsCfg.maxNumConsideredActions)
          (gumbelScale := task.mctsCfg.gumbelScale)
      catch e =>
        return .error e.toString
    let mut newActive : Array Nat := #[]
    for idx in [:active.size] do
      let global := active.getD idx 0
      let state := activeStates.getD idx s0
      let t := transition task.envCfg state (out.action.getD idx 0)
      let currentPending := pending.getD global #[]
      pending := pending.set! global (currentPending.push {
        features := evalOut.observations.getD idx #[]
        policyTarget := out.actionWeights.getD idx #[]
        reward := t.reward
      })
      totals := totals.set! global (totals.getD global 0.0 + t.reward)
      states := states.set! global t.nextState
      if !t.done then
        newActive := newActive.push global
    active := newActive
    key := mix (key + UInt64.ofNat (step + 1))
    step := step + 1
  if !active.isEmpty then
    return .error s!"AlphaZero batched rollout did not terminate within {maxSteps} steps."
  let mut samples : Array ReplaySample := #[]
  for envIdx in [:batchSize] do
    let envSteps := pending.getD envIdx #[]
    let returns := discountedReturns (envSteps.map (·.reward)) cfg.gamma
    for i in [:envSteps.size] do
      let st := envSteps.getD i default
      samples := samples.push {
        kind := .alphazero
        features := st.features
        reward := st.reward
        valueTarget := returns.getD i 0.0
        policyTarget := st.policyTarget
        done := i + 1 = envSteps.size
      }
  pure (.ok (samples, totals, key))

private def collectPPOEpisodeBatch
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (caps : ObservationCaps)
    (net : AlphaGradNet vertexDim tokenDim)
    (cfg : AlphaGradTrainerConfig)
    (batchSize : Nat)
    (seed : UInt64) :
    IO (Except String (Array ReplaySample × Array Float × UInt64)) := do
  if batchSize = 0 then
    return .error "PPO batch collector requires batchSize > 0."
  let s0 ←
    match initAlphaGradState? task.graph task.numVertices with
    | .error msg => return .error msg
    | .ok s0 => pure s0
  let maxSteps := task.envCfg.maxEpisodeSteps.getD task.numEliminableVertices
  let mut states := Array.replicate batchSize s0
  let mut active : Array Nat := (List.range batchSize).toArray
  let mut pending : Array (Array PendingPPOStep) := Array.replicate batchSize #[]
  let mut totals : Array Float := Array.replicate batchSize 0.0
  let mut key := seed
  let mut step := 0
  while !active.isEmpty && step < maxSteps do
    let activeStates := active.map (fun idx => states.getD idx s0)
    let evalOut ←
      match (← evalStatesPolicyValueTensor? net task caps activeStates) with
      | .error msg => return .error msg
      | .ok out => pure out
    let logitsBatch := evalOut.actionLogits
    let valuesBatch := evalOut.values
    let mut actions : Array ActionId0 := #[]
    let mut logProbs : Array Float := #[]
    let mut keyAfter := key
    for idx in [:active.size] do
      let logits := logitsBatch.getD idx #[]
      let probs := softmax logits
      let (action, nextKey) ←
        match sampleCategorical? probs (keyAfter + UInt64.ofNat (idx + 1)) with
        | .error msg => return .error msg
        | .ok out => pure out
      actions := actions.push action
      logProbs := logProbs.push (logSafe (probs.getD action 0.0))
      keyAfter := nextKey
    let transitions := (List.range active.size).toArray.map fun idx =>
      transition task.envCfg (activeStates.getD idx s0) (actions.getD idx 0)
    let aliveLocals := (List.range transitions.size).toArray.filter fun i =>
      !(transitions.getD i default).done
    let aliveStates := aliveLocals.map fun i => (transitions.getD i default).nextState
    let nextValuesAlive ←
      if aliveStates.isEmpty then
        pure #[]
      else
        match (← evalStatesPolicyValueTensor? net task caps aliveStates) with
        | .error msg => return .error msg
        | .ok out => pure out.values
    let mut nextValueMap : Std.HashMap Nat Float := {}
    for i in [:aliveLocals.size] do
      nextValueMap := nextValueMap.insert (aliveLocals.getD i 0) (nextValuesAlive.getD i 0.0)
    let mut newActive : Array Nat := #[]
    for idx in [:active.size] do
      let global := active.getD idx 0
      let t := transitions.getD idx default
      let nextValue := if t.done then 0.0 else nextValueMap.getD idx 0.0
      let currentPending := pending.getD global #[]
      pending := pending.set! global (currentPending.push {
        features := evalOut.observations.getD idx #[]
        action := actions.getD idx 0
        reward := t.reward
        value := valuesBatch.getD idx 0.0
        nextValue := nextValue
        oldLogProb := logProbs.getD idx 0.0
        done := t.done
      })
      totals := totals.set! global (totals.getD global 0.0 + t.reward)
      states := states.set! global t.nextState
      if !t.done then
        newActive := newActive.push global
    active := newActive
    key := mix (keyAfter + UInt64.ofNat (step + 1))
    step := step + 1
  if !active.isEmpty then
    return .error s!"PPO batched rollout did not terminate within {maxSteps} steps."
  let mut samples : Array ReplaySample := #[]
  for envIdx in [:batchSize] do
    let envSteps := pending.getD envIdx #[]
    let (advantages, returns) := computeGAE envSteps cfg.gamma cfg.gaeLambda
    for i in [:envSteps.size] do
      let st := envSteps.getD i default
      samples := samples.push {
        kind := .ppo
        features := st.features
        action := st.action
        oldLogProb := st.oldLogProb
        reward := st.reward
        valueTarget := returns.getD i 0.0
        advantage := advantages.getD i 0.0
        done := st.done
      }
  pure (.ok (samples, totals, key))

private def ppoUpdateFromReplay
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (steps : Array ReplaySample)
    (cfg : AlphaGradTrainerConfig) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim × Optim.AdamWState (AlphaGradNet vertexDim tokenDim) × Float × Float × Float)) := do
  if steps.isEmpty then
    return .error "PPO update received an empty replay batch."
  let rows := steps.map (·.features)
  let actions : Array Int64 := steps.map (fun s => Int64.ofNat s.action)
  let oldLogProbs := steps.map (·.oldLogProb)
  let advantages := normalizeArray (steps.map (·.advantage))
  let returnsRaw := steps.map (·.valueTarget)
  match obsRowsToTensor3d? rows vertexDim tokenDim with
  | .error msg => return .error msg
  | .ok ⟨n, obs⟩ =>
    let attnMaskT ←
      match attentionMaskTensor? rows vertexDim tokenDim with
      | .error msg => return .error msg
      | .ok ⟨nMask, mask⟩ =>
        if nMask != n then
          return .error s!"Attention mask batch mismatch: expected {n}, got {nMask}."
        pure mask
    let actionDyn := data.fromInt64Array actions
    let actionT0 : T #[n, 1] := reshape actionDyn #[n, 1]
    let actionT : T #[n, 1] := data.toLong actionT0
    let actionIndexT ←
      match actionIndexTensor n task.graph.actionVertices with
      | .error msg => return .error msg
      | .ok ⟨actionDim, idxs⟩ =>
        if actionDim.toNat != task.numActions then
          return .error s!"Action-surface width mismatch: expected {task.numActions}, got {actionDim}."
        pure idxs
    let oldLogT : T #[n, 1] := reshape (data.fromFloatArray oldLogProbs) #[n, 1]
    let advT : T #[n, 1] := reshape (data.fromFloatArray advantages) #[n, 1]
    let retT : T #[n, 1] := reshape (data.fromFloatArray returnsRaw) #[n, 1]
    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0
    let mut entropyV := 0.0
    for _ in [:cfg.ppoUpdateEpochs] do
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
        mul_scalar valueLoss cfg.ppoValueCoef -
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

private def alphaZeroUpdateFromReplay
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (samples : Array ReplaySample)
    (cfg : AlphaGradTrainerConfig) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim × Optim.AdamWState (AlphaGradNet vertexDim tokenDim) × Float × Float)) := do
  if samples.isEmpty then
    return .error "AlphaZero update received an empty replay batch."
  let obsRows := samples.map (·.features)
  let policyRows := samples.map (·.policyTarget)
  let valueTargets := samples.map (fun s => applyValueTransform cfg.valueTransform s.valueTarget)
  match obsRowsToTensor3d? obsRows vertexDim tokenDim with
  | .error msg => return .error msg
  | .ok ⟨n, obs⟩ =>
    let attnMaskT ←
      match attentionMaskTensor? obsRows vertexDim tokenDim with
      | .error msg => return .error msg
      | .ok ⟨nMask, mask⟩ =>
        if nMask != n then
          return .error s!"Attention mask batch mismatch: expected {n}, got {nMask}."
        pure mask
    let actionNat := task.numActions
    let mut flatPolicy : Array Float := #[]
    for i in [:policyRows.size] do
      let row := policyRows.getD i #[]
      if row.size != actionNat then
        return .error s!"AlphaZero policy row {i} width mismatch: expected {actionNat}, got {row.size}."
      for x in row do
        flatPolicy := flatPolicy.push x
    let policyT : T #[n, task.numActions.toUInt64] :=
      reshape (data.fromFloatArray flatPolicy) #[n, task.numActions.toUInt64]
    let actionIndexT ←
      match actionIndexTensor n task.graph.actionVertices with
      | .error msg => return .error msg
      | .ok ⟨actionDim, idxs⟩ =>
        if actionDim.toNat != task.numActions then
          return .error s!"Action-surface width mismatch: expected {task.numActions}, got {actionDim}."
        pure idxs
    let valueT : T #[n, 1] := reshape (data.fromFloatArray valueTargets) #[n, 1]
    let mut netCur := net
    let mut stateCur := optState
    let mut policyLossV := 0.0
    let mut valueLossV := 0.0
    for _ in [:cfg.alphaZeroUpdateEpochs] do
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

private def defaultRunDir (cfg : AlphaGradTrainerConfig) : String :=
  let explicit := cfg.runDir.trimAscii.toString
  if explicit.isEmpty then
    s!"runs/alphagrad/{toString cfg.mode}/{toString cfg.task}"
  else
    explicit

private def latestCheckpointDir (artifacts : RunArtifacts) : String :=
  (artifacts.checkpointsDir / "latest").toString

private def namedCheckpointDir (artifacts : RunArtifacts) (name : String) : String :=
  (artifacts.checkpointsDir / name).toString

private def trainerStatePath (dir : String) : System.FilePath :=
  ⟨s!"{dir}/trainer_state.json"⟩

private def replayStatePath (dir : String) : System.FilePath :=
  ⟨s!"{dir}/replay.json"⟩

private def writeJsonFile [ToJson α] (path : System.FilePath) (payload : α) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path (Lean.toJson payload).pretty

private def readJsonFile [FromJson α] (path : System.FilePath) : IO α := do
  let content ← IO.FS.readFile path
  match Json.parse content with
  | .error err =>
    throw <| IO.userError s!"JSON parse failed at {path}: {err}"
  | .ok json =>
    match (Lean.fromJson? json : Except String α) with
    | .error err =>
      throw <| IO.userError s!"JSON decode failed at {path}: {err}"
    | .ok value =>
      pure value

private def saveTrainerCheckpoint
    {vertexDim tokenDim : UInt64}
    (artifacts : RunArtifacts)
    (name : String)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (replay : ReplayBuffer)
    (snapshot : TrainerSnapshot) :
    IO Unit := do
  let dir := namedCheckpointDir artifacts name
  IO.FS.createDirAll ⟨dir⟩
  saveParams (TensorStruct.detach net) dir "model"
  saveOptimizerState optState.fst.mu optState.fst.nu optState.fst.count dir
  writeJsonFile (trainerStatePath dir) snapshot
  saveReplayBuffer (replayStatePath dir) replay
  appendCheckpointEvent artifacts {
    name := name
    path := dir
    kind := "alphagrad-trainer"
    step := some snapshot.completedEpochs
    metadata := [
      metricStr "mode" (toString snapshot.mode),
      metricStr "task" (toString snapshot.task),
      metricNat "globalEpisodes" snapshot.globalEpisodes,
      metricNat "globalSamples" snapshot.globalSamples
    ]
  }

private def loadTrainerCheckpoint?
    {vertexDim tokenDim : UInt64}
    (dir : String)
    (template : AlphaGradNet vertexDim tokenDim) :
    IO (Option
      (AlphaGradNet vertexDim tokenDim ×
       Optim.AdamWState (AlphaGradNet vertexDim tokenDim) ×
       ReplayBuffer ×
       TrainerSnapshot)) := do
  let stateExists ← System.FilePath.pathExists (trainerStatePath dir)
  if !stateExists then
    pure none
  else
    let net ← loadParams template dir "model"
    let (mu, nu, count) ← loadOptimizerState template dir
    let replay ← loadReplayBuffer (replayStatePath dir)
    let snapshot : TrainerSnapshot ← readJsonFile (trainerStatePath dir)
    let optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim) := {
      fst := { count := count, mu := mu, nu := nu }
      snd := { fst := {}, snd := {} }
    }
    pure (some (net, optState, replay, snapshot))

private def collectEpochReplay
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (caps : ObservationCaps)
    (net : AlphaGradNet vertexDim tokenDim)
    (cfg : AlphaGradTrainerConfig)
    (seed : UInt64) :
    IO (Except String (Array ReplaySample × Float × UInt64)) := do
  let mut remaining := cfg.episodesPerEpoch
  let mut key := seed
  let mut samples : Array ReplaySample := #[]
  let mut rewardTotal := 0.0
  let mut episodesDone := 0
  while remaining > 0 do
    let batch := Nat.min (Nat.max cfg.numEnvs 1) remaining
    let (batchSamples, batchRewards, key') ←
      match cfg.mode with
      | .ppo =>
        match (← collectPPOEpisodeBatch task caps net cfg batch key) with
        | .error msg => return .error msg
        | .ok out => pure out
      | .alphazero =>
        match (← collectAlphaZeroEpisodeBatch task caps net cfg batch key) with
        | .error msg => return .error msg
        | .ok out => pure out
    key := key'
    samples := samples ++ batchSamples
    rewardTotal := rewardTotal + batchRewards.foldl (init := 0.0) (· + ·)
    episodesDone := episodesDone + batch
    remaining := remaining - batch
  let avgReward := if episodesDone = 0 then 0.0 else rewardTotal / Float.ofNat episodesDone
  pure (.ok (samples, avgReward, key))

private def runTrainUpdates
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (replay : ReplayBuffer)
    (cfg : AlphaGradTrainerConfig)
    (seed : UInt64) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim ×
       Optim.AdamWState (AlphaGradNet vertexDim tokenDim) ×
       UInt64 × Float × Float × Float)) := do
  let pool := replay.filterByKind (match cfg.mode with | .ppo => .ppo | .alphazero => .alphazero)
  if pool.isEmpty then
    return .error "Replay buffer has no samples for the selected training mode."
  let mut netCur := net
  let mut optCur := optState
  let mut key := seed
  let mut policyLoss := 0.0
  let mut valueLoss := 0.0
  let mut entropy := 0.0
  let batchSize := Nat.min (Nat.max cfg.sampleBatchSize 1) pool.size
  for _ in [:cfg.updateBatchesPerEpoch] do
    let (batch, key') ←
      match replay.sampleBatch key batchSize (some (match cfg.mode with | .ppo => .ppo | .alphazero => .alphazero)) with
      | .error msg => return .error msg
      | .ok out => pure out
    key := key'
    match cfg.mode with
    | .ppo =>
      match (← ppoUpdateFromReplay task netCur optCur batch cfg) with
      | .error msg => return .error msg
      | .ok (netNext, optNext, p, v, e) =>
        netCur := netNext
        optCur := optNext
        policyLoss := p
        valueLoss := v
        entropy := e
    | .alphazero =>
      match (← alphaZeroUpdateFromReplay task netCur optCur batch cfg) with
      | .error msg => return .error msg
      | .ok (netNext, optNext, p, v) =>
        netCur := netNext
        optCur := optNext
        policyLoss := p
        valueLoss := v
        entropy := 0.0
  return .ok (netCur, optCur, key, policyLoss, valueLoss, entropy)

private def evaluateModel
    {vertexDim tokenDim : UInt64}
    (task : TaskSpec)
    (caps : ObservationCaps)
    (net : AlphaGradNet vertexDim tokenDim)
    (cfg : AlphaGradTrainerConfig)
    (seed : UInt64) :
    IO (Except String (Float × Float × UInt64)) := do
  let greedy ←
    match (← evalGreedyReward? task net caps) with
    | .error msg => return .error msg
    | .ok value => pure value
  let (_samples, rewards, key) ←
    match cfg.mode with
    | .ppo =>
      match (← collectPPOEpisodeBatch task caps net cfg 1 seed) with
      | .error msg => return .error msg
      | .ok out => pure out
    | .alphazero =>
      match (← collectAlphaZeroEpisodeBatch task caps net cfg 1 seed) with
      | .error msg => return .error msg
      | .ok out => pure out
  let searchReward := rewards.getD 0 greedy
  pure (.ok (greedy, searchReward, key))

private def writeTrainerReport
    (artifacts : RunArtifacts)
    (cfg : AlphaGradTrainerConfig)
    (snapshot : TrainerSnapshot)
    (replay : ReplayBuffer) :
    IO Unit := do
  let lines : List String := [
    "# AlphaGrad Trainer Report",
    "",
    s!"- mode: {toString cfg.mode}",
    s!"- task: {toString cfg.task}",
    s!"- run_dir: {artifacts.baseDir}",
    s!"- completed_epochs: {snapshot.completedEpochs}",
    s!"- global_episodes: {snapshot.globalEpisodes}",
    s!"- global_samples: {snapshot.globalSamples}",
    s!"- replay_size: {replay.size}",
    s!"- best_eval_reward: {snapshot.bestEvalReward}",
    s!"- best_search_reward: {snapshot.bestSearchReward}",
    s!"- last_train_reward: {snapshot.lastTrainReward}",
    s!"- last_policy_loss: {snapshot.lastPolicyLoss}",
    s!"- last_value_loss: {snapshot.lastValueLoss}",
    s!"- last_entropy: {snapshot.lastEntropy}"
  ]
  IO.FS.writeFile artifacts.reportPath (String.intercalate "\n" lines)

def trainWithConfig (cfg : AlphaGradTrainerConfig) : IO (Except String TrainerSnapshot) := do
  match (← materializeTask cfg.task) with
  | .error msg =>
    pure (.error s!"AlphaGrad trainer task materialization failed: {msg}")
  | .ok task =>
    let caps := taskObservationCaps task
    let vertexDim := capsVertexDim caps
    let tokenDim := capsTokenDim caps
    let runDir := defaultRunDir cfg
    let artifacts := RunArtifacts.ofBaseDir runDir
    let policy :=
      if cfg.resume then ExistingRunPolicy.resume
      else if cfg.overwrite then ExistingRunPolicy.overwrite
      else ExistingRunPolicy.failIfExists
    prepare artifacts cfg policy
    let net0 ← AlphaGradNet.init vertexDim tokenDim
    let net0 := TensorStruct.makeLeafParams net0
    let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
    let optState0 := opt.init net0
    let replay0 := ReplayBuffer.empty cfg.replayCapacity
    let snapshot0 : TrainerSnapshot := {
      mode := cfg.mode
      task := cfg.task
      nextSeed := cfg.seed
    }
    let latestDir := latestCheckpointDir artifacts
    let initialBundle ←
      if cfg.resume then
        match (← loadTrainerCheckpoint? latestDir net0) with
        | some loaded => pure loaded
        | none => pure (net0, optState0, replay0, snapshot0)
      else
        pure (net0, optState0, replay0, snapshot0)
    let mut net := initialBundle.1
    let mut optState := initialBundle.2.1
    let mut replay := initialBundle.2.2.1
    let mut snapshot := initialBundle.2.2.2
    IO.println s!"[AlphaGradTrainer] mode={toString cfg.mode} task={task.name} run_dir={runDir} epochs={cfg.epochs} episodes/epoch={cfg.episodesPerEpoch} num_envs={cfg.numEnvs} replay_capacity={cfg.replayCapacity}"
    let mut seed := UInt64.ofNat snapshot.nextSeed
    for epoch in [snapshot.completedEpochs:cfg.epochs] do
      let (collected, avgReward, seed') ←
        match (← collectEpochReplay task caps (TensorStruct.detach net) cfg seed) with
        | .error msg => return .error s!"AlphaGrad trainer collection failed at epoch {epoch + 1}: {msg}"
        | .ok out => pure out
      seed := seed'
      replay := replay.pushBatch collected
      snapshot := {
        snapshot with
        completedEpochs := epoch + 1
        globalEpisodes := snapshot.globalEpisodes + cfg.episodesPerEpoch
        globalSamples := snapshot.globalSamples + collected.size
        nextSeed := seed.toNat
        lastTrainReward := avgReward
      }
      match (← runTrainUpdates task net optState replay cfg seed) with
      | .error msg =>
        return .error s!"AlphaGrad trainer update failed at epoch {epoch + 1}: {msg}"
      | .ok (netNext, optNext, seed'', policyLoss, valueLoss, entropy) =>
        net := netNext
        optState := optNext
        seed := seed''
        snapshot := {
          snapshot with
          nextSeed := seed.toNat
          lastPolicyLoss := policyLoss
          lastValueLoss := valueLoss
          lastEntropy := entropy
        }
      if cfg.evalEvery > 0 && ((epoch + 1) % cfg.evalEvery = 0 || epoch + 1 = cfg.epochs) then
        match (← evaluateModel task caps (TensorStruct.detach net) cfg seed) with
        | .error msg =>
          return .error s!"AlphaGrad trainer evaluation failed at epoch {epoch + 1}: {msg}"
        | .ok (greedyReward, searchReward, seed') =>
          seed := seed'
          snapshot := {
            snapshot with
            nextSeed := seed.toNat
            bestEvalReward := max snapshot.bestEvalReward greedyReward
            bestSearchReward := max snapshot.bestSearchReward searchReward
          }
          appendMetricEvent artifacts {
            scope := "train"
            step := some (epoch + 1)
            metrics := [
              metricStr "mode" (toString cfg.mode),
              metricStr "task" task.name,
              metricFloat "avgEpisodeReward" avgReward,
              metricFloat "policyLoss" snapshot.lastPolicyLoss,
              metricFloat "valueLoss" snapshot.lastValueLoss,
              metricFloat "entropy" snapshot.lastEntropy,
              metricFloat "greedyReward" greedyReward,
              metricFloat "searchReward" searchReward,
              metricNat "replaySize" replay.size,
              metricNat "samplesCollected" collected.size
            ]
          }
          IO.println s!"[AlphaGradTrainer] epoch={epoch + 1}/{cfg.epochs} avg_reward={avgReward} greedy={greedyReward} search={searchReward} replay={replay.size} policy_loss={snapshot.lastPolicyLoss} value_loss={snapshot.lastValueLoss}"
          if greedyReward >= snapshot.bestEvalReward || searchReward >= snapshot.bestSearchReward then
            saveTrainerCheckpoint artifacts "best" net optState replay snapshot
      if cfg.checkpointEvery > 0 && ((epoch + 1) % cfg.checkpointEvery = 0 || epoch + 1 = cfg.epochs) then
        saveTrainerCheckpoint artifacts s!"epoch_{epoch + 1}" net optState replay snapshot
        saveTrainerCheckpoint artifacts "latest" net optState replay snapshot
      writeTrainerReport artifacts cfg snapshot replay
    pure (.ok snapshot)

def evalWithConfig
    (cfg : AlphaGradTrainerConfig)
    (checkpointDir? : Option String := none)
    (seedOverride? : Option Nat := none) :
    IO (Except String (Float × Float)) := do
  match (← materializeTask cfg.task) with
  | .error msg =>
    pure (.error s!"AlphaGrad eval task materialization failed: {msg}")
  | .ok task =>
    let caps := taskObservationCaps task
    let vertexDim := capsVertexDim caps
    let tokenDim := capsTokenDim caps
    let net0 ← AlphaGradNet.init vertexDim tokenDim
    let runDir := defaultRunDir cfg
    let artifacts := RunArtifacts.ofBaseDir runDir
    let ckptDir := checkpointDir?.getD (latestCheckpointDir artifacts)
    match (← loadTrainerCheckpoint? ckptDir net0) with
    | none =>
      pure (.error s!"AlphaGrad trainer checkpoint not found at {ckptDir}")
    | some (net, _optState, replay, snapshot) =>
      let evalSeed := UInt64.ofNat (seedOverride?.getD snapshot.nextSeed)
      match (← evaluateModel task caps (TensorStruct.detach net) cfg evalSeed) with
      | .error msg =>
        pure (.error s!"AlphaGrad trainer eval failed: {msg}")
      | .ok (greedyReward, searchReward, _seed) =>
        appendMetricEvent artifacts {
          scope := "eval"
          step := some snapshot.completedEpochs
          metrics := [
            metricStr "mode" (toString cfg.mode),
            metricStr "task" task.name,
            metricFloat "greedyReward" greedyReward,
            metricFloat "searchReward" searchReward,
            metricNat "replaySize" replay.size
          ]
        }
        IO.println s!"[AlphaGradTrainer][eval] checkpoint={ckptDir} greedy={greedyReward} search={searchReward} replay={replay.size}"
        pure (.ok (greedyReward, searchReward))

inductive CurriculumMode where
  | roundRobin
  | progressive
  | random
  deriving Repr, Inhabited, BEq

instance : ToJson CurriculumMode where
  toJson
    | .roundRobin => Json.str "round-robin"
    | .progressive => Json.str "progressive"
    | .random => Json.str "random"

instance : FromJson CurriculumMode where
  fromJson?
    | .str "round-robin" => pure .roundRobin
    | .str "roundrobin" => pure .roundRobin
    | .str "progressive" => pure .progressive
    | .str "random" => pure .random
    | .str s => throw s!"Unknown curriculum mode '{s}'."
    | _ => throw "Expected curriculum mode as JSON string."

structure MultiTaskTrainerConfig where
  mode : TrainMode := .alphazero
  tasks : Array TaskName := taskSequence
  evalTasks : Array TaskName := #[]
  curriculum : CurriculumMode := .roundRobin
  curriculumWindow : Nat := 4
  epochs : Nat := 64
  episodesPerEpoch : Nat := 16
  numEnvs : Nat := 8
  replayCapacityPerTask : Nat := 4096
  sampleBatchSize : Nat := 256
  updateBatchesPerEpoch : Nat := 4
  ppoUpdateEpochs : Nat := 4
  alphaZeroUpdateEpochs : Nat := 8
  gamma : Float := 1.0
  gaeLambda : Float := 0.95
  clipEps : Float := 0.2
  ppoValueCoef : Float := 0.5
  entropyCoef : Float := 0.01
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
  evalEvery : Nat := 1
  checkpointEvery : Nat := 1
  runDir : String := ""
  seed : Nat := 250197
  resume : Bool := false
  overwrite : Bool := false
  deriving Repr, Inhabited, ToJson, FromJson

structure MultiTaskTrainerSnapshot where
  mode : TrainMode
  tasks : Array TaskName
  evalTasks : Array TaskName
  curriculum : CurriculumMode
  completedEpochs : Nat := 0
  globalEpisodes : Nat := 0
  globalSamples : Nat := 0
  nextSeed : Nat := 0
  lastTask : TaskName := .perceptron
  bestMeanEvalReward : Float := -1.0e30
  bestMeanSearchReward : Float := -1.0e30
  lastMeanEvalReward : Float := 0.0
  lastMeanSearchReward : Float := 0.0
  lastTrainReward : Float := 0.0
  lastPolicyLoss : Float := 0.0
  lastValueLoss : Float := 0.0
  lastEntropy : Float := 0.0
  deriving Repr, Inhabited, ToJson, FromJson

private def multiTaskBaseConfig
    (cfg : MultiTaskTrainerConfig)
    (task : TaskName) :
    AlphaGradTrainerConfig := {
  mode := cfg.mode
  task := task
  epochs := cfg.epochs
  episodesPerEpoch := cfg.episodesPerEpoch
  numEnvs := cfg.numEnvs
  replayCapacity := cfg.replayCapacityPerTask
  sampleBatchSize := cfg.sampleBatchSize
  updateBatchesPerEpoch := cfg.updateBatchesPerEpoch
  ppoUpdateEpochs := cfg.ppoUpdateEpochs
  alphaZeroUpdateEpochs := cfg.alphaZeroUpdateEpochs
  gamma := cfg.gamma
  gaeLambda := cfg.gaeLambda
  clipEps := cfg.clipEps
  ppoValueCoef := cfg.ppoValueCoef
  entropyCoef := cfg.entropyCoef
  valueWeight := cfg.valueWeight
  learningRate := cfg.learningRate
  weightDecay := cfg.weightDecay
  numSimulations := cfg.numSimulations
  maxDepth := cfg.maxDepth
  valueTransform := cfg.valueTransform
  qtransformValueScale := cfg.qtransformValueScale
  qtransformMaxvisitInit := cfg.qtransformMaxvisitInit
  qtransformRescaleValues := cfg.qtransformRescaleValues
  qtransformUseMixedValue := cfg.qtransformUseMixedValue
  evalEvery := cfg.evalEvery
  checkpointEvery := cfg.checkpointEvery
  runDir := cfg.runDir
  seed := cfg.seed
  resume := cfg.resume
  overwrite := cfg.overwrite
}

private def defaultMultiRunDir (cfg : MultiTaskTrainerConfig) : String :=
  let explicit := cfg.runDir.trimAscii.toString
  if explicit.isEmpty then
    s!"runs/alphagrad/multi/{toString cfg.mode}"
  else
    explicit

private def multiTrainerStatePath (dir : String) : System.FilePath :=
  ⟨s!"{dir}/multitask_trainer_state.json"⟩

private def multiReplayStatePath (dir : String) (task : TaskName) : System.FilePath :=
  ⟨s!"{dir}/replay_{toString task}.json"⟩

private def materializeTaskPairs
    (names : Array TaskName) :
    IO (Except String (Array (TaskName × TaskSpec))) := do
  let mut out : Array (TaskName × TaskSpec) := #[]
  for name in names do
    match (← materializeTask name) with
    | .error msg => return .error s!"AlphaGrad task materialization failed for {name}: {msg}"
    | .ok task => out := out.push (name, task)
  pure (.ok out)

private def eligibleTaskIndices
    (cfg : MultiTaskTrainerConfig)
    (numTasks epoch : Nat) :
    Array Nat :=
  if numTasks = 0 then
    #[]
  else
    match cfg.curriculum with
    | .roundRobin
    | .random => (List.range numTasks).toArray
    | .progressive =>
      let window := Nat.max cfg.curriculumWindow 1
      let count := min numTasks (epoch / window + 1)
      (List.range count).toArray

private def chooseTaskIndex
    (cfg : MultiTaskTrainerConfig)
    (eligible : Array Nat)
    (epochHint : Nat)
    (seed : UInt64) :
    Nat × UInt64 :=
  if eligible.isEmpty then
    (0, seed)
  else
    match cfg.curriculum with
    | .random =>
      let seed' := mix seed
      let idx := (seed'.toNat % eligible.size)
      (eligible.getD idx 0, seed')
    | _ =>
      let idx := epochHint % eligible.size
      (eligible.getD idx 0, seed)

private def meanArrayOrZero (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size

private def replayAt
    (buffers : Array ReplayBuffer)
    (idx capacity : Nat) :
    ReplayBuffer :=
  buffers.getD idx (ReplayBuffer.empty capacity)

private def saveMultiTrainerCheckpoint
    {vertexDim tokenDim : UInt64}
    (artifacts : RunArtifacts)
    (name : String)
    (taskNames : Array TaskName)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (replays : Array ReplayBuffer)
    (snapshot : MultiTaskTrainerSnapshot) :
    IO Unit := do
  let dir := namedCheckpointDir artifacts name
  IO.FS.createDirAll ⟨dir⟩
  saveParams (TensorStruct.detach net) dir "model"
  saveOptimizerState optState.fst.mu optState.fst.nu optState.fst.count dir
  writeJsonFile (multiTrainerStatePath dir) snapshot
  for i in [:taskNames.size] do
    let task := taskNames.getD i .perceptron
    let replay := replays.getD i (ReplayBuffer.empty 1)
    saveReplayBuffer (multiReplayStatePath dir task) replay
  appendCheckpointEvent artifacts {
    name := name
    path := dir
    kind := "alphagrad-multitask-trainer"
    step := some snapshot.completedEpochs
    metadata := [
      metricStr "mode" (toString snapshot.mode),
      metricNat "globalEpisodes" snapshot.globalEpisodes,
      metricNat "globalSamples" snapshot.globalSamples
    ]
  }

private def loadMultiTrainerCheckpoint?
    {vertexDim tokenDim : UInt64}
    (dir : String)
    (taskNames : Array TaskName)
    (capacity : Nat)
    (template : AlphaGradNet vertexDim tokenDim) :
    IO (Option
      (AlphaGradNet vertexDim tokenDim ×
       Optim.AdamWState (AlphaGradNet vertexDim tokenDim) ×
       Array ReplayBuffer ×
       MultiTaskTrainerSnapshot)) := do
  let stateExists ← System.FilePath.pathExists (multiTrainerStatePath dir)
  if !stateExists then
    pure none
  else
    let net ← loadParams template dir "model"
    let (mu, nu, count) ← loadOptimizerState template dir
    let mut replays : Array ReplayBuffer := #[]
    for task in taskNames do
      let path := multiReplayStatePath dir task
      let hasReplay ← System.FilePath.pathExists path
      let replay ←
        if hasReplay then
          loadReplayBuffer path
        else
          pure (ReplayBuffer.empty capacity)
      replays := replays.push replay
    let snapshot : MultiTaskTrainerSnapshot ← readJsonFile (multiTrainerStatePath dir)
    let optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim) := {
      fst := { count := count, mu := mu, nu := nu }
      snd := { fst := {}, snd := {} }
    }
    pure (some (net, optState, replays, snapshot))

private def runMultiTaskUpdates
    {vertexDim tokenDim : UInt64}
    (taskPairs : Array (TaskName × TaskSpec))
    (eligible : Array Nat)
    (net : AlphaGradNet vertexDim tokenDim)
    (optState : Optim.AdamWState (AlphaGradNet vertexDim tokenDim))
    (replays : Array ReplayBuffer)
    (cfg : MultiTaskTrainerConfig)
    (epoch : Nat)
    (seed : UInt64) :
    IO (Except String
      (AlphaGradNet vertexDim tokenDim ×
       Optim.AdamWState (AlphaGradNet vertexDim tokenDim) ×
       UInt64 × Float × Float × Float)) := do
  let modeKind := match cfg.mode with | .ppo => ReplayKind.ppo | .alphazero => .alphazero
  let candidates := eligible.filter fun idx =>
    let replay := replayAt replays idx cfg.replayCapacityPerTask
    !(replay.filterByKind modeKind).isEmpty
  if candidates.isEmpty then
    return .error "Multi-task replay banks have no samples for the selected mode."
  let mut netCur := net
  let mut optCur := optState
  let mut key := seed
  let mut policyLoss := 0.0
  let mut valueLoss := 0.0
  let mut entropy := 0.0
  for updateIdx in [:cfg.updateBatchesPerEpoch] do
    let (taskIdx, key') := chooseTaskIndex cfg candidates (epoch + updateIdx) key
    key := key'
    let taskPair := taskPairs.getD taskIdx (.perceptron, default)
    let taskName := taskPair.1
    let task := taskPair.2
    let replay := replayAt replays taskIdx cfg.replayCapacityPerTask
    let batchSize := Nat.min (Nat.max cfg.sampleBatchSize 1) (replay.filterByKind modeKind).size
    let (batch, key'') ←
      match replay.sampleBatch key batchSize (some modeKind) with
      | .error msg => return .error msg
      | .ok out => pure out
    key := key''
    let baseCfg := multiTaskBaseConfig cfg taskName
    match cfg.mode with
    | .ppo =>
      match (← ppoUpdateFromReplay task netCur optCur batch baseCfg) with
      | .error msg => return .error msg
      | .ok (netNext, optNext, p, v, e) =>
        netCur := netNext
        optCur := optNext
        policyLoss := p
        valueLoss := v
        entropy := e
    | .alphazero =>
      match (← alphaZeroUpdateFromReplay task netCur optCur batch baseCfg) with
      | .error msg => return .error msg
      | .ok (netNext, optNext, p, v) =>
        netCur := netNext
        optCur := optNext
        policyLoss := p
        valueLoss := v
        entropy := 0.0
  pure (.ok (netCur, optCur, key, policyLoss, valueLoss, entropy))

private def evaluateMultiTaskModel
    {vertexDim tokenDim : UInt64}
    (taskPairs : Array (TaskName × TaskSpec))
    (caps : ObservationCaps)
    (net : AlphaGradNet vertexDim tokenDim)
    (cfg : MultiTaskTrainerConfig)
    (seed : UInt64) :
    IO (Except String (Array (TaskName × Float × Float) × UInt64)) := do
  let evalNames := if cfg.evalTasks.isEmpty then cfg.tasks else cfg.evalTasks
  let mut results : Array (TaskName × Float × Float) := #[]
  let mut key := seed
  for name in evalNames do
    let pair? := taskPairs.find? (fun pair => pair.1 == name)
    let taskPair ←
      match pair? with
      | some pair => pure pair
      | none =>
        match (← materializeTask name) with
        | .error msg => return .error msg
        | .ok task => pure (name, task)
    let baseCfg := multiTaskBaseConfig cfg name
    match (← evaluateModel taskPair.2 caps (TensorStruct.detach net) baseCfg key) with
    | .error msg => return .error msg
    | .ok (greedyReward, searchReward, key') =>
      key := key'
      results := results.push (name, greedyReward, searchReward)
  pure (.ok (results, key))

def multiTrainWithConfig (cfg : MultiTaskTrainerConfig) : IO (Except String MultiTaskTrainerSnapshot) := do
  if cfg.tasks.isEmpty then
    return .error "Multi-task trainer requires at least one task."
  let taskPairsRes ← materializeTaskPairs cfg.tasks
  match taskPairsRes with
  | .error msg => pure (.error msg)
  | .ok taskPairs =>
    let caps := taskSetObservationCaps (taskPairs.map fun pair => pair.2)
    let vertexDim := capsVertexDim caps
    let tokenDim := capsTokenDim caps
    let runDir := defaultMultiRunDir cfg
    let artifacts := RunArtifacts.ofBaseDir runDir
    let policy :=
      if cfg.resume then ExistingRunPolicy.resume
      else if cfg.overwrite then ExistingRunPolicy.overwrite
      else ExistingRunPolicy.failIfExists
    prepare artifacts cfg policy
    let net0 ← AlphaGradNet.init vertexDim tokenDim
    let net0 := TensorStruct.makeLeafParams net0
    let opt := Optim.adamw (lr := cfg.learningRate) (weight_decay := cfg.weightDecay)
    let optState0 := opt.init net0
    let replays0 := Array.replicate taskPairs.size (ReplayBuffer.empty cfg.replayCapacityPerTask)
    let snapshot0 : MultiTaskTrainerSnapshot := {
      mode := cfg.mode
      tasks := cfg.tasks
      evalTasks := if cfg.evalTasks.isEmpty then cfg.tasks else cfg.evalTasks
      curriculum := cfg.curriculum
      nextSeed := cfg.seed
      lastTask := cfg.tasks.getD 0 .perceptron
    }
    let latestDir := latestCheckpointDir artifacts
    let initialBundle ←
      if cfg.resume then do
        match (← loadMultiTrainerCheckpoint? latestDir cfg.tasks cfg.replayCapacityPerTask net0) with
        | some loaded => pure loaded
        | none => pure (net0, optState0, replays0, snapshot0)
      else
        pure (net0, optState0, replays0, snapshot0)
    let mut net := initialBundle.1
    let mut optState := initialBundle.2.1
    let mut replays := initialBundle.2.2.1
    let mut snapshot := initialBundle.2.2.2
    let mut seed := UInt64.ofNat snapshot.nextSeed
    let taskLabels := String.intercalate "," (cfg.tasks.toList.map toString)
    IO.println s!"[AlphaGradMulti] mode={toString cfg.mode} tasks={taskLabels} run_dir={runDir} epochs={cfg.epochs}"
    for epoch in [snapshot.completedEpochs:cfg.epochs] do
      let eligible := eligibleTaskIndices cfg taskPairs.size epoch
      let (activeIdx, seedAfterPick) := chooseTaskIndex cfg eligible epoch seed
      seed := seedAfterPick
      let activePair := taskPairs.getD activeIdx (.perceptron, default)
      let activeName := activePair.1
      let activeTask := activePair.2
      let baseCfg := multiTaskBaseConfig cfg activeName
      let collectRes ← collectEpochReplay activeTask caps (TensorStruct.detach net) baseCfg seed
      let (collected, avgReward, seedAfterCollect) ←
        match collectRes with
        | .error msg => return .error s!"Multi-task collection failed at epoch {epoch + 1}: {msg}"
        | .ok out => pure out
      seed := seedAfterCollect
      let replay := replayAt replays activeIdx cfg.replayCapacityPerTask
      replays := replays.set! activeIdx (replay.pushBatch collected)
      snapshot := {
        snapshot with
        completedEpochs := epoch + 1
        globalEpisodes := snapshot.globalEpisodes + cfg.episodesPerEpoch
        globalSamples := snapshot.globalSamples + collected.size
        nextSeed := seed.toNat
        lastTask := activeName
        lastTrainReward := avgReward
      }
      let updateRes ← runMultiTaskUpdates taskPairs eligible net optState replays cfg epoch seed
      let (netNext, optNext, seedAfterUpdate, policyLoss, valueLoss, entropy) ←
        match updateRes with
        | .error msg => return .error s!"Multi-task update failed at epoch {epoch + 1}: {msg}"
        | .ok out => pure out
      net := netNext
      optState := optNext
      seed := seedAfterUpdate
      snapshot := {
        snapshot with
        nextSeed := seed.toNat
        lastPolicyLoss := policyLoss
        lastValueLoss := valueLoss
        lastEntropy := entropy
      }
      let shouldEval := cfg.evalEvery > 0 && ((epoch + 1) % cfg.evalEvery = 0 || epoch + 1 = cfg.epochs)
      if shouldEval then do
        let evalRes ← evaluateMultiTaskModel taskPairs caps (TensorStruct.detach net) cfg seed
        match evalRes with
        | .error msg => return .error s!"Multi-task evaluation failed at epoch {epoch + 1}: {msg}"
        | .ok (results, seedEval) =>
          let meanGreedy := meanArrayOrZero (results.map fun triple => triple.2.1)
          let meanSearch := meanArrayOrZero (results.map fun triple => triple.2.2)
          seed := seedEval
          snapshot := {
            snapshot with
            nextSeed := seed.toNat
            lastMeanEvalReward := meanGreedy
            lastMeanSearchReward := meanSearch
            bestMeanEvalReward := max snapshot.bestMeanEvalReward meanGreedy
            bestMeanSearchReward := max snapshot.bestMeanSearchReward meanSearch
          }
          appendMetricEvent artifacts {
            scope := "multitask-train"
            step := some (epoch + 1)
            metrics := [
              metricStr "mode" (toString cfg.mode),
              metricStr "activeTask" (toString activeName),
              metricFloat "avgEpisodeReward" avgReward,
              metricFloat "policyLoss" snapshot.lastPolicyLoss,
              metricFloat "valueLoss" snapshot.lastValueLoss,
              metricFloat "entropy" snapshot.lastEntropy,
              metricFloat "meanGreedyReward" meanGreedy,
              metricFloat "meanSearchReward" meanSearch
            ]
          }
          IO.println s!"[AlphaGradMulti] epoch={epoch + 1}/{cfg.epochs} active={activeName} avg_reward={avgReward} mean_greedy={meanGreedy} mean_search={meanSearch}"
          if meanGreedy >= snapshot.bestMeanEvalReward || meanSearch >= snapshot.bestMeanSearchReward then
            saveMultiTrainerCheckpoint artifacts "best" cfg.tasks net optState replays snapshot
      let shouldCheckpoint := cfg.checkpointEvery > 0 && ((epoch + 1) % cfg.checkpointEvery = 0 || epoch + 1 = cfg.epochs)
      if shouldCheckpoint then do
        saveMultiTrainerCheckpoint artifacts s!"epoch_{epoch + 1}" cfg.tasks net optState replays snapshot
        saveMultiTrainerCheckpoint artifacts "latest" cfg.tasks net optState replays snapshot
    pure (.ok snapshot)

def multiEvalWithConfig
    (cfg : MultiTaskTrainerConfig)
    (checkpointDir? : Option String := none)
    (seedOverride? : Option Nat := none) :
    IO (Except String (Array (TaskName × Float × Float))) := do
  match (← materializeTaskPairs cfg.tasks) with
  | .error msg => pure (.error msg)
  | .ok taskPairs =>
    let caps := taskSetObservationCaps (taskPairs.map fun pair => pair.2)
    let vertexDim := capsVertexDim caps
    let tokenDim := capsTokenDim caps
    let net0 ← AlphaGradNet.init vertexDim tokenDim
    let runDir := defaultMultiRunDir cfg
    let artifacts := RunArtifacts.ofBaseDir runDir
    let ckptDir := checkpointDir?.getD (latestCheckpointDir artifacts)
    match (← loadMultiTrainerCheckpoint? ckptDir cfg.tasks cfg.replayCapacityPerTask net0) with
    | none =>
      pure (.error s!"AlphaGrad multi-task checkpoint not found at {ckptDir}")
    | some (net, _optState, _replays, snapshot) =>
      let evalSeed := UInt64.ofNat (seedOverride?.getD snapshot.nextSeed)
      match (← evaluateMultiTaskModel taskPairs caps (TensorStruct.detach net) cfg evalSeed) with
      | .error msg => pure (.error msg)
      | .ok (results, _seed) =>
        let metrics : List (String × Json) :=
          results.foldl (init := ([] : List (String × Json))) fun acc (name, greedyReward, searchReward) =>
            acc ++ [metricFloat s!"{toString name}.greedy" greedyReward, metricFloat s!"{toString name}.search" searchReward]
        appendMetricEvent artifacts {
          scope := "multitask-eval"
          step := some snapshot.completedEpochs
          metrics := metrics
        }
        pure (.ok results)

structure TrainerArgs where
  command : String := "train"
  mode : TrainMode := .alphazero
  task : TaskName := .perceptron
  epochs : Option Nat := none
  episodesPerEpoch : Option Nat := none
  numEnvs : Option Nat := none
  replayCapacity : Option Nat := none
  sampleBatchSize : Option Nat := none
  updateBatchesPerEpoch : Option Nat := none
  checkpointEvery : Option Nat := none
  evalEvery : Option Nat := none
  numSimulations : Option Nat := none
  seed : Option Nat := none
  runDir : Option String := none
  checkpointDir : Option String := none
  resume : Bool := false
  overwrite : Bool := false
  deriving Repr, Inhabited

private def parseMode? (s : String) : Option TrainMode :=
  match s.trimAscii.toString.toLower with
  | "ppo" => some .ppo
  | "az" => some .alphazero
  | "alphazero" => some .alphazero
  | _ => none

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def parseTrainerFlags
    (args : List String)
    (st : TrainerArgs) :
    Except String TrainerArgs :=
  match args with
  | [] => pure st
  | "--epochs" :: v :: rest =>
    parseTrainerFlags rest { st with epochs := parseNatArg? v }
  | "--episodes-per-epoch" :: v :: rest =>
    parseTrainerFlags rest { st with episodesPerEpoch := parseNatArg? v }
  | "--num-envs" :: v :: rest =>
    parseTrainerFlags rest { st with numEnvs := parseNatArg? v }
  | "--replay-capacity" :: v :: rest =>
    parseTrainerFlags rest { st with replayCapacity := parseNatArg? v }
  | "--batch-size" :: v :: rest =>
    parseTrainerFlags rest { st with sampleBatchSize := parseNatArg? v }
  | "--update-batches" :: v :: rest =>
    parseTrainerFlags rest { st with updateBatchesPerEpoch := parseNatArg? v }
  | "--checkpoint-every" :: v :: rest =>
    parseTrainerFlags rest { st with checkpointEvery := parseNatArg? v }
  | "--eval-every" :: v :: rest =>
    parseTrainerFlags rest { st with evalEvery := parseNatArg? v }
  | "--num-simulations" :: v :: rest =>
    parseTrainerFlags rest { st with numSimulations := parseNatArg? v }
  | "--seed" :: v :: rest =>
    parseTrainerFlags rest { st with seed := parseNatArg? v }
  | "--run-dir" :: v :: rest =>
    parseTrainerFlags rest { st with runDir := some v }
  | "--checkpoint" :: v :: rest =>
    parseTrainerFlags rest { st with checkpointDir := some v }
  | "--resume" :: rest =>
    parseTrainerFlags rest { st with resume := true }
  | "--overwrite" :: rest =>
    parseTrainerFlags rest { st with overwrite := true }
  | flag :: _ =>
    throw s!"Unknown AlphaGrad trainer flag '{flag}'."

private def usage : String :=
  String.intercalate "\n" ([
    "Usage:",
    "  lake exe AlphaGradTrainer train <mode> <task> [flags]",
    "  lake exe AlphaGradTrainer eval <mode> <task> [flags]",
    "  lake exe AlphaGradTrainer multitrain <mode> <task1,task2,...|all> [flags]",
    "  lake exe AlphaGradTrainer multieval <mode> <task1,task2,...|all> [flags]",
    "Flags:",
    "  --epochs <n>",
    "  --episodes-per-epoch <n>",
    "  --num-envs <n>",
    "  --replay-capacity <n>",
    "  --batch-size <n>",
    "  --update-batches <n>",
    "  --checkpoint-every <n>",
    "  --eval-every <n>",
    "  --num-simulations <n>",
    "  --seed <n>",
    "  --run-dir <dir>",
    "  --checkpoint <dir>",
    "  --eval-tasks <task1,task2,...|all>",
    "  --curriculum <round-robin|progressive|random>",
    "  --curriculum-window <n>",
    "  --resume",
    "  --overwrite"
  ] : List String)

structure MultiTrainerArgs where
  command : String := "multitrain"
  mode : TrainMode := .alphazero
  tasks : Array TaskName := taskSequence
  evalTasks : Option (Array TaskName) := none
  curriculum : Option CurriculumMode := none
  curriculumWindow : Option Nat := none
  epochs : Option Nat := none
  episodesPerEpoch : Option Nat := none
  numEnvs : Option Nat := none
  replayCapacityPerTask : Option Nat := none
  sampleBatchSize : Option Nat := none
  updateBatchesPerEpoch : Option Nat := none
  checkpointEvery : Option Nat := none
  evalEvery : Option Nat := none
  numSimulations : Option Nat := none
  seed : Option Nat := none
  runDir : Option String := none
  checkpointDir : Option String := none
  resume : Bool := false
  overwrite : Bool := false
  deriving Repr, Inhabited

private def parseTaskListArg?
    (s : String) :
    Except String (Array TaskName) := do
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then
    throw "Expected non-empty task list."
  if trimmed = "all" then
    pure taskSequence
  else
    let parts := trimmed.splitOn ","
    let mut out : Array TaskName := #[]
    for raw in parts do
      match parseTaskName? raw with
      | some task => out := out.push task
      | none => throw s!"Unknown AlphaGrad task '{raw}'."
    if out.isEmpty then
      throw "Expected at least one AlphaGrad task."
    pure out

private def parseCurriculumMode?
    (s : String) :
    Option CurriculumMode :=
  match s.trimAscii.toString.toLower with
  | "round-robin"
  | "roundrobin" => some .roundRobin
  | "progressive" => some .progressive
  | "random" => some .random
  | _ => none

private def parseMultiTrainerFlags
    (args : List String)
    (st : MultiTrainerArgs) :
    Except String MultiTrainerArgs :=
  match args with
  | [] => pure st
  | "--epochs" :: v :: rest =>
    parseMultiTrainerFlags rest { st with epochs := parseNatArg? v }
  | "--episodes-per-epoch" :: v :: rest =>
    parseMultiTrainerFlags rest { st with episodesPerEpoch := parseNatArg? v }
  | "--num-envs" :: v :: rest =>
    parseMultiTrainerFlags rest { st with numEnvs := parseNatArg? v }
  | "--replay-capacity" :: v :: rest =>
    parseMultiTrainerFlags rest { st with replayCapacityPerTask := parseNatArg? v }
  | "--batch-size" :: v :: rest =>
    parseMultiTrainerFlags rest { st with sampleBatchSize := parseNatArg? v }
  | "--update-batches" :: v :: rest =>
    parseMultiTrainerFlags rest { st with updateBatchesPerEpoch := parseNatArg? v }
  | "--checkpoint-every" :: v :: rest =>
    parseMultiTrainerFlags rest { st with checkpointEvery := parseNatArg? v }
  | "--eval-every" :: v :: rest =>
    parseMultiTrainerFlags rest { st with evalEvery := parseNatArg? v }
  | "--num-simulations" :: v :: rest =>
    parseMultiTrainerFlags rest { st with numSimulations := parseNatArg? v }
  | "--seed" :: v :: rest =>
    parseMultiTrainerFlags rest { st with seed := parseNatArg? v }
  | "--run-dir" :: v :: rest =>
    parseMultiTrainerFlags rest { st with runDir := some v }
  | "--checkpoint" :: v :: rest =>
    parseMultiTrainerFlags rest { st with checkpointDir := some v }
  | "--eval-tasks" :: v :: rest =>
    match parseTaskListArg? v with
    | .error msg => throw msg
    | .ok tasks => parseMultiTrainerFlags rest { st with evalTasks := some tasks }
  | "--curriculum" :: v :: rest =>
    match parseCurriculumMode? v with
    | some mode => parseMultiTrainerFlags rest { st with curriculum := some mode }
    | none => throw s!"Unknown curriculum mode '{v}'."
  | "--curriculum-window" :: v :: rest =>
    parseMultiTrainerFlags rest { st with curriculumWindow := parseNatArg? v }
  | "--resume" :: rest =>
    parseMultiTrainerFlags rest { st with resume := true }
  | "--overwrite" :: rest =>
    parseMultiTrainerFlags rest { st with overwrite := true }
  | flag :: _ =>
    throw s!"Unknown AlphaGrad multi-trainer flag '{flag}'."

private def mkMultiTrainerConfig (args : MultiTrainerArgs) : MultiTaskTrainerConfig := {
  mode := args.mode
  tasks := args.tasks
  evalTasks := args.evalTasks.getD #[]
  curriculum := args.curriculum.getD .roundRobin
  curriculumWindow := args.curriculumWindow.getD 4
  epochs := args.epochs.getD 64
  episodesPerEpoch := args.episodesPerEpoch.getD 16
  numEnvs := args.numEnvs.getD 8
  replayCapacityPerTask := args.replayCapacityPerTask.getD 4096
  sampleBatchSize := args.sampleBatchSize.getD 256
  updateBatchesPerEpoch := args.updateBatchesPerEpoch.getD 4
  checkpointEvery := args.checkpointEvery.getD 1
  evalEvery := args.evalEvery.getD 1
  numSimulations := args.numSimulations.getD 48
  runDir := args.runDir.getD ""
  seed := args.seed.getD 250197
  resume := args.resume
  overwrite := args.overwrite
}

def multiTrainerMain (args : List String) : IO UInt32 := do
  match args with
  | command :: modeStr :: tasksStr :: rest =>
    match parseMode? modeStr, parseTaskListArg? tasksStr with
    | some mode, .ok tasks =>
      match parseMultiTrainerFlags rest { command := command, mode := mode, tasks := tasks } with
      | .error msg =>
        IO.eprintln msg
        IO.eprintln usage
        pure 1
      | .ok parsed =>
        let cfg := mkMultiTrainerConfig parsed
        match command.trimAscii.toString.toLower with
        | "multitrain" =>
          match (← multiTrainWithConfig cfg) with
          | .error msg =>
            IO.eprintln s!"[AlphaGradMulti][train] failed: {msg}"
            pure 1
          | .ok snapshot =>
            IO.println s!"[AlphaGradMulti][train] completed epochs={snapshot.completedEpochs} best_mean_eval={snapshot.bestMeanEvalReward} best_mean_search={snapshot.bestMeanSearchReward}"
            pure 0
        | "multieval" =>
          match (← multiEvalWithConfig cfg parsed.checkpointDir) with
          | .error msg =>
            IO.eprintln s!"[AlphaGradMulti][eval] failed: {msg}"
            pure 1
          | .ok results =>
            let greedyMean := meanArrayOrZero (results.map fun triple => triple.2.1)
            let searchMean := meanArrayOrZero (results.map fun triple => triple.2.2)
            IO.println s!"[AlphaGradMulti][eval] tasks={results.size} mean_greedy={greedyMean} mean_search={searchMean}"
            pure 0
        | other =>
          IO.eprintln s!"Unknown AlphaGrad multi-trainer command '{other}'."
          IO.eprintln usage
          pure 1
    | none, _ =>
      IO.eprintln s!"Invalid AlphaGrad mode '{modeStr}'."
      IO.eprintln usage
      pure 1
    | _, .error msg =>
      IO.eprintln msg
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

private def mkTrainerConfig (args : TrainerArgs) : AlphaGradTrainerConfig := {
  mode := args.mode
  task := args.task
  epochs := args.epochs.getD 64
  episodesPerEpoch := args.episodesPerEpoch.getD 16
  numEnvs := args.numEnvs.getD 8
  replayCapacity := args.replayCapacity.getD 4096
  sampleBatchSize := args.sampleBatchSize.getD 256
  updateBatchesPerEpoch := args.updateBatchesPerEpoch.getD 4
  checkpointEvery := args.checkpointEvery.getD 1
  evalEvery := args.evalEvery.getD 1
  numSimulations := args.numSimulations.getD 48
  runDir := args.runDir.getD ""
  seed := args.seed.getD 250197
  resume := args.resume
  overwrite := args.overwrite
}

def trainerMain (args : List String) : IO UInt32 := do
  match args with
  | command :: modeStr :: taskStr :: rest =>
    let commandNorm := command.trimAscii.toString.toLower
    if commandNorm = "multitrain" || commandNorm = "multieval" then
      multiTrainerMain args
    else
    match parseMode? modeStr, parseTaskName? taskStr with
    | some mode, some task =>
      match parseTrainerFlags rest { command := command, mode := mode, task := task } with
      | .error msg =>
        IO.eprintln msg
        IO.eprintln usage
        pure 1
      | .ok parsed =>
        let cfg := mkTrainerConfig parsed
        match command.trimAscii.toString.toLower with
        | "train" =>
          match (← trainWithConfig cfg) with
          | .error msg =>
            IO.eprintln s!"[AlphaGradTrainer][train] failed: {msg}"
            pure 1
          | .ok snapshot =>
            IO.println s!"[AlphaGradTrainer][train] completed epochs={snapshot.completedEpochs} best_eval={snapshot.bestEvalReward} best_search={snapshot.bestSearchReward}"
            pure 0
        | "eval" =>
          match (← evalWithConfig cfg parsed.checkpointDir) with
          | .error msg =>
            IO.eprintln s!"[AlphaGradTrainer][eval] failed: {msg}"
            pure 1
          | .ok (greedyReward, searchReward) =>
            IO.println s!"[AlphaGradTrainer][eval] greedy={greedyReward} search={searchReward}"
            pure 0
        | other =>
          IO.eprintln s!"Unknown AlphaGrad trainer command '{other}'."
          IO.eprintln usage
          pure 1
    | _, _ =>
      IO.eprintln s!"Invalid AlphaGrad trainer arguments: {modeStr} {taskStr}"
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

end Examples.AlphaGradPort
