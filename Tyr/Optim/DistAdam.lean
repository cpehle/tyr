/-
  Tyr/Optim/DistAdam.lean

  Distributed Adam optimizer for embedding parameters.

  DistAdam is used in modded-nanogpt for:
  - Token embeddings
  - Value embeddings
  - Language model head
  - Scalar parameters

  Features:
  - Standard Adam with bias correction
  - Type-safe parameter sharding via ShardedTensor
  - Proper reduce-scatter/all-gather pattern
  - Per-parameter learning rate multipliers
  - Decoupled weight decay

  Based on modded-nanogpt's DistAdam implementation.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Sharding

namespace torch.Optim.DistAdam

open torch
open torch.Sharding

/-- DistAdam configuration -/
structure Config where
  /-- Base learning rate -/
  lr : Float := 0.008
  /-- First moment decay (beta1) -/
  beta1 : Float := 0.65
  /-- Second moment decay (beta2) -/
  beta2 : Float := 0.95
  /-- Epsilon for numerical stability -/
  eps : Float := 1e-8
  /-- Weight decay (decoupled, AdamW-style) -/
  weightDecay : Float := 0.005
  /-- Whether to use distributed training -/
  distributed : Bool := false
  deriving Repr

instance : Inhabited Config where
  default := { lr := 0.008, beta1 := 0.65, beta2 := 0.95, eps := 1e-8,
               weightDecay := 0.005, distributed := false }

/-- State for a single parameter in DistAdam -/
structure ParamState (s : Shape) where
  /-- First moment (mean of gradients) -/
  expAvg : T s
  /-- Second moment (mean of squared gradients) -/
  expAvgSq : T s
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

/-- Full DistAdam optimizer state -/
structure State (α : Type) where
  /-- Configuration -/
  config : Config
  /-- Step count -/
  step : Nat := 0
  deriving Repr

/-- Initialize DistAdam state -/
def State.init (cfg : Config) : State α := {
  config := cfg
  step := 0
}

/-- Initialize parameter state with zeros -/
def initParamState {s : Shape} (param : T s) : ParamState s := {
  expAvg := zeros_like param
  expAvgSq := zeros_like param
  step := 0
}

instance {s : Shape} : TensorStruct (ParamState s) where
  map f ps := { ps with expAvg := f ps.expAvg, expAvgSq := f ps.expAvgSq }
  mapM f ps := do
    let expAvg ← f ps.expAvg
    let expAvgSq ← f ps.expAvgSq
    return { ps with expAvg, expAvgSq }
  zipWith f ps1 ps2 := {
    expAvg := f ps1.expAvg ps2.expAvg
    expAvgSq := f ps1.expAvgSq ps2.expAvgSq
    step := ps1.step
  }
  fold f acc ps :=
    let acc := f ps.expAvg acc
    f ps.expAvgSq acc

/-- Single Adam update step for one parameter.

    Standard Adam with bias correction:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g
    v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    update = lr * m_hat / (sqrt(v_hat) + eps)
    param = param - update - lr * wd * param  (decoupled WD)
-/
def stepSingle {s : Shape} (param : T s) (grad : T s) (state : ParamState s)
    (cfg : Config) (lrMul : Float := 1.0) (wdMul : Float := 1.0) : IO (T s × ParamState s) := do
  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  -- Detach gradient for updates
  let g := autograd.detach grad

  -- Update first moment: m = beta1 * m + (1 - beta1) * g
  let expAvg := mul_scalar state.expAvg beta1 + mul_scalar g (1.0 - beta1)

  -- Update second moment: v = beta2 * v + (1 - beta2) * g^2
  let gSquared := g * g
  let expAvgSq := mul_scalar state.expAvgSq beta2 + mul_scalar gSquared (1.0 - beta2)

  -- Bias correction
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat

  -- Compute step: lr * m_hat / (sqrt(v_hat) + eps)
  let mHat := div_scalar expAvg biasCorrection1
  let vHat := div_scalar expAvgSq biasCorrection2
  let denominator := nn.sqrt vHat + eps
  let update := nn.div mHat denominator
  let scaledUpdate := mul_scalar update lr

  -- Apply weight decay (decoupled)
  let paramDecayed := if wd > 0.0 then
      let decay := mul_scalar (autograd.detach param) (lr * wd)
      autograd.detach param - decay
    else
      autograd.detach param

  -- Apply update
  let newParam := paramDecayed - scaledUpdate
  let newParam := autograd.set_requires_grad newParam true

  let newState : ParamState s := {
    expAvg := expAvg
    expAvgSq := expAvgSq
    step := step
  }

  return (newParam, newState)

/-- Distributed Adam step with gradient aggregation.

    For distributed training:
    1. Reduce-scatter gradients across ranks
    2. Each rank updates its shard of parameters
    3. All-gather updated parameters

    This is more efficient than all-reduce as each rank only
    updates a portion of the parameters.
-/
private def canShardFirstDim (s : Shape) (worldSize : UInt64) : Bool :=
  worldSize > 0 && s.size > 0 && s[0]! % worldSize == 0

private def firstDimShardShape (s : Shape) (worldSize : UInt64) : Shape :=
  if s.size == 0 then
    s
  else
    #[s[0]! / worldSize] ++ s[1:].toArray

def stepDistributed {s : Shape} (param : T s) (grad : T s) (state : ParamState s)
    (cfg : Config) (lrMul : Float := 1.0) (wdMul : Float := 1.0) : IO (T s × ParamState s) := do
  if !cfg.distributed then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let isDist ← dist.isInitialized
  if !isDist then
    return (← stepSingle param grad state cfg lrMul wdMul)

  let worldSize ← dist.getWorldSize
  let rank ← dist.getRank
  if worldSize <= 1 then
    return (← stepSingle param grad state cfg lrMul wdMul)

  -- Match nanochat DistAdamW behavior where the first dimension must shard evenly.
  -- If not shardable, fall back to all-reduce + local step for compatibility.
  if !canShardFirstDim s worldSize then
    let gradReduced := autograd.detach grad
    dist.allReduce gradReduced .avg
    return (← stepSingle param gradReduced state cfg lrMul wdMul)

  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  let firstDim := s[0]!
  let localRows := firstDim / worldSize
  let start := rank * localRows
  let stop := start + localRows
  let localShape := firstDimShardShape s worldSize

  -- 1) materialize local parameter/state slices on the parameter device
  let paramSliceRaw := param.slice 0 start.toInt64 stop.toInt64
  let paramLocal : T localShape := autograd.detach (reshape paramSliceRaw localShape)
  let expAvgBase := if state.expAvg.device == param.device then state.expAvg else state.expAvg.to param.device
  let expAvgSqBase := if state.expAvgSq.device == param.device then state.expAvgSq else state.expAvgSq.to param.device
  let expAvgSliceRaw := expAvgBase.slice 0 start.toInt64 stop.toInt64
  let expAvgLocal : T localShape := autograd.detach (reshape expAvgSliceRaw localShape)
  let expAvgSqSliceRaw := expAvgSqBase.slice 0 start.toInt64 stop.toInt64
  let expAvgSqLocal : T localShape := autograd.detach (reshape expAvgSqSliceRaw localShape)

  -- 2) reduce-scatter averaged gradient shard
  let gradLocal : T localShape := zeros_like paramLocal
  dist.reduceScatter gradLocal grad .avg
  let g := autograd.detach gradLocal

  -- 3) local Adam update on shard
  let newExpAvg := mul_scalar expAvgLocal beta1 + mul_scalar g (1.0 - beta1)
  let gSquared := g * g
  let newExpAvgSq := mul_scalar expAvgSqLocal beta2 + mul_scalar gSquared (1.0 - beta2)
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat
  let mHat := div_scalar newExpAvg biasCorrection1
  let vHat := div_scalar newExpAvgSq biasCorrection2
  let denominator := nn.sqrt vHat + eps
  let update := nn.div mHat denominator
  let scaledUpdate := mul_scalar update lr

  let paramDecayed :=
    if wd > 0.0 then
      let decay := mul_scalar paramLocal (lr * wd)
      paramLocal - decay
    else
      paramLocal
  let newLocalParam := paramDecayed - scaledUpdate

  -- 4) all-gather updated local shard and optimizer shards back to replicated form
  let gatheredParam : T s := zeros_like param
  dist.allGather gatheredParam newLocalParam
  let gatheredExpAvg : T s := zeros_like expAvgBase
  dist.allGather gatheredExpAvg newExpAvg
  let gatheredExpAvgSq : T s := zeros_like expAvgSqBase
  dist.allGather gatheredExpAvgSq newExpAvgSq

  let newParam := autograd.set_requires_grad (autograd.detach gatheredParam) true
  let newState : ParamState s := {
    expAvg := gatheredExpAvg
    expAvgSq := gatheredExpAvgSq
    step := step
  }
  return (newParam, newState)

/-- Get learning rate multiplier for embedding parameters.

    modded-nanogpt uses 75x for embeddings.
-/
def embeddingLrMul : Float := 75.0

/-- Get learning rate multiplier for scalar parameters.

    modded-nanogpt uses 5x for scalars.
-/
def scalarLrMul : Float := 5.0

/-- Initialize a map of parameter states from a model structure -/
def initParamStates [TensorStruct α] (_model : α) : Array (ParamState #[]) :=
  -- Simplified: in a full implementation, this would create states
  -- for each parameter in the model
  #[]

/-! ## Type-Safe Sharded Adam -/

/-- State for a sharded parameter in DistAdam.

    The optimizer state (moments) is sharded in the same way as the parameter,
    so each rank only stores and updates its portion.
-/
structure ShardedParamState (fullShape : Shape) (rank worldSize : UInt64) where
  /-- First moment (sharded) -/
  expAvg : ShardedTensor fullShape rank worldSize
  /-- Second moment (sharded) -/
  expAvgSq : ShardedTensor fullShape rank worldSize
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

/-- Initialize sharded parameter state with zeros -/
def initShardedParamState {fullShape : Shape} {rank worldSize : UInt64}
    (_param : ShardedTensor fullShape rank worldSize) : ShardedParamState fullShape rank worldSize :=
  let localShape := shardedShape fullShape ⟨.first, worldSize, rank⟩
  {
    expAvg := ⟨zeros localShape, none, true⟩
    expAvgSq := ⟨zeros localShape, none, true⟩
    step := 0
  }

/-- Adam step for a sharded parameter.

    The sharding pattern for DistAdam:
    1. fullGrad arrives from backward pass
    2. Reduce-scatter: each rank gets its gradient shard (summed across ranks)
    3. Each rank updates its parameter shard with its gradient shard
    4. State (moments) stays sharded - no communication needed
    5. Parameters are gathered when needed for forward pass

    This is more memory-efficient than all-reduce because:
    - Each rank only stores 1/worldSize of the parameter
    - Gradient communication is reduce-scatter (same bandwidth as all-reduce)
    - Parameter gather only happens once per forward pass
-/
def stepSharded {fullShape : Shape} {rank worldSize : UInt64}
    (param : ShardedTensor fullShape rank worldSize)
    (fullGrad : T fullShape)
    (state : ShardedParamState fullShape rank worldSize)
    (cfg : Config) (lrMul wdMul : Float := 1.0)
    : IO (ShardedTensor fullShape rank worldSize × ShardedParamState fullShape rank worldSize) := do

  let step := state.step + 1
  let beta1 := cfg.beta1
  let beta2 := cfg.beta2
  let eps := cfg.eps
  let lr := cfg.lr * lrMul
  let wd := cfg.weightDecay * wdMul

  -- 1. Reduce-scatter gradient to get this rank's portion
  let localGrad ← param.scatterGrad fullGrad

  -- Detach for optimizer updates
  let g := autograd.detach localGrad

  -- 2. Update first moment: m = beta1 * m + (1 - beta1) * g
  let expAvg := mul_scalar state.expAvg.shard beta1 + mul_scalar g (1.0 - beta1)

  -- 3. Update second moment: v = beta2 * v + (1 - beta2) * g^2
  let gSquared := g * g
  let expAvgSq := mul_scalar state.expAvgSq.shard beta2 + mul_scalar gSquared (1.0 - beta2)

  -- 4. Bias correction
  let biasCorrection1 := 1.0 - Float.pow beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow beta2 step.toFloat

  -- 5. Compute update: lr * m_hat / (sqrt(v_hat) + eps)
  let mHat := div_scalar expAvg biasCorrection1
  let vHat := div_scalar expAvgSq biasCorrection2
  let denominator := nn.sqrt vHat + eps
  let update := nn.div mHat denominator
  let scaledUpdate := mul_scalar update lr

  -- 6. Apply weight decay (decoupled) to local shard
  let paramLocal := autograd.detach param.shard
  let paramDecayed := if wd > 0.0 then
      let decay := mul_scalar paramLocal (lr * wd)
      paramLocal - decay
    else
      paramLocal

  -- 7. Apply update to local shard
  let newLocal := paramDecayed - scaledUpdate
  let newLocal := autograd.set_requires_grad newLocal true

  -- 8. Update sharded parameter (mark cache as stale)
  let newParam := param.updateShard newLocal

  -- 9. Update sharded state
  let newState : ShardedParamState fullShape rank worldSize := {
    expAvg := state.expAvg.updateShard expAvg
    expAvgSq := state.expAvgSq.updateShard expAvgSq
    step := step
  }

  return (newParam, newState)

/-- Sharded embedding with integrated optimizer state.

    This bundles together:
    - The sharded embedding weights
    - The sharded Adam moments
    - Learning rate and weight decay multipliers

    Usage:
    ```lean
    let (output, newEmbed) ← embed.forward tokens
    -- ... backward pass produces fullGrad ...
    let newEmbed ← embed.step fullGrad cfg
    ```
-/
structure ShardedEmbeddingAdam (vocabSize dim : UInt64) (rank worldSize : UInt64) where
  /-- Sharded embedding weights -/
  weight : ShardedTensor #[vocabSize, dim] rank worldSize
  /-- Sharded optimizer state -/
  optState : ShardedParamState #[vocabSize, dim] rank worldSize
  /-- Learning rate multiplier (modded-nanogpt uses 75x) -/
  lrMul : Float := 75.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr

/-- Initialize sharded embedding with Adam state -/
def ShardedEmbeddingAdam.init {vocabSize dim rank worldSize : UInt64}
    (fullWeight : T #[vocabSize, dim]) (h : ValidShard rank worldSize)
    : ShardedEmbeddingAdam vocabSize dim rank worldSize :=
  let shardedWeight := ShardedTensor.fromFull fullWeight h
  let optState := initShardedParamState shardedWeight
  {
    weight := shardedWeight
    optState := optState
    lrMul := 75.0
    wdMul := 1.0
  }

/-- Forward pass: gather weights and apply embedding lookup -/
def ShardedEmbeddingAdam.forward {vocabSize dim rank worldSize batch seq : UInt64}
    (embed : ShardedEmbeddingAdam vocabSize dim rank worldSize)
    (tokens : T #[batch, seq])
    : IO (T #[batch, seq, dim] × ShardedEmbeddingAdam vocabSize dim rank worldSize) := do
  let (fullWeight, newSharded) ← embed.weight.gather
  let output := nn.embedding tokens fullWeight
  return (output, { embed with weight := newSharded })

/-- Backward/update step: reduce-scatter grad and update local shard -/
def ShardedEmbeddingAdam.step {vocabSize dim rank worldSize : UInt64}
    (embed : ShardedEmbeddingAdam vocabSize dim rank worldSize)
    (fullGrad : T #[vocabSize, dim])
    (cfg : Config)
    : IO (ShardedEmbeddingAdam vocabSize dim rank worldSize) := do
  let (newWeight, newOptState) ← stepSharded
    embed.weight fullGrad embed.optState cfg embed.lrMul embed.wdMul
  return { embed with
    weight := newWeight
    optState := newOptState
  }

end torch.Optim.DistAdam
