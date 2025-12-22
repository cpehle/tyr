/-
  Tyr/Sharding.lean

  Type-safe distributed parameter sharding for data-parallel training.

  Key design decisions:
  1. **Type-level sharding** - ShardedTensor encodes rank and worldSize in the type,
     ensuring operations preserve sharding semantics
  2. **Automatic gather/scatter** - Operations that need the full tensor trigger
     implicit all-gather, and gradient updates use reduce-scatter
  3. **Zero-copy sharding** - ShardSpec allows computing shard boundaries without copying
  4. **Proof-carrying shards** - Proofs ensure rank < worldSize

  This enables modded-nanogpt's DistAdam pattern:
  - Reduce-scatter gradients (each rank updates its shard)
  - All-gather parameters after update

  Example:
  ```lean
  -- Full embedding: [vocab, dim]
  -- Each rank owns [vocab/worldSize, dim]
  let shardedEmbed : ShardedTensor #[vocabSize, dim] rank worldSize := ...

  -- Forward: need full tensor, triggers all-gather
  let fullEmbed ← shardedEmbed.gather
  let output := nn.embedding tokens fullEmbed

  -- Backward: gradients are reduced, each rank updates its shard
  let gradShard ← shardedEmbed.scatterGrad fullGrad
  ```
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed

namespace torch.Sharding

open torch

/-! ## Sharding Specification -/

/-- Dimension along which to shard (for multi-dimensional tensors) -/
inductive ShardDim where
  | first : ShardDim   -- Shard along first dimension
  | last : ShardDim    -- Shard along last dimension
  | dim (n : Nat) : ShardDim
  deriving Repr, BEq, Inhabited

/-- Specification for how a tensor is sharded across ranks -/
structure ShardSpec where
  /-- Which dimension to shard along -/
  shardDim : ShardDim := .first
  /-- Total number of shards (world size) -/
  numShards : UInt64
  /-- This shard's index (rank) -/
  shardIdx : UInt64
  deriving Repr, BEq

/-- Compute the shard size for a given full size -/
def shardSize (fullSize numShards shardIdx : UInt64) : UInt64 :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  if shardIdx < remainder then baseSize + 1 else baseSize

/-- Compute shard offset (starting index) -/
def shardOffset (fullSize numShards shardIdx : UInt64) : UInt64 :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  -- First `remainder` shards get one extra element
  if shardIdx < remainder then
    shardIdx * (baseSize + 1)
  else
    remainder * (baseSize + 1) + (shardIdx - remainder) * baseSize

/-! ## Type-level Shape Computation -/

/-- Compute the sharded shape from full shape -/
def shardedShape (s : Shape) (spec : ShardSpec) : Shape :=
  match s, spec.shardDim with
  | #[n], _ => #[shardSize n spec.numShards spec.shardIdx]
  | #[n, m], .first => #[shardSize n spec.numShards spec.shardIdx, m]
  | #[n, m], .last => #[n, shardSize m spec.numShards spec.shardIdx]
  | #[n, m, k], .first => #[shardSize n spec.numShards spec.shardIdx, m, k]
  | #[n, m, k], .last => #[n, m, shardSize k spec.numShards spec.shardIdx]
  | #[n, m, k], .dim 1 => #[n, shardSize m spec.numShards spec.shardIdx, k]
  | _, _ => s  -- Fallback: no sharding

/-! ## Sharded Tensor -/

/-- A tensor that is sharded across distributed ranks.

    The type encodes:
    - `fullShape`: The shape of the full (gathered) tensor
    - `rank`: This process's rank (shard index)
    - `worldSize`: Total number of processes (shards)

    The actual local data is stored as a regular tensor with shape
    `shardedShape fullShape spec`.
-/
structure ShardedTensor (fullShape : Shape) (rank worldSize : UInt64)
    (shardDim : ShardDim := .first) where
  /-- The local shard (this rank's portion) -/
  shard : T (shardedShape fullShape ⟨shardDim, worldSize, rank⟩)
  /-- Cached full tensor (populated after gather) -/
  cachedFull : Option (T fullShape) := none
  /-- Whether the cache is stale (shard was modified) -/
  cacheIsStale : Bool := true
  deriving Repr

/-- Proof obligation for valid sharding -/
abbrev ValidShard (rank worldSize : UInt64) := rank < worldSize

/-! ## Sharded Tensor Operations -/

/-- Create a sharded tensor by slicing a full tensor.

    This is used during initialization when one rank has the full tensor
    and needs to extract its shard.
-/
def ShardedTensor.fromFull {s : Shape} {rank worldSize : UInt64}
    (full : T s) (_h : ValidShard rank worldSize)
    (shardDim : ShardDim := .first) : ShardedTensor s rank worldSize shardDim :=
  -- Compute this rank's slice
  let spec : ShardSpec := ⟨shardDim, worldSize, rank⟩
  let shardShape := shardedShape s spec

  -- For sharding, we slice along the first dimension
  -- Get the first dimension size
  let firstDim := if h : s.size > 0 then s[0] else 1
  let offset := shardOffset firstDim worldSize rank
  let size := shardSize firstDim worldSize rank

  -- Extract the shard using T.slice method
  let sliced := full.slice 0 offset.toInt64 (offset + size).toInt64
  let sharded := reshape sliced shardShape

  { shard := sharded
    cachedFull := some full
    cacheIsStale := false
  }

/-- Gather all shards to reconstruct the full tensor.

    Uses all-gather collective operation.
    The result is cached for repeated access.
-/
def ShardedTensor.gather {s : Shape} {rank worldSize : UInt64} {sd : ShardDim}
    (sharded : ShardedTensor s rank worldSize sd) : IO (T s × ShardedTensor s rank worldSize sd) := do
  -- Check cache first
  if !sharded.cacheIsStale then
    match sharded.cachedFull with
    | some full => return (full, sharded)
    | none => pure ()

  -- Need to gather
  let full := zeros s  -- Allocate output
  dist.allGather full sharded.shard

  let newSharded := { sharded with
    cachedFull := some full
    cacheIsStale := false
  }
  return (full, newSharded)

/-- Scatter gradients to get this rank's portion.

    Used during backward pass: full gradient -> local gradient shard
    Uses reduce-scatter collective operation.
-/
def ShardedTensor.scatterGrad {s : Shape} {rank worldSize : UInt64} {sd : ShardDim}
    (_sharded : ShardedTensor s rank worldSize sd)
    (fullGrad : T s) : IO (T (shardedShape s ⟨sd, worldSize, rank⟩)) := do
  let localGrad := zeros (shardedShape s ⟨sd, worldSize, rank⟩)
  dist.reduceScatter localGrad fullGrad .sum
  return localGrad

/-- Update the local shard and mark cache as stale.

    This is called after the optimizer updates the local parameters.
-/
def ShardedTensor.updateShard {s : Shape} {rank worldSize : UInt64} {sd : ShardDim}
    (sharded : ShardedTensor s rank worldSize sd)
    (newShard : T (shardedShape s ⟨sd, worldSize, rank⟩))
    : ShardedTensor s rank worldSize sd :=
  { sharded with
    shard := newShard
    cacheIsStale := true
  }

/-! ## Sharded Parameter Group -/

/-- A parameter that may or may not be sharded.

    Some parameters (like embeddings) are sharded for memory efficiency.
    Others (like small scalars) are replicated.
-/
inductive MaybeSharded (s : Shape) (rank worldSize : UInt64) where
  | sharded : ShardedTensor s rank worldSize → MaybeSharded s rank worldSize
  | replicated : T s → MaybeSharded s rank worldSize
  deriving Repr

/-- Get the tensor for forward pass (gathers if sharded) -/
def MaybeSharded.get {s : Shape} {rank worldSize : UInt64}
    (p : MaybeSharded s rank worldSize) : IO (T s × MaybeSharded s rank worldSize) := do
  match p with
  | .replicated t => return (t, p)
  | .sharded st =>
    let (full, newSt) ← st.gather
    return (full, .sharded newSt)

/-- Update parameter with gradient -/
def MaybeSharded.updateWithGrad {s : Shape} {rank worldSize : UInt64}
    (p : MaybeSharded s rank worldSize)
    (fullGrad : T s)
    (updateFn : ∀ localShape, T localShape → T localShape → IO (T localShape))
    : IO (MaybeSharded s rank worldSize) := do
  match p with
  | .replicated t =>
    -- Replicated: all-reduce gradient, update locally
    dist.allReduce fullGrad .avg
    let newT ← updateFn s t fullGrad
    return .replicated newT
  | .sharded st =>
    -- Sharded: reduce-scatter gradient, update local shard
    let localGrad ← st.scatterGrad fullGrad
    let newShard ← updateFn _ st.shard localGrad
    return .sharded (st.updateShard newShard)

/-! ## Sharded Model Parameters -/

/-- Sharded embedding table for distributed training.

    Embeddings are sharded along the vocabulary dimension.
    Each rank owns vocab_size/world_size tokens.
-/
structure ShardedEmbedding (vocabSize dim : UInt64) (rank worldSize : UInt64) where
  /-- Local portion of embedding table -/
  weight : ShardedTensor #[vocabSize, dim] rank worldSize .first
  /-- Learning rate multiplier -/
  lrMul : Float := 75.0
  /-- Weight decay multiplier -/
  wdMul : Float := 1.0
  deriving Repr

/-- Forward pass for sharded embedding.

    1. Gather full embedding
    2. Apply embedding lookup
-/
def ShardedEmbedding.forward {vocabSize dim rank worldSize batch seq : UInt64}
    (emb : ShardedEmbedding vocabSize dim rank worldSize)
    (tokens : T #[batch, seq])
    : IO (T #[batch, seq, dim] × ShardedEmbedding vocabSize dim rank worldSize) := do
  let (fullWeight, newSharded) ← emb.weight.gather
  let output := nn.embedding tokens fullWeight
  return (output, { emb with weight := newSharded })

/-! ## Sharded Optimizer State -/

/-- Optimizer state that mirrors parameter sharding.

    If a parameter is sharded, its optimizer state (momentum, variance) is too.
-/
structure ShardedAdamState (s : Shape) (rank worldSize : UInt64) where
  /-- First moment (sharded if param is sharded) -/
  expAvg : MaybeSharded s rank worldSize
  /-- Second moment (sharded if param is sharded) -/
  expAvgSq : MaybeSharded s rank worldSize
  /-- Step count -/
  step : Nat
  deriving Repr

/-- Initialize sharded Adam state with zeros -/
def ShardedAdamState.init {s : Shape} {rank worldSize : UInt64}
    (param : MaybeSharded s rank worldSize) : ShardedAdamState s rank worldSize :=
  match param with
  | .replicated _ => {
      expAvg := .replicated (zeros s)
      expAvgSq := .replicated (zeros s)
      step := 0
    }
  | .sharded _ =>
      let shardShape := shardedShape s ⟨.first, worldSize, rank⟩
      {
        expAvg := .sharded ⟨zeros shardShape, none, true⟩
        expAvgSq := .sharded ⟨zeros shardShape, none, true⟩
        step := 0
      }

/-! ## Helper Functions -/

/-- Check if distributed training is active -/
def isShardingActive : IO Bool := dist.isInitialized

/-- Get sharding info for current process -/
def getShardingInfo : IO (UInt64 × UInt64) := do
  let isActive ← isShardingActive
  if isActive then
    dist.getRankAndWorldSize
  else
    pure (0, 1)

/-- Create shard spec from current distributed state -/
def currentShardSpec (shardDim : ShardDim := .first) : IO ShardSpec := do
  let (rank, worldSize) ← getShardingInfo
  return ⟨shardDim, worldSize, rank⟩

end torch.Sharding
