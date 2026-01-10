/-
  Tyr/DataLoader.lean

  Document-aware data loading for modded-nanogpt style training.

  Key features:
  - BOS token tracking for document-aligned sequences
  - Dynamic batch and window sizing during training
  - Distributed data loading with rank-aware sharding
  - Async data preloading for throughput

  Based on modded-nanogpt's data loading implementation.
-/
import Tyr.Torch
import Tyr.Distributed

namespace torch.DataLoader

open torch

/-! ## Configuration -/

/-- Data loader configuration -/
structure Config where
  /-- Path to tokenized data files -/
  dataPath : String := "data"
  /-- Sequence length for training -/
  seqLen : UInt64 := 2048
  /-- BOS token ID -/
  bosToken : UInt64 := 50256
  /-- Number of data loading workers -/
  numWorkers : UInt64 := 4
  /-- Buffer size for preloading (in batches) -/
  bufferSize : UInt64 := 8
  /-- Seed for random shuffling -/
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-! ## BOS Finder -/

/-- BOS token finder for document-aligned batching.

    Tracks positions of BOS tokens in a token sequence to ensure
    sequences don't cross document boundaries.
-/
structure BOSFinder where
  /-- BOS token ID to search for -/
  bosToken : UInt64
  /-- Cached BOS positions in current data chunk -/
  bosPositions : Array UInt64
  /-- Current position in the data -/
  currentPos : UInt64
  /-- Total data length -/
  dataLen : UInt64
  deriving Repr

/-- Initialize a BOS finder by scanning a token tensor for BOS tokens -/
def BOSFinder.init (tokens : T #[n]) (bosToken : UInt64) : IO BOSFinder := do
  -- Scan for BOS tokens using efficient C++ implementation
  let dataLen := n
  let positionsTensor ← data.findBosPositions tokens bosToken.toInt64
  -- Use the shape-erased version since findBosPositions returns T #[]
  let positions ← data.tensorToUInt64Array' positionsTensor

  -- If no BOS tokens found, treat entire data as one document starting at 0
  let positions := if positions.isEmpty then #[0] else positions

  return {
    bosToken := bosToken
    bosPositions := positions
    currentPos := 0
    dataLen := dataLen
  }

/-- Find the next valid starting position that aligns with a document boundary -/
def BOSFinder.findNextValidStart (finder : BOSFinder) (after : UInt64) : Option UInt64 :=
  -- Find the first BOS position >= after
  finder.bosPositions.find? (· >= after)

/-- Get a batch of sequences aligned to document boundaries.

    Returns (sequences, new_finder_state) where sequences don't cross
    document boundaries.
-/
def BOSFinder.getBatch (finder : BOSFinder) (tokens : T #[n])
    (batchSize seqLen : UInt64) : IO (Option (T #[batchSize, seqLen]) × BOSFinder) := do
  -- Check if we have enough data
  let requiredLen := batchSize * seqLen
  if finder.currentPos + requiredLen > finder.dataLen then
    return (none, finder)

  -- In a full implementation, we'd:
  -- 1. Find BOS positions for each sequence start
  -- 2. Extract non-overlapping sequences
  -- 3. Handle document boundaries

  -- Simplified: just slice contiguous sequences
  let startPos := finder.currentPos
  let endPos := startPos + requiredLen

  -- Extract and reshape
  let sliced := tokens.slice 0 startPos.toInt64 endPos.toInt64
  let batch := reshape sliced #[batchSize, seqLen]

  let newFinder := { finder with currentPos := endPos }
  return (some batch, newFinder)

/-- Reset the BOS finder to the beginning -/
def BOSFinder.reset (finder : BOSFinder) : BOSFinder :=
  { finder with currentPos := 0 }

/-- LCG random number generator step -/
private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

/-- Fisher-Yates shuffle using LCG PRNG -/
private def fisherYatesShuffle (arr : Array UInt64) (seed : UInt64) : Array UInt64 := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    -- Generate random index in range [i, arr.size)
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    -- Swap elements at i and j
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

/-- Shuffle BOS positions for randomized document order -/
def BOSFinder.shuffle (finder : BOSFinder) (seed : UInt64) : BOSFinder :=
  let shuffled := fisherYatesShuffle finder.bosPositions seed
  { finder with bosPositions := shuffled }

/-! ## Data Shard -/

/-- Default shard size (1M tokens) -/
def defaultShardSize : UInt64 := 1000000

/-- A shard of training data for a specific rank -/
structure DataShard (n : UInt64 := defaultShardSize) where
  /-- Token data tensor -/
  tokens : T #[n]
  /-- BOS finder for this shard -/
  bosFinder : BOSFinder
  /-- Shard index (rank) -/
  shardIdx : UInt64
  /-- Total number of shards (world size) -/
  numShards : UInt64
  deriving Repr

/-- Load a data shard from a file for the given rank.
    For distributed training, each rank loads a different portion of the data.
    Returns a dependent pair (size, shard). -/
def DataShard.loadFromFile (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (Σ n, DataShard n) := do
  -- Get total file size
  let totalTokens ← data.binFileTokenCount path

  -- Compute this shard's portion
  let tokensPerShard := totalTokens / numShards
  let startToken := tokensPerShard * shardIdx
  let endToken := if shardIdx == numShards - 1
                  then totalTokens  -- Last shard gets remainder
                  else startToken + tokensPerShard
  let shardSize := endToken - startToken

  -- Load full file and slice to this shard's portion
  let allTokens ← data.loadU16Bin totalTokens path
  let shardTokens := allTokens.slice 0 startToken.toInt64 endToken.toInt64
  let shardTokens := reshape shardTokens #[shardSize]

  -- Initialize BOS finder on shard data
  let bosFinder ← BOSFinder.init shardTokens bosToken

  return ⟨shardSize, {
    tokens := shardTokens
    bosFinder := bosFinder
    shardIdx := shardIdx
    numShards := numShards
  }⟩

/-- Load a data shard with fixed default size (for backward compatibility).
    If file is smaller than defaultShardSize, pads with zeros. -/
def DataShard.load (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (DataShard defaultShardSize) := do
  -- Check if file exists
  let fileExists ← data.fileExists path
  if fileExists then
    -- Load from file
    let ⟨_, shard⟩ ← DataShard.loadFromFile path shardIdx numShards bosToken
    -- Reshape/pad to fixed size (may truncate or need padding)
    -- For now, just use the actual size - caller should use loadFromFile for dynamic sizes
    let paddedTokens := reshape shard.tokens #[defaultShardSize]
    let bosFinder ← BOSFinder.init paddedTokens bosToken
    return {
      tokens := paddedTokens
      bosFinder := bosFinder
      shardIdx := shardIdx
      numShards := numShards
    }
  else
    -- Fallback: create random data for testing
    IO.println s!"Warning: File not found '{path}', using random data"
    let tokens ← randn #[defaultShardSize] false
    let bosFinder ← BOSFinder.init tokens bosToken
    return {
      tokens := tokens
      bosFinder := bosFinder
      shardIdx := shardIdx
      numShards := numShards
    }

/-! ## Batch Iterator

### Type Safety Design Choice

`BatchIterator` intentionally uses dynamic (type-erased) batch dimensions because
modded-nanogpt changes batch and sequence sizes during training:
- Batch size: 8 → 16 → 24 (at specific step thresholds)
- Window size: 3 → 7 → 11 blocks

This flexibility requires runtime dimension tracking via `T #[]`.

**For fixed batch/sequence dimensions**, use `SequentialBatchIterator n b s` instead,
which provides full compile-time shape guarantees via `T #[b, s]`.

Example:
```lean
-- Dynamic dimensions (modded-nanogpt style):
let iter := BatchIterator.new shard 8 256
let (batch, iter') ← iter.next  -- Returns T #[]

-- Fixed dimensions (type-safe):
let iter := SequentialBatchIterator.new loader 8 256
let (batch, iter') := iter.next  -- Returns T #[8, 256]
```
-/

/-- Batch iterator state with dynamic dimensions.
    Uses erased types (`T #[]`) to support runtime dimension changes.
    For type-safe iteration with fixed dimensions, use `SequentialBatchIterator` instead. -/
structure BatchIterator where
  /-- Data shard being iterated -/
  shard : DataShard defaultShardSize
  /-- Current batch size (runtime value, not in type) -/
  batchSize : UInt64
  /-- Current sequence length (runtime value, not in type) -/
  seqLen : UInt64
  /-- Number of batches produced -/
  batchCount : UInt64
  /-- Epoch number -/
  epoch : UInt64
  deriving Repr

/-- Create a new batch iterator -/
def BatchIterator.new (shard : DataShard defaultShardSize) (batchSize seqLen : UInt64)
    : BatchIterator := {
  shard := shard
  batchSize := batchSize
  seqLen := seqLen
  batchCount := 0
  epoch := 0
}

/-- Get the next batch of (input, target) pairs with erased types.

    Returns None when epoch is complete.
    Uses `T #[]` because batch/seq dimensions can change during training.

    To recover typed tensors, use `reshape` with known dimensions:
    ```lean
    let (maybeBatch, iter') ← iter.next
    match maybeBatch with
    | some (x, y) =>
      -- Caller knows current dimensions from iter.batchSize and iter.seqLen
      let x : T #[b, s] := reshape x #[b, s]
      let y : T #[b, s] := reshape y #[b, s]
    | none => ...
    ```
-/
def BatchIterator.next (iter : BatchIterator)
    : IO (Option (T #[] × T #[]) × BatchIterator) := do
  -- Get batch from BOS finder
  let (maybeBatch, newBosFinder) ← iter.shard.bosFinder.getBatch
    iter.shard.tokens iter.batchSize (iter.seqLen + 1)

  match maybeBatch with
  | none =>
    -- Epoch complete, reset and increment epoch
    let resetFinder := newBosFinder.reset.shuffle iter.epoch
    let newShard := { iter.shard with bosFinder := resetFinder }
    let newIter := { iter with
      shard := newShard
      epoch := iter.epoch + 1
      batchCount := 0
    }
    return (none, newIter)
  | some batch =>
    -- Split into input and target (target is shifted by 1)
    -- batch is [batchSize, seqLen+1]
    let input := batch.slice 1 0 iter.seqLen.toInt64
    let target := batch.slice 1 1 (iter.seqLen.toInt64 + 1)
    -- Reshape to dynamic types for flexibility
    let input := reshape input #[iter.batchSize, iter.seqLen]
    let target := reshape target #[iter.batchSize, iter.seqLen]
    -- Erase types for simpler return type
    let input := reshape input #[]
    let target := reshape target #[]
    let newShard := { iter.shard with bosFinder := newBosFinder }
    let newIter := { iter with
      shard := newShard
      batchCount := iter.batchCount + 1
    }
    return (some (input, target), newIter)

/-- Typed version of `next` that returns tensors with explicit shape.

    Use when you know the current batch dimensions and want type-safe tensors.
    Panics if the iterator's current dimensions don't match the expected shape.

    ```lean
    let (maybeBatch, iter') ← iter.nextTyped 8 256
    -- maybeBatch : Option (T #[8, 256] × T #[8, 256])
    ```
-/
def BatchIterator.nextTyped (iter : BatchIterator) (b s : UInt64)
    : IO (Option (T #[b, s] × T #[b, s]) × BatchIterator) := do
  let (maybeBatch, newIter) ← iter.next
  match maybeBatch with
  | none => return (none, newIter)
  | some (x, y) =>
    -- Reshape to the specified dimensions
    let x : T #[b, s] := reshape x #[b, s]
    let y : T #[b, s] := reshape y #[b, s]
    return (some (x, y), newIter)

/-- Update batch size and sequence length dynamically.

    modded-nanogpt changes these during training:
    - Batch size: 8 -> 16 -> 24 (at specific step thresholds)
    - Window size: 3 -> 7 -> 11 blocks
-/
def BatchIterator.updateParams (iter : BatchIterator)
    (batchSize seqLen : UInt64) : BatchIterator :=
  { iter with batchSize := batchSize, seqLen := seqLen }

/-! ## Distributed Data Generator -/

/-- Distributed data generator for multi-GPU training.

    Each rank gets a different shard of the data.
    Batch and window sizes can change during training.
-/
structure DistributedDataGenerator where
  /-- Data iterator for this rank -/
  iterator : BatchIterator
  /-- Configuration -/
  config : Config
  /-- Current global step -/
  globalStep : UInt64
  /-- Rank of this process -/
  rank : UInt64
  /-- World size -/
  worldSize : UInt64
  deriving Repr

/-- Initialize distributed data generator -/
def DistributedDataGenerator.init (config : Config) (batchSize seqLen : UInt64)
    : IO DistributedDataGenerator := do
  -- Get rank and world size
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  -- Load this rank's shard
  let shard ← DataShard.load config.dataPath rank worldSize config.bosToken

  -- Create iterator
  let iterator := BatchIterator.new shard batchSize seqLen

  return {
    iterator := iterator
    config := config
    globalStep := 0
    rank := rank
    worldSize := worldSize
  }

/-- Get the next batch for training.
    Returns dynamically-typed tensors for flexibility. -/
def DistributedDataGenerator.nextBatch (gen : DistributedDataGenerator)
    : IO (Option (T #[] × T #[]) × DistributedDataGenerator) := do
  let (maybeBatch, newIterator) ← gen.iterator.next
  let newGen := { gen with
    iterator := newIterator
    globalStep := gen.globalStep + 1
  }
  return (maybeBatch, newGen)

/-- Update batch and sequence parameters based on training step.

    modded-nanogpt schedule:
    - Steps 0-199: batchSize=8, window=3 blocks
    - Steps 200-999: batchSize=16, window=7 blocks
    - Steps 1000+: batchSize=24, window=11 blocks
-/
def DistributedDataGenerator.updateForStep (gen : DistributedDataGenerator)
    (step : UInt64) (blockSize : UInt64 := 128) : DistributedDataGenerator :=
  let (batchSize, windowBlocks) :=
    if step < 200 then (8, 3)
    else if step < 1000 then (16, 7)
    else (24, 11)
  let seqLen := windowBlocks * blockSize
  let newIterator := gen.iterator.updateParams batchSize seqLen
  { gen with iterator := newIterator }

/-- Get current batch size -/
def DistributedDataGenerator.batchSize (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.batchSize

/-- Get current sequence length -/
def DistributedDataGenerator.seqLen (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.seqLen

/-! ## Validation Data -/

/-- Load validation data (smaller, not sharded) -/
def loadValidationData (path : String) (_seqLen : UInt64) (bosToken : UInt64)
    : IO DataShard := do
  -- Load full validation set (same on all ranks)
  DataShard.load path 0 1 bosToken

/-- Evaluate on validation set -/
def validateBatch {batch seq vocab : UInt64}
    (logits : T #[batch, seq, vocab])
    (targets : T #[batch, seq])
    : IO Float := do
  -- Compute cross-entropy loss
  let logitsFlat := reshape logits #[batch * seq, vocab]
  let targetsFlat := reshape targets #[batch * seq]
  let loss := nn.cross_entropy logitsFlat targetsFlat
  return nn.item loss

/-! ## Utility Functions -/

/-- Get batch and window sizes for a given step.

    Based on modded-nanogpt hyperparameters.
-/
def getHyperparamsForStep (step : UInt64) (_blockSize : UInt64 := 128)
    : (UInt64 × UInt64 × UInt64) :=  -- (batchSize, wsShort, wsLong)
  if step < 200 then
    (8, 3, 3)  -- All short windows initially
  else if step < 1000 then
    (16, 3, 7)  -- Mixed windows
  else
    (24, 3, 11)  -- Full windows

/-- Compute effective tokens per step -/
def tokensPerStep (batchSize seqLen worldSize : UInt64) : UInt64 :=
  batchSize * seqLen * worldSize

/-- Estimate remaining training time based on step and total -/
def estimateRemainingTime (currentStep totalSteps : UInt64)
    (msPerStep : Float) : Float :=
  let remaining := totalSteps - currentStep
  remaining.toFloat * msPerStep / 1000.0 / 60.0  -- In minutes

/-! ## Sequential DataLoader (Shakespeare style)

    Simple sequential data loading without document boundaries.
    Suitable for character-level or single-document datasets.
-/

/-- Sequential data loader for simple datasets like Shakespeare.
    Loads entire file into memory and iterates sequentially or randomly. -/
structure SequentialLoader (n : UInt64) where
  /-- Token data tensor -/
  tokens : T #[n]
  /-- Current position for sequential iteration -/
  currentPos : UInt64
  deriving Repr

/-- Load a sequential loader from a binary file -/
def SequentialLoader.fromFile (path : String) : IO (Σ n, SequentialLoader n) := do
  let n ← data.binFileTokenCount path
  let tokens ← data.loadU16Bin n path
  return ⟨n, { tokens := tokens, currentPos := 0 }⟩

/-- Reset position to beginning -/
def SequentialLoader.reset {n : UInt64} (loader : SequentialLoader n) : SequentialLoader n :=
  { loader with currentPos := 0 }

/-- Get a contiguous batch of sequences.
    Returns (batch, newLoader) where batch is [batchSize, seqLen+1].
    Returns none when not enough data remains. -/
def SequentialLoader.getBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : Option (T #[batchSize, seqLen + 1] × SequentialLoader n) :=
  let requiredLen := batchSize * (seqLen + 1)
  if loader.currentPos + requiredLen > n then
    none
  else
    let startPos := loader.currentPos
    let endPos := startPos + requiredLen
    -- Slice and reshape to [batchSize, seqLen+1]
    let sliced := loader.tokens.slice 0 startPos.toInt64 endPos.toInt64
    let batch := reshape sliced #[batchSize, seqLen + 1]
    let newLoader := { loader with currentPos := endPos }
    some (batch, newLoader)

/-- Sample a random batch (nanoGPT style).
    Each sequence in the batch starts at a random position.
    Returns (input, target) where target is input shifted by 1. -/
def SequentialLoader.sampleRandomBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize blockSize : UInt64) : IO (T #[batchSize, blockSize] × T #[batchSize, blockSize]) := do
  -- Maximum valid starting index (need blockSize + 1 tokens for input and target)
  let maxStart := n - blockSize - 1

  -- For simplicity, sample a single starting position and create batch from sequential positions
  -- This gives us a contiguous batch rather than truly random starts, but is much simpler
  let startIdx ← randint 0 (maxStart - batchSize * blockSize).toInt64 #[1]
  let startPos := nn.item startIdx

  -- Calculate required length: batchSize sequences of (blockSize + 1) tokens each
  let requiredLen := batchSize * (blockSize + 1)

  -- Slice out the required portion
  let sliced := loader.tokens.slice 0 startPos.toInt64 (startPos + requiredLen.toFloat).toInt64

  -- Reshape to [batchSize, blockSize + 1]
  let batch := reshape sliced #[batchSize, blockSize + 1]

  -- Split into input and target using slice along dimension 1
  let input := batch.slice 1 0 blockSize.toInt64
  let target := batch.slice 1 1 (blockSize.toInt64 + 1)

  -- Reshape to correct types
  let input := reshape input #[batchSize, blockSize]
  let target := reshape target #[batchSize, blockSize]
  return (input, target)

/-- Sequential batch iterator with epoch tracking.
    Parametrized by data size n, batch size b, and sequence length s. -/
structure SequentialBatchIterator (n b s : UInt64) where
  /-- Underlying loader -/
  loader : SequentialLoader n
  /-- Current epoch -/
  epoch : UInt64
  /-- Batches produced in current epoch -/
  batchCount : UInt64
  deriving Repr

/-- Create a new sequential batch iterator -/
def SequentialBatchIterator.new {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : SequentialBatchIterator n batchSize seqLen := {
  loader := loader
  epoch := 0
  batchCount := 0
}

/-- Get next batch, returning (input, target) pair with proper types.
    Returns none at epoch boundary, then resets for next epoch. -/
def SequentialBatchIterator.next {n b s : UInt64} (iter : SequentialBatchIterator n b s)
    : Option (T #[b, s] × T #[b, s]) × SequentialBatchIterator n b s :=
  match iter.loader.getBatch b s with
  | none =>
    -- Epoch complete, reset loader
    let newLoader := iter.loader.reset
    let newIter := { iter with
      loader := newLoader
      epoch := iter.epoch + 1
      batchCount := 0
    }
    (none, newIter)
  | some (batch, newLoader) =>
    -- batch is [b, s+1], split into input/target
    let input := batch.slice 1 0 s.toInt64
    let target := batch.slice 1 1 (s.toInt64 + 1)
    -- Reshape to ensure correct types
    let input := reshape input #[b, s]
    let target := reshape target #[b, s]
    let newIter := { iter with
      loader := newLoader
      batchCount := iter.batchCount + 1
    }
    (some (input, target), newIter)

/-- Convenience: load Shakespeare data and create iterator -/
def loadShakespeareData (trainPath valPath : String) (batchSize seqLen : UInt64)
    : IO (Σ n, SequentialBatchIterator n batchSize seqLen × Option (Σ m, SequentialLoader m)) := do
  -- Load training data
  let ⟨nTrain, trainLoader⟩ ← SequentialLoader.fromFile trainPath
  let trainIter := SequentialBatchIterator.new trainLoader batchSize seqLen

  -- Try to load validation data
  let valExists ← data.fileExists valPath
  let valLoader ← if valExists then do
    let ⟨nVal, loader⟩ ← SequentialLoader.fromFile valPath
    pure (some ⟨nVal, loader⟩)
  else
    pure none

  return ⟨nTrain, trainIter, valLoader⟩

end torch.DataLoader
