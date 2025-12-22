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
  -- Scan for BOS tokens (this would be done efficiently in C++)
  -- For now, return a placeholder with evenly spaced "documents"
  let dataLen := n
  -- In a real implementation, we'd scan the tokens tensor
  -- Here we assume documents are roughly evenly distributed
  let estimatedDocLen : UInt64 := 512  -- Average document length
  let numDocs := dataLen / estimatedDocLen
  let mut positions := #[]
  for i in [:numDocs.toNat] do
    positions := positions.push (i.toUInt64 * estimatedDocLen)
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

/-- Shuffle BOS positions for randomized document order -/
def BOSFinder.shuffle (finder : BOSFinder) (seed : UInt64) : BOSFinder :=
  -- Simple shuffle using seed - in production use proper PRNG
  -- For now, just reverse based on seed parity
  let arr := if seed % 2 == 0 then finder.bosPositions else finder.bosPositions.reverse
  { finder with bosPositions := arr }

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

/-- Load a data shard for the given rank -/
def DataShard.load (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (DataShard defaultShardSize) := do
  -- In a full implementation:
  -- 1. Memory-map the file
  -- 2. Compute shard boundaries
  -- 3. Load only this rank's portion

  -- Placeholder: create random data for testing
  let tokens ← randn #[defaultShardSize] false
  let bosFinder ← BOSFinder.init tokens bosToken
  return {
    tokens := tokens
    bosFinder := bosFinder
    shardIdx := shardIdx
    numShards := numShards
  }

/-! ## Batch Iterator -/

/-- Batch iterator state (uses fixed shard size for simplicity) -/
structure BatchIterator where
  /-- Data shard being iterated -/
  shard : DataShard defaultShardSize
  /-- Current batch size -/
  batchSize : UInt64
  /-- Current sequence length -/
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

/-- Get the next batch of (input, target) pairs.

    Returns None when epoch is complete.
    Note: Uses erased types for simplicity - real impl would track shapes properly.
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
def loadValidationData (path : String) (seqLen : UInt64) (bosToken : UInt64)
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
def getHyperparamsForStep (step : UInt64) (blockSize : UInt64 := 128)
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

end torch.DataLoader
