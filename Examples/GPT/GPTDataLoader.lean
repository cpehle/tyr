/-
  Examples/GPT/GPTDataLoader.lean

  GPT-specific data loading extensions for Tyr.
  
  Includes:
  - Language modeling batching (input/target shifting)
  - Modded-nanogpt training schedules
  - Simple sequential loading for character-level data (Shakespeare)
-/
import Tyr.DataLoader
import Examples.GPT.GPT

namespace torch.DataLoader

open torch

/-! ## GPT Batching -/

/-- Get the next batch of (input, target) pairs with erased types.
    For causal language modeling, target is input shifted by 1.
-/
def BatchIterator.nextGPT (iter : BatchIterator)
    : IO (Option (T #[] × T #[]) × BatchIterator) := do
  -- Get batch from BOS finder
  -- We need seqLen + 1 to split into input and target
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

/-- Get the next batch for GPT training. -/
def DistributedDataGenerator.nextBatchGPT (gen : DistributedDataGenerator)
    : IO (Option (T #[] × T #[]) × DistributedDataGenerator) := do
  let (maybeBatch, newIterator) ← gen.iterator.nextGPT
  match maybeBatch with
  | some batch =>
    let newGen := { gen with
      iterator := newIterator
      globalStep := gen.globalStep + 1
    }
    return (some batch, newGen)
  | none =>
    let nextIdx := (gen.trainPathIdx + 1) % gen.trainPaths.size
    let nextPath := gen.trainPaths[nextIdx]!
    let shard ← DataShard.load nextPath gen.rank gen.worldSize gen.config.bosToken
    let iter0 := BatchIterator.new shard newIterator.batchSize newIterator.seqLen
    let (maybeBatch', iter1) ← iter0.nextGPT
    let newGen := { gen with
      iterator := iter1
      globalStep := gen.globalStep + 1
      trainPathIdx := nextIdx
    }
    return (maybeBatch', newGen)

/-- Update batch and sequence parameters based on training step for modded-nanogpt. -/
def DistributedDataGenerator.updateForStepGPT (gen : DistributedDataGenerator)
    (step : UInt64) (blockSize : UInt64 := 128) : DistributedDataGenerator :=
  let (batchSize, windowBlocks) :=
    if step < 200 then (8, 3)
    else if step < 1000 then (16, 7)
    else (24, 11)
  let seqLen := windowBlocks * blockSize
  let newIterator := gen.iterator.updateParams batchSize seqLen
  { gen with iterator := newIterator }

/-! ## Utility Functions -/

/-- Get batch and window sizes for a given step based on modded-nanogpt. -/
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

/-! ## Sequential DataLoader (Shakespeare style) -/

/-- Sequential data loader for simple datasets like Shakespeare. -/
structure SequentialLoader (n : UInt64) where
  tokens : T #[n]
  currentPos : UInt64
  deriving Repr

def SequentialLoader.fromFile (path : String) : IO (Σ n, SequentialLoader n) := do
  let n ← data.binFileTokenCount path
  let tokens ← data.loadU16Bin n path
  return ⟨n, { tokens := tokens, currentPos := 0 }⟩

def SequentialLoader.reset {n : UInt64} (loader : SequentialLoader n) : SequentialLoader n :=
  { loader with currentPos := 0 }

def SequentialLoader.getBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : Option (T #[batchSize, seqLen + 1] × SequentialLoader n) :=
  let requiredLen := batchSize * (seqLen + 1)
  if loader.currentPos + requiredLen > n then
    none
  else
    let startPos := loader.currentPos
    let endPos := startPos + requiredLen
    let sliced := loader.tokens.slice 0 startPos.toInt64 endPos.toInt64
    let batch := reshape sliced #[batchSize, seqLen + 1]
    let newLoader := { loader with currentPos := endPos }
    some (batch, newLoader)

def SequentialLoader.sampleRandomBatch {n : UInt64} (loader : SequentialLoader n)
    (batchSize blockSize : UInt64) : IO (T #[batchSize, blockSize] × T #[batchSize, blockSize]) := do
  let maxStart := n - blockSize - 1
  let startIdx ← randint 0 (maxStart - batchSize * blockSize).toInt64 #[1]
  let startPos := nn.item startIdx
  let requiredLen := batchSize * (blockSize + 1)
  let sliced := loader.tokens.slice 0 startPos.toInt64 (startPos + requiredLen.toFloat).toInt64
  let batch := reshape sliced #[batchSize, blockSize + 1]
  let input := batch.slice 1 0 blockSize.toInt64
  let target := batch.slice 1 1 (blockSize.toInt64 + 1)
  let input := reshape input #[batchSize, blockSize]
  let target := reshape target #[batchSize, blockSize]
  return (input, target)

structure SequentialBatchIterator (n b s : UInt64) where
  loader : SequentialLoader n
  epoch : UInt64
  batchCount : UInt64
  deriving Repr

def SequentialBatchIterator.new {n : UInt64} (loader : SequentialLoader n)
    (batchSize seqLen : UInt64) : SequentialBatchIterator n batchSize seqLen := {
  loader := loader
  epoch := 0
  batchCount := 0
}

def SequentialBatchIterator.next {n b s : UInt64} (iter : SequentialBatchIterator n b s)
    : Option (T #[b, s] × T #[b, s]) × SequentialBatchIterator n b s :=
  match iter.loader.getBatch b s with
  | none =>
    let newLoader := iter.loader.reset
    let newIter := { iter with
      loader := newLoader
      epoch := iter.epoch + 1
      batchCount := 0
    }
    (none, newIter)
  | some (batch, newLoader) =>
    let input := batch.slice 1 0 s.toInt64
    let target := batch.slice 1 1 (s.toInt64 + 1)
    let input := reshape input #[b, s]
    let target := reshape target #[b, s]
    let newIter := { iter with
      loader := newLoader
      batchCount := iter.batchCount + 1
    }
    (some (input, target), newIter)

def loadShakespeareData (trainPath valPath : String) (batchSize seqLen : UInt64)
    : IO (Σ n, SequentialBatchIterator n batchSize seqLen × Option (Σ m, SequentialLoader m)) := do
  let ⟨nTrain, trainLoader⟩ ← SequentialLoader.fromFile trainPath
  let trainIter := SequentialBatchIterator.new trainLoader batchSize seqLen
  let valExists ← data.fileExists valPath
  let valLoader ← if valExists then do
    let ⟨nVal, loader⟩ ← SequentialLoader.fromFile valPath
    pure (some ⟨nVal, loader⟩)
  else
    pure none
  return ⟨nTrain, trainIter, valLoader⟩

end torch.DataLoader
