/-
  Tyr/DataLoader.lean

  General purpose data loading infrastructure for Tyr.
  
  Provides:
  - Configuration and metadata tracking
  - BOS (Beginning Of Sequence) tracking for document boundaries
  - Distributed sharding logic
  - Dynamic batch and sequence sizing support
-/
import Tyr.Torch
import Tyr.Distributed

namespace torch.DataLoader

open torch

/-! ## Configuration -/

structure Config where
  dataPath : String := "data"
  seqLen : UInt64 := 2048
  bosToken : UInt64 := 50256
  numWorkers : UInt64 := 4
  bufferSize : UInt64 := 8
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-! ## BOS Finder -/

structure BOSFinder where
  bosToken : UInt64
  bosPositions : Array UInt64
  currentPos : UInt64
  dataLen : UInt64
  deriving Repr

def BOSFinder.init (tokens : T #[n]) (bosToken : UInt64) : IO BOSFinder := do
  let dataLen := n
  let positionsTensor ← data.findBosPositions tokens bosToken.toInt64
  let positions ← data.tensorToUInt64Array' positionsTensor
  let positions := if positions.isEmpty then #[0] else positions
  return {
    bosToken := bosToken
    bosPositions := positions
    currentPos := 0
    dataLen := dataLen
  }

def BOSFinder.findNextValidStart (finder : BOSFinder) (after : UInt64) : Option UInt64 :=
  finder.bosPositions.find? (· >= after)

def BOSFinder.getBatch (finder : BOSFinder) (tokens : T #[n])
    (batchSize seqLen : UInt64) : IO (Option (T #[batchSize, seqLen]) × BOSFinder) := do
  let requiredLen := batchSize * seqLen
  if finder.currentPos + requiredLen > finder.dataLen then
    return (none, finder)

  let startPos := finder.currentPos
  let endPos := startPos + requiredLen
  let sliced := tokens.slice 0 startPos.toInt64 endPos.toInt64
  let batch := reshape sliced #[batchSize, seqLen]
  let newFinder := { finder with currentPos := endPos }
  return (some batch, newFinder)

def BOSFinder.reset (finder : BOSFinder) : BOSFinder :=
  { finder with currentPos := 0 }

/-! ## Randomization Utilities -/

private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

private def fisherYatesShuffle (arr : Array UInt64) (seed : UInt64) : Array UInt64 := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

def BOSFinder.shuffle (finder : BOSFinder) (seed : UInt64) : BOSFinder :=
  let shuffled := fisherYatesShuffle finder.bosPositions seed
  { finder with bosPositions := shuffled }

/-! ## Data Shard -/

def defaultShardSize : UInt64 := 1000000

structure DataShard (n : UInt64 := defaultShardSize) where
  tokens : T #[n]
  bosFinder : BOSFinder
  shardIdx : UInt64
  numShards : UInt64
  deriving Repr

def DataShard.loadFromFile (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (Σ n, DataShard n) := do
  let totalTokens ← data.binFileTokenCount path
  let tokensPerShard := totalTokens / numShards
  let startToken := tokensPerShard * shardIdx
  let endToken := if shardIdx == numShards - 1 then totalTokens else startToken + tokensPerShard
  let shardSize := endToken - startToken
  let allTokens ← data.loadU16Bin totalTokens path
  let shardTokens := allTokens.slice 0 startToken.toInt64 endToken.toInt64
  let shardTokens := reshape shardTokens #[shardSize]
  let bosFinder ← BOSFinder.init shardTokens bosToken
  return ⟨shardSize, { tokens := shardTokens, bosFinder, shardIdx, numShards }⟩

def DataShard.load (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (DataShard defaultShardSize) := do
  let fileExists ← data.fileExists path
  if fileExists then
    let ⟨_, shard⟩ ← DataShard.loadFromFile path shardIdx numShards bosToken
    let paddedTokens := reshape shard.tokens #[defaultShardSize]
    let bosFinder ← BOSFinder.init paddedTokens bosToken
    return { tokens := paddedTokens, bosFinder, shardIdx, numShards }
  else
    let tokens ← randn #[defaultShardSize] false
    let bosFinder ← BOSFinder.init tokens bosToken
    return { tokens := tokens, bosFinder, shardIdx, numShards }

/-! ## Batch Iterator -/

structure BatchIterator where
  shard : DataShard defaultShardSize
  batchSize : UInt64
  seqLen : UInt64
  batchCount : UInt64
  epoch : UInt64
  deriving Repr

def BatchIterator.new (shard : DataShard defaultShardSize) (batchSize seqLen : UInt64)
    : BatchIterator := { shard, batchSize, seqLen, batchCount := 0, epoch := 0 }

def BatchIterator.next (iter : BatchIterator)
    : IO (Option (T #[] ) × BatchIterator) := do
  let (maybeBatch, newBosFinder) ← iter.shard.bosFinder.getBatch
    iter.shard.tokens iter.batchSize iter.seqLen
  match maybeBatch with
  | none =>
    let resetFinder := newBosFinder.reset.shuffle iter.epoch
    let newShard := { iter.shard with bosFinder := resetFinder }
    let newIter := { iter with shard := newShard, epoch := iter.epoch + 1, batchCount := 0 }
    return (none, newIter)
  | some batch =>
    let batchDynamic := reshape batch #[]
    let newShard := { iter.shard with bosFinder := newBosFinder }
    let newIter := { iter with shard := newShard, batchCount := iter.batchCount + 1 }
    return (some batchDynamic, newIter)

def BatchIterator.updateParams (iter : BatchIterator)
    (batchSize seqLen : UInt64) : BatchIterator :=
  { iter with batchSize, seqLen }

/-! ## Distributed Data Generator -/

structure DistributedDataGenerator where
  iterator : BatchIterator
  config : Config
  globalStep : UInt64
  rank : UInt64
  worldSize : UInt64
  deriving Repr

def DistributedDataGenerator.init (config : Config) (batchSize seqLen : UInt64)
    : IO DistributedDataGenerator := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let shard ← DataShard.load config.dataPath rank worldSize config.bosToken
  let iterator := BatchIterator.new shard batchSize seqLen
  return { iterator, config, globalStep := 0, rank, worldSize }

def DistributedDataGenerator.nextBatch (gen : DistributedDataGenerator)
    : IO (Option (T #[]) × DistributedDataGenerator) := do
  let (maybeBatch, newIterator) ← gen.iterator.next
  return (maybeBatch, { gen with iterator := newIterator, globalStep := gen.globalStep + 1 })

def DistributedDataGenerator.batchSize (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.batchSize

def DistributedDataGenerator.seqLen (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.seqLen

/-! ## Validation and Utilities -/

def loadValidationData (path : String) (_seqLen : UInt64) (bosToken : UInt64)
    : IO DataShard := DataShard.load path 0 1 bosToken

def estimateRemainingTime (currentStep totalSteps : UInt64)
    (msPerStep : Float) : Float :=
  (totalSteps - currentStep).toFloat * msPerStep / 1000.0 / 60.0

end torch.DataLoader