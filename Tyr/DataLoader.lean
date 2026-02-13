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
  valPath : Option String := none
  seqLen : UInt64 := 2048
  bosToken : UInt64 := 50256
  numWorkers : UInt64 := 4
  bufferSize : UInt64 := 8
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-! ## Path Resolution -/

inductive ShardKind where
  | train
  | val

private def shardPrefix : ShardKind → String
  | .train => "fineweb_train_"
  | .val => "fineweb_val_"

private def sortPaths (paths : Array String) : Array String :=
  paths.qsort (· < ·)

private def listShardFilesInDir (dir : System.FilePath) (kind : ShardKind)
    : IO (Array String) := do
  let entries ← dir.readDir
  let mut preferred : Array String := #[]
  let mut anyBin : Array String := #[]
  for e in entries do
    if e.fileName.endsWith ".bin" then
      let path := e.path.toString
      anyBin := anyBin.push path
      if e.fileName.startsWith (shardPrefix kind) then
        preferred := preferred.push path
  let chosen := if preferred.isEmpty then anyBin else preferred
  return sortPaths chosen

/--
Resolve a shard path specification into concrete `.bin` files.

Supports:
1. Exact file path.
2. Directory path (prefers `fineweb_train_*.bin` / `fineweb_val_*.bin`).
3. Prefix path whose parent exists (e.g. `data/fineweb_val` -> `data/fineweb_val_*.bin`).
-/
def resolveShardPaths (pathSpec : String) (kind : ShardKind) : IO (Array String) := do
  let p : System.FilePath := ⟨pathSpec⟩
  if ← p.pathExists then
    if ← p.isDir then
      let files ← listShardFilesInDir p kind
      if files.isEmpty then
        throw <| IO.userError s!"No .bin files found under directory: {pathSpec}"
      return files
    else
      return #[pathSpec]

  let cwd : System.FilePath := ⟨"."⟩
  let parent := p.parent.getD cwd
  let stem := p.fileName.getD pathSpec
  if (← parent.pathExists) && (← parent.isDir) then
    let entries ← parent.readDir
    let mut prefixed : Array String := #[]
    for e in entries do
      if e.fileName.startsWith stem && e.fileName.endsWith ".bin" then
        prefixed := prefixed.push e.path.toString
    let files := sortPaths prefixed
    if !files.isEmpty then
      return files

  throw <| IO.userError s!"Could not resolve shard path: {pathSpec}"

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

/-- fineweb/modded-nanogpt binary format: 256 int32 header words. -/
def finewebHeaderI32Words : UInt64 := 256

/-- Header size in uint16 words (= 1024 bytes = 512 uint16 entries). -/
def finewebHeaderU16Words : UInt64 := finewebHeaderI32Words * 2

/-- Magic/version used by modded-nanogpt fineweb shards. -/
def finewebMagic : UInt64 := 20240520
def finewebVersion : UInt64 := 1

/-- Decode one little-endian u32 value from two u16 words. -/
private def decodeLEU32FromU16Words (lo hi : UInt64) : UInt64 :=
  (lo &&& (0xFFFF : UInt64)) ||| ((hi &&& (0xFFFF : UInt64)) <<< 16)

structure FinewebHeader where
  magic : UInt64
  version : UInt64
  tokenCount : UInt64
  deriving Repr

/-- Parse the leading modded-nanogpt header values (magic, version, tokenCount). -/
def parseFinewebHeader? {n : UInt64} (tokens : T #[n]) : IO (Option FinewebHeader) := do
  if n < 6 then
    return none

  let hdr6 : T #[6] := tokens.slice 0 0 6
  let vals ← data.tensorToUInt64Array' hdr6
  if vals.size < 6 then
    return none

  let magic := decodeLEU32FromU16Words vals[0]! vals[1]!
  let version := decodeLEU32FromU16Words vals[2]! vals[3]!
  let tokenCount := decodeLEU32FromU16Words vals[4]! vals[5]!
  return some { magic, version, tokenCount }

/-- Split a tensor loaded from .bin into payload tokens, parsing fineweb header when present. -/
def splitFinewebPayload {n : UInt64} (tokens : T #[n]) : IO (Σ m, T #[m]) := do
  if n < finewebHeaderU16Words then
    return ⟨n, tokens⟩

  let some header ← parseFinewebHeader? tokens
    | return ⟨n, tokens⟩

  if header.magic != finewebMagic then
    return ⟨n, tokens⟩

  if header.version != finewebVersion then
    throw <| IO.userError s!"Unsupported fineweb shard version {header.version} in header (expected {finewebVersion})"

  let payloadWords := n - finewebHeaderU16Words
  let payloadCount ←
    if header.tokenCount == payloadWords then
      pure header.tokenCount
    else if header.tokenCount > payloadWords then
      IO.eprintln s!"Warning: Fineweb shard appears truncated. header={header.tokenCount}, payload={payloadWords}. Using payload size."
      pure payloadWords
    else
      IO.eprintln s!"Warning: Fineweb shard has trailing payload. header={header.tokenCount}, payload={payloadWords}. Using header size."
      pure header.tokenCount

  let payloadStart := finewebHeaderU16Words
  let payloadEnd := payloadStart + payloadCount
  let payloadRaw := tokens.slice 0 payloadStart.toInt64 payloadEnd.toInt64
  let payload : T #[payloadCount] := reshape payloadRaw #[payloadCount]
  return ⟨payloadCount, payload⟩

def DataShard.loadFromFile (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (Σ n, DataShard n) := do
  let rawTokensCount ← data.binFileTokenCount path
  let rawTokens ← data.loadU16Bin rawTokensCount path
  let ⟨totalTokens, allTokens⟩ ← splitFinewebPayload rawTokens
  let tokensPerShard := totalTokens / numShards
  let startToken := tokensPerShard * shardIdx
  let endToken := if shardIdx == numShards - 1 then totalTokens else startToken + tokensPerShard
  let shardSize := endToken - startToken
  let shardTokens := allTokens.slice 0 startToken.toInt64 endToken.toInt64
  let shardTokens := reshape shardTokens #[shardSize]
  let bosFinder ← BOSFinder.init shardTokens bosToken
  return ⟨shardSize, { tokens := shardTokens, bosFinder, shardIdx, numShards }⟩

def DataShard.load (path : String) (shardIdx numShards : UInt64)
    (bosToken : UInt64) : IO (DataShard defaultShardSize) := do
  let fileExists ← data.fileExists path
  if fileExists then
    let ⟨n, shard⟩ ← DataShard.loadFromFile path shardIdx numShards bosToken
    if n == 0 then
      throw <| IO.userError s!"Data shard is empty: {path}"
    let tokens :=
      if n == defaultShardSize then
        shard.tokens
      else if n > defaultShardSize then
        let sliced := shard.tokens.slice 0 0 defaultShardSize.toInt64
        reshape sliced #[defaultShardSize]
      else
        -- Small fixtures are supported by tiling to the working shard size.
        let reps := (defaultShardSize + n - 1) / n
        let tiledDyn := nn.tensor_repeat (reshape shard.tokens #[]) #[reps]
        let tiled := reshape tiledDyn #[reps * n]
        let sliced := tiled.slice 0 0 defaultShardSize.toInt64
        reshape sliced #[defaultShardSize]
    let bosFinder ← BOSFinder.init tokens bosToken
    return { tokens := tokens, bosFinder, shardIdx, numShards }
  else
    throw <| IO.userError s!"Data shard not found: {path}"

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
  trainPaths : Array String
  trainPathIdx : Nat
  deriving Repr

def DistributedDataGenerator.init (config : Config) (batchSize seqLen : UInt64)
    : IO DistributedDataGenerator := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let trainPaths ← resolveShardPaths config.dataPath .train
  let trainPathIdx := 0
  let shard ← DataShard.load trainPaths[trainPathIdx]! rank worldSize config.bosToken
  let iterator := BatchIterator.new shard batchSize seqLen
  return { iterator, config, globalStep := 0, rank, worldSize, trainPaths, trainPathIdx }

def DistributedDataGenerator.nextBatch (gen : DistributedDataGenerator)
    : IO (Option (T #[]) × DistributedDataGenerator) := do
  let (maybeBatch, newIterator) ← gen.iterator.next
  match maybeBatch with
  | some batch =>
    return (some batch, { gen with iterator := newIterator, globalStep := gen.globalStep + 1 })
  | none =>
    let nextIdx := (gen.trainPathIdx + 1) % gen.trainPaths.size
    let nextPath := gen.trainPaths[nextIdx]!
    let shard ← DataShard.load nextPath gen.rank gen.worldSize gen.config.bosToken
    let iter0 := BatchIterator.new shard newIterator.batchSize newIterator.seqLen
    let (maybeBatch', iter1) ← iter0.next
    return (maybeBatch', {
      gen with
        iterator := iter1
        globalStep := gen.globalStep + 1
        trainPathIdx := nextIdx
    })

def DistributedDataGenerator.batchSize (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.batchSize

def DistributedDataGenerator.seqLen (gen : DistributedDataGenerator) : UInt64 :=
  gen.iterator.seqLen

/-! ## Validation and Utilities -/

def loadValidationData (path : String) (_seqLen : UInt64) (bosToken : UInt64)
    : IO DataShard := do
  let valPaths ← resolveShardPaths path .val
  DataShard.load valPaths[0]! 0 1 bosToken

def estimateRemainingTime (currentStep totalSteps : UInt64)
    (msPerStep : Float) : Float :=
  (totalSteps - currentStep).toFloat * msPerStep / 1000.0 / 60.0

end torch.DataLoader
