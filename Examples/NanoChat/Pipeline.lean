/-
  Examples/NanoChat/Pipeline.lean

  NanoChat training pipeline orchestration.

  This implements the full nanochat training pipeline:
  1. Tokenizer training and evaluation
  2. Base model pretraining
  3. Midtraining (conversation tokens, tool use)
  4. Supervised fine-tuning
  5. Reinforcement learning (optional)

  Based on nanochat's speedrun.sh orchestration.
-/
import Tyr.Pipeline
import Tyr.Distributed
import Tyr.DataLoader
import Tyr.Data.Pretraining
import Tyr.Data.Task
import Tyr.Data.Download
import Tyr.Data.HuggingFace
import Tyr.Tokenizer
import Examples.NanoChat.ModdedGPT
import Examples.NanoChat.ModdedTrain
import Examples.NanoChat.ChatSFT
import Examples.NanoChat.GRPO
import Examples.NanoChat.Eval.CORE
import Examples.NanoChat.Eval.COREData
import Examples.NanoChat.Tasks.LLM

namespace torch.NanoChat.Pipeline

open torch
open torch.Pipeline
open torch.ModdedTrain
open torch.moddedGpt
open torch.Data.Pretraining
open torch.Data.Task
open torch.Data.Download
open torch.Data.HuggingFace
open torch.Train.ChatSFT
open torch.RL.GRPO
open torch.Eval.CORE
open torch.Eval.COREData
open torch.Tasks.LLM
open tokenizer

/-! ## Pipeline Configuration -/

/-- NanoChat-specific pipeline configuration -/
structure NanoChatConfig extends PipelineConfig where
  /-- Model depth (number of transformer layers) -/
  modelDepth : Nat := 20
  /-- Target parameter-to-data ratio (Chinchilla scaling) -/
  paramDataRatio : Nat := 20
  /-- Tokenizer vocabulary size -/
  vocabSize : Nat := 65536
  /-- Maximum characters for tokenizer training -/
  tokenizerMaxChars : Nat := 2000000000
  /-- Number of data shards for pretraining -/
  numDataShards : Nat := 240
  /-- Number of data shards for initial tokenizer training -/
  initialDataShards : Nat := 8
  /-- Path to data directory -/
  dataPath : String := "base_data"
  /-- HuggingFace dataset URL for FineWeb-Edu -/
  dataUrl : String := "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
  /-- Maximum download retry attempts -/
  maxDownloadRetries : Nat := 5
  /-- SFT learning rate for embeddings -/
  sftEmbeddingLr : Float := 0.2
  /-- SFT learning rate for matrix weights -/
  sftMatrixLr : Float := 0.02
  /-- Number of SFT epochs -/
  sftEpochs : Nat := 1
  /-- Number of GRPO samples per prompt -/
  grpoNumSamples : Nat := 16
  /-- GRPO max new tokens -/
  grpoMaxNewTokens : Nat := 256
  /-- Enable reinforcement-learning stages in full pipeline runs. -/
  enableRL : Bool := false
  deriving Repr, Inhabited

/-- Convert to model config -/
def NanoChatConfig.toModelConfig (cfg : NanoChatConfig) : moddedGpt.Config := {
  vocabSize := cfg.vocabSize.toUInt64
  nLayer := cfg.modelDepth.toUInt64
  nHead := 16
  headDim := 64
  modelDim := 1024
  maxSeqLen := 2048
  blockSize := 128
  ropeBase := 10000.0
}

/-- Get full path for data directory -/
def NanoChatConfig.getDataDir (cfg : NanoChatConfig) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/{cfg.dataPath}"

/-- Get tokenizer path -/
def NanoChatConfig.getTokenizerPath (cfg : NanoChatConfig) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/tokenizer"

/-- Get checkpoint directory for a stage -/
def NanoChatConfig.getCheckpointDir (cfg : NanoChatConfig) (stage : String) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/checkpoints/{stage}"

private def resolveDataDirWithOverride (cfg : NanoChatConfig) (envVar : String) : IO String := do
  match ← IO.getEnv envVar with
  | some overridePath =>
    if overridePath.startsWith "/" then
      pure overridePath
    else
      let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
      pure s!"{baseDir}/{overridePath}"
  | none =>
    cfg.getDataDir

private def envNat (name : String) : IO (Option Nat) := do
  pure <| (← IO.getEnv name).bind String.toNat?

private def envUInt64 (name : String) : IO (Option UInt64) := do
  pure <| (← envNat name).map (·.toUInt64)

private def capArray {α : Type} (xs : Array α) (cap? : Option Nat) : Array α :=
  match cap? with
  | some n =>
    let m := min n xs.size
    xs.extract 0 m
  | none => xs

private def runOnMaster (action : PipelineM Unit) : PipelineM Unit := do
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  if rank == 0 then
    action
    if isDistributed then
      dist.barrier
  else
    if isDistributed then
      dist.barrier

private def runOnMasterIO (action : IO Unit) : IO Unit := do
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  if rank == 0 then
    action

/-! ## Device & Data Adapters -/

/-- Get local rank from environment (for torchrun). -/
def getLocalRankFromEnv : IO UInt64 := do
  let envVar ← IO.getEnv "LOCAL_RANK"
  match envVar with
  | some r => pure r.toNat!.toUInt64
  | none => pure 0

/-- Resolve training device from TYR_DEVICE.
    In distributed mode, CUDA ordinal follows LOCAL_RANK (per-node index). -/
def resolveTrainingDevice (_rank localRank : UInt64) (isDistributed : Bool) : IO Device := do
  let cudaOrdinal := if isDistributed then localRank else 0
  let requested? := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested? with
  | some "cpu" => pure Device.CPU
  | some "mps" => pure Device.MPS
  | some "cuda" => pure (Device.CUDA cudaOrdinal)
  | some "auto" | none =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA cudaOrdinal) else pure Device.CPU
    else
      getBestDevice
  | some _ =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA cudaOrdinal) else pure Device.CPU
    else
      getBestDevice

/-- Move YaRN rotary cache tensors to the target device. -/
private def moveYarnToDevice {headDim maxSeqLen : UInt64}
    (yarn : YarnRotary headDim maxSeqLen) (device : Device) : YarnRotary headDim maxSeqLen :=
  { yarn with
    cos := yarn.cos.to device
    sin := yarn.sin.to device
    angularFreq := yarn.angularFreq.to device
  }

/-- Load ARC task from HuggingFace parquet rows. -/
def loadARCFromHuggingFace (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO ARCTask := do
  let rows ← loadARC subset split cacheDir
  let examples := rows.filterMap ARCExample.fromJson?
  pure { subset, split, examples, config := {} }

/-- Load GSM8K task from HuggingFace parquet rows. -/
def loadGSM8KFromHuggingFace (subset : String := "main") (split : String := "train")
    (cacheDir : String := "~/.cache/huggingface") : IO GSM8KTask := do
  let rows ← loadGSM8K subset split cacheDir
  let examples := rows.filterMap GSM8KExample.fromJson?
  pure { subset, split, examples, config := {} }

/-- Create spelling-bee task after ensuring the shared word list is available. -/
def createSpellingBeeTaskWithDownload (cacheDir : String) (size : Nat) (split : String) : IO SpellingBeeTask := do
  let wordListPath ← ensureWordList cacheDir
  createSpellingBeeTask ⟨wordListPath⟩ size split

/-- Create simple-spelling task after ensuring the shared word list is available. -/
def createSimpleSpellingTaskWithDownload (cacheDir : String) (size : Nat) (split : String) : IO SimpleSpellingTask := do
  let wordListPath ← ensureWordList cacheDir
  createSimpleSpellingTask ⟨wordListPath⟩ size split

/-! ## Data Download with Retry Logic -/

/-- Download a single file with exponential backoff retry -/
def downloadSingleFile (url : String) (filepath : String) (maxAttempts : Nat := 5) : IO Bool := do
  -- Check if file already exists
  if ← System.FilePath.pathExists ⟨filepath⟩ then
    IO.println s!"  Skipping (already exists): {filepath}"
    return true

  IO.println s!"  Downloading: {url}"

  for attempt in [1:maxAttempts+1] do
    let tempPath := s!"{filepath}.tmp"
    let result ← IO.Process.output {
      cmd := "curl"
      args := #["-fsSL", "-o", tempPath, url]
    }

    if result.exitCode == 0 then
      -- Move temp file to final location
      IO.FS.rename ⟨tempPath⟩ ⟨filepath⟩
      IO.println s!"  Downloaded successfully: {filepath}"
      return true
    else
      IO.println s!"  Attempt {attempt}/{maxAttempts} failed: {result.stderr}"
      -- Clean up temp file if it exists
      if ← System.FilePath.pathExists ⟨tempPath⟩ then
        IO.FS.removeFile ⟨tempPath⟩

      if attempt < maxAttempts then
        -- Exponential backoff: 2^attempt seconds
        let waitTime := (2 ^ attempt) * 1000
        IO.println s!"  Waiting {waitTime / 1000} seconds before retry..."
        IO.sleep waitTime.toUInt32

  IO.println s!"  Failed to download after {maxAttempts} attempts: {filepath}"
  return false

/-- Left-pad a string with zeros -/
def leftPadZeros (s : String) (width : Nat) : String :=
  let padding := String.ofList (List.replicate (width - s.length) '0')
  padding ++ s

/-- Download data shards with progress tracking -/
def downloadDataShards (cfg : NanoChatConfig) (numShards : Nat) : IO Unit := do
  let dataDir ← cfg.getDataDir
  IO.FS.createDirAll ⟨dataDir⟩

  IO.println s!"Downloading {numShards} shards to {dataDir}..."

  let mut successful := 0
  let mut failed := 0

  for i in [:numShards] do
    let filename := s!"shard_{leftPadZeros (toString i) 5}.parquet"
    let filepath := s!"{dataDir}/{filename}"
    let url := s!"{cfg.dataUrl}/{filename}"

    if ← downloadSingleFile url filepath cfg.maxDownloadRetries then
      successful := successful + 1
    else
      failed := failed + 1

    -- Progress update every 10 shards
    if (i + 1) % 10 == 0 then
      IO.println s!"Progress: {i + 1}/{numShards} shards processed"

  IO.println s!"Download complete: {successful} succeeded, {failed} failed"

  if failed > 0 then
    throw $ IO.userError s!"Failed to download {failed} shards"

/-! ## Tokenizer Training -/

/-- Load text from parquet files for tokenizer training -/
def loadParquetTexts (dataDir : String) (maxChars : Nat) : IO (Array String) := do
  let files ← listParquetFiles dataDir

  if files.isEmpty then
    throw $ IO.userError s!"No parquet files found in {dataDir}"

  IO.println s!"Found {files.size} parquet files for tokenizer training"

  let mut texts : Array String := #[]
  let mut totalChars := 0

  for file in files do
    if totalChars >= maxChars then break

    let metadata ← getParquetMetadata file
    IO.println s!"  Processing {file}: {metadata.numRowGroups} row groups"

    for rgIdx in [:metadata.numRowGroups] do
      if totalChars >= maxChars then break

      let rgData ← readRowGroup file rgIdx.toUInt64 "text"
      for doc in rgData.documents do
        if totalChars >= maxChars then break
        texts := texts.push doc
        totalChars := totalChars + doc.length

  IO.println s!"Loaded {texts.size} documents, {totalChars} characters"
  return texts

/-- Train tokenizer on parquet data -/
def trainTokenizer (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log s!"Training tokenizer with vocab_size={cfg.vocabSize} on up to {cfg.tokenizerMaxChars} chars..."

  let dataDir ← cfg.getDataDir
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Check if tokenizer already exists
  if ← System.FilePath.pathExists ⟨tokenizerFile⟩ then
    log "Tokenizer already exists, skipping training"
    recordMetrics [("status", "cached")]
    return

  -- Load text from parquet files
  let texts ← loadParquetTexts dataDir cfg.tokenizerMaxChars

  -- Create training config
  let trainConfig : TrainConfig := {
    vocabSize := cfg.vocabSize
    maxChars := cfg.tokenizerMaxChars
    specialTokens := defaultChatSpecialTokens
  }

  -- Train the tokenizer
  let result ← tokenizer.trainBPE texts trainConfig

  -- Create tokenizer directory and save
  IO.FS.createDirAll ⟨tokenizerPath⟩
  log s!"Saving tokenizer to {tokenizerFile}"
  tokenizer.save result.tokenizer tokenizerFile

  recordMetrics [
    ("vocab_size", toString result.stats.finalVocabSize),
    ("total_chars", toString result.stats.totalChars),
    ("num_merges", toString result.stats.numMerges),
    ("compression_ratio", toString result.stats.compressionRatio)
  ]

/-- Evaluate tokenizer compression ratio -/
def evalTokenizer (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Evaluating tokenizer compression ratio..."

  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load the trained tokenizer
  let tok ← tokenizer.load tokenizerFile

  -- Load a sample of text for evaluation
  let dataDir ← cfg.getDataDir
  let sampleTexts ← loadParquetTexts dataDir 10000000  -- 10M chars sample

  -- Count total chars and tokens using actual tokenization
  let mut totalChars := 0
  let mut totalTokens := 0

  for text in sampleTexts do
    totalChars := totalChars + text.length
    totalTokens := totalTokens + tokenizer.countTokens tok text

  let compressionRatio := if totalTokens > 0
    then totalChars.toFloat / totalTokens.toFloat
    else 0.0

  recordMetrics [
    ("compression_ratio", toString compressionRatio),
    ("sample_chars", toString totalChars),
    ("sample_tokens", toString totalTokens),
    ("vocab_size", toString tok.vocabSize)
  ]

/-! ## Pretraining Stage -/

/-- Pretrain base model using ModdedTrain infrastructure -/
def pretrainBaseModel (cfg : NanoChatConfig) : PipelineM Unit := do
  log s!"Pretraining d{cfg.modelDepth} model..."

  let modelCfg := cfg.toModelConfig
  let defaultHp := Hyperparameters.default
  let hp : Hyperparameters := {
    defaultHp with
    numIterations := (← envUInt64 "PRETRAIN_ITERS").getD defaultHp.numIterations
    extensionIterations := (← envUInt64 "PRETRAIN_EXTENSION_ITERS").getD defaultHp.extensionIterations
    valInterval := (← envUInt64 "PRETRAIN_VAL_INTERVAL").getD defaultHp.valInterval
    logInterval := (← envUInt64 "PRETRAIN_LOG_INTERVAL").getD defaultHp.logInterval
    checkpointInterval := (← envUInt64 "PRETRAIN_CHECKPOINT_INTERVAL").getD defaultHp.checkpointInterval
  }
  let dataDir ← resolveDataDirWithOverride cfg "PRETRAIN_DATA_PATH"
  let checkpointDir ← cfg.getCheckpointDir "base"
  IO.FS.createDirAll ⟨checkpointDir⟩

  -- Create data config
  let dataConfig : DataLoader.Config := {
    dataPath := dataDir
    bosToken := 0  -- <|bos|> token ID
    seqLen := hp.maxSeqLen
  }

  -- Check for existing checkpoint to resume
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let hasCheckpoint ← checkpointExists ckptPath

  -- Initialize or resume training state
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed
  let isMaster := rank == 0

  -- Run distributed training
  if isMaster then
    log s!"Starting training with {worldSize} GPUs"
    log s!"Data path: {dataDir}"
    log s!"Pretrain schedule: iters={hp.numIterations}, ext={hp.extensionIterations}, val_int={hp.valInterval}, ckpt_int={hp.checkpointInterval}"
    if hasCheckpoint then
      log s!"Resuming base pretraining from {ckptPath}"

  let _finalState ← trainDistributed modelCfg hp dataConfig trainDevice checkpointDir none

  -- Estimate parameters for logging
  let numParams := modelCfg.vocabSize.toNat * modelCfg.modelDim.toNat +  -- embeddings
                   modelCfg.nLayer.toNat * (4 * modelCfg.modelDim.toNat * modelCfg.modelDim.toNat)  -- approx
  let targetTokens := numParams * cfg.paramDataRatio

  recordMetrics [
    ("model_params", s!"{numParams / 1000000}M"),
    ("target_tokens", s!"{targetTokens / 1000000000}B"),
    ("num_layers", toString cfg.modelDepth),
    ("checkpoint_dir", checkpointDir)
  ]

/-- Evaluate base model on CORE benchmark -/
def evalBaseModel (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Evaluating base model on CORE tasks..."

  let modelCfg := cfg.toModelConfig
  let checkpointDir ← cfg.getCheckpointDir "base"
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load tokenizer for evaluation
  let tok ← tokenizer.load tokenizerFile

  -- Load model checkpoint
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath

  match maybeCkpt with
  | none =>
    log "Warning: No checkpoint found, skipping evaluation"
    recordMetrics [("CORE", "N/A"), ("checkpoint", "not found")]
  | some ckpt =>
    log s!"Loaded checkpoint from {ckptPath} at step {ckpt.step}"
    let evalDevice := ckpt.params.embed.device

    -- Create tokenization function
    let tokenizeFn := fun (text : String) =>
      (tokenizer.encode tok text).map (·.toUInt64)

    -- Initialize YaRN rotary embeddings for evaluation
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase
    let yarn := moveYarnToDevice yarn evalDevice

    -- Create model forward function for evaluation
    let runModel := fun (input : T #[]) => do
      let shape := input.runtimeShape
      let batch := shape.getD 0 1
      let seq := shape.getD 1 1
      let inputTyped : T #[batch, seq] := cast rfl (input.to evalDevice)
      let logits ← moddedGpt.forward (batch := batch) (seq := seq) ckpt.params yarn inputTyped
      pure (cast rfl logits : T #[])

    -- Evaluate on CORE tasks
    let evalConfig : EvalConfig := {
      bosToken := 0  -- <|bos|>
      maxSeqLen := some modelCfg.maxSeqLen
      device := evalDevice
      parallelAcrossRanks := false
    }

    let mut taskResults : Array (String × Float) := #[]

    -- Load CORE eval data
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let evalBundleDir ← ensureCOREData cacheDir
    let coreConfigs ← loadCOREConfig evalBundleDir
    let evalCap := (← envNat "CORE_MAX_EXAMPLES")

    for taskMeta in coreTasks do
      log s!"  Evaluating {taskMeta.taskName}..."
      -- Load task-specific data from CORE bundle
      let examplesRaw ← match coreTaskMetaToConfig taskMeta coreConfigs with
        | some tc => loadCORETaskData evalBundleDir tc
        | none => pure #[]
      let examples := capArray examplesRaw evalCap
      let score ← evaluateTask examples taskMeta tokenizeFn runModel evalConfig
      taskResults := taskResults.push (taskMeta.taskName, score)

    let coreScore := computeCOREScore taskResults

    recordMetrics [
      ("CORE", toString coreScore),
      ("checkpoint", checkpointDir),
      ("step", toString ckpt.step)
    ]

/-! ## Midtraining Stage -/

/-- Create task mixture for midtraining -/
def createMidtrainTaskMixture (cfg : NanoChatConfig) : IO AnyTaskMixture := do
  -- Get cache directory
  let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))

  IO.println "Creating midtraining task mixture..."

  -- Load ARC tasks from HuggingFace
  let arcEasy ← loadARCFromHuggingFace "ARC-Easy" "train" cacheDir
  let arcChallenge ← loadARCFromHuggingFace "ARC-Challenge" "train" cacheDir

  -- Load GSM8K from HuggingFace
  let gsm8k ← loadGSM8KFromHuggingFace "main" "train" cacheDir

  -- Create spelling tasks with downloaded word list
  let spellingBee ← createSpellingBeeTaskWithDownload cacheDir 300 "train"
  let simpleSpelling ← createSimpleSpellingTaskWithDownload cacheDir 300 "train"

  let entries : Array AnyMixtureEntry := #[
    { task := .arc arcEasy, weight := 1 },
    { task := .arc arcChallenge, weight := 1 },
    { task := .gsm8k gsm8k, weight := 2 },
    { task := .spellingBee spellingBee, weight := 1 },
    { task := .simpleSpelling simpleSpelling, weight := 1 }
  ]

  IO.println s!"Created mixture with {entries.size} tasks"
  return AnyTaskMixture.create entries

/-- Run midtraining with conversation data and task mixture -/
def midtrain (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running midtraining (conversation tokens, tool use)..."

  let modelCfg := cfg.toModelConfig
  let baseCheckpointDir ← cfg.getCheckpointDir "base"
  let midCheckpointDir ← cfg.getCheckpointDir "mid"
  IO.FS.createDirAll ⟨midCheckpointDir⟩

  -- Midtraining hyperparameters (lower LR, shorter training)
  let midDefaultHp : Hyperparameters := {
    Hyperparameters.default with
    numIterations := 500
    extensionIterations := 0
    cooldownFrac := 0.3
    valInterval := 25
    logInterval := 5
  }
  let hp : Hyperparameters := {
    midDefaultHp with
    numIterations := (← envUInt64 "MIDTRAIN_ITERS").getD midDefaultHp.numIterations
    extensionIterations := (← envUInt64 "MIDTRAIN_EXTENSION_ITERS").getD midDefaultHp.extensionIterations
    valInterval := (← envUInt64 "MIDTRAIN_VAL_INTERVAL").getD midDefaultHp.valInterval
    logInterval := (← envUInt64 "MIDTRAIN_LOG_INTERVAL").getD midDefaultHp.logInterval
    checkpointInterval := (← envUInt64 "MIDTRAIN_CHECKPOINT_INTERVAL").getD midDefaultHp.checkpointInterval
  }

  -- Create data config for task data
  -- TODO: Replace with task mixture data (conversation tokens, tool use)
  let dataDir ← resolveDataDirWithOverride cfg "MIDTRAIN_DATA_PATH"
  let dataConfig : DataLoader.Config := {
    dataPath := dataDir
    bosToken := 0
    seqLen := hp.maxSeqLen
  }

  -- Resume policy:
  -- 1) If mid checkpoint exists, continue from it.
  -- 2) Otherwise bootstrap midtraining from base/latest checkpoint.
  let baseCkptPath := s!"{baseCheckpointDir}/latest.ckpt"
  let midCkptPath := s!"{midCheckpointDir}/latest.ckpt"
  let hasMid ← checkpointExists midCkptPath
  let hasBase ← checkpointExists baseCkptPath

  if !hasMid && !hasBase then
    throw <| IO.userError s!"Base checkpoint not found at {baseCkptPath}"

  let resumePath : Option String :=
    if hasMid then none else some baseCkptPath

  -- Run training loop
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed
  let isMaster := rank == 0

  if isMaster then
    log s!"Starting midtraining with {worldSize} GPUs"
    log s!"Midtrain schedule: iters={hp.numIterations}, ext={hp.extensionIterations}, val_int={hp.valInterval}, ckpt_int={hp.checkpointInterval}"
    if hasMid then
      log s!"Resuming midtraining from {midCkptPath}"
    else
      log s!"Bootstrapping midtraining from base checkpoint {baseCkptPath}"

  let finalState ← trainDistributed modelCfg hp dataConfig trainDevice midCheckpointDir resumePath

  -- If no new step was run (e.g. resume already at horizon), ensure mid/latest exists.
  if isMaster && !(← checkpointExists midCkptPath) then
    let fallbackCkpt : Checkpoint modelCfg := {
      params := finalState.params
      optState := finalState.optState
      step := finalState.step
      bestValLoss := finalState.bestValLoss
    }
    saveCheckpoint fallbackCkpt midCkptPath
    log s!"Materialized fallback mid checkpoint at {midCkptPath}"

  recordMetrics [
    ("steps", toString hp.numIterations),
    ("base_checkpoint", baseCheckpointDir),
    ("mid_checkpoint", midCheckpointDir)
  ]

/-- Evaluate chat model at a checkpoint -/
def evalChat (cfg : NanoChatConfig) (checkpoint : String) : PipelineM Unit := runOnMaster do
  log s!"Evaluating chat model ({checkpoint})..."

  let modelCfg := cfg.toModelConfig
  let checkpointDir ← cfg.getCheckpointDir checkpoint
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load tokenizer
  let tok ← tokenizer.load tokenizerFile

  -- Load checkpoint
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath

  match maybeCkpt with
  | none =>
    log s!"Warning: No checkpoint found at {ckptPath}"
    recordMetrics [("status", "checkpoint not found")]
  | some ckpt =>
    log s!"Loaded checkpoint at step {ckpt.step}"
    let evalDevice := ckpt.params.embed.device

    -- Create tokenization function
    let tokenizeFn := fun (text : String) =>
      (tokenizer.encode tok text).map (·.toUInt64)

    -- Initialize model inference components
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase
    let yarn := moveYarnToDevice yarn evalDevice

    -- Create model forward function for evaluation
    let runModel := fun (input : T #[]) => do
      let shape := input.runtimeShape
      let batch := shape.getD 0 1
      let seq := shape.getD 1 1
      let inputTyped : T #[batch, seq] := cast rfl (input.to evalDevice)
      let logits ← moddedGpt.forward (batch := batch) (seq := seq) ckpt.params yarn inputTyped
      pure (cast rfl logits : T #[])

    -- Evaluate on multiple task categories
    let evalConfig : EvalConfig := {
      bosToken := 0
      maxSeqLen := some modelCfg.maxSeqLen
      device := evalDevice
      parallelAcrossRanks := false
    }

    let mut results : Array (String × Float) := #[]

    -- Define chat eval tasks
    let chatTasks : Array TaskMeta := #[
      { taskType := .multipleChoice, numFewshot := 0, taskName := "ARC-Easy" },
      { taskType := .multipleChoice, numFewshot := 0, taskName := "ARC-Challenge" },
      { taskType := .multipleChoice, numFewshot := 5, taskName := "MMLU" },
      { taskType := .languageModeling, numFewshot := 0, taskName := "GSM8K" },
      { taskType := .languageModeling, numFewshot := 0, taskName := "HumanEval" }
    ]

    -- Load CORE eval data for chat evaluation
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let evalBundleDir ← ensureCOREData cacheDir
    let coreConfigs ← loadCOREConfig evalBundleDir
    let evalCap := (← envNat "CORE_MAX_EXAMPLES")

    for taskMeta in chatTasks do
      log s!"  Evaluating {taskMeta.taskName}..."
      -- Load task-specific data from CORE bundle
      let examplesRaw ← match coreTaskMetaToConfig taskMeta coreConfigs with
        | some tc => loadCORETaskData evalBundleDir tc
        | none => pure #[]
      let examples := capArray examplesRaw evalCap
      let score ← evaluateTask examples taskMeta tokenizeFn runModel evalConfig
      results := results.push (taskMeta.taskName, score)

    -- Compute aggregate ChatCORE score
    let chatCoreScore := computeCOREScore results

    let resultsList := results.map (fun (name, score) => (name, toString score)) |>.toList
    recordMetrics (resultsList ++ [("ChatCORE", toString chatCoreScore), ("checkpoint", checkpointDir)])

/-! ## Supervised Fine-tuning Stage -/

/-- Convert ConversationBatch to SFTBatch for training -/
def convertConversationBatchToSFT (batch : ConversationBatch) : IO SFTBatch := do
  -- Get shape info
  let batchSize := batch.lengths.size.toUInt64
  let seqLen := if batch.lengths.isEmpty then 0
                else batch.lengths.foldl max 0 |>.toUInt64
  let numValidTokens := batch.lengths.foldl (· + ·) 0

  -- SFTBatch wraps the token and mask tensors
  return {
    inputs := batch.tokens
    targets := batch.tokens  -- Shifted internally by loss function
    mask := batch.mask
    batchSize := batchSize
    seqLen := seqLen
    numValidTokens := numValidTokens
  }

/-- Run supervised fine-tuning using ChatSFT infrastructure -/
def supervisedFineTune (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Running supervised fine-tuning..."

  let modelCfg := cfg.toModelConfig
  let midCheckpointDir ← cfg.getCheckpointDir "mid"
  let sftCheckpointDir ← cfg.getCheckpointDir "sft"
  IO.FS.createDirAll ⟨sftCheckpointDir⟩
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed

  -- SFT configuration
  let sftCfg : SFTConfig := {
    numEpochs := cfg.sftEpochs
    deviceBatchSize := 4
    targetExamplesPerStep := 32
    embeddingLr := cfg.sftEmbeddingLr
    matrixLr := cfg.sftMatrixLr
    maxSeqLen := 2048
    evalEvery := 100
    logInterval := 10
    device := trainDevice
  }

  let deviceStr := match trainDevice with
    | Device.CPU => "cpu"
    | Device.MPS => "mps"
    | Device.CUDA idx => s!"cuda:{idx}"
  log s!"SFT config: epochs={sftCfg.numEpochs}, batch={sftCfg.deviceBatchSize}"
  log s!"Learning rates: embed={sftCfg.embeddingLr}, matrix={sftCfg.matrixLr}"
  log s!"SFT device: {deviceStr}"

  -- Load mid checkpoint
  let ckptPath := s!"{midCheckpointDir}/latest.ckpt"
  log s!"Loading midtraining checkpoint from {ckptPath}"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath

  match maybeCkpt with
  | none =>
    throw $ IO.userError s!"Midtraining checkpoint not found at {ckptPath}"
  | some midCkpt =>
    log s!"Loaded checkpoint at step {midCkpt.step}"
    let paramsOnDeviceRaw ← TensorStruct.mapM (fun t => pure (t.to trainDevice)) midCkpt.params
    let paramsOnDevice := TensorStruct.makeLeafParams paramsOnDeviceRaw

    -- Initialize YaRN rotary embeddings
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase
    let yarn := moveYarnToDevice yarn trainDevice

    -- Create forward function for SFT
    -- Note: T s is the same type for all s (TSpec.type), so we can cast between shapes
    let forwardFn := fun (params : ModdedGPTParams modelCfg) (input : T #[]) => do
      -- Get runtime shape to determine batch and sequence length
      let shape := input.runtimeShape
      let batch := shape.getD 0 1
      let seq := shape.getD 1 1
      -- Call forward with explicit shape parameters using cast
      let inputTyped : T #[batch, seq] := cast rfl (input.to trainDevice)
      let logits ← moddedGpt.forward (batch := batch) (seq := seq) params yarn inputTyped
      pure (cast rfl logits : T #[])

    -- Create masked loss function
    let lossFn := fun (logits : T #[]) (targets : T #[]) (mask : T #[]) =>
      let shape := targets.runtimeShape
      let batch := shape.getD 0 1
      let seq := shape.getD 1 1
      let logitsReshaped : T #[batch, seq, modelCfg.vocabSize] := cast rfl logits
      let targetsReshaped : T #[batch, seq] := cast rfl targets
      let maskReshaped : T #[batch, seq] := cast rfl mask
      maskedCrossEntropy logitsReshaped targetsReshaped maskReshaped

    -- Initialize AdamW optimizer
    let opt := Optim.adamw (lr := sftCfg.embeddingLr) (b1 := 0.9) (b2 := 0.999)
    let optState := opt.init paramsOnDevice

    -- Load tokenizer
    let tokenizerPath ← cfg.getTokenizerPath
    let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
    let tok ← tokenizer.load tokenizerFile

    -- Create task mixture from HuggingFace data
    let sftMixture ← createMidtrainTaskMixture cfg

    -- Convert AnyTaskMixture to LoadedTask-based TaskMixture
    let loadedEntries : Array MixtureEntry ← sftMixture.entries.mapM fun entry => do
      -- Get all conversations from this task
      let taskSize := entry.task.size
      let mut convs : Array Conversation := #[]
      for i in [:taskSize] do
        if let some conv := entry.task.getExample i then
          convs := convs.push conv
      let loadedTask : LoadedTask := {
        name := match entry.task with
          | .arc t => s!"arc_{t.subset}"
          | .mmlu t => s!"mmlu_{t.subset}"
          | .gsm8k _ => "gsm8k"
          | .spellingBee _ => "spelling_bee"
          | .simpleSpelling _ => "simple_spelling"
          | .humanEval _ => "humaneval"
        conversations := convs
        config := {}
      }
      return { task := loadedTask, weight := entry.weight }

    let taskMixture := TaskMixture.create loadedEntries

    -- Create chat tokens from tokenizer special tokens
    let chatTokens : ChatTokens := {
      bos := 0  -- <|bos|>
      userStart := 1
      userEnd := 2
      assistantStart := 3
      assistantEnd := 4
      systemStart := 5
      systemEnd := 6
      toolStart := 7
      toolEnd := 8
      outputStart := 9
      outputEnd := 10
    }

    -- Create encode function
    let encode := fun (text : String) => (tokenizer.encode tok text).map (·.toUInt64)

    -- Create task iterator
    let taskIterator := TaskIterator.new taskMixture sftCfg.deviceBatchSize sftCfg.maxSeqLen chatTokens encode
    let iterRef ← IO.mkRef taskIterator

    -- Create data generator that yields SFTBatches
    let trainDataFn : IO SFTBatch := do
      let iter ← iterRef.get
      let (maybeBatch, newIter) ← iter.nextBatch
      iterRef.set newIter
      match maybeBatch with
      | none =>
        -- Epoch ended, reset iterator
        let resetIter := newIter.reset
        let (batch, newerIter) ← resetIter.nextBatch
        iterRef.set newerIter
        match batch with
        | none => pure SFTBatch.empty
        | some b => convertConversationBatchToSFT b
      | some batch =>
        convertConversationBatchToSFT batch

    -- Get number of training examples from mixture (optionally capped for fast validation runs)
    let totalTrainExamples := taskMixture.size
    let sftMaxExamples := (← envNat "SFT_MAX_EXAMPLES")
    let numTrainExamples :=
      match sftMaxExamples with
      | some n => max 1 (min n totalTrainExamples)
      | none => totalTrainExamples
    log s!"Training on {numTrainExamples}/{totalTrainExamples} task examples"

    -- Run SFT training loop
    log "Starting SFT training..."
    let (finalParams, finalOptState, finalState) ← trainLoop sftCfg paramsOnDevice optState
      forwardFn lossFn trainDataFn none numTrainExamples

    log s!"SFT complete: {finalState.totalTokens} tokens trained"

    -- Save SFT checkpoint
    if rank == 0 then
      let sftOptState : OptimizerState modelCfg := {
        midCkpt.optState with
        adamState := finalOptState
        step := finalState.step.toUInt64
      }
      let sftCkpt : Checkpoint modelCfg := {
        params := finalParams
        optState := sftOptState
        step := finalState.step.toUInt64
        bestValLoss := finalState.bestValLoss
      }
      saveCheckpoint sftCkpt s!"{sftCheckpointDir}/latest.ckpt"

    recordMetrics [
      ("epochs", toString sftCfg.numEpochs),
      ("batch_size", toString sftCfg.deviceBatchSize),
      ("embedding_lr", toString sftCfg.embeddingLr),
      ("matrix_lr", toString sftCfg.matrixLr),
      ("total_tokens", toString finalState.totalTokens),
      ("best_val_loss", toString finalState.bestValLoss),
      ("mid_checkpoint", midCheckpointDir),
      ("sft_checkpoint", sftCheckpointDir)
    ]

/-! ## Reinforcement Learning Stage -/

/-- Run GRPO reinforcement learning on GSM8K -/
def reinforcementLearning (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Running reinforcement learning (GRPO on GSM8K)..."

  let modelCfg := cfg.toModelConfig
  let sftCheckpointDir ← cfg.getCheckpointDir "sft"
  let rlCheckpointDir ← cfg.getCheckpointDir "rl"
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
  IO.FS.createDirAll ⟨rlCheckpointDir⟩
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed

  let grpoExamplesPerStep := max 1 <| (← envNat "GRPO_EXAMPLES_PER_STEP").getD 16
  let grpoEvalEvery := max 1 <| (← envNat "GRPO_EVAL_EVERY").getD 100
  let grpoLogEvery := max 1 <| (← envNat "GRPO_LOG_EVERY").getD 10
  let grpoEpochs := max 1 <| (← envNat "GRPO_EPOCHS").getD 3
  let grpoMaxPrompts := (← envNat "GRPO_MAX_PROMPTS")

  -- GRPO configuration
  let grpoCfg : GRPOConfig := {
    numSamples := cfg.grpoNumSamples
    maxNewTokens := cfg.grpoMaxNewTokens
    temperature := 1.0
    topK := 50
    examplesPerStep := grpoExamplesPerStep
    evalEvery := grpoEvalEvery
    logEvery := grpoLogEvery
  }

  log s!"GRPO config: samples={grpoCfg.numSamples}, max_tokens={grpoCfg.maxNewTokens}, examples_per_step={grpoCfg.examplesPerStep}, eval_every={grpoCfg.evalEvery}, log_every={grpoCfg.logEvery}, epochs={grpoEpochs}"

  -- Load tokenizer
  let tok ← tokenizer.load tokenizerFile

  -- Load SFT checkpoint
  let ckptPath := s!"{sftCheckpointDir}/latest.ckpt"
  log s!"Loading SFT checkpoint from {ckptPath}"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath

  match maybeCkpt with
  | none =>
    throw $ IO.userError s!"SFT checkpoint not found at {ckptPath}"
  | some sftCkpt =>
    log s!"Loaded checkpoint at step {sftCkpt.step}"
    let paramsOnDeviceRaw ← TensorStruct.mapM (fun t => pure (t.to trainDevice)) sftCkpt.params
    let paramsOnDevice := TensorStruct.makeLeafParams paramsOnDeviceRaw
    let adamStateOnDevice ← TensorStruct.mapM (fun t => pure (t.to trainDevice)) sftCkpt.optState.adamState

    -- Initialize model components
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase
    let yarn := moveYarnToDevice yarn trainDevice

    -- Create decode function
    let decodeFn := fun (tokens : Array UInt64) =>
      tokenizer.decode tok (tokens.map (·.toUInt32))

    -- Create generation function (single token sampling)
    let generateOneFn := fun (params : ModdedGPTParams modelCfg) (context : Array UInt64) => do
      let inputTensor := data.fromInt64Array (context.map (·.toInt64))
      let inputReshaped := (reshape inputTensor #[1, context.size.toUInt64]).to trainDevice
      let logits ← moddedGpt.forward params yarn inputReshaped
      -- Flatten logits and extract last position's vocab slice
      let logitsFlat := reshape logits #[context.size.toUInt64 * modelCfg.vocabSize]
      let startIdx := ((context.size - 1).toUInt64 * modelCfg.vocabSize).toInt64
      let endIdx := (context.size.toUInt64 * modelCfg.vocabSize).toInt64
      let lastLogits := data.slice1d' logitsFlat startIdx endIdx
      let rowLogits := reshape lastLogits #[1, modelCfg.vocabSize]
      let scaled := if grpoCfg.temperature == 1.0 then rowLogits
                    else mul_scalar rowLogits (1.0 / grpoCfg.temperature)
      let filtered := if grpoCfg.topK == 0 then scaled
                      else nn.topKFilter scaled grpoCfg.topK.toUInt64
      let probs := nn.softmax filtered (-1)
      let sampled ← nn.multinomial probs 1
      let sampled := nn.squeezeDim sampled (-1)
      return (nn.itemInt sampled).toUInt64

    -- Create forward function for policy gradient
    -- Returns per-token NLL: (input, target) → nll_per_token
    let forwardFn := fun (params : ModdedGPTParams modelCfg) (input : T #[1, 256]) (target : T #[1, 256]) => do
      let logits ← moddedGpt.forward params yarn (input.to trainDevice)
      -- Compute per-token cross-entropy (NLL)
      let logitsFlat := reshape logits #[256, modelCfg.vocabSize]
      let targetFlat := reshape (target.to trainDevice) #[256]
      let nllFlat := nn.cross_entropy_none logitsFlat targetFlat
      return reshape nllFlat #[1, 256]

    -- Load GSM8K prompts from HuggingFace
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let gsm8kTask ← loadGSM8KFromHuggingFace "main" "train" cacheDir

    -- Convert GSM8K examples to (tokenized_prompt, question, expected_answer) tuples
    let gsm8kPromptsAll : Array (Array UInt64 × String × String) :=
      gsm8kTask.examples.filterMap fun ex => do
        let promptTokens := (tokenizer.encode tok ex.question).map (·.toUInt64)
        -- Extract the numerical answer after "#### " marker
        let expectedAnswer ← extractHashAnswer ex.answer
        some (promptTokens, ex.question, expectedAnswer)

    let gsm8kPrompts :=
      match grpoMaxPrompts with
      | some n =>
        let cap := min n gsm8kPromptsAll.size
        gsm8kPromptsAll.extract 0 cap
      | none => gsm8kPromptsAll

    if gsm8kPrompts.isEmpty then
      throw <| IO.userError "No GSM8K prompts available for GRPO"

    log s!"Loaded {gsm8kPrompts.size} GSM8K prompts for GRPO"

    -- Math reward function (from GRPO module)
    let rewardFn := fun (expected : String) (response : String) =>
      (mathReward expected response).reward

    -- Run GRPO training
    log "Starting GRPO training..."
    let numEpochs := grpoEpochs  -- Multiple passes over GSM8K

    let initAdamState := adamStateOnDevice
    let (finalParams, finalAdamState, result) ← trainOnPromptsWithUpdates 1 256
      gsm8kPrompts
      generateOneFn
      forwardFn
      paramsOnDevice
      initAdamState
      decodeFn
      rewardFn
      grpoCfg
      numEpochs

    log s!"GRPO complete: best pass@1 = {result.bestPass1}"

    -- Save RL checkpoint
    if rank == 0 then
      let rlOptState : OptimizerState modelCfg := {
        sftCkpt.optState with
        adamState := finalAdamState
        step := result.totalSteps.toUInt64
      }
      let rlCkpt : Checkpoint modelCfg := {
        params := finalParams
        optState := rlOptState
        step := result.totalSteps.toUInt64
        bestValLoss := sftCkpt.bestValLoss
      }
      saveCheckpoint rlCkpt s!"{rlCheckpointDir}/latest.ckpt"

    recordMetrics [
      ("num_samples", toString grpoCfg.numSamples),
      ("max_new_tokens", toString grpoCfg.maxNewTokens),
      ("temperature", toString grpoCfg.temperature),
      ("best_pass1", toString result.bestPass1),
      ("final_pass1", toString result.finalPass1),
      ("total_steps", toString result.totalSteps),
      ("sft_checkpoint", sftCheckpointDir),
      ("rl_checkpoint", rlCheckpointDir)
    ]

/-! ## Utility Functions -/

/-- Print banner (master only) -/
def printBanner : PipelineM Unit := do
  log "                                                       █████                █████"
  log "                                                      ░░███                ░░███"
  log "     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████"
  log "    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░"
  log "     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███"
  log "     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███"
  log "     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████"
  log "    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░"

/-- Estimate training cost based on GPU type and duration -/
def estimateCost (gpuType : String) (numGpus : Nat) (hoursEstimate : Float) : PipelineM Unit := do
  let hourlyRate := match gpuType with
    | "H100" => 3.00
    | "A100" => 1.79
    | "V100" => 0.55
    | _ => 2.00
  let totalCost := hourlyRate * numGpus.toFloat * hoursEstimate
  log s!"Estimated cost: ${totalCost} ({numGpus}x {gpuType} @ ${hourlyRate}/GPU/hour)"

/-! ## Full Pipeline -/

/-- Run the complete NanoChat training pipeline -/
def runNanoChatPipeline (cfg : NanoChatConfig) : PipelineM Unit := do
  printBanner

  log ""
  log "Starting NanoChat training pipeline..."
  log s!"Model: d{cfg.modelDepth}, vocab: {cfg.vocabSize}"
  log s!"Base directory: {cfg.baseDir}"
  log ""

  -- Stage 1: Data Download
  -- Download initial data for tokenizer training
  stage "data-download-initial" do
    runOnMaster do
      downloadDataShards cfg cfg.initialDataShards

  -- Spawn background download of remaining shards while tokenizer trains
  let moreDataTask ← background "download-remaining-shards" do
    runOnMasterIO do
      downloadDataShards cfg cfg.numDataShards

  -- Stage 2: Tokenizer
  stage "tokenizer-training" do
    trainTokenizer cfg

  stage "tokenizer-eval" do
    evalTokenizer cfg

  -- Stage 3: Base model pretraining
  -- Wait for all data shards before pretraining
  stage "data-download-complete" do
    await moreDataTask
    let isDistributed ← dist.isInitialized
    if isDistributed then
      dist.barrier
    log s!"All {cfg.numDataShards} data shards ready"

  stage "pretrain" do
    pretrainBaseModel cfg

  stage "pretrain-eval" do
    evalBaseModel cfg

  -- Stage 4: Midtraining
  stage "midtrain" do
    midtrain cfg

  stage "midtrain-eval" do
    evalChat cfg "mid"

  -- Stage 5: Supervised Fine-tuning
  stage "sft" do
    supervisedFineTune cfg

  stage "sft-eval" do
    evalChat cfg "sft"

  -- Stage 6: Reinforcement Learning (optional)
  if cfg.enableRL then
    stage "rl" do
      reinforcementLearning cfg
    stage "rl-eval" do
      evalChat cfg "rl"
  else
    log "RL stage disabled (set ENABLE_RL=1 to enable)"

  log ""
  log "Pipeline complete!"

/-- Quick pipeline for testing (reduced settings) -/
def runQuickPipeline (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running QUICK pipeline (testing mode)..."
  let quickCfg := { cfg with modelDepth := 4 }

  -- Skip data download, use existing data
  stage "tokenizer-training" do
    trainTokenizer quickCfg

  stage "pretrain" do
    pretrainBaseModel quickCfg

  stage "midtrain" do
    midtrain quickCfg

  stage "sft" do
    supervisedFineTune quickCfg

  log "Quick pipeline complete!"

/-- Main entry point for NanoChat pipeline -/
def main : IO Unit := do
  -- Parse environment for configuration
  let baseDir := (← IO.getEnv "NANOCHAT_DIR").getD "~/.cache/nanochat"
  let numGpus := (← IO.getEnv "WORLD_SIZE").bind (·.toNat?) |>.getD 1
  let modelDepth := (← IO.getEnv "MODEL_DEPTH").bind (·.toNat?) |>.getD 20
  let paramDataRatio := (← IO.getEnv "PARAM_DATA_RATIO").bind (·.toNat?) |>.getD 20
  let vocabSize := (← IO.getEnv "VOCAB_SIZE").bind (·.toNat?) |>.getD 65536
  let numShards := (← IO.getEnv "NUM_SHARDS").bind (·.toNat?) |>.getD 240
  let initialDataShards := (← IO.getEnv "INITIAL_DATA_SHARDS").bind (·.toNat?) |>.getD 8
  let tokenizerMaxChars := (← IO.getEnv "TOKENIZER_MAX_CHARS").bind (·.toNat?) |>.getD 2000000000
  let dataPath := (← IO.getEnv "DATA_PATH").getD "base_data"
  let sftEpochs := (← IO.getEnv "SFT_EPOCHS").bind (·.toNat?) |>.getD 1
  let grpoNumSamples := (← IO.getEnv "GRPO_NUM_SAMPLES").bind (·.toNat?) |>.getD 16
  let grpoMaxNewTokens := (← IO.getEnv "GRPO_MAX_NEW_TOKENS").bind (·.toNat?) |>.getD 256
  let quickMode := (← IO.getEnv "QUICK_MODE").isSome
  let enableRL := (← IO.getEnv "ENABLE_RL").isSome

  let cfg : NanoChatConfig := {
    baseDir := baseDir
    wandbEnabled := (← IO.getEnv "WANDB_ENABLED").isSome
    wandbRun := (← IO.getEnv "WANDB_RUN").getD "nanochat"
    numGpus := numGpus
    modelDepth := modelDepth
    paramDataRatio := paramDataRatio
    vocabSize := vocabSize
    tokenizerMaxChars := tokenizerMaxChars
    dataPath := dataPath
    numDataShards := numShards
    initialDataShards := initialDataShards
    sftEpochs := sftEpochs
    grpoNumSamples := grpoNumSamples
    grpoMaxNewTokens := grpoMaxNewTokens
    resumeFromCheckpoint := true
    checkpointAfterStage := true
    enableRL := enableRL
  }

  IO.println "╔════════════════════════════════════════╗"
  IO.println "║           NanoChat Pipeline            ║"
  IO.println "╚════════════════════════════════════════╝"
  IO.println s!"Configuration:"
  IO.println s!"  Model depth: {cfg.modelDepth}"
  IO.println s!"  Vocab size: {cfg.vocabSize}"
  IO.println s!"  Param/data ratio: {cfg.paramDataRatio}"
  IO.println s!"  Data path: {cfg.dataPath}"
  IO.println s!"  Num GPUs: {cfg.numGpus}"
  IO.println s!"  Data shards: {cfg.numDataShards}"
  IO.println s!"  Initial shards: {cfg.initialDataShards}"
  IO.println s!"  Tokenizer max chars: {cfg.tokenizerMaxChars}"
  IO.println s!"  SFT epochs: {cfg.sftEpochs}"
  IO.println s!"  GRPO samples: {cfg.grpoNumSamples}, max_new_tokens: {cfg.grpoMaxNewTokens}"
  IO.println s!"  Quick mode: {quickMode}"
  IO.println s!"  RL enabled: {cfg.enableRL}"
  IO.println ""

  withDistributed cfg.toPipelineConfig do
    if quickMode then
      runQuickPipeline cfg
    else
      runNanoChatPipeline cfg

end torch.NanoChat.Pipeline

/-- Executable entrypoint wrapper. -/
unsafe def main (_args : List String) : IO UInt32 := do
  torch.NanoChat.Pipeline.main
  pure 0
