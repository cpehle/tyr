/-
  Tyr/Pipeline.lean

  Multi-stage training pipeline orchestration.

  Models the bash/shell orchestration pattern from nanochat's speedrun.sh:
  - Sequential stages with dependencies
  - Background tasks with async await
  - Distributed coordination (master-only logging)
  - Markdown report generation
  - Checkpointing and resume
  - Failure recovery with retry and resume

  Example:
  ```lean
  def trainingPipeline : Pipeline PipelineState := do
    -- Stage 1: Tokenizer
    stage "tokenizer" do
      tokenizerTraining
      tokenizerEval

    -- Stage 2: Pretraining (can spawn background task)
    let dataDownload ← background do
      downloadMoreData 240

    stage "pretrain" do
      baseModelTraining
      await dataDownload  -- Wait for background download
      baseModelEval

    -- Stage 3: Fine-tuning
    stage "sft" do
      supervisedFineTuning
      chatEval "sft"
  ```
-/
import Tyr.Torch
import Tyr.Distributed
import Lean.Data.Json.Basic
import Lean.Data.Json.Parser
import Lean.Data.Json.Printer
import Lean.Data.Json.FromToJson

namespace torch.Pipeline

open torch
open Lean (Json)

-- Re-export needed functions
def toJson [Lean.ToJson α] (a : α) : Json := Lean.toJson a
def fromJson? [Lean.FromJson α] (j : Json) : Except String α := Lean.fromJson? j

/-! ## Retry Policy -/

/-- Retry configuration for transient failures -/
structure RetryPolicy where
  /-- Maximum retry attempts -/
  maxAttempts : Nat := 3
  /-- Initial delay in milliseconds -/
  initialDelayMs : Nat := 1000
  /-- Exponential backoff multiplier -/
  backoffMultiplier : Float := 2.0
  /-- Maximum delay cap in milliseconds -/
  maxDelayMs : Nat := 60000
  deriving Repr, Inhabited

/-- Calculate delay for a given attempt using exponential backoff -/
def RetryPolicy.delayForAttempt (policy : RetryPolicy) (attempt : Nat) : Nat :=
  let delay := policy.initialDelayMs.toFloat * (policy.backoffMultiplier ^ attempt.toFloat)
  min policy.maxDelayMs delay.toUInt64.toNat

/-! ## Pipeline Stage Tracking -/

/-- Stage status for checkpointing and resume -/
inductive StageStatus where
  | pending
  | running
  | completed
  | failed (error : String)
  deriving Repr, BEq, Inhabited

instance : Lean.ToJson StageStatus where
  toJson
    | .pending => .str "pending"
    | .running => .str "running"
    | .completed => .str "completed"
    | .failed err => .mkObj [("failed", .str err)]

instance : Lean.FromJson StageStatus where
  fromJson? json := do
    match json with
    | .str "pending" => pure .pending
    | .str "running" => pure .running
    | .str "completed" => pure .completed
    | _ =>
      -- Try to parse as failed with error message
      match json.getObjValAs? String "failed" with
      | .ok err => pure (.failed err)
      | .error _ => throw "Invalid StageStatus"

/-- Information about a pipeline stage -/
structure StageInfo where
  /-- Stage name/identifier -/
  name : String
  /-- Current status -/
  status : StageStatus
  /-- Start timestamp (ms since epoch) -/
  startTime : Option Nat := none
  /-- End timestamp -/
  endTime : Option Nat := none
  /-- Metrics collected during this stage -/
  metrics : List (String × String) := []
  /-- Number of retry attempts made -/
  retryCount : Nat := 0
  deriving Repr, Inhabited

instance : Lean.ToJson StageInfo where
  toJson info := .mkObj [
    ("name", .str info.name),
    ("status", toJson info.status),
    ("startTime", match info.startTime with | some t => .num t | none => .null),
    ("endTime", match info.endTime with | some t => .num t | none => .null),
    ("retryCount", .num info.retryCount)
  ]

instance : Lean.FromJson StageInfo where
  fromJson? json := do
    let name ← json.getObjValAs? String "name"
    let status ← json.getObjValAs? StageStatus "status"
    let startTime := (json.getObjValAs? Nat "startTime").toOption
    let endTime := (json.getObjValAs? Nat "endTime").toOption
    let retryCount := (json.getObjValAs? Nat "retryCount").toOption.getD 0
    pure { name, status, startTime, endTime, metrics := [], retryCount }

/-- Get stage duration in milliseconds -/
def StageInfo.durationMs (info : StageInfo) : Option Nat :=
  match info.startTime, info.endTime with
  | some start, some end_ => some (end_ - start)
  | _, _ => none

/-- Format duration as human-readable string -/
def formatDuration (ms : Nat) : String :=
  let seconds := ms / 1000
  let minutes := seconds / 60
  let hours := minutes / 60
  if hours > 0 then
    s!"{hours}h{minutes % 60}m{seconds % 60}s"
  else if minutes > 0 then
    s!"{minutes}m{seconds % 60}s"
  else
    s!"{seconds}s"

/-! ## Background Task Management -/

/-- Handle for a background task -/
structure BackgroundTask (α : Type) where
  /-- Task identifier -/
  id : UInt64
  /-- Description for logging -/
  description : String
  /-- IO action to get result (blocks until complete) -/
  await : IO α

/-- Spawn a background task -/
def spawnBackground (description : String) (action : IO α) : IO (BackgroundTask α) := do
  -- In a real implementation, this would spawn a thread
  -- For now, we use IO.asTask
  let task ← IO.asTask action
  let id := 0  -- Would be a real task ID
  return {
    id := id
    description := description
    await := IO.ofExcept (← IO.wait task)
  }

/-- Await a background task -/
def BackgroundTask.get (task : BackgroundTask α) : IO α :=
  task.await

/-! ## Report Generation -/

/-- Report section types -/
inductive ReportSection where
  | header (content : String)
  | stage (name : String) (content : String)
  | metrics (name : String) (values : List (String × String))
  | summary (content : String)
  deriving Repr

/-- Report accumulator -/
structure Report where
  /-- Base directory for report files -/
  baseDir : String
  /-- Sections in order -/
  sections : Array ReportSection := #[]
  /-- Start timestamp -/
  startTime : Nat
  deriving Inhabited

/-- Create a new report -/
def Report.create (baseDir : String) : IO Report := do
  let startTime ← IO.monoMsNow
  return { baseDir, startTime := startTime }

/-- Log a stage section to the report -/
def Report.logStage (report : Report) (stageName : String) (content : String) : Report :=
  { report with sections := report.sections.push (.stage stageName content) }

/-- Log metrics to the report -/
def Report.logMetrics (report : Report) (name : String) (values : List (String × String)) : Report :=
  { report with sections := report.sections.push (.metrics name values) }

/-- Format a single metric value -/
def formatMetric (key : String) (value : String) : String :=
  s!"- {key}: {value}"

/-- Generate markdown for a report section -/
def ReportSection.toMarkdown : ReportSection → String
  | .header content => s!"# Training Report\n\n{content}\n"
  | .stage name content => s!"## {name}\n\n{content}\n"
  | .metrics name values =>
    let metricLines := values.map (fun (k, v) => formatMetric k v)
    s!"### {name}\n\n{String.intercalate "\n" metricLines}\n"
  | .summary content => s!"## Summary\n\n{content}\n"

/-- Generate full markdown report -/
def Report.toMarkdown (report : Report) : IO String := do
  let endTime ← IO.monoMsNow
  let duration := formatDuration (endTime - report.startTime)

  let mut content := ""
  for sect in report.sections do
    content := content ++ sect.toMarkdown ++ "\n"

  content := content ++ s!"\n---\nTotal time: {duration}\n"
  return content

/-- Write report to file -/
def Report.write (report : Report) (filename : String := "report.md") : IO Unit := do
  let content ← report.toMarkdown
  let path := s!"{report.baseDir}/{filename}"
  IO.FS.writeFile ⟨path⟩ content
  IO.println s!"Report written to {path}"

/-! ## Checkpoint Persistence -/

/-- Checkpoint file structure for pipeline resume -/
structure PipelineCheckpoint where
  /-- Stage information for resume -/
  stages : Array StageInfo
  /-- Timestamp of checkpoint -/
  timestamp : Nat
  /-- Pipeline run identifier -/
  runId : String
  deriving Repr, Inhabited

instance : Lean.ToJson PipelineCheckpoint where
  toJson cp := .mkObj [
    ("stages", .arr (cp.stages.map toJson)),
    ("timestamp", .num cp.timestamp),
    ("runId", .str cp.runId)
  ]

instance : Lean.FromJson PipelineCheckpoint where
  fromJson? json := do
    let stagesArr ← json.getObjValAs? (Array Json) "stages"
    let stages ← stagesArr.mapM fromJson?
    let timestamp ← json.getObjValAs? Nat "timestamp"
    let runId := (json.getObjValAs? String "runId").toOption.getD ""
    pure { stages, timestamp, runId }

/-- Path for checkpoint file -/
def checkpointPath (baseDir : String) : String :=
  s!"{baseDir}/.pipeline_checkpoint.json"

/-- Save checkpoint to disk -/
def saveCheckpoint (baseDir : String) (stages : Array StageInfo) (runId : String) : IO Unit := do
  let timestamp ← IO.monoMsNow
  let checkpoint : PipelineCheckpoint := { stages, timestamp, runId }
  let path := checkpointPath baseDir
  IO.FS.writeFile ⟨path⟩ (toJson checkpoint).pretty

/-- Load checkpoint from disk -/
def loadCheckpoint (baseDir : String) : IO (Option PipelineCheckpoint) := do
  let path := checkpointPath baseDir
  if ← System.FilePath.pathExists ⟨path⟩ then
    let content ← IO.FS.readFile ⟨path⟩
    match Json.parse content >>= fromJson? with
    | .ok checkpoint => return some checkpoint
    | .error _ => return none
  else
    return none

/-- Clear checkpoint file -/
def clearCheckpoint (baseDir : String) : IO Unit := do
  let path := checkpointPath baseDir
  if ← System.FilePath.pathExists ⟨path⟩ then
    IO.FS.removeFile ⟨path⟩

/-! ## Pipeline State -/

/-- Pipeline configuration -/
structure PipelineConfig where
  /-- Base directory for artifacts -/
  baseDir : String := "~/.cache/tyr"
  /-- Whether to enable wandb logging -/
  wandbEnabled : Bool := false
  /-- Wandb run name -/
  wandbRun : String := "dummy"
  /-- Number of GPUs -/
  numGpus : Nat := 1
  /-- Resume from checkpoint if available -/
  resumeFromCheckpoint : Bool := true
  /-- Default retry policy for stages -/
  retryPolicy : RetryPolicy := {}
  /-- Checkpoint after each stage -/
  checkpointAfterStage : Bool := true
  /-- Fail fast: stop on first error -/
  failFast : Bool := true
  /-- Run identifier for this pipeline execution -/
  runId : String := ""
  deriving Repr, Inhabited

/-- Pipeline execution state -/
structure PipelineState where
  /-- Configuration -/
  config : PipelineConfig
  /-- Stages and their status -/
  stages : Array StageInfo := #[]
  /-- Current stage index -/
  currentStage : Nat := 0
  /-- Report accumulator -/
  report : Report
  /-- Distributed rank -/
  rank : UInt64 := 0
  /-- World size -/
  worldSize : UInt64 := 1
  /-- Background tasks -/
  backgroundTasks : Array (BackgroundTask Unit) := #[]
  deriving Inhabited

/-- Check if this is the master rank -/
def PipelineState.isMaster (state : PipelineState) : Bool :=
  state.rank == 0

/-- Print only on master rank -/
def printMaster (state : PipelineState) (msg : String) : IO Unit := do
  if state.isMaster then
    IO.println msg

/-! ## Pipeline Monad -/

/-- Pipeline monad for sequencing stages -/
abbrev PipelineM := StateT PipelineState IO

/-- Run pipeline action only on master rank -/
def masterOnly (action : PipelineM α) (default : α) : PipelineM α := do
  let state ← get
  if state.isMaster then
    action
  else
    pure default

/-- Log a message (master only) -/
def log (msg : String) : PipelineM Unit := do
  let state ← get
  if state.isMaster then
    IO.println msg

/-- Log a stage start -/
def logStageStart (name : String) : PipelineM Unit := do
  let state ← get
  let timestamp ← IO.monoMsNow
  if state.isMaster then
    IO.println s!"[{formatDuration (timestamp - state.report.startTime)}] Starting stage: {name}"

/-- Log a stage end -/
def logStageEnd (name : String) (durationMs : Nat) : PipelineM Unit := do
  let state ← get
  if state.isMaster then
    IO.println s!"[+{formatDuration durationMs}] Completed stage: {name}"

/-! ## Stage Execution -/

/-- Log a stage failure -/
def logStageFailed (name : String) (error : String) : PipelineM Unit := do
  let state ← get
  if state.isMaster then
    IO.eprintln s!"[FAILED] Stage '{name}': {error}"

/-- Define and execute a pipeline stage with error handling. -/
def stage (name : String) (action : PipelineM Unit) : PipelineM Unit := do
  let mut state ← get
  let config := state.config

  -- Check if should skip (resume support)
  let existingStage := state.stages.find? (·.name == name)
  match existingStage with
  | some info =>
    if info.status == .completed then
      log s!"[SKIP] Already completed: {name}"
      return ()
    else if let .failed err := info.status then
      log s!"[RESUME] Retrying previously failed stage: {name} (was: {err})"
  | none => pure ()

  -- Record stage start
  let startTime ← IO.monoMsNow
  let stageInfo : StageInfo := {
    name := name
    status := .running
    startTime := some startTime
  }
  state := { state with stages := state.stages.push stageInfo }
  set state

  logStageStart name

  -- Execute stage action with exception handling
  try
    action

    -- Record stage completion
    let endTime ← IO.monoMsNow
    let durationMs := endTime - startTime
    state ← get
    let stages := state.stages.map fun info =>
      if info.name == name then
        { info with status := .completed, endTime := some endTime }
      else info
    let report := state.report.logStage name s!"Duration: {formatDuration durationMs}"
    set { state with stages := stages, report := report }

    -- Checkpoint on success
    if config.checkpointAfterStage then
      state ← get
      let baseDir := config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
      Pipeline.saveCheckpoint baseDir state.stages config.runId

    logStageEnd name durationMs
    return ()

  catch e =>
    -- Record failure
    let endTime ← IO.monoMsNow
    state ← get
    let errorMsg := toString e
    let stages := state.stages.map fun info =>
      if info.name == name then
        { info with status := .failed errorMsg, endTime := some endTime }
      else info
    let report := state.report.logStage name s!"FAILED: {errorMsg}"
    set { state with stages := stages, report := report }

    -- Checkpoint failure state
    if config.checkpointAfterStage then
      state ← get
      let baseDir := config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
      Pipeline.saveCheckpoint baseDir state.stages config.runId

    logStageFailed name errorMsg

    if config.failFast then
      throw e
    else
      throw e  -- For now, always propagate

/-- Execute a stage with retry logic. -/
def stageWithRetry (name : String) (policy : RetryPolicy) (action : PipelineM Unit) : PipelineM Unit := do
  let mut lastError := ""
  for attempt in [:policy.maxAttempts] do
    try
      stage name action
      return ()
    catch e =>
      lastError := toString e
      let state ← get

      -- Log retry attempt
      if state.isMaster then
        IO.eprintln s!"[WARN] Stage '{name}' failed (attempt {attempt + 1}/{policy.maxAttempts}): {lastError}"

      -- Update retry count in stage info
      let stages := state.stages.map fun info =>
        if info.name == name then
          { info with retryCount := attempt + 1 }
        else info
      set { state with stages := stages }

      if attempt + 1 < policy.maxAttempts then
        let delayMs := policy.delayForAttempt attempt
        if state.isMaster then
          IO.println s!"[RETRY] Waiting {delayMs}ms before retry..."
        IO.sleep delayMs.toUInt32
      else
        -- Final failure
        throw e

  throw (IO.userError s!"Stage '{name}' failed after {policy.maxAttempts} attempts: {lastError}")

/-- Spawn a background task within the pipeline -/
def background (description : String) (action : IO α) : PipelineM (BackgroundTask α) := do
  log s!"Spawning background task: {description}"
  spawnBackground description action

/-- Await a background task -/
def await (task : BackgroundTask α) : PipelineM α := do
  log s!"Waiting for: {task.description}"
  task.get

/-! ## Tracked Background Tasks -/

/-- Background task with error tracking -/
structure TrackedTask (α : Type) where
  /-- Underlying task -/
  task : BackgroundTask (Except String α)
  /-- Task description -/
  description : String

/-- Spawn a background task with error capture -/
def backgroundTracked (description : String) (action : IO α) : PipelineM (TrackedTask α) := do
  log s!"Spawning tracked background task: {description}"
  let wrapped : IO (Except String α) := do
    try
      let result ← action
      return .ok result
    catch e =>
      return .error (toString e)
  let task ← spawnBackground description wrapped
  return { task := task, description := description }

/-- Await a tracked task with error handling -/
def awaitTracked (task : TrackedTask α) (onError : String → PipelineM α) : PipelineM α := do
  let result ← task.task.get
  match result with
  | .ok value => return value
  | .error msg =>
    log s!"[ERROR] Background task '{task.description}' failed: {msg}"
    onError msg

/-- Await a tracked task, throwing on error -/
def awaitTrackedOrThrow (task : TrackedTask α) : PipelineM α := do
  let result ← task.task.get
  match result with
  | .ok value => return value
  | .error msg =>
    throw (IO.userError s!"Background task '{task.description}' failed: {msg}")

/-! ## Distributed Failure Coordination -/

/-- Check if any stage has failed locally -/
def hasLocalFailure : PipelineM Bool := do
  let state ← get
  return state.stages.any fun s =>
    match s.status with
    | .failed _ => true
    | _ => false

/-- Synchronize all ranks with a barrier -/
def syncBarrier : PipelineM Unit := do
  let state ← get
  if state.worldSize > 1 then
    dist.barrier

/-- Check if all ranks are healthy (barrier with failure check).
    Returns true if this rank has no failures.
    Note: In distributed mode, each rank checks its own failure status. -/
def checkAllRanksHealthy : PipelineM Bool := do
  let state ← get
  if state.worldSize == 1 then
    return !(← hasLocalFailure)

  -- Barrier to synchronize
  dist.barrier
  -- Check local failure status
  let failed ← hasLocalFailure
  if failed && state.isMaster then
    IO.eprintln "[DISTRIBUTED] This rank has failures"
  return !failed

/-- Record metrics for the current stage -/
def recordMetrics (metrics : List (String × String)) : PipelineM Unit := do
  let state ← get
  if state.stages.size > 0 then
    let lastIdx := state.stages.size - 1
    let stages := state.stages.modify lastIdx fun info =>
      { info with metrics := info.metrics ++ metrics }
    let stageName := state.stages[lastIdx]!.name
    let report := state.report.logMetrics stageName metrics
    set { state with stages := stages, report := report }

/-! ## Pipeline Initialization and Cleanup -/

/-- Initialize pipeline state -/
def initPipeline (config : PipelineConfig) : IO PipelineState := do
  -- Check distributed status
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  -- Create base directory
  let baseDir := config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  IO.FS.createDirAll ⟨baseDir⟩

  -- Create report
  let report ← Report.create baseDir

  -- Generate run ID if not provided
  let runId ← if config.runId.isEmpty then do
    let ts ← IO.monoMsNow
    pure s!"run_{ts}"
  else
    pure config.runId

  -- Try to load checkpoint for resume
  let resumedStages ← if config.resumeFromCheckpoint then
    match ← loadCheckpoint baseDir with
    | some checkpoint =>
      if rank == 0 then
        IO.println s!"[RESUME] Loading checkpoint from {checkpointPath baseDir}"
        IO.println s!"[RESUME] Found {checkpoint.stages.size} stages"
      pure checkpoint.stages
    | none =>
      pure #[]
  else
    pure #[]

  if rank == 0 then
    IO.println "╔════════════════════════════════════════╗"
    IO.println "║         Tyr Training Pipeline          ║"
    IO.println "╚════════════════════════════════════════╝"
    IO.println s!"Base directory: {baseDir}"
    IO.println s!"World size: {worldSize}"
    IO.println s!"Run ID: {runId}"
    if resumedStages.size > 0 then
      let completed := resumedStages.filter (·.status == .completed)
      IO.println s!"[RESUME] {completed.size}/{resumedStages.size} stages already completed"

  return {
    config := { config with runId := runId }
    report := report
    rank := rank
    worldSize := worldSize
    stages := resumedStages
  }

/-- Finalize pipeline and generate report -/
def finalizePipeline (state : PipelineState) : IO Unit := do
  if state.isMaster then
    -- Wait for any remaining background tasks
    for task in state.backgroundTasks do
      let _ ← task.get

    -- Generate summary
    let completedStages := state.stages.filter (·.status == .completed)
    let failedStages := state.stages.filter fun s =>
      match s.status with | .failed _ => true | _ => false
    let totalDuration := completedStages.foldl (fun acc info =>
      acc + info.durationMs.getD 0) 0

    IO.println ""
    if failedStages.size > 0 then
      IO.println "╔════════════════════════════════════════╗"
      IO.println "║         Pipeline Failed                ║"
      IO.println "╚════════════════════════════════════════╝"
      for s in failedStages do
        match s.status with
        | .failed err => IO.eprintln s!"  - {s.name}: {err}"
        | _ => pure ()
    else
      IO.println "╔════════════════════════════════════════╗"
      IO.println "║           Pipeline Complete            ║"
      IO.println "╚════════════════════════════════════════╝"

    IO.println s!"Completed stages: {completedStages.size}/{state.stages.size}"
    if failedStages.size > 0 then
      IO.println s!"Failed stages: {failedStages.size}"
    IO.println s!"Total duration: {formatDuration totalDuration}"

    -- Write report
    state.report.write

    -- Clear checkpoint on successful completion
    if failedStages.size == 0 && state.config.checkpointAfterStage then
      let baseDir := state.config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
      clearCheckpoint baseDir

/-- Run a pipeline action -/
def runPipeline (config : PipelineConfig) (action : PipelineM α) : IO α := do
  let initialState ← initPipeline config
  let (result, finalState) ← action.run initialState
  finalizePipeline finalState
  return result

/-! ## Distributed Execution Helpers -/

/-- Run with torchrun-style distributed setup -/
def withDistributed (config : PipelineConfig) (action : PipelineM α) : IO α := do
  -- Initialize distributed if env vars are set
  let worldSize := (← IO.getEnv "WORLD_SIZE").bind (·.toNat?) |>.getD 1
  let rank := (← IO.getEnv "RANK").bind (·.toNat?) |>.getD 0
  let localRank := (← IO.getEnv "LOCAL_RANK").bind (·.toNat?) |>.getD rank
  let masterAddr := (← IO.getEnv "MASTER_ADDR").getD "localhost"
  let masterPort := (← IO.getEnv "MASTER_PORT").bind (·.toNat?) |>.getD 29500

  if worldSize > 1 then
    if ← cuda_is_available then
      dist.setCudaDevice localRank.toUInt64
    dist.initProcessGroup "nccl" masterAddr masterPort.toUInt64 rank.toUInt64 worldSize.toUInt64

  let result ← runPipeline config action

  if worldSize > 1 then
    dist.destroyProcessGroup

  return result

/-! ## Pre-built Pipeline Stages -/

/-- Standard pretraining stage template -/
def pretrainStage (name : String) (trainFn : PipelineM Unit) (evalFn : PipelineM Unit) : PipelineM Unit := do
  stage s!"{name}-training" trainFn
  stage s!"{name}-eval" evalFn

/-- Standard fine-tuning stage template -/
def finetuneStage (name : String) (trainFn : PipelineM Unit) (evalFn : PipelineM Unit) : PipelineM Unit := do
  stage s!"{name}-training" trainFn
  stage s!"{name}-eval" evalFn

end torch.Pipeline
