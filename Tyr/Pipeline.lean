/-
  Tyr/Pipeline.lean

  Multi-stage training pipeline orchestration.

  Models the bash/shell orchestration pattern from nanochat's speedrun.sh:
  - Sequential stages with dependencies
  - Background tasks with async await
  - Distributed coordination (master-only logging)
  - Markdown report generation
  - Checkpointing and resume

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

namespace torch.Pipeline

open torch

/-! ## Pipeline Stage Tracking -/

/-- Stage status for checkpointing and resume -/
inductive StageStatus where
  | pending
  | running
  | completed
  | failed (error : String)
  deriving Repr, BEq, Inhabited

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
  deriving Repr, Inhabited

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
  /-- Resume from checkpoint -/
  resumeFrom : Option String := none
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

/-- Define and execute a pipeline stage -/
def stage (name : String) (action : PipelineM α) : PipelineM α := do
  let mut state ← get

  -- Check if should skip (resume support)
  let existingStage := state.stages.find? (·.name == name)
  match existingStage with
  | some info =>
    if info.status == .completed then
      log s!"Skipping completed stage: {name}"
      -- Still run the action (could optimize to skip in future)
      return ← action
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

  -- Execute stage action
  let result ← action

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

  logStageEnd name durationMs
  return result

/-- Spawn a background task within the pipeline -/
def background (description : String) (action : IO α) : PipelineM (BackgroundTask α) := do
  log s!"Spawning background task: {description}"
  spawnBackground description action

/-- Await a background task -/
def await (task : BackgroundTask α) : PipelineM α := do
  log s!"Waiting for: {task.description}"
  task.get

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

  if rank == 0 then
    IO.println "╔════════════════════════════════════════╗"
    IO.println "║         Tyr Training Pipeline          ║"
    IO.println "╚════════════════════════════════════════╝"
    IO.println s!"Base directory: {baseDir}"
    IO.println s!"World size: {worldSize}"

  return {
    config := config
    report := report
    rank := rank
    worldSize := worldSize
  }

/-- Finalize pipeline and generate report -/
def finalizePipeline (state : PipelineState) : IO Unit := do
  if state.isMaster then
    -- Wait for any remaining background tasks
    for task in state.backgroundTasks do
      let _ ← task.get

    -- Generate summary
    let completedStages := state.stages.filter (·.status == .completed)
    let totalDuration := completedStages.foldl (fun acc info =>
      acc + info.durationMs.getD 0) 0

    IO.println ""
    IO.println "╔════════════════════════════════════════╗"
    IO.println "║           Pipeline Complete            ║"
    IO.println "╚════════════════════════════════════════╝"
    IO.println s!"Completed stages: {completedStages.size}/{state.stages.size}"
    IO.println s!"Total duration: {formatDuration totalDuration}"

    -- Write report
    state.report.write

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
  let masterAddr := (← IO.getEnv "MASTER_ADDR").getD "localhost"
  let masterPort := (← IO.getEnv "MASTER_PORT").bind (·.toNat?) |>.getD 29500

  if worldSize > 1 then
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
