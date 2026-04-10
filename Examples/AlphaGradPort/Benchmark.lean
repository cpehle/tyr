import Examples.AlphaGradPort.Trainer
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace Examples.AlphaGradPort

open Lean

structure BenchmarkConfig where
  mode : TrainMode := .alphazero
  tasks : Array TaskName := taskSequence
  multitask : Bool := true
  numSeeds : Nat := 3
  baseSeed : Nat := 0
  numSimulations : Nat := 48
  runDir : String := ""
  checkpointDir : Option String := none
  outputPath : Option String := none
  deriving Repr, Inhabited, ToJson, FromJson

structure BenchmarkEntry where
  task : TaskName
  seed : Nat
  greedyReward : Float
  searchReward : Float
  deriving Repr, Inhabited, ToJson, FromJson

structure BenchmarkSummary where
  mode : TrainMode
  multitask : Bool
  entries : Array BenchmarkEntry
  meanGreedyReward : Float
  meanSearchReward : Float
  deriving Repr, Inhabited, ToJson, FromJson

private def meanFloatArray (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size

private def benchmarkOutputPath (cfg : BenchmarkConfig) : System.FilePath :=
  ⟨cfg.outputPath.getD s!"{cfg.runDir}/benchmark.json"⟩

private def writeBenchmarkSummary
    (cfg : BenchmarkConfig)
    (summary : BenchmarkSummary) :
    IO Unit := do
  let path := benchmarkOutputPath cfg
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path (Lean.toJson summary).pretty

def runBenchmarkWithConfig
    (cfg : BenchmarkConfig) :
    IO (Except String BenchmarkSummary) := do
  if cfg.tasks.isEmpty then
    return .error "Benchmark requires at least one task."
  let mut entries : Array BenchmarkEntry := #[]
  for offset in [:cfg.numSeeds] do
    let seed := cfg.baseSeed + offset
    if cfg.multitask then
      let evalCfg : MultiTaskTrainerConfig := {
        mode := cfg.mode
        tasks := cfg.tasks
        evalTasks := cfg.tasks
        numSimulations := cfg.numSimulations
        runDir := cfg.runDir
      }
      match (← multiEvalWithConfig evalCfg cfg.checkpointDir (some seed)) with
      | .error msg => return .error msg
      | .ok results =>
        for (task, greedyReward, searchReward) in results do
          entries := entries.push {
            task := task
            seed := seed
            greedyReward := greedyReward
            searchReward := searchReward
          }
    else
      for task in cfg.tasks do
        let taskRunDir := s!"{cfg.runDir}/{toString task}"
        let evalCfg : AlphaGradTrainerConfig := {
          mode := cfg.mode
          task := task
          numSimulations := cfg.numSimulations
          runDir := taskRunDir
        }
        match (← evalWithConfig evalCfg cfg.checkpointDir (some seed)) with
        | .error msg => return .error msg
        | .ok (greedyReward, searchReward) =>
          entries := entries.push {
            task := task
            seed := seed
            greedyReward := greedyReward
            searchReward := searchReward
          }
  let summary : BenchmarkSummary := {
    mode := cfg.mode
    multitask := cfg.multitask
    entries := entries
    meanGreedyReward := meanFloatArray (entries.map (·.greedyReward))
    meanSearchReward := meanFloatArray (entries.map (·.searchReward))
  }
  writeBenchmarkSummary cfg summary
  pure (.ok summary)

private def parseMode? (s : String) : Option TrainMode :=
  match s.trimAscii.toString.toLower with
  | "ppo" => some .ppo
  | "az" => some .alphazero
  | "alphazero" => some .alphazero
  | _ => none

private def parseTaskListArg?
    (s : String) :
    Except String (Array TaskName) := do
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then
    throw "Expected non-empty task list."
  if trimmed = "all" then
    pure taskSequence
  else
    let parts := trimmed.splitOn ","
    let mut out : Array TaskName := #[]
    for raw in parts do
      match parseTaskName? raw with
      | some task => out := out.push task
      | none => throw s!"Unknown AlphaGrad task '{raw}'."
    pure out

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" ([
    "Usage:",
    "  lake exe AlphaGradBenchmark <mode> <task1,task2,...|all> [flags]",
    "Flags:",
    "  --multitask",
    "  --seeds <n>",
    "  --base-seed <n>",
    "  --num-simulations <n>",
    "  --run-dir <dir>",
    "  --checkpoint <dir>",
    "  --output <path>"
  ] : List String)

private def parseFlags
    (args : List String)
    (cfg : BenchmarkConfig) :
    Except String BenchmarkConfig :=
  match args with
  | [] => pure cfg
  | "--multitask" :: rest =>
    parseFlags rest { cfg with multitask := true }
  | "--seeds" :: v :: rest =>
    parseFlags rest { cfg with numSeeds := parseNatArg? v |>.getD cfg.numSeeds }
  | "--base-seed" :: v :: rest =>
    parseFlags rest { cfg with baseSeed := parseNatArg? v |>.getD cfg.baseSeed }
  | "--num-simulations" :: v :: rest =>
    parseFlags rest { cfg with numSimulations := parseNatArg? v |>.getD cfg.numSimulations }
  | "--run-dir" :: v :: rest =>
    parseFlags rest { cfg with runDir := v }
  | "--checkpoint" :: v :: rest =>
    parseFlags rest { cfg with checkpointDir := some v }
  | "--output" :: v :: rest =>
    parseFlags rest { cfg with outputPath := some v }
  | flag :: _ =>
    throw s!"Unknown AlphaGrad benchmark flag '{flag}'."

def benchmarkMain (args : List String) : IO UInt32 := do
  match args with
  | modeStr :: tasksStr :: rest =>
    match parseMode? modeStr, parseTaskListArg? tasksStr with
    | some mode, .ok tasks =>
      match parseFlags rest { mode := mode, tasks := tasks } with
      | .error msg =>
        IO.eprintln msg
        IO.eprintln usage
        pure 1
      | .ok cfg =>
        match (← runBenchmarkWithConfig cfg) with
        | .error msg =>
          IO.eprintln s!"[AlphaGradBenchmark] failed: {msg}"
          pure 1
        | .ok summary =>
          IO.println s!"[AlphaGradBenchmark] entries={summary.entries.size} mean_greedy={summary.meanGreedyReward} mean_search={summary.meanSearchReward}"
          pure 0
    | none, _ =>
      IO.eprintln s!"Invalid AlphaGrad mode '{modeStr}'."
      IO.eprintln usage
      pure 1
    | _, .error msg =>
      IO.eprintln msg
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

end Examples.AlphaGradPort
