import Examples.AlphaGradPort.PolicyTrain

/-!
  Examples/AlphaGradPort/PolicySweep.lean

  Sweep runner for AlphaGrad policy training across one task or all tasks.
  Supports PPO, AlphaZero, or both sequentially.
-/

namespace Examples.AlphaGradPort

inductive SweepMode where
  | ppo
  | alphazero
  | both
  deriving Repr, Inhabited

instance : ToString SweepMode where
  toString
    | .ppo => "ppo"
    | .alphazero => "alphazero"
    | .both => "both"

private def parseSweepMode? (s : String) : Option SweepMode :=
  match s.trimAscii.toString.toLower with
  | "ppo" => some .ppo
  | "az" => some .alphazero
  | "alphazero" => some .alphazero
  | "both" => some .both
  | _ => none

private def parseNatArg? (s : String) : Option Nat :=
  s.toNat?

private def usage : String :=
  String.intercalate "\n" <| ([
    "Usage:",
    "  lake exe AlphaGradPolicySweep",
    "  lake exe AlphaGradPolicySweep <mode>",
    "  lake exe AlphaGradPolicySweep <mode> <task-name|all>",
    "  lake exe AlphaGradPolicySweep <mode> <task-name|all> <epochs>",
    "  lake exe AlphaGradPolicySweep <mode> <task-name|all> <epochs> <episodes-per-epoch>",
    "Modes: ppo, alphazero (or az), both",
    s!"Tasks: {taskNamesCsv}"
  ] : List String)

private def runPPO
    (task : TaskSpec)
    (epochs episodesPerEpoch : Nat) :
    IO Bool := do
  let cfg : PPOTrainConfig := {
    epochs := epochs
    episodesPerEpoch := episodesPerEpoch
  }
  match (← trainPPO task cfg) with
  | .ok _ => pure true
  | .error msg =>
    IO.eprintln s!"[AlphaGradPolicySweep][ppo] task={task.name} failed: {msg}"
    pure false

private def runAlphaZero
    (task : TaskSpec)
    (epochs episodesPerEpoch : Nat) :
    IO Bool := do
  let cfg : AlphaZeroTrainConfig := {
    epochs := epochs
    episodesPerEpoch := episodesPerEpoch
    numSimulations := task.mctsCfg.numSimulations
  }
  match (← trainAlphaZero task cfg) with
  | .ok _ => pure true
  | .error msg =>
    IO.eprintln s!"[AlphaGradPolicySweep][alphazero] task={task.name} failed: {msg}"
    pure false

private def runOneMode
    (mode : SweepMode)
    (task : TaskSpec)
    (epochs episodesPerEpoch : Nat) :
    IO Bool := do
  match mode with
  | .ppo =>
    runPPO task epochs episodesPerEpoch
  | .alphazero =>
    runAlphaZero task epochs episodesPerEpoch
  | .both =>
    let ppoOk ← runPPO task epochs episodesPerEpoch
    let azOk ← runAlphaZero task epochs episodesPerEpoch
    pure (ppoOk && azOk)

private def runOneTask
    (mode : SweepMode)
    (taskName : TaskName)
    (epochs episodesPerEpoch : Nat) :
    IO Bool := do
  match (← materializeTask taskName) with
  | .error msg =>
    IO.eprintln s!"[AlphaGradPolicySweep] task={taskName} materialization failed: {msg}"
    pure false
  | .ok task =>
    IO.println s!"[AlphaGradPolicySweep] mode={mode} task={task.name} epochs={epochs} episodes/epoch={episodesPerEpoch}"
    runOneMode mode task epochs episodesPerEpoch

private def runAllTasks
    (mode : SweepMode)
    (epochs episodesPerEpoch : Nat) :
    IO Bool := do
  let mut ok := true
  for h : i in [:taskSequence.size] do
    let taskName := taskSequence[i]
    IO.println s!"[AlphaGradPolicySweep] ({i + 1}/{taskSequence.size}) targeting task {taskName}"
    let thisOk ← runOneTask mode taskName epochs episodesPerEpoch
    ok := ok && thisOk
  pure ok

def policySweepMain (args : List String) : IO UInt32 := do
  let defaultMode : SweepMode := .ppo
  let defaultEpochs : Nat := 8
  let defaultEpisodesPerEpoch : Nat := 4

  match args with
  | [] =>
    let ok ← runAllTasks defaultMode defaultEpochs defaultEpisodesPerEpoch
    pure (if ok then 0 else 1)
  | [modeArg] =>
    match parseSweepMode? modeArg with
    | some mode =>
      let ok ← runAllTasks mode defaultEpochs defaultEpisodesPerEpoch
      pure (if ok then 0 else 1)
    | none =>
      IO.eprintln s!"Unknown mode: {modeArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, targetArg] =>
    match parseSweepMode? modeArg with
    | none =>
      IO.eprintln s!"Unknown mode: {modeArg}"
      IO.eprintln usage
      pure 1
    | some mode =>
      if targetArg.trimAscii.toString = "all" then
        let ok ← runAllTasks mode defaultEpochs defaultEpisodesPerEpoch
        pure (if ok then 0 else 1)
      else
        match parseTaskName? targetArg with
        | some taskName =>
          let ok ← runOneTask mode taskName defaultEpochs defaultEpisodesPerEpoch
          pure (if ok then 0 else 1)
        | none =>
          IO.eprintln s!"Unknown task: {targetArg}"
          IO.eprintln usage
          pure 1
  | [modeArg, targetArg, epochsArg] =>
    match parseSweepMode? modeArg, parseNatArg? epochsArg with
    | some mode, some epochs =>
      if targetArg.trimAscii.toString = "all" then
        let ok ← runAllTasks mode epochs defaultEpisodesPerEpoch
        pure (if ok then 0 else 1)
      else
        match parseTaskName? targetArg with
        | some taskName =>
          let ok ← runOneTask mode taskName epochs defaultEpisodesPerEpoch
          pure (if ok then 0 else 1)
        | none =>
          IO.eprintln s!"Unknown task: {targetArg}"
          IO.eprintln usage
          pure 1
    | _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {targetArg} {epochsArg}"
      IO.eprintln usage
      pure 1
  | [modeArg, targetArg, epochsArg, epPerEpochArg] =>
    match parseSweepMode? modeArg, parseNatArg? epochsArg, parseNatArg? epPerEpochArg with
    | some mode, some epochs, some episodesPerEpoch =>
      if targetArg.trimAscii.toString = "all" then
        let ok ← runAllTasks mode epochs episodesPerEpoch
        pure (if ok then 0 else 1)
      else
        match parseTaskName? targetArg with
        | some taskName =>
          let ok ← runOneTask mode taskName epochs episodesPerEpoch
          pure (if ok then 0 else 1)
        | none =>
          IO.eprintln s!"Unknown task: {targetArg}"
          IO.eprintln usage
          pure 1
    | _, _, _ =>
      IO.eprintln s!"Invalid arguments: {modeArg} {targetArg} {epochsArg} {epPerEpochArg}"
      IO.eprintln usage
      pure 1
  | _ =>
    IO.eprintln usage
    pure 1

end Examples.AlphaGradPort
