import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows.MoleculeTransformer
import Tyr.Model.BranchingFlows.QM9

/-!
  Dataset-backed BranchingFlows molecule training and generation.

  This executable trains the molecule transformer on freshly sampled
  BranchingFlows bridge batches, evaluates the trained model on held-out bridge
  samples, then runs the learned model through the forward molecule generator
  and writes XYZ artifacts.
-/

namespace Examples.BranchingFlows

open torch
open torch.branching

private def noEventTimeDist : TimeDist :=
  { cdf := fun _ => 0.0,
    pdf := fun _ => 0.0,
    quantile := fun _ => 0.0 }

private def defaultJsonl : String :=
  "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.95,0.0,0.0]},{\"label\":1,\"coord\":[-0.24,0.93,0.0]}]}\n" ++
  "{\"name\":\"ammonia\",\"smiles\":\"N\",\"atoms\":[{\"label\":7,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.94,0.0,0.0]},{\"label\":1,\"coord\":[-0.31,0.89,0.0]},{\"label\":1,\"coord\":[-0.31,-0.89,0.0]}]}\n" ++
  "{\"name\":\"methane\",\"smiles\":\"C\",\"atoms\":[{\"label\":6,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.63,0.63,0.63]},{\"label\":1,\"coord\":[-0.63,-0.63,0.63]},{\"label\":1,\"coord\":[-0.63,0.63,-0.63]},{\"label\":1,\"coord\":[0.63,-0.63,-0.63]}]}\n" ++
  "{\"name\":\"methanol\",\"smiles\":\"CO\",\"atoms\":[{\"label\":6,\"coord\":[0.0,0.0,0.0]},{\"label\":8,\"coord\":[1.43,0.0,0.0]},{\"label\":1,\"coord\":[-0.54,0.94,0.0]},{\"label\":1,\"coord\":[-0.54,-0.47,0.81]},{\"label\":1,\"coord\":[-0.54,-0.47,-0.81]},{\"label\":1,\"coord\":[1.76,0.9,0.0]}]}\n"

structure RunOptions where
  dataPath? : Option String := none
  outputPrefix : String := "examples_branching_molecule_trained"
  steps : Nat := 260
  batchSize : Nat := 4
  seed : UInt64 := 20260618
  deriving Repr

private def usage : String :=
  "usage: lake exe BranchingFlowsMoleculeTrainGenerate [--data molecules.jsonl|json] " ++
  "[--out-prefix prefix] [--steps n] [--batch-size n] [--seed n]"

private def parseNatArg (name value : String) : IO Nat := do
  match value.toNat? with
  | some n => pure n
  | none => throw (IO.userError s!"{name} expects a natural number, got '{value}'")

private def parseSeedArg (value : String) : IO UInt64 := do
  let n ← parseNatArg "--seed" value
  pure n.toUInt64

partial def parseArgsLoop (args : List String) (opts : RunOptions) : IO RunOptions := do
  match args with
  | [] => pure opts
  | "--help" :: _ => throw (IO.userError usage)
  | "--data" :: value :: rest =>
      parseArgsLoop rest { opts with dataPath? := some value }
  | "--out-prefix" :: value :: rest =>
      parseArgsLoop rest { opts with outputPrefix := value }
  | "--steps" :: value :: rest =>
      parseArgsLoop rest { opts with steps := (← parseNatArg "--steps" value) }
  | "--batch-size" :: value :: rest =>
      parseArgsLoop rest { opts with batchSize := (← parseNatArg "--batch-size" value) }
  | "--seed" :: value :: rest =>
      parseArgsLoop rest { opts with seed := (← parseSeedArg value) }
  | flag :: rest =>
      if flag.startsWith "--" then
        throw (IO.userError s!"unknown option '{flag}'\n{usage}")
      else
        match opts.dataPath? with
        | none => parseArgsLoop rest { opts with dataPath? := some flag }
        | some _ => throw (IO.userError s!"unexpected positional argument '{flag}'\n{usage}")

private def parseOptions (args : List String) : IO RunOptions :=
  parseArgsLoop args {}

private def exceptToIO {α : Type} (context : String) (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw (IO.userError s!"{context}: {e}")

private def parseDatasetRaw (raw : String) : Except String (Array QM9MoleculeRecord) :=
  match parseQM9MoleculeJsonl raw with
  | .ok records =>
      if records.isEmpty then
        parseQM9MoleculeDatasetJson raw
      else
        .ok records
  | .error jsonlErr =>
      match parseQM9MoleculeDatasetJson raw with
      | .ok records => .ok records
      | .error jsonErr => .error s!"as JSONL: {jsonlErr}; as JSON: {jsonErr}"

private def loadRecords (opts : RunOptions) : IO (Array QM9MoleculeRecord) := do
  match opts.dataPath? with
  | none => exceptToIO "parse embedded molecule fixture" (parseDatasetRaw defaultJsonl)
  | some "-" => exceptToIO "parse embedded molecule fixture" (parseDatasetRaw defaultJsonl)
  | some path =>
      let raw ← IO.FS.readFile ⟨path⟩
      exceptToIO s!"parse molecule dataset {path}" (parseDatasetRaw raw)

private def splitTrainEval (states : Array (BranchingState MoleculeAtom)) :
    Array (BranchingState MoleculeAtom) × Array (BranchingState MoleculeAtom) :=
  if states.size <= 1 then
    (states, states)
  else
    let split := states.size - 1
    (states.extract 0 split, states.extract split states.size)

private def countSplits (events : Array (Array BranchingStepEvent)) : Nat :=
  events.foldl (init := 0) (fun acc step =>
    step.foldl (init := acc) (fun acc ev => acc + ev.splitCount))

private def countDeletes (events : Array (Array BranchingStepEvent)) : Nat :=
  events.foldl (init := 0) (fun acc step =>
    step.foldl (init := acc) (fun acc ev => if ev.deleted then acc + 1 else acc))

private def labelSymbol (maskToken label : Nat) : String :=
  if label == maskToken then "X" else qm9AtomicNumberSymbol label

def run (opts : RunOptions) : IO Unit := do
  if opts.steps == 0 then
    throw (IO.userError "training steps must be positive")
  if opts.batchSize == 0 then
    throw (IO.userError "batch size must be positive")

  let vocabSize : Nat := 10
  let vocab : UInt64 := 10
  let maskToken : Nat := 9
  let maxLen : UInt64 := 16
  let heads : UInt64 := 2
  let headDim : UInt64 := 8
  let mlp : UInt64 := 48
  let lr : Float := 8.0e-3
  let deletionPad : Float := 1.2

  let bridgeCfg := MoleculeBridgeConfig.qm9 vocabSize maskToken
  let labelDFM := DistNoisyDiscreteConfig.qm9 vocabSize maskToken
  let records ← loadRecords opts
  let allStates ← exceptToIO "convert molecule records"
    (qm9RecordsToBranchingStates records { vocabSize? := some vocabSize, maskToken? := some maskToken })
  let states := allStates.filter (fun state => state.state.size <= maxLen.toNat)
  if states.isEmpty then
    throw (IO.userError s!"no molecules fit maxLen={maxLen}")
  let (trainStates, evalStates) := splitTrainEval states
  let evalStates := if evalStates.isEmpty then trainStates else evalStates

  let trainCfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.25
    delWeight := 0.25
    weightDecay := 1.0e-4
    gradClip := 0.0
  }
  let model :=
    moleculeTransformerModel (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)

  torch.manualSeed opts.seed
  let initParams ← MoleculeTransformerParams.init vocab heads headDim mlp
  let opt := Optim.adamw (lr := lr) (weight_decay := trainCfg.weightDecay)
  let initOptState := opt.init initParams
  let mut params := initParams
  let mut optState := initOptState
  let mut rng : Rng := { state := opts.seed + 17 }

  let (fixedTrainBridge, rng') ←
    sampleMoleculeBridgeBatch bridgeCfg trainStates opts.batchSize rng
      (maxLen := some maxLen.toNat) (deletionPad := deletionPad)
  rng := rng'
  let initTrain ←
    evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params fixedTrainBridge
      (some labelDFM)
  let mut lastReport := initTrain

  for step in [:opts.steps] do
    let (bridgeBatch, rng') ←
      sampleMoleculeBridgeBatch bridgeCfg trainStates opts.batchSize rng
        (maxLen := some maxLen.toNat) (deletionPad := deletionPad)
    rng := rng'
    let (params', optState', report) ←
      trainStepMolecule (maxLen := maxLen) (vocab := vocab) trainCfg model
        params optState bridgeBatch lr (some labelDFM)
    params := params'
    optState := optState'
    lastReport := report
    if step % 50 == 0 then
      IO.println s!"molecule_train step={step} total={report.total} coord={report.coord} label={report.label} splits={report.splits} del={report.del}"

  let finalTrain ←
    evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params fixedTrainBridge
      (some labelDFM)
  if !(Float.isFinite finalTrain.total) then
    throw (IO.userError "final train loss is not finite")
  if !(finalTrain.total < initTrain.total) then
    throw (IO.userError s!"training did not reduce fixed train loss: initial={initTrain.total}, final={finalTrain.total}")

  let (evalBridge, rng') ←
    sampleMoleculeBridgeBatch bridgeCfg evalStates opts.batchSize rng
      (maxLen := some maxLen.toNat) (deletionPad := deletionPad)
  rng := rng'
  let evalReport ←
    evalMoleculeLoss (maxLen := maxLen) (vocab := vocab) trainCfg model params evalBridge
      (some labelDFM)

  let target := evalStates[0]!
  writeMoleculeXYZ ⟨opts.outputPrefix ++ "_target.xyz"⟩ target
    "target molecule from evaluation split" (labelSymbol maskToken)

  let (x0Atom, rng') := MoleculeBridgeConfig.sampleInitialAtom bridgeCfg rng
  rng := rng'
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x0Atom] #[0]
  writeMoleculeXYZ ⟨opts.outputPrefix ++ "_source.xyz"⟩ x0
    "masked source molecule for learned generation" (labelSymbol maskToken)

  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault bridgeCfg TimeDist.betaOneThreeHalves noEventTimeDist
  let learnedModel :=
    moleculeTransformerIOModel (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      trainCfg.padToken params (splitLogitCap? := some 1.25)
  let schedule : Array Float := #[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
  let (generated, _rng) ←
    moleculeBranchingGenerateIO flow x0 learnedModel schedule
      (maxStateLen? := some maxLen.toNat) (rng := rng)
  if generated.finalState.state.isEmpty then
    throw (IO.userError "learned molecule generation produced an empty state")
  writeMoleculeXYZ ⟨opts.outputPrefix ++ "_generated.xyz"⟩ generated.finalState
    "learned BranchingFlows molecule generation" (labelSymbol maskToken)

  IO.println s!"molecule_dataset records={records.size} usable={states.size} train={trainStates.size} eval={evalStates.size}"
  IO.println s!"molecule_train fixed_initial_total={initTrain.total} fixed_final_total={finalTrain.total}"
  IO.println s!"molecule_train last_batch_total={lastReport.total} eval_total={evalReport.total}"
  IO.println s!"molecule_generate target_atoms={target.state.size} source_atoms={x0.state.size} generated_atoms={generated.finalState.state.size}"
  IO.println s!"molecule_generate splits={countSplits generated.events} deletes={countDeletes generated.events}"
  IO.println s!"wrote {opts.outputPrefix}_target.xyz"
  IO.println s!"wrote {opts.outputPrefix}_source.xyz"
  IO.println s!"wrote {opts.outputPrefix}_generated.xyz"

def _root_.main (args : List String) : IO UInt32 := do
  let opts ← parseOptions args
  run opts
  pure 0

end Examples.BranchingFlows
