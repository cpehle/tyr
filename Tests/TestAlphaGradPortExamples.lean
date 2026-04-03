import LeanTest
import Examples.AlphaGradPort.A0Train
import Examples.AlphaGradPort.Benchmark
import Examples.AlphaGradPort.PolicySweep
import Examples.AlphaGradPort.PolicyTrain
import Examples.AlphaGradPort.Replay
import Examples.AlphaGradPort.Trainer

namespace Tests.AlphaGradPortExamples

open LeanTest
open Examples.AlphaGradPort
open Tyr.AD.Elim

private def assertSemanticEdges (label : String) (edges : Array Tyr.AD.JaxprLike.LocalJacEdge) : IO Unit := do
  let hasNonSemantic := edges.any (fun e =>
    match e.map.repr with
    | Tyr.AD.Sparse.SparseMapTag.semantic _ => false
    | _ => true)
  LeanTest.assertTrue (!hasNonSemantic)
    s!"{label} should only contain semantic local-Jac edges (no placeholder/hybrid fallback tags)."

@[test]
def testPerceptronSearchParityAcrossPolicies : IO Unit := do
  let task ←
    match (← materializeTask .perceptron) with
    | .error msg => LeanTest.fail s!"Perceptron materialization failed: {msg}"
    | .ok task => pure task

  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 6
    maxDepth := some task.numEliminableVertices
    maxNumConsideredActions := 6
    gumbelScale := 0.0
    dagDirichletFraction := 0.0
    dagTemperature := 1.0
  }

  let checkEpisode (label : String) (res : Except String AlphaGradEpisodeResult) : IO Unit := do
    match res with
    | .error msg =>
      LeanTest.fail s!"{label} failed: {msg}"
    | .ok out =>
      LeanTest.assertEqual out.actions0.size task.numEliminableVertices
        s!"{label} should emit one action per eliminable vertex."
      LeanTest.assertEqual out.order1.size task.numEliminableVertices
        s!"{label} should emit one vertex per eliminable action."
      LeanTest.assertTrue (hasNoDuplicates out.actions0)
        s!"{label} action trace must be duplicate-free."
      LeanTest.assertTrue (hasNoDuplicates out.order1)
        s!"{label} vertex order must be duplicate-free."
      LeanTest.assertTrue (out.finalState.violation?.isNone)
        s!"{label} should complete without constraint violations."

  let gumbel :=
    searchEpisodeFromGraph? task.envCfg mctsCfg 2026030601 task.graph task.numVertices
  let dagAlphaZero :=
    searchEpisodeDagWithPolicyFromGraph? .alphaZero task.envCfg mctsCfg 2026030602 task.graph task.numVertices
  let dagGumbel :=
    searchEpisodeDagWithPolicyFromGraph? .gumbelMuZero task.envCfg mctsCfg 2026030603 task.graph task.numVertices

  checkEpisode "gumbel" gumbel
  checkEpisode "dag-alphaZero" dagAlphaZero
  checkEpisode "dag-gumbelMuZero" dagGumbel

@[test]
def testKStmtLoweredTasksUseSemanticMaterialization : IO Unit := do
  let loweredTasks : Array TaskName := #[
    .perceptron, .encoder, .robotArm6DOF, .blackScholesJacobian,
    .humanHeartDipole, .propaneCombustion
  ]

  for taskName in loweredTasks do
    let task ←
      match (← materializeTask taskName) with
      | .error msg => LeanTest.fail s!"{taskName} materialization failed: {msg}"
      | .ok task => pure task

    LeanTest.assertTrue (task.numVertices > 0)
      s!"{task.name} should expose at least one vertex after KStmt lowering."
    LeanTest.assertTrue (task.numActions > 0)
      s!"{task.name} should expose at least one action after KStmt lowering."
    LeanTest.assertTrue (task.numEliminableVertices > 0)
      s!"{task.name} should expose at least one eliminable vertex after KStmt lowering."
    LeanTest.assertEqual task.numActions task.graph.actionVertices.size
      s!"{task.name} should size task action width from the graph action table."
    LeanTest.assertTrue (!task.edges.isEmpty)
      s!"{task.name} should expose non-empty local-Jac edges after KStmt lowering."
    LeanTest.assertTrue
      (task.graph.actionVertices.all (fun vertex => (producerInfo? task.graph vertex).isSome))
      s!"{task.name} should preserve normalized producer semantics on its explicit action surface."
    assertSemanticEdges task.name task.edges

@[test]
def testAllAlphaGradTasksMaterialize : IO Unit := do
  for taskName in taskSequence do
    let task ←
      match (← materializeTask taskName) with
      | .error msg => LeanTest.fail s!"{taskName} materialization failed: {msg}"
      | .ok task => pure task
    LeanTest.assertTrue (task.numVertices > 0)
      s!"{task.name} should report a positive vertex count."
    LeanTest.assertTrue (task.numActions > 0)
      s!"{task.name} should report a positive action count."
    LeanTest.assertTrue (task.numEliminableVertices > 0)
      s!"{task.name} should report a positive eliminable vertex count."

@[test]
def testPolicySweepPPOPerceptronTinyRun : IO Unit := do
  let code ← Examples.AlphaGradPort.policySweepMain ["ppo", "Perceptron", "1", "1"]
  LeanTest.assertEqual code 0
    "AlphaGradPolicySweep should complete a tiny PPO run for Perceptron."

@[test]
def testAlphaGradObservationExportMatchesTaskShape : IO Unit := do
  let task ←
    match (← materializeTask .perceptron) with
    | .error msg => LeanTest.fail s!"Perceptron materialization failed: {msg}"
    | .ok task => pure task

  let s0 ←
    match initAlphaGradState? task.graph task.numVertices with
    | .error msg => LeanTest.fail s!"Perceptron state init failed: {msg}"
    | .ok s => pure s

  let flat := exportObservationFlat task.envCfg s0
  let expected := task.numVertices * observationTokenDim task.graph task.numVertices
  LeanTest.assertEqual flat.size expected
    "AlphaGrad observation export should match (numVertices * tokenDim)."

@[test]
def testPolicyTrainAlphaZeroPerceptronTinyRun : IO Unit := do
  let code ← Examples.AlphaGradPort.policyTrainMain ["alphazero", "Perceptron", "1", "1"]
  LeanTest.assertEqual code 0
    "AlphaGradPolicyTrain should complete a tiny AlphaZero/Gumbel run for Perceptron."

@[test]
def testAlphaGradReplayRoundtrip : IO Unit := do
  let path : System.FilePath := ⟨s!"/tmp/alphagrad_replay_{← IO.monoMsNow}.json"⟩
  let buf := ReplayBuffer.empty 4
  let buf := buf.push {
    kind := .alphazero
    features := #[1.0, 2.0]
    reward := -3.0
    valueTarget := -2.5
    policyTarget := #[0.25, 0.75]
  }
  saveReplayBuffer path buf
  let loaded ← loadReplayBuffer path
  LeanTest.assertEqual loaded.size 1
    "Replay roundtrip should preserve sample count."
  LeanTest.assertEqual (loaded.orderedSamples.getD 0 default).policyTarget.size 2
    "Replay roundtrip should preserve policy targets."

@[test]
def testAlphaGradTrainerTinyCheckpointCycle : IO Unit := do
  let runDir := s!"/tmp/alphagrad_trainer_{← IO.monoMsNow}"
  let trainCode ← Examples.AlphaGradPort.trainerMain [
    "train", "alphazero", "Perceptron",
    "--epochs", "1",
    "--episodes-per-epoch", "2",
    "--num-envs", "2",
    "--num-simulations", "2",
    "--batch-size", "4",
    "--update-batches", "1",
    "--checkpoint-every", "1",
    "--eval-every", "1",
    "--run-dir", runDir,
    "--overwrite"
  ]
  LeanTest.assertEqual trainCode 0
    "AlphaGradTrainer should complete a tiny AlphaZero training run."
  let latestExists ← System.FilePath.pathExists ⟨s!"{runDir}/checkpoints/latest/trainer_state.json"⟩
  LeanTest.assertTrue latestExists
    "AlphaGradTrainer should materialize a latest trainer checkpoint."
  let evalCode ← Examples.AlphaGradPort.trainerMain [
    "eval", "alphazero", "Perceptron",
    "--num-simulations", "2",
    "--run-dir", runDir
  ]
  LeanTest.assertEqual evalCode 0
    "AlphaGradTrainer should load and evaluate the latest checkpoint."

@[test]
def testAlphaGradObservationCapsPadAcrossTasks : IO Unit := do
  let taskA ←
    match (← materializeTask .roeFlux1d) with
    | .error msg => LeanTest.fail s!"RoeFlux_1d materialization failed: {msg}"
    | .ok task => pure task
  let taskB ←
    match (← materializeTask .perceptron) with
    | .error msg => LeanTest.fail s!"Perceptron materialization failed: {msg}"
    | .ok task => pure task
  let caps := taskSetObservationCaps #[taskA, taskB]
  let s0 ←
    match initAlphaGradState? taskA.graph taskA.numVertices with
    | .error msg => LeanTest.fail s!"RoeFlux_1d state init failed: {msg}"
    | .ok s => pure s
  let flat ←
    match exportObservationFlatWithCaps? taskA.envCfg s0 caps with
    | .error msg => LeanTest.fail s!"Capped observation export failed: {msg}"
    | .ok flat => pure flat
  LeanTest.assertEqual flat.size (caps.vertexCap * caps.tokenDim)
    "Capped observation export should match (vertexCap * tokenDim)."
  let attnMask := attentionMaskFromObservationFlat flat caps.vertexCap caps.tokenDim
  LeanTest.assertEqual attnMask.size caps.vertexCap
    "Capped observation attention mask should match the vertex cap."
  LeanTest.assertTrue (attnMask.extract taskA.numVertices caps.vertexCap |>.all (fun x => x < 0.5))
    "Capped observation export should mask padded vertices."

@[test]
def testAlphaGradMultiTrainerTinyCheckpointCycle : IO Unit := do
  let runDir := s!"/tmp/alphagrad_multi_trainer_{← IO.monoMsNow}"
  let trainCode ← Examples.AlphaGradPort.trainerMain [
    "multitrain", "alphazero", "RoeFlux_1d,Perceptron",
    "--epochs", "1",
    "--episodes-per-epoch", "1",
    "--num-envs", "1",
    "--num-simulations", "2",
    "--batch-size", "4",
    "--update-batches", "1",
    "--checkpoint-every", "1",
    "--eval-every", "1",
    "--run-dir", runDir,
    "--overwrite"
  ]
  LeanTest.assertEqual trainCode 0
    "AlphaGrad multi-trainer should complete a tiny curriculum run."
  let latestExists ← System.FilePath.pathExists ⟨s!"{runDir}/checkpoints/latest/multitask_trainer_state.json"⟩
  LeanTest.assertTrue latestExists
    "AlphaGrad multi-trainer should materialize a latest multi-task checkpoint."
  let evalCode ← Examples.AlphaGradPort.trainerMain [
    "multieval", "alphazero", "RoeFlux_1d,Perceptron",
    "--num-simulations", "2",
    "--run-dir", runDir
  ]
  LeanTest.assertEqual evalCode 0
    "AlphaGrad multi-trainer should load and evaluate the latest multi-task checkpoint."
  let benchmarkCode ← Examples.AlphaGradPort.benchmarkMain [
    "alphazero", "RoeFlux_1d,Perceptron",
    "--multitask",
    "--seeds", "1",
    "--base-seed", "7",
    "--num-simulations", "2",
    "--run-dir", runDir,
    "--output", s!"{runDir}/benchmark.json"
  ]
  LeanTest.assertEqual benchmarkCode 0
    "AlphaGrad benchmark should evaluate the multi-task checkpoint."
  let benchmarkExists ← System.FilePath.pathExists ⟨s!"{runDir}/benchmark.json"⟩
  LeanTest.assertTrue benchmarkExists
    "AlphaGrad benchmark should materialize a benchmark summary JSON file."

end Tests.AlphaGradPortExamples
