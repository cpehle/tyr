import LeanTest
import Examples.AlphaGradPort.A0Train

namespace Tests.AlphaGradPortExamples

open LeanTest
open Examples.AlphaGradPort
open Tyr.AD.Elim

@[test]
def testMaterializeAllAlphaGradPortTasksAndBaseline : IO Unit := do
  for taskName in taskSequence do
    match (← materializeTask taskName) with
    | .error msg =>
      LeanTest.fail s!"Task materialization failed for {taskName}: {msg}"
    | .ok task =>
      LeanTest.assertTrue (task.numVertices > 0)
        s!"Task {task.name} must declare at least one vertex."
      LeanTest.assertTrue (!task.edges.isEmpty)
        s!"Task {task.name} must expose at least one local-Jac edge."

      let cfg : Examples.AlphaGradPort.RunConfig := {
        episodes := 0
        backend := .dagGumbel
        logEvery := 0
      }
      match (← runTask task cfg) with
      | .error msg =>
        LeanTest.fail s!"Baseline evaluation failed for {task.name}: {msg}"
      | .ok summary =>
        LeanTest.assertEqual summary.taskName task.name
          "Run summary should preserve task name."
        LeanTest.assertEqual summary.episodes 0
          "Baseline run should preserve requested episode count."
        LeanTest.assertTrue (summary.bestActions0.isEmpty && summary.bestOrder1.isEmpty)
          "Zero-episode baseline should not emit sampled action/order traces."

@[test]
def testPerceptronSearchParityAcrossPolicies : IO Unit := do
  let task ←
    match (← materializeTask .perceptron) with
    | .error msg => LeanTest.fail s!"Perceptron materialization failed: {msg}"
    | .ok task => pure task

  let mctsCfg : AlphaGradMctsConfig := {
    numSimulations := 6
    maxDepth := some task.numVertices
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
      LeanTest.assertEqual out.actions0.size task.numVertices
        s!"{label} should emit one action per vertex."
      LeanTest.assertEqual out.order1.size task.numVertices
        s!"{label} should emit one vertex per action."
      LeanTest.assertTrue (hasNoDuplicates out.actions0)
        s!"{label} action trace must be duplicate-free."
      LeanTest.assertTrue (hasNoDuplicates out.order1)
        s!"{label} vertex order must be duplicate-free."
      LeanTest.assertTrue (out.finalState.violation?.isNone)
        s!"{label} should complete without constraint violations."

  let gumbel :=
    searchEpisodeFromEdges? task.envCfg mctsCfg 2026030601 task.edges task.numVertices
  let dagAlphaZero :=
    searchEpisodeDagWithPolicyFromEdges? .alphaZero task.envCfg mctsCfg 2026030602 task.edges task.numVertices
  let dagGumbel :=
    searchEpisodeDagWithPolicyFromEdges? .gumbelMuZero task.envCfg mctsCfg 2026030603 task.edges task.numVertices

  checkEpisode "gumbel" gumbel
  checkEpisode "dag-alphaZero" dagAlphaZero
  checkEpisode "dag-gumbelMuZero" dagGumbel

end Tests.AlphaGradPortExamples
