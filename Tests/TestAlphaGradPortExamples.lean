import LeanTest
import Examples.AlphaGradPort.A0Train

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
    LeanTest.assertTrue (!task.edges.isEmpty)
      s!"{task.name} should expose non-empty local-Jac edges after KStmt lowering."
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

end Tests.AlphaGradPortExamples
