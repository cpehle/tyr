import Lean.CoreM
import LeanTest
import Tyr.AD.Elim
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADElimFromJaxpr

open Lean
open LeanTest
open Tyr.AD.Elim
open Tyr.AD.JaxprLike
open Tyr.GPU.Codegen

def runCoreMResult (x : CoreM α) : IO (Except String α) := do
  let env ← mkEmptyEnvironment
  let ctx : Core.Context := { fileName := "<test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure (.ok value)
  | .error err =>
    let msg ← err.toMessageData.toString
    pure (.error msg)

def runCoreM (x : CoreM α) : IO α := do
  match (← runCoreMResult x) with
  | .ok value => pure value
  | .error msg => throw (IO.userError msg)

private def sampleKStmtLoweredJaxpr : LeanJaxpr :=
  {
    invars := #[{ id := 0 }, { id := 1 }]
    eqns := #[
      {
        op := kstmtUnaryOpName .Exp
        invars := #[{ id := 0 }]
        outvars := #[{ id := 2 }]
        source := { decl := `test.elim_from_jaxpr, line? := some 10 }
      },
      {
        op := kstmtBinaryOpName .Add
        invars := #[{ id := 2 }, { id := 1 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.elim_from_jaxpr, line? := some 11 }
      }
    ]
    outvars := #[{ id := 3 }]
  }

@[test]
def testRunEliminationOnJaxprRequiresRules : IO Unit := do
  let res ← runCoreM (runEliminationOnJaxpr sampleKStmtLoweredJaxpr #[2])
  match res with
  | .ok _ =>
    LeanTest.fail "runEliminationOnJaxpr should fail when equation rules are missing"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "Local-Jacobian extraction failed")
      s!"Expected extraction-failure header, got: {msg}"
    LeanTest.assertTrue (msg.contains "eqn #0")
      s!"Expected equation-index context in diagnostic, got: {msg}"

@[test]
def testRunEliminationOnJaxprSucceedsWithRulePack : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxpr sampleKStmtLoweredJaxpr #[2]
  )
  match res with
  | .error msg =>
    LeanTest.fail s!"runEliminationOnJaxpr should succeed with rules registered, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual out.steps.size 1 "Expected one elimination step"
    LeanTest.assertTrue (!(hasVertex out.graph 2))
      "Eliminated intermediate vertex should be removed from incident edges"

@[test]
def testBuildElimGraphFromJaxprTracksPartitions : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    buildElimGraphFromJaxpr sampleKStmtLoweredJaxpr
  )
  match res with
  | .error msg =>
    LeanTest.fail s!"buildElimGraphFromJaxpr should succeed with rules registered, got: {msg}"
  | .ok g =>
    LeanTest.assertEqual g.inputs #[0, 1]
      "Partitioned graph should keep input variables as explicit boundary inputs"
    LeanTest.assertEqual g.outputs #[3]
      "Partitioned graph should keep final outputs as explicit boundary outputs"
    LeanTest.assertEqual g.eliminable #[2]
      "Partitioned graph should expose only non-output intermediates as eliminable vertices"

@[test]
def testForwardAndReverseEliminationOnJaxpr : IO Unit := do
  let fwd ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runForwardEliminationOnJaxpr sampleKStmtLoweredJaxpr
  )
  match fwd with
  | .error msg =>
    LeanTest.fail s!"runForwardEliminationOnJaxpr should succeed, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual (out.steps.map (fun s => s.vertex)) #[2]
      "Forward Graphax-style elimination should consume the sole eliminable vertex"

  let rev ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runReverseEliminationOnJaxpr sampleKStmtLoweredJaxpr
  )
  match rev with
  | .error msg =>
    LeanTest.fail s!"runReverseEliminationOnJaxpr should succeed, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual (out.steps.map (fun s => s.vertex)) #[2]
      "Reverse Graphax-style elimination should match forward order for a single eliminable vertex"

@[test]
def testRunEliminationOnJaxprWithPolicy : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxprWithPolicy sampleKStmtLoweredJaxpr .forward
  )
  match res with
  | .error msg =>
    LeanTest.fail s!"runEliminationOnJaxprWithPolicy should execute forward policy, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual (out.steps.map (fun s => s.vertex)) #[2]
      "Forward policy should execute over the graph's eliminable order"

@[test]
def testRunEliminationOnJaxprWithExplicitPolicyRejectsBoundaryVertex : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxprWithPolicy sampleKStmtLoweredJaxpr (.explicitVertex #[1])
  )
  match res with
  | .ok _ =>
    LeanTest.fail "Explicit policy should reject non-eliminable boundary vertices"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "non-eliminable vertex 1")
      s!"Expected non-eliminable-vertex diagnostic, got: {msg}"

@[test]
def testRunEliminationOnJaxprWithUnresolvedHeuristicPolicy : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxprWithPolicy sampleKStmtLoweredJaxpr (.heuristic "markowitz")
  )
  match res with
  | .ok _ =>
    LeanTest.fail "Unresolved heuristic policy should fail until a scheduler is implemented"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "does not resolve to a concrete elimination order")
      s!"Expected unresolved-policy diagnostic, got: {msg}"

@[test]
def testRunEliminationOnKStmtsCoverageBoundary : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let noRules ← runCoreM (runEliminationOnKStmts {} stmts #[1])
  match noRules with
  | .ok _ =>
    LeanTest.fail "runEliminationOnKStmts should fail coverage when no rules are registered"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "Rule coverage error")
      s!"Expected coverage failure, got: {msg}"

  let withRules ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnKStmts {} stmts #[1]
  )
  match withRules with
  | .error msg =>
    LeanTest.fail s!"runEliminationOnKStmts should succeed with rule-pack, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual out.steps.size 1 "Expected one elimination step after lowering+extraction"
    LeanTest.assertTrue (!(hasVertex out.graph 1))
      "Eliminated vertex should be removed after successful run"

@[test]
def testRunEliminationOnKStmtsWithPolicy : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnKStmtsWithPolicy {} stmts .forward
  )
  match res with
  | .error msg =>
    LeanTest.fail s!"runEliminationOnKStmtsWithPolicy should execute forward policy, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual (out.steps.map (fun s => s.vertex)) #[1]
      "Forward policy on lowered KStmt graph should eliminate the sole interior vertex"

end Tests.ADElimFromJaxpr
