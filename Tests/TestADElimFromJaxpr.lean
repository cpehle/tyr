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

end Tests.ADElimFromJaxpr
