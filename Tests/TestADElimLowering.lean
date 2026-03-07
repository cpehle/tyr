import Lean.CoreM
import LeanTest
import Tyr.AD.Elim
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADElimLowering

open Lean
open Lean.IR
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

private def renderErrors (errs : Array String) : String :=
  String.intercalate "\n" errs.toList

private def arithmeticJaxpr : LeanJaxpr :=
  {
    invars := #[{ id := 1 }, { id := 2 }]
    eqns := #[
      {
        op := kstmtUnaryOpName .Exp
        invars := #[{ id := 1 }]
        outvars := #[{ id := 3 }]
      },
      {
        op := kstmtBinaryOpName .Add
        invars := #[{ id := 3 }, { id := 2 }]
        outvars := #[{ id := 4 }]
      }
    ]
    outvars := #[{ id := 4 }]
  }

@[test]
def testKStmtRoundtripArithmeticSubset : IO Unit := do
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
  let v4 : Tyr.GPU.Codegen.VarId := { idx := 4 }
  let stmts : Array KStmt := #[
    .unary .Exp v3 v1,
    .binary .Add v4 v3 v2
  ]

  let jaxpr ←
    match fromKStmts stmts with
    | .ok jaxpr => pure jaxpr
    | .error errs =>
      LeanTest.fail s!"fromKStmts failed unexpectedly:\n{renderErrors errs}"

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"lowerToKStmts failed unexpectedly:\n{renderErrors errs}"
  | .ok lowered =>
    LeanTest.assertTrue (lowered == stmts)
      "KStmt -> LeanJaxpr -> KStmt should preserve arithmetic subset statements"

@[test]
def testFnBodyRoundtripArithmeticSubset : IO Unit := do
  let lowered ←
    match lowerToFnBody arithmeticJaxpr `test.ad_elim_lowering.fnbody_roundtrip with
    | .ok lowered => pure lowered
    | .error errs =>
      LeanTest.fail s!"lowerToFnBody failed unexpectedly:\n{renderErrors errs}"

  let jaxprRoundtrip ←
    match fromFnBody lowered.declName lowered.params lowered.body with
    | .ok jaxpr => pure jaxpr
    | .error msg =>
      LeanTest.fail s!"fromFnBody failed after lowering:\n{msg}"

  LeanTest.assertEqual (jaxprRoundtrip.invars.map (fun v => v.id)) #[1, 2]
    "FnBody roundtrip should preserve arithmetic input IDs under normalized binder order"
  LeanTest.assertEqual (jaxprRoundtrip.outvars.map (fun v => v.id)) #[4]
    "FnBody roundtrip should preserve arithmetic output ID under normalized binder order"
  LeanTest.assertEqual (jaxprRoundtrip.eqns.map (fun e => e.op))
    #[kstmtUnaryOpName .Exp, kstmtBinaryOpName .Add]
    "FnBody roundtrip should preserve arithmetic op sequence"
  LeanTest.assertEqual (jaxprRoundtrip.eqns[0]!.invars.map (fun v => v.id)) #[1]
    "First arithmetic equation input should roundtrip"
  LeanTest.assertEqual (jaxprRoundtrip.eqns[0]!.outvars.map (fun v => v.id)) #[3]
    "First arithmetic equation output should roundtrip"
  LeanTest.assertEqual (jaxprRoundtrip.eqns[1]!.invars.map (fun v => v.id)) #[3, 2]
    "Second arithmetic equation inputs should roundtrip"
  LeanTest.assertEqual (jaxprRoundtrip.eqns[1]!.outvars.map (fun v => v.id)) #[4]
    "Second arithmetic equation output should roundtrip"

@[test]
def testEliminationDifferentialAcrossLoweringPaths : IO Unit := do
  let directResult ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxpr arithmeticJaxpr #[3]
  )

  let stmts ←
    match lowerToKStmts arithmeticJaxpr with
    | .ok stmts => pure stmts
    | .error errs =>
      LeanTest.fail s!"KStmt lowering failed:\n{renderErrors errs}"

  let viaKStmtResult ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnKStmts {} stmts #[3]
  )

  let loweredFn ←
    match lowerToFnBody arithmeticJaxpr `test.ad_elim_lowering.differential_fnbody with
    | .ok lowered => pure lowered
    | .error errs =>
      LeanTest.fail s!"FnBody lowering failed:\n{renderErrors errs}"

  let jaxprViaFnBody ←
    match fromFnBody loweredFn.declName loweredFn.params loweredFn.body with
    | .ok jaxpr => pure jaxpr
    | .error msg =>
      LeanTest.fail s!"FnBody roundtrip to LeanJaxpr failed: {msg}"

  let viaFnBodyResult ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    runEliminationOnJaxpr jaxprViaFnBody #[3]
  )

  let direct ←
    match directResult with
    | .ok result => pure result
    | .error msg => LeanTest.fail s!"Direct Jaxpr elimination failed: {msg}"

  let viaKStmt ←
    match viaKStmtResult with
    | .ok result => pure result
    | .error msg => LeanTest.fail s!"KStmt-path elimination failed: {msg}"

  let viaFnBody ←
    match viaFnBodyResult with
    | .ok result => pure result
    | .error msg => LeanTest.fail s!"FnBody-path elimination failed: {msg}"

  LeanTest.assertEqual direct.steps.size 1
    "Direct arithmetic elimination should have exactly one step"
  LeanTest.assertEqual viaKStmt.steps.size 1
    "KStmt-path arithmetic elimination should have exactly one step"
  LeanTest.assertEqual viaFnBody.steps.size 1
    "FnBody-path arithmetic elimination should have exactly one step"

  LeanTest.assertTrue (!(hasVertex direct.graph 3))
    "Direct arithmetic elimination should remove vertex 3"
  LeanTest.assertTrue (!(hasVertex viaKStmt.graph 3))
    "KStmt-path arithmetic elimination should remove vertex 3"
  LeanTest.assertTrue (!(hasVertex viaFnBody.graph 3))
    "FnBody-path arithmetic elimination should remove vertex 3"

  let outDirect1 := outNeighbors direct.graph 1 |>.map (fun p => p.1)
  let outKStmt1 := outNeighbors viaKStmt.graph 1 |>.map (fun p => p.1)
  let outFnBody1 := outNeighbors viaFnBody.graph 1 |>.map (fun p => p.1)
  LeanTest.assertEqual outKStmt1 outDirect1
    "KStmt-path elimination should match direct outgoing connectivity from source 1"
  LeanTest.assertEqual outFnBody1 outDirect1
    "FnBody-path elimination should match direct outgoing connectivity from source 1"

  let inDirect4 := inNeighbors direct.graph 4 |>.map (fun p => p.1)
  let inKStmt4 := inNeighbors viaKStmt.graph 4 |>.map (fun p => p.1)
  let inFnBody4 := inNeighbors viaFnBody.graph 4 |>.map (fun p => p.1)
  LeanTest.assertEqual inKStmt4 inDirect4
    "KStmt-path elimination should match direct incoming connectivity into sink 4"
  LeanTest.assertEqual inFnBody4 inDirect4
    "FnBody-path elimination should match direct incoming connectivity into sink 4"

end Tests.ADElimLowering
