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
def testLowerToKStmtsDotGeneralAndMma : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 1 }, { id := 2 }, { id := 3 }]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[{ id := 1 }, { id := 2 }]
        outvars := #[{ id := 4 }]
        params := #[
          OpParam.mkNats .lhsContract #[1],
          OpParam.mkNats .rhsContract #[0],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
      },
      {
        op := kstmtMmaOpName .AB
        invars := #[{ id := 1 }, { id := 2 }, { id := 3 }]
        outvars := #[{ id := 5 }]
      }
    ]
    outvars := #[{ id := 5 }]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"dot_general/mma lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    let v4 : Tyr.GPU.Codegen.VarId := { idx := 4 }
    let v5 : Tyr.GPU.Codegen.VarId := { idx := 5 }
    let expected : Array KStmt := #[
      .mm .AB v4 v1 v2,
      .mma .AB v5 v1 v2 v3
    ]
    LeanTest.assertTrue (lowered == expected)
      s!"dot_general/mma lowering mismatch; expected={reprStr expected}, got={reprStr lowered}"

@[test]
def testLowerToKStmtsOuterLikeDotGeneralAlias : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 7 }, { id := 8 }]
    eqns := #[
      {
        op := `Graphax.dot_general
        invars := #[{ id := 7 }, { id := 8 }]
        outvars := #[{ id := 9 }]
      }
    ]
    outvars := #[{ id := 9 }]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"outer-like dot_general alias lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v7 : Tyr.GPU.Codegen.VarId := { idx := 7 }
    let v8 : Tyr.GPU.Codegen.VarId := { idx := 8 }
    let v9 : Tyr.GPU.Codegen.VarId := { idx := 9 }
    LeanTest.assertTrue (lowered == #[.outer v9 v7 v8])
      s!"outer-like dot_general alias should lower to KStmt.outer, got={reprStr lowered}"

@[test]
def testLowerToKStmtsUnitBatchDotGeneralToMM : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[1, 2, 3] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[1, 3, 4] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[1, 2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[1],
          OpParam.mkNats .lhsBatch #[0],
          OpParam.mkNats .rhsBatch #[0]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"unit-batch dot_general lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    LeanTest.assertTrue (lowered == #[.mm .AB v3 v1 v2])
      s!"unit-batch dot_general should lower to mm.AB, got={reprStr lowered}"

@[test]
def testLowerToKStmtsMultiLeadingUnitBatchDotGeneralToMM : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[1, 1, 2, 3] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[1, 1, 3, 4] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[1, 1, 2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[3],
          OpParam.mkNats .rhsContract #[2],
          OpParam.mkNats .lhsBatch #[0, 1],
          OpParam.mkNats .rhsBatch #[0, 1]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"multi-leading-unit-batch dot_general lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    LeanTest.assertTrue (lowered == #[.mm .AB v3 v1 v2])
      s!"multi-leading-unit-batch dot_general should lower to mm.AB, got={reprStr lowered}"

@[test]
def testLowerToKStmtsNonLeadingUnitBatchDotGeneralToMM : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[2, 1, 3] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[1, 3, 4] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[1, 2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[1],
          OpParam.mkNats .lhsBatch #[1],
          OpParam.mkNats .rhsBatch #[0]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"non-leading unit-batch dot_general lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    LeanTest.assertTrue (lowered == #[.mm .AB v3 v1 v2])
      s!"non-leading unit-batch dot_general should lower to mm.AB, got={reprStr lowered}"

@[test]
def testLowerToKStmtsUnitBatchDotGeneralTransposeVariants : IO Unit := do
  let lhsABt : JVar := { id := 1, metaInfo := { shape := some #[2, 1, 3] } }
  let rhsABt : JVar := { id := 2, metaInfo := { shape := some #[4, 3, 1] } }
  let outABt : JVar := { id := 3, metaInfo := { shape := some #[1, 2, 4] } }
  let lhsAtB : JVar := { id := 4, metaInfo := { shape := some #[3, 2, 1] } }
  let rhsAtB : JVar := { id := 5, metaInfo := { shape := some #[1, 3, 4] } }
  let outAtB : JVar := { id := 6, metaInfo := { shape := some #[1, 2, 4] } }
  let lhsAtBt : JVar := { id := 7, metaInfo := { shape := some #[3, 2, 1] } }
  let rhsAtBt : JVar := { id := 8, metaInfo := { shape := some #[4, 1, 3] } }
  let outAtBt : JVar := { id := 9, metaInfo := { shape := some #[1, 2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhsABt, rhsABt, lhsAtB, rhsAtB, lhsAtBt, rhsAtBt]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhsABt, rhsABt]
        outvars := #[outABt]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[1],
          OpParam.mkNats .lhsBatch #[1],
          OpParam.mkNats .rhsBatch #[2]
        ]
      },
      {
        op := kstmtDotGeneralOpName
        invars := #[lhsAtB, rhsAtB]
        outvars := #[outAtB]
        params := #[
          OpParam.mkNats .lhsContract #[0],
          OpParam.mkNats .rhsContract #[1],
          OpParam.mkNats .lhsBatch #[2],
          OpParam.mkNats .rhsBatch #[0]
        ]
      },
      {
        op := kstmtDotGeneralOpName
        invars := #[lhsAtBt, rhsAtBt]
        outvars := #[outAtBt]
        params := #[
          OpParam.mkNats .lhsContract #[0],
          OpParam.mkNats .rhsContract #[2],
          OpParam.mkNats .lhsBatch #[2],
          OpParam.mkNats .rhsBatch #[1]
        ]
      }
    ]
    outvars := #[outABt, outAtB, outAtBt]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"unit-batch transpose dot_general lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    let v4 : Tyr.GPU.Codegen.VarId := { idx := 4 }
    let v5 : Tyr.GPU.Codegen.VarId := { idx := 5 }
    let v6 : Tyr.GPU.Codegen.VarId := { idx := 6 }
    let v7 : Tyr.GPU.Codegen.VarId := { idx := 7 }
    let v8 : Tyr.GPU.Codegen.VarId := { idx := 8 }
    let v9 : Tyr.GPU.Codegen.VarId := { idx := 9 }
    LeanTest.assertTrue
      (lowered == #[
        .mm .ABt v3 v1 v2,
        .mm .AtB v6 v4 v5,
        .mm .AtBt v9 v7 v8
      ])
      s!"unit-batch transpose dot_general variants should lower to the expected mm transposes, got={reprStr lowered}"

@[test]
def testLowerToKStmtsDotGeneralDropsUnitFreeAndContractAxesToMM : IO Unit := do
  let lhsFreeUnit : JVar := { id := 1, metaInfo := { shape := some #[1, 2, 3] } }
  let rhsFreeUnit : JVar := { id := 2, metaInfo := { shape := some #[3, 4] } }
  let outFreeUnit : JVar := { id := 3, metaInfo := { shape := some #[1, 2, 4] } }
  let lhsContractUnit : JVar := { id := 4, metaInfo := { shape := some #[2, 1, 3] } }
  let rhsContractUnit : JVar := { id := 5, metaInfo := { shape := some #[1, 3, 4] } }
  let outContractUnit : JVar := { id := 6, metaInfo := { shape := some #[2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhsFreeUnit, rhsFreeUnit, lhsContractUnit, rhsContractUnit]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhsFreeUnit, rhsFreeUnit]
        outvars := #[outFreeUnit]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[0],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
      },
      {
        op := kstmtDotGeneralOpName
        invars := #[lhsContractUnit, rhsContractUnit]
        outvars := #[outContractUnit]
        params := #[
          OpParam.mkNats .lhsContract #[1, 2],
          OpParam.mkNats .rhsContract #[0, 1],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
      }
    ]
    outvars := #[outFreeUnit, outContractUnit]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"degenerate singleton dot_general lowering to mm should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    let v4 : Tyr.GPU.Codegen.VarId := { idx := 4 }
    let v5 : Tyr.GPU.Codegen.VarId := { idx := 5 }
    let v6 : Tyr.GPU.Codegen.VarId := { idx := 6 }
    LeanTest.assertTrue
      (lowered == #[
        .mm .AB v3 v1 v2,
        .mm .AB v6 v4 v5
      ])
      s!"singleton free/contract axes should collapse to mm.AB, got={reprStr lowered}"

@[test]
def testLowerToKStmtsDotGeneralDropsUnitContractAxesToOuter : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[5, 1, 1] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[1, 1, 7] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[5, 7] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[1, 2],
          OpParam.mkNats .rhsContract #[0, 1],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"degenerate singleton contract dot_general lowering to outer should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    LeanTest.assertTrue (lowered == #[.outer v3 v1 v2])
      s!"singleton contract axes should collapse to outer, got={reprStr lowered}"

@[test]
def testLowerToKStmtsUnitBatchDotGeneralToOuter : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[5, 1] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[1, 7] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[1, 5, 7] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := `Graphax.dot_general
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[],
          OpParam.mkNats .rhsContract #[],
          OpParam.mkNats .lhsBatch #[1],
          OpParam.mkNats .rhsBatch #[0]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .error errs =>
    LeanTest.fail s!"unit-batch outer-like dot_general lowering should succeed, got:\n{renderErrors errs}"
  | .ok lowered =>
    let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
    let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
    let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
    LeanTest.assertTrue (lowered == #[.outer v3 v1 v2])
      s!"unit-batch outer-like dot_general should lower to outer, got={reprStr lowered}"

@[test]
def testLowerToKStmtsUnitBatchDotGeneralRejectsNonCanonicalOutputShape : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[2, 1, 3] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[2, 1, 4] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[2, 1, 3] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[2],
          OpParam.mkNats .lhsBatch #[1],
          OpParam.mkNats .rhsBatch #[1]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .ok lowered =>
    LeanTest.fail s!"non-canonical unit-batch dot_general should be rejected for KStmt lowering, got={reprStr lowered}"
  | .error errs =>
    LeanTest.assertTrue (!errs.isEmpty) "Expected non-empty lowering errors."
    LeanTest.assertTrue ((errs[0]!).contains "not representable as KStmt mm/outer")
      s!"Expected representability error, got: {errs[0]!}"

@[test]
def testLowerToKStmtsNonUnitBatchDotGeneralRejects : IO Unit := do
  let lhs : JVar := { id := 1, metaInfo := { shape := some #[2, 2, 3] } }
  let rhs : JVar := { id := 2, metaInfo := { shape := some #[2, 3, 4] } }
  let out : JVar := { id := 3, metaInfo := { shape := some #[2, 2, 4] } }
  let jaxpr : LeanJaxpr := {
    invars := #[lhs, rhs]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[lhs, rhs]
        outvars := #[out]
        params := #[
          OpParam.mkNats .lhsContract #[2],
          OpParam.mkNats .rhsContract #[1],
          OpParam.mkNats .lhsBatch #[0],
          OpParam.mkNats .rhsBatch #[0]
        ]
      }
    ]
    outvars := #[out]
  }

  match lowerToKStmts jaxpr with
  | .ok lowered =>
    LeanTest.fail s!"non-unit-batch dot_general should be rejected for KStmt lowering, got={reprStr lowered}"
  | .error errs =>
    LeanTest.assertTrue (!errs.isEmpty) "Expected non-empty lowering errors."
    LeanTest.assertTrue ((errs[0]!).contains "not representable as KStmt mm/outer")
      s!"Expected representability error, got: {errs[0]!}"

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
