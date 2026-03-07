import Lean.CoreM
import LeanTest
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADJaxprLikeExtract

open Lean
open LeanTest
open Tyr.AD.JaxprLike
open Tyr.AD.Sparse
open Tyr.GPU.Codegen

private def approx (a b : Float) (tol : Float := 1e-9) : Bool :=
  Float.abs (a - b) < tol

private def findEdge? (edges : Array LocalJacEdge) (src dst : Nat) : Option LocalJacEdge :=
  edges.find? (fun e => e.src = src && e.dst = dst)

private def edgeWeight? (e : LocalJacEdge) : Option Float :=
  if e.map.entries.size = 1 then
    match e.map.entries[0]? with
    | some entry => some entry.weight
    | none => none
  else
    none

private def atomName (value : String) : Lean.Name :=
  Lean.Name.str Lean.Name.anonymous value

private def reduceOpTagName (op : ReduceOp) : Lean.Name :=
  atomName (kstmtReduceOpTag op)

private def reduceAxisTagName (axis : ReduceAxis) : Lean.Name :=
  atomName (kstmtReduceAxisTag axis)

private def broadcastAxisTagName (axis : BroadcastAxis) : Lean.Name :=
  atomName (kstmtBroadcastAxisTag axis)

private def unaryTag (op : UnaryOp) (mode : JacMode := .none) : SparseMapTag :=
  .semantic (.unary (kstmtUnaryOpName op) .x mode)

private def binaryTag (op : BinaryOp) (arg : JacArgRole) (mode : JacMode := .none) : SparseMapTag :=
  .semantic (.binary (kstmtBinaryOpName op) arg mode)

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
        source := { decl := `test.extract, line? := some 10 }
      },
      {
        op := kstmtBinaryOpName .Add
        invars := #[{ id := 2 }, { id := 1 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.extract, line? := some 11 }
      }
    ]
    outvars := #[{ id := 3 }]
  }

@[test]
def testExtractLocalJacEdgesReportsMissingRules : IO Unit := do
  let res ← runCoreM (extractLocalJacEdges sampleKStmtLoweredJaxpr)
  match res with
  | .ok _ =>
    LeanTest.fail "extractLocalJacEdges should fail when no rules are registered"
  | .error errs =>
    LeanTest.assertEqual errs.size 2 "Expected one rule-execution error per equation"
    let first := errs.getD 0 { eqnIndex0 := 999, op := .anonymous, source := {}, message := "" }
    LeanTest.assertEqual first.eqnIndex0 0 "First missing rule should correspond to equation #0"
    LeanTest.assertTrue (first.message.contains "No local-Jacobian rule")
      s!"Expected unsupported-op diagnostic, got: {first.message}"

@[test]
def testExtractLocalJacEdgesSucceedsWithKStmtRulePack : IO Unit := do
  let res ← runCoreM (do
    registerKStmtUnaryBinaryPlaceholderRules
    extractLocalJacEdges sampleKStmtLoweredJaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"extractLocalJacEdges should succeed after rule-pack registration, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 3 "Default placeholder rules should emit one edge per input var"
    let pairs := edges.map (fun e => (e.src, e.dst))
    LeanTest.assertEqual pairs #[(0, 2), (2, 3), (1, 3)]
      "Edge endpoints should follow equation/invar traversal order"
    LeanTest.assertTrue (edges.all (fun e => e.map.repr == Tyr.AD.Sparse.SparseMapTag.identityLike))
      "Registered KStmt placeholder rule-pack should emit identity-like maps"

@[test]
def testBuildAndExtractFromKStmtsHonorsCoverage : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let noRules ← runCoreM (buildAndExtractFromKStmts {} stmts)
  match noRules with
  | .ok _ =>
    LeanTest.fail "buildAndExtractFromKStmts should fail coverage when no rules are registered"
  | .error (.build (.coverage msgs)) =>
    LeanTest.assertTrue (!msgs.isEmpty) "Coverage failure should contain diagnostics"
  | .error err =>
    LeanTest.fail s!"Expected build coverage failure, got: {buildExtractErrorToString err}"

  let withRules ← runCoreM (do
    registerKStmtUnaryBinaryPlaceholderRules
    buildAndExtractFromKStmts {} stmts
  )
  match withRules with
  | .error err =>
    LeanTest.fail s!"buildAndExtractFromKStmts should succeed with rule-pack, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 3 "Expected extracted local-Jac edges from two lowered equations"

@[test]
def testBuildAndExtractFromKStmtsExtendedSubsetWithAllSupportedPack : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let v3 : VarId := { idx := 3 }
  let v4 : VarId := { idx := 4 }
  let stmts := #[
    KStmt.broadcast .Row v1 v0,
    KStmt.reduce .Sum .Col v2 v1,
    KStmt.transpose v3 v2,
    KStmt.concatCols v4 v3 v0
  ]

  let noRules ← runCoreM (buildAndExtractFromKStmts {} stmts)
  match noRules with
  | .ok _ =>
    LeanTest.fail "Extended KStmt lowering should fail coverage when no rules are registered"
  | .error (.build (.coverage msgs)) =>
    LeanTest.assertTrue (!msgs.isEmpty) "Coverage failure should contain diagnostics for structural ops"
  | .error err =>
    LeanTest.fail s!"Expected build coverage failure for structural subset, got: {buildExtractErrorToString err}"

  let withRules ← runCoreM (do
    registerKStmtAllSupportedPlaceholderRules
    buildAndExtractFromKStmts {} stmts
  )
  match withRules with
  | .error err =>
    LeanTest.fail s!"All-supported placeholder pack should cover structural subset, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 5
      "Expected placeholder edges for broadcast/reduce/transpose/concatCols (1+1+1+2)"

@[test]
def testBuildAndExtractFromKStmtsLinearSemanticsRules : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Neg v1 v0,
    KStmt.binary .Sub v2 v1 v0
  ]

  let res ← runCoreM (do
    registerKStmtLinearSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Linear semantics rule-pack should support Neg/Sub path, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 3 "Expected one unary edge + two binary edges"
    match findEdge? edges 0 1 with
    | none => LeanTest.fail "Missing expected Neg edge 0 -> 1"
    | some e =>
      LeanTest.assertTrue (e.map.repr == unaryTag .Neg)
        s!"Unexpected unary Neg map repr: {e.map.repr}"
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) (-1.0))
        s!"Unexpected unary Neg weight: {edgeWeight? e}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing expected Sub edge 1 -> 2"
    | some e =>
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) 1.0)
        s!"Unexpected left-Sub weight: {edgeWeight? e}"

    match findEdge? edges 0 2 with
    | none => LeanTest.fail "Missing expected Sub edge 0 -> 2"
    | some e =>
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) (-1.0))
        s!"Unexpected right-Sub weight: {edgeWeight? e}"

@[test]
def testBuildAndExtractFromKStmtsHybridRulesCoverExpAndAdd : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let res ← runCoreM (do
    registerKStmtUnaryBinaryHybridRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Hybrid rule-pack should cover Exp/Add path, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 3 "Expected one unary edge + two binary edges"
    match findEdge? edges 0 1 with
    | none => LeanTest.fail "Missing expected Exp edge 0 -> 1"
    | some e =>
      LeanTest.assertTrue (e.map.repr == unaryTag .Exp)
        s!"Exp should use symbolic semantic map in hybrid pack, got: {e.map.repr}"
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) 1.0)
        s!"Unexpected Exp weight: {edgeWeight? e}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing expected Add edge 1 -> 2"
    | some e =>
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) 1.0)
        s!"Unexpected left-Add weight: {edgeWeight? e}"

    match findEdge? edges 0 2 with
    | none => LeanTest.fail "Missing expected Add edge 0 -> 2"
    | some e =>
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) 1.0)
        s!"Unexpected right-Add weight: {edgeWeight? e}"

@[test]
def testBuildAndExtractFromKStmtsStructuralSemanticsInHybridPack : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let v3 : VarId := { idx := 3 }
  let v4 : VarId := { idx := 4 }
  let v5 : VarId := { idx := 5 }
  let v6 : VarId := { idx := 6 }
  let v7 : VarId := { idx := 7 }
  let stmts := #[
    KStmt.broadcast .Row v1 v0,
    KStmt.reduce .Sum .Col v2 v1,
    KStmt.transpose v3 v2,
    KStmt.concatCols v4 v3 v0,
    KStmt.outer v5 v4 v2,
    KStmt.cumsum .Row v6 v5,
    KStmt.cumprod .Row v7 v6
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedHybridRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Hybrid pack should include structural semantics, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 9
      "Expected structural-edge extraction (1+1+1+2+2+1+1)"

    match findEdge? edges 0 1 with
    | none => LeanTest.fail "Missing broadcast edge 0 -> 1"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.broadcast (broadcastAxisTagName .Row) .x .expand))
        s!"Unexpected broadcast repr: {e.map.repr}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing reduce edge 1 -> 2"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.reduce (reduceOpTagName .Sum) (reduceAxisTagName .Col) .x .contract))
        s!"Unexpected reduce repr: {e.map.repr}"

    match findEdge? edges 2 3 with
    | none => LeanTest.fail "Missing transpose edge 2 -> 3"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.transpose .x .permute))
        s!"Unexpected transpose repr: {e.map.repr}"

    match findEdge? edges 3 4 with
    | none => LeanTest.fail "Missing concat-left edge 3 -> 4"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.concatCols .lhs .inject))
        s!"Unexpected concat-left repr: {e.map.repr}"

    match findEdge? edges 0 4 with
    | none => LeanTest.fail "Missing concat-right edge 0 -> 4"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.concatCols .rhs .inject))
        s!"Unexpected concat-right repr: {e.map.repr}"

    match findEdge? edges 4 5 with
    | none => LeanTest.fail "Missing outer-left edge 4 -> 5"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.outer .a .kronOther))
        s!"Unexpected outer-left repr: {e.map.repr}"

    match findEdge? edges 2 5 with
    | none => LeanTest.fail "Missing outer-right edge 2 -> 5"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.outer .b .kronOther))
        s!"Unexpected outer-right repr: {e.map.repr}"

    match findEdge? edges 5 6 with
    | none => LeanTest.fail "Missing cumsum edge 5 -> 6"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.cumsum (reduceAxisTagName .Row) .x .prefix))
        s!"Unexpected cumsum repr: {e.map.repr}"

    match findEdge? edges 6 7 with
    | none => LeanTest.fail "Missing cumprod edge 6 -> 7"
    | some e =>
      LeanTest.assertTrue (e.map.repr == .semantic (.cumprod (reduceAxisTagName .Row) .x .prefixProduct))
        s!"Unexpected cumprod repr: {e.map.repr}"

@[test]
def testBuildAndExtractFromKStmtsGraphaxAlphaGradSemanticsRules : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let v3 : VarId := { idx := 3 }
  let v4 : VarId := { idx := 4 }
  let stmts := #[
    KStmt.binary .Mul v2 v0 v1,
    KStmt.binary .Div v3 v2 v1,
    KStmt.binary .Max v4 v3 v0
  ]

  let res ← runCoreM (do
    registerKStmtGraphaxAlphaGradSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Graphax/AlphaGrad semantics pack should cover Mul/Div/Max, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 6 "Expected two edges per binary equation"

    match findEdge? edges 0 2 with
    | none => LeanTest.fail "Missing Mul edge 0 -> 2"
    | some e =>
      LeanTest.assertTrue (e.map.repr == binaryTag .Mul .a .rhsValue)
        s!"Unexpected Mul-left repr: {e.map.repr}"
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) 1.0)
        s!"Unexpected Mul-left weight: {edgeWeight? e}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing Mul edge 1 -> 2"
    | some e =>
      LeanTest.assertTrue (e.map.repr == binaryTag .Mul .b .lhsValue)
        s!"Unexpected Mul-right repr: {e.map.repr}"

    match findEdge? edges 1 3 with
    | none => LeanTest.fail "Missing Div-right edge 1 -> 3"
    | some e =>
      LeanTest.assertTrue (e.map.repr == binaryTag .Div .b .negLhsOverRhsSq)
        s!"Unexpected Div-right repr: {e.map.repr}"
      LeanTest.assertTrue (approx ((edgeWeight? e).getD 0.0) (-1.0))
        s!"Unexpected Div-right weight: {edgeWeight? e}"

    match findEdge? edges 3 4 with
    | none => LeanTest.fail "Missing Max-left edge 3 -> 4"
    | some e =>
      LeanTest.assertTrue (e.map.repr == binaryTag .Max .a .mask)
        s!"Unexpected Max-left repr: {e.map.repr}"

    match findEdge? edges 0 4 with
    | none => LeanTest.fail "Missing Max-right edge 0 -> 4"
    | some e =>
      LeanTest.assertTrue (e.map.repr == binaryTag .Max .b .complementMask)
        s!"Unexpected Max-right repr: {e.map.repr}"

@[test]
def testExtractNoGradControlRulesStopGradientAndIota : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }]
    eqns := #[
      {
        op := stopGradientOpName
        invars := #[{ id := 0 }]
        outvars := #[{ id := 1 }]
        source := { decl := `test.no_grad_control, line? := some 40 }
      },
      {
        op := iotaOpName
        invars := #[]
        outvars := #[{ id := 2 }]
        source := { decl := `test.no_grad_control, line? := some 41 }
      },
      {
        op := kstmtBinaryOpName .Add
        invars := #[{ id := 1 }, { id := 2 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.no_grad_control, line? := some 42 }
      }
    ]
    outvars := #[{ id := 3 }]
  }

  let res ← runCoreM (do
    registerKStmtGraphaxAlphaGradSemanticsRules
    registerGraphaxAlphaGradNoGradControlRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"No-grad/control rule-pack should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 2
      "stop_gradient and iota should contribute zero local-Jac edges; Add should contribute two."
    LeanTest.assertTrue ((findEdge? edges 0 1).isNone)
      "stop_gradient should block local-Jac edge from input to its output."
    LeanTest.assertTrue ((findEdge? edges 1 3).isSome)
      "Add should include lhs edge from stop_gradient output."
    LeanTest.assertTrue ((findEdge? edges 2 3).isSome)
      "Add should include rhs edge from iota output."

@[test]
def testExtractDotGeneralRules : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }]
    eqns := #[
      {
        op := kstmtDotGeneralOpName
        invars := #[{ id := 0 }, { id := 1 }]
        outvars := #[{ id := 2 }]
        params := #[
          OpParam.mkName .variant `matmul,
          OpParam.mkNats .lhsContract #[1],
          OpParam.mkNats .rhsContract #[0],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
        source := { decl := `test.dot_general, line? := some 60 }
      }
    ]
    outvars := #[{ id := 2 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradDotGeneralRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"dot_general rule-pack should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 2 "dot_general should emit lhs/rhs local-Jac edges"
    match findEdge? edges 0 2 with
    | none => LeanTest.fail "Missing dot_general lhs edge 0 -> 2"
    | some e =>
      LeanTest.assertTrue
        (e.map.repr == .semantic (.dotGeneral {
          variant := `matmul
          arg := .lhs
          lhsContract := #[1]
          rhsContract := #[0]
          lhsBatch := #[]
          rhsBatch := #[]
        }))
        s!"Unexpected dot_general lhs repr: {e.map.repr}"
    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing dot_general rhs edge 1 -> 2"
    | some e =>
      LeanTest.assertTrue
        (e.map.repr == .semantic (.dotGeneral {
          variant := `matmul
          arg := .rhs
          lhsContract := #[1]
          rhsContract := #[0]
          lhsBatch := #[]
          rhsBatch := #[]
        }))
        s!"Unexpected dot_general rhs repr: {e.map.repr}"

end Tests.ADJaxprLikeExtract
