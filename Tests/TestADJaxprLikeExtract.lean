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

private def hasEntry (entries : Array SparseEntry) (src dst : Nat) (weight : Float := 1.0) : Bool :=
  entries.any (fun e => e.src = src && e.dst = dst && approx e.weight weight)

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

private def jaxprFingerprint (jaxpr : LeanJaxpr) : String :=
  reprStr jaxpr

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
def testBuildFromKStmtsWrapperParityWithManualPath : IO Unit := do
  let cfg : BuildConfig := { requireRuleCoverage := false }
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let built ← runCoreM (buildFromKStmts cfg stmts)
  let manual : Except BuildError LeanJaxpr :=
    match fromKStmts stmts with
    | .error msgs =>
      .error (.conversion (String.intercalate "\n" msgs.toList))
    | .ok jaxpr =>
      match validate jaxpr with
      | .error errs => .error (.validation errs)
      | .ok () => .ok jaxpr

  match built, manual with
  | .ok lhs, .ok rhs =>
    LeanTest.assertEqual (jaxprFingerprint lhs) (jaxprFingerprint rhs)
      "buildFromKStmts should match manual fromKStmts+validate path when coverage is disabled."
  | .error lhsErr, .error rhsErr =>
    LeanTest.assertEqual (buildErrorToString lhsErr) (buildErrorToString rhsErr)
      "buildFromKStmts should preserve manual conversion/validation error messages."
  | .ok _, .error rhsErr =>
    LeanTest.fail s!"buildFromKStmts succeeded but manual path failed: {buildErrorToString rhsErr}"
  | .error lhsErr, .ok _ =>
    LeanTest.fail s!"buildFromKStmts failed but manual path succeeded: {buildErrorToString lhsErr}"

@[test]
def testBuildFromFnBodyWrapperParityWithManualPath : IO Unit := do
  let cfg : BuildConfig := { requireRuleCoverage := false }
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Lean.IR.Param := { x := x, borrow := false, ty := Lean.IR.IRType.object }
  let declName := `test.build_from_fnbody_wrapper_parity
  let body : Lean.IR.FnBody :=
    .vdecl y Lean.IR.IRType.object (Lean.IR.Expr.fap declName #[Lean.IR.Arg.var x]) (
      .ret (.var y)
    )

  let built ← runCoreM (buildFromFnBody cfg declName #[p] body)
  let manual : Except BuildError LeanJaxpr :=
    match fromFnBody declName #[p] body with
    | .error msg =>
      .error (.conversion msg)
    | .ok jaxpr =>
      match validate jaxpr with
      | .error errs => .error (.validation errs)
      | .ok () => .ok jaxpr

  match built, manual with
  | .ok lhs, .ok rhs =>
    LeanTest.assertEqual (jaxprFingerprint lhs) (jaxprFingerprint rhs)
      "buildFromFnBody should match manual fromFnBody+validate path when coverage is disabled."
  | .error lhsErr, .error rhsErr =>
    LeanTest.assertEqual (buildErrorToString lhsErr) (buildErrorToString rhsErr)
      "buildFromFnBody should preserve manual conversion/validation error messages."
  | .ok _, .error rhsErr =>
    LeanTest.fail s!"buildFromFnBody succeeded but manual path failed: {buildErrorToString rhsErr}"
  | .error lhsErr, .ok _ =>
    LeanTest.fail s!"buildFromFnBody failed but manual path succeeded: {buildErrorToString lhsErr}"

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
def testExtractLocalJacEdgesSucceedsWithKStmtSemanticsPack : IO Unit := do
  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    extractLocalJacEdges sampleKStmtLoweredJaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"extractLocalJacEdges should succeed after semantics-pack registration, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 3 "Semantics pack should emit one edge per input var"
    let pairs := edges.map (fun e => (e.src, e.dst))
    LeanTest.assertEqual pairs #[(0, 2), (2, 3), (1, 3)]
      "Edge endpoints should follow equation/invar traversal order"
    LeanTest.assertTrue (edges.all (fun e =>
      match e.map.repr with
      | .semantic _ => true
      | _ => false))
      "Registered KStmt semantics pack should emit structured semantic maps."

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
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match withRules with
  | .error err =>
    LeanTest.fail s!"buildAndExtractFromKStmts should succeed with semantics-pack, got: {buildExtractErrorToString err}"
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
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match withRules with
  | .error err =>
    LeanTest.fail s!"All-supported semantics pack should cover structural subset, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 5
      "Expected semantic edges for broadcast/reduce/transpose/concatCols (1+1+1+2)"

@[test]
def testBuildAndExtractFromKStmtsMMAAndDotGeneralWithShapeAwareMaps : IO Unit := do
  let a : VarId := { idx := 0 }
  let b : VarId := { idx := 1 }
  let c : VarId := { idx := 2 }
  let mmOut : VarId := { idx := 3 }
  let mmaOut : VarId := { idx := 4 }
  let stmts := #[
    KStmt.declRT a .Float16 8 16 .Row,
    KStmt.declRT b .Float16 16 4 .Col,
    KStmt.declRT c .Float16 8 4 .Row,
    KStmt.declRT mmOut .Float16 8 4 .Row,
    KStmt.declRT mmaOut .Float16 8 4 .Row,
    KStmt.mm .AB mmOut a b,
    KStmt.mma .AB mmaOut a b c
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"All-semantics pack should cover mm/mma path, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 5
      "Expected `mm` (2 edges) + `mma` (3 edges) local-Jac edges."

    match findEdge? edges a.idx mmOut.idx with
    | none => LeanTest.fail "Missing dot-general lhs edge a -> mmOut"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 128)
        "dot-general lhs edge should carry flattened source dim 8*16=128."
      LeanTest.assertEqual e.map.outDim? (some 32)
        "dot-general lhs edge should carry flattened output dim 8*4=32."
      LeanTest.assertTrue
        (match e.map.repr with
        | .semantic (.dotGeneral _) => true
        | _ => false)
        s!"Expected dot-general semantic repr, got: {e.map.repr}"

    match findEdge? edges c.idx mmaOut.idx with
    | none => LeanTest.fail "Missing mma accumulation edge c -> mmaOut"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 32)
        "mma accumulation edge should carry flattened source dim 8*4=32."
      LeanTest.assertEqual e.map.outDim? (some 32)
        "mma accumulation edge should carry flattened output dim 8*4=32."
      LeanTest.assertTrue
        (e.map.repr == .semantic (.binary (kstmtMmaOpName .AB) .accum .inject))
        s!"Expected mma accumulation semantic repr, got: {e.map.repr}"

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
def testBuildAndExtractFromKStmtsSemanticsPackCoverExpAndAdd : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Semantics pack should cover Exp/Add path, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 3 "Expected one unary edge + two binary edges"
    match findEdge? edges 0 1 with
    | none => LeanTest.fail "Missing expected Exp edge 0 -> 1"
    | some e =>
      LeanTest.assertTrue (e.map.repr == unaryTag .Exp)
        s!"Exp should use symbolic semantic map in semantics pack, got: {e.map.repr}"
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
def testBuildAndExtractFromKStmtsStructuralSemanticsInAllSemanticsPack : IO Unit := do
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
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"All-semantics pack should include structural semantics, got: {buildExtractErrorToString err}"
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
def testBuildAndExtractFromKStmtsStructuralExactSparsePayloads : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let v3 : VarId := { idx := 3 }
  let v4 : VarId := { idx := 4 }
  let v5 : VarId := { idx := 5 }
  let v6 : VarId := { idx := 6 }
  let v7 : VarId := { idx := 7 }
  let stmts := #[
    KStmt.declRV v0 .Float32 3,
    KStmt.declRT v1 .Float32 2 3 .Row,
    KStmt.broadcast .Row v1 v0,
    KStmt.declRV v2 .Float32 2,
    KStmt.reduce .Sum .Col v2 v1,
    KStmt.declRT v3 .Float32 3 2 .Row,
    KStmt.transpose v3 v1,
    KStmt.declRT v4 .Float32 1 3 .Row,
    KStmt.sliceRows v4 v1 1 1,
    KStmt.declRT v6 .Float32 1 3 .Row,
    KStmt.sliceRows v6 v1 0 1,
    KStmt.declRT v5 .Float32 1 6 .Row,
    KStmt.concatCols v5 v4 v6,
    KStmt.declRT v7 .Float32 2 3 .Row,
    KStmt.cumsum .Col v7 v1
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Structural exact-payload extraction should succeed, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    match findEdge? edges 0 1 with
    | none => LeanTest.fail "Missing broadcast edge 0 -> 1"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 3) "Broadcast input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "Broadcast output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 6 "Broadcast map should materialize six sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 0)
        s!"Broadcast map missing expected entry src=0,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 0 3)
        s!"Broadcast map missing expected entry src=0,dst=3: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 2 5)
        s!"Broadcast map missing expected entry src=2,dst=5: {reprStr e.map.entries}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing reduce edge 1 -> 2"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 6) "Reduce input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 2) "Reduce output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 6 "Reduce-sum map should materialize six sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 0)
        s!"Reduce map missing expected entry src=0,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 2 0)
        s!"Reduce map missing expected entry src=2,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 5 1)
        s!"Reduce map missing expected entry src=5,dst=1: {reprStr e.map.entries}"

    match findEdge? edges 1 3 with
    | none => LeanTest.fail "Missing transpose edge 1 -> 3"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 6) "Transpose input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "Transpose output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 6 "Transpose map should materialize six sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 1 2)
        s!"Transpose map missing expected entry src=1,dst=2: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 3 1)
        s!"Transpose map missing expected entry src=3,dst=1: {reprStr e.map.entries}"

    match findEdge? edges 1 4 with
    | none => LeanTest.fail "Missing sliceRows edge 1 -> 4"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 6) "sliceRows input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 3) "sliceRows output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 3 "sliceRows map should materialize three sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 3 0)
        s!"sliceRows map missing expected entry src=3,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 5 2)
        s!"sliceRows map missing expected entry src=5,dst=2: {reprStr e.map.entries}"

    match findEdge? edges 4 5 with
    | none => LeanTest.fail "Missing concat-left edge 4 -> 5"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 3) "concat-left input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "concat-left output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 3 "concat-left map should materialize three sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 2 2)
        s!"concat-left map missing expected entry src=2,dst=2: {reprStr e.map.entries}"

    match findEdge? edges 6 5 with
    | none => LeanTest.fail "Missing concat-right edge 6 -> 5"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 3) "concat-right input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "concat-right output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 3 "concat-right map should materialize three sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 3)
        s!"concat-right map missing expected entry src=0,dst=3: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 2 5)
        s!"concat-right map missing expected entry src=2,dst=5: {reprStr e.map.entries}"

    match findEdge? edges 1 7 with
    | none => LeanTest.fail "Missing cumsum edge 1 -> 7"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 6) "cumsum input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "cumsum output dim should be explicit."
      LeanTest.assertEqual e.map.entries.size 12 "cumsum Col map should materialize prefix sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 2)
        s!"cumsum map missing expected entry src=0,dst=2: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 3 4)
        s!"cumsum map missing expected entry src=3,dst=4: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 5 5)
        s!"cumsum map missing expected entry src=5,dst=5: {reprStr e.map.entries}"

@[test]
def testBuildAndExtractFromKStmtsBinaryBroadcastExactSparsePayloads : IO Unit := do
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmts := #[
    KStmt.declRT v0 .Float32 2 3 .Row,
    KStmt.declRV v1 .Float32 3,
    KStmt.declRT v2 .Float32 2 3 .Row,
    KStmt.binaryBroadcast .Sub .Row v2 v0 v1
  ]

  let res ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    buildAndExtractFromKStmts {} stmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"binaryBroadcast exact-payload extraction should succeed, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 2 "binaryBroadcast should emit tile/vec edges."

    match findEdge? edges 0 2 with
    | none => LeanTest.fail "Missing binaryBroadcast tile edge 0 -> 2"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 6) "binaryBroadcast tile input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "binaryBroadcast tile output dim should be explicit."
      LeanTest.assertEqual e.map.repr
        (.semantic (.binaryBroadcast (kstmtBinaryOpName .Sub) (broadcastAxisTagName .Row) .tile .tileBroadcast))
        s!"Unexpected binaryBroadcast tile repr: {e.map.repr}"
      LeanTest.assertEqual e.map.entries.size 6 "binaryBroadcast tile map should materialize six sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 0)
        s!"binaryBroadcast tile map missing expected entry src=0,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 5 5)
        s!"binaryBroadcast tile map missing expected entry src=5,dst=5: {reprStr e.map.entries}"

    match findEdge? edges 1 2 with
    | none => LeanTest.fail "Missing binaryBroadcast vec edge 1 -> 2"
    | some e =>
      LeanTest.assertEqual e.map.inDim? (some 3) "binaryBroadcast vec input dim should be explicit."
      LeanTest.assertEqual e.map.outDim? (some 6) "binaryBroadcast vec output dim should be explicit."
      LeanTest.assertEqual e.map.repr
        (.semantic (.binaryBroadcast (kstmtBinaryOpName .Sub) (broadcastAxisTagName .Row) .vec .vecBroadcast))
        s!"Unexpected binaryBroadcast vec repr: {e.map.repr}"
      LeanTest.assertEqual e.map.entries.size 6 "binaryBroadcast vec map should materialize six sparse entries."
      LeanTest.assertTrue (hasEntry e.map.entries 0 0 (-1.0))
        s!"binaryBroadcast vec map missing expected weighted entry src=0,dst=0: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 0 3 (-1.0))
        s!"binaryBroadcast vec map missing expected weighted entry src=0,dst=3: {reprStr e.map.entries}"
      LeanTest.assertTrue (hasEntry e.map.entries 2 5 (-1.0))
        s!"binaryBroadcast vec map missing expected weighted entry src=2,dst=5: {reprStr e.map.entries}"

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
def testExtractNoGradControlRulesStopGradientIotaDevicePutPjit : IO Unit := do
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
        op := devicePutOpName
        invars := #[{ id := 1 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.no_grad_control, line? := some 42 }
      },
      {
        op := pjitOpName
        invars := #[{ id := 3 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.no_grad_control, line? := some 43 }
      },
      {
        op := kstmtBinaryOpName .Add
        invars := #[{ id := 4 }, { id := 2 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.no_grad_control, line? := some 44 }
      }
    ]
    outvars := #[{ id := 5 }]
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
      "stop_gradient/iota/device_put/pjit should contribute zero local-Jac edges; Add should contribute two."
    LeanTest.assertTrue ((findEdge? edges 0 1).isNone)
      "stop_gradient should block local-Jac edge from input to its output."
    LeanTest.assertTrue ((findEdge? edges 1 3).isNone)
      "device_put should emit no local-Jac edge."
    LeanTest.assertTrue ((findEdge? edges 3 4).isNone)
      "pjit should emit no local-Jac edge."
    LeanTest.assertTrue ((findEdge? edges 4 5).isSome)
      "Add should include lhs edge from pjit output."
    LeanTest.assertTrue ((findEdge? edges 2 5).isSome)
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

@[test]
def testExtractCommunicationAliasRules : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }]
    eqns := #[
      {
        op := allGatherOpName
        invars := #[{ id := 0 }]
        outvars := #[{ id := 1 }]
        source := { decl := `test.communication_alias, line? := some 70 }
      },
      {
        op := `jax.lax.reduce_scatter
        invars := #[{ id := 1 }]
        outvars := #[{ id := 2 }]
        source := { decl := `test.communication_alias, line? := some 71 }
      }
    ]
    outvars := #[{ id := 2 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradCommunicationRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Communication alias rules should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 2
      "Communication unary aliases should preserve one edge per equation."
    LeanTest.assertTrue ((findEdge? edges 0 1).isSome)
      "all_gather should emit a unary local-Jac edge."
    LeanTest.assertTrue ((findEdge? edges 1 2).isSome)
      "reduce_scatter should emit a unary local-Jac edge."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | .semantic (.unary _ .x .none) => true
        | _ => false))
      "Communication aliases should use unary structured semantic tags."

@[test]
def testExtractStructuralAliasRules : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }]
    eqns := #[
      {
        op := reshapeAliasOpName
        invars := #[{ id := 0 }]
        outvars := #[{ id := 2 }]
        source := { decl := `test.structural_alias, line? := some 80 }
      },
      {
        op := `jax.lax.slice_in_dim_p
        invars := #[{ id := 2 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.structural_alias, line? := some 81 }
      },
      {
        op := concatenateAliasOpName
        invars := #[{ id := 3 }, { id := 1 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.structural_alias, line? := some 82 }
      }
    ]
    outvars := #[{ id := 4 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradStructuralAliasRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Structural alias rules should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 4
      "reshape(1) + slice_in_dim(1) + concatenate(2) should produce four local-Jac edges."
    LeanTest.assertTrue ((findEdge? edges 0 2).isSome)
      "reshape alias should emit unary local-Jac edge."
    LeanTest.assertTrue ((findEdge? edges 2 3).isSome)
      "slice_in_dim alias should emit unary local-Jac edge."
    LeanTest.assertTrue ((findEdge? edges 3 4).isSome)
      "concatenate alias should emit edge for first input."
    LeanTest.assertTrue ((findEdge? edges 1 4).isSome)
      "concatenate alias should emit edge for second input."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | .semantic _ => true
        | _ => false))
      "Structural aliases should emit structured semantic tags."

@[test]
def testBuildAndExtractFromFnBodyGraphaxAliasPath : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let z : Lean.IR.VarId := { idx := 2 }
  let w : Lean.IR.VarId := { idx := 3 }
  let u : Lean.IR.VarId := { idx := 4 }
  let v : Lean.IR.VarId := { idx := 5 }
  let p : Lean.IR.Param := { x := x, borrow := false, ty := Lean.IR.IRType.object }
  let body : Lean.IR.FnBody :=
    .vdecl y Lean.IR.IRType.object (Lean.IR.Expr.fap stopGradientOpName #[Lean.IR.Arg.var x]) (
      .vdecl z Lean.IR.IRType.object (Lean.IR.Expr.fap allGatherOpName #[Lean.IR.Arg.var y]) (
        .vdecl w Lean.IR.IRType.object (Lean.IR.Expr.fap reshapeAliasOpName #[Lean.IR.Arg.var z]) (
          .vdecl u Lean.IR.IRType.object (Lean.IR.Expr.fap iotaOpName #[]) (
            .vdecl v Lean.IR.IRType.object (Lean.IR.Expr.fap `jax.lax.dot_general #[Lean.IR.Arg.var w, Lean.IR.Arg.var u]) (
              .ret (.var v)
            )
          )
        )
      )
    )

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    buildAndExtractFromFnBody {} `test.fnbody_graphax_alias_path #[p] body
  )
  match res with
  | .error err =>
    LeanTest.fail s!"FnBody alias path should build+extract successfully, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 4
      "stop_gradient/iota should emit no edges; all_gather/reshape/dot_general should emit 1+1+2 edges."
    LeanTest.assertTrue ((findEdge? edges 2 3).isSome)
      "all_gather should emit unary edge."
    LeanTest.assertTrue ((findEdge? edges 3 4).isSome)
      "reshape should emit unary edge."
    LeanTest.assertTrue ((findEdge? edges 4 6).isSome)
      "dot_general should emit lhs edge."
    LeanTest.assertTrue ((findEdge? edges 5 6).isSome)
      "dot_general should emit rhs edge."
    LeanTest.assertTrue ((findEdge? edges 1 2).isNone)
      "stop_gradient should emit no edge."

@[test]
def testBuildAndExtractFromFnBodyPadSelectDynamicUpdateIndexAliasPath : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let s : Lean.IR.VarId := { idx := 2 }
  let p0 : Lean.IR.Param := { x := x, borrow := false, ty := Lean.IR.IRType.object }
  let p1 : Lean.IR.Param := { x := y, borrow := false, ty := Lean.IR.IRType.object }
  let p2 : Lean.IR.Param := { x := s, borrow := false, ty := Lean.IR.IRType.object }
  let a : Lean.IR.VarId := { idx := 3 }
  let b : Lean.IR.VarId := { idx := 4 }
  let c : Lean.IR.VarId := { idx := 5 }
  let body : Lean.IR.FnBody :=
    .vdecl a Lean.IR.IRType.object (Lean.IR.Expr.fap `jax.lax.pad_p #[Lean.IR.Arg.var x, Lean.IR.Arg.var y]) (
      .vdecl b Lean.IR.IRType.object (Lean.IR.Expr.fap `jax.lax.select_p #[Lean.IR.Arg.var s, Lean.IR.Arg.var a, Lean.IR.Arg.var x]) (
        .vdecl c Lean.IR.IRType.object (Lean.IR.Expr.fap `jax.lax.dynamic_update_index_in_dim_p #[Lean.IR.Arg.var b, Lean.IR.Arg.var y, Lean.IR.Arg.var s]) (
          .ret (.var c)
        )
      )
    )

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    buildAndExtractFromFnBody {} `test.fnbody_pad_select_dynamic_update_index_alias_path #[p0, p1, p2] body
  )
  match res with
  | .error err =>
    LeanTest.fail s!"FnBody pad/select/dynamic_update_index alias path should build+extract successfully, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 6
      "pad(2) + select(data-only=2) + dynamic_update_index_in_dim(2) should produce six edges."
    LeanTest.assertTrue ((findEdge? edges 1 4).isSome)
      "pad should emit edge from base operand."
    LeanTest.assertTrue ((findEdge? edges 2 4).isSome)
      "pad should emit edge from padding-value operand."
    LeanTest.assertTrue ((findEdge? edges 4 5).isSome)
      "select should emit edge from first data operand."
    LeanTest.assertTrue ((findEdge? edges 1 5).isSome)
      "select should emit edge from second data operand."
    LeanTest.assertTrue ((findEdge? edges 3 5).isNone)
      "select should not emit edge from selector operand."
    LeanTest.assertTrue ((findEdge? edges 5 6).isSome)
      "dynamic_update_index_in_dim should emit edge from base operand."
    LeanTest.assertTrue ((findEdge? edges 2 6).isSome)
      "dynamic_update_index_in_dim should emit edge from update operand."
    LeanTest.assertTrue ((findEdge? edges 3 6).isNone)
      "dynamic_update_index_in_dim should not emit edge from index operand."

@[test]
def testExtractDynamicAliasRules : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }, { id := 2 }]
    eqns := #[
      {
        op := dynamicSliceAliasOpName
        invars := #[{ id := 0 }, { id := 2 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.dynamic_alias, line? := some 90 }
      },
      {
        op := dynamicUpdateSliceAliasOpName
        invars := #[{ id := 3 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.dynamic_alias, line? := some 91 }
      },
      {
        op := `jax.lax.dynamic_update_index_in_dim_p
        invars := #[{ id := 4 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.dynamic_alias, line? := some 92 }
      }
    ]
    outvars := #[{ id := 5 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradDynamicAliasRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Dynamic alias rules should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 5
      "dynamic_slice(1) + dynamic_update_slice(2) + dynamic_update_index_in_dim(2) should produce five local-Jac edges."
    LeanTest.assertTrue ((findEdge? edges 0 3).isSome)
      "dynamic_slice alias should emit base-operand edge."
    LeanTest.assertTrue ((findEdge? edges 2 3).isNone)
      "dynamic_slice alias should not emit index edge."
    LeanTest.assertTrue ((findEdge? edges 3 4).isSome)
      "dynamic_update_slice alias should emit base edge."
    LeanTest.assertTrue ((findEdge? edges 1 4).isSome)
      "dynamic_update_slice alias should emit update edge."
    LeanTest.assertTrue ((findEdge? edges 2 4).isNone)
      "dynamic_update_slice alias should not emit index edge."
    LeanTest.assertTrue ((findEdge? edges 4 5).isSome)
      "dynamic_update_index_in_dim alias should emit base edge."
    LeanTest.assertTrue ((findEdge? edges 1 5).isSome)
      "dynamic_update_index_in_dim alias should emit update edge."
    LeanTest.assertTrue ((findEdge? edges 2 5).isNone)
      "dynamic_update_index_in_dim alias should not emit index edge."

@[test]
def testExtractScanAliasRuleSubjaxprPartition : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 10 }, { id := 11 }, { id := 14 }]
    eqns := #[
      {
        op := scanAliasOpName
        invars := #[{ id := 10 }, { id := 11 }, { id := 14 }]
        outvars := #[{ id := 12 }, { id := 13 }]
        params := #[
          OpParam.mkNat .scanCarryInputCount 1,
          OpParam.mkNat .scanDataInputCount 1,
          OpParam.mkNat .scanCarryOutputCount 1
        ]
        source := { decl := `test.scan_subjaxpr_partition, line? := some 100 }
      }
    ]
    outvars := #[{ id := 12 }, { id := 13 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradReductionControlAliasRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"scan alias subjaxpr partition should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 4
      "scan with one carry/data input and one carry/data output should produce four dependency edges."
    LeanTest.assertTrue (edges.any (fun e => e.src == 10 && e.dst == 12))
      "scan carry input should connect to carry output."
    LeanTest.assertTrue (edges.any (fun e => e.src == 10 && e.dst == 13))
      "scan carry input should connect to data output."
    LeanTest.assertTrue (edges.any (fun e => e.src == 11 && e.dst == 12))
      "scan data input should connect to carry output."
    LeanTest.assertTrue (edges.any (fun e => e.src == 11 && e.dst == 13))
      "scan data input should connect to data output."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 14)))
      "scan should ignore trailing non-data inputs when `scanDataInputCount` excludes them."
    LeanTest.assertTrue
      (edges.any (fun e =>
        e.src == 10 && e.dst == 12 &&
        match e.map.repr with
        | .semantic (.unary op .x .carry) => op == scanAliasOpName
        | _ => false))
      "scan carry->carry edge should be carry-tagged."
    LeanTest.assertTrue
      (edges.any (fun e =>
        e.src == 11 && e.dst == 13 &&
        match e.map.repr with
        | .semantic (.unary op .x .projection) => op == scanAliasOpName
        | _ => false))
      "scan data->data edge should be projection-tagged."

@[test]
def testExtractCondAliasRuleSubjaxprPartition : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 20 }, { id := 21 }, { id := 22 }, { id := 25 }]
    eqns := #[
      {
        op := condAliasOpName
        invars := #[{ id := 20 }, { id := 21 }, { id := 22 }, { id := 25 }]
        outvars := #[{ id := 23 }, { id := 24 }]
        params := #[
          OpParam.mkNat .condPredicateCount 1,
          OpParam.mkNat .condDataInputCount 2
        ]
        source := { decl := `test.cond_subjaxpr_partition, line? := some 110 }
      }
    ]
    outvars := #[{ id := 23 }, { id := 24 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradReductionControlAliasRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"cond alias subjaxpr partition should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 4
      "cond with two data inputs and two outputs should produce four dependency edges."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 20)))
      "cond should not emit predicate-input local-Jac edges."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 25)))
      "cond should ignore trailing non-data inputs when `condDataInputCount` excludes them."
    LeanTest.assertTrue (edges.any (fun e => e.src == 21 && e.dst == 23))
      "cond should connect first data input to first output."
    LeanTest.assertTrue (edges.any (fun e => e.src == 22 && e.dst == 24))
      "cond should connect second data input to second output."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | .semantic (.unary op .x .projection) => op == condAliasOpName
        | _ => false))
      "cond data edges should be projection-tagged."

end Tests.ADJaxprLikeExtract
