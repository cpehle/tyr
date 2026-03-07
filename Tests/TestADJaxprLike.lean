import LeanTest
import Lean.CoreM
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADJaxprLike

open Lean
open Lean.IR
open LeanTest
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

private def jaxprFingerprint (jaxpr : LeanJaxpr) : String :=
  reprStr jaxpr

private def passthroughUnaryRule (op : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let outv ←
      match eqn.outvars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Rule `{op}` requires one output variable.")
    if eqn.outvars.size != 1 then
      .error (.malformedEqn s!"Rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
    else
      let inv ←
        match eqn.invars[0]? with
        | some v => .ok v
        | none => .error (.malformedEqn s!"Rule `{op}` requires one input variable.")
      if eqn.invars.size != 1 then
        .error (.malformedEqn s!"Rule `{op}` expects exactly one input variable, got {eqn.invars.size}.")
      else
        .ok #[{
          src := inv.id
          dst := outv.id
          map := {
            repr := .semantic (.unary op .x .none)
            inDim? := some 1
            outDim? := some 1
            entries := #[{ src := 0, dst := 0, weight := 1.0 }]
          }
        }]

@[test]
def testFromKStmtsUnaryBinary : IO Unit := do
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]
  match fromKStmts stmts with
  | .error errs =>
    LeanTest.fail s!"fromKStmts should succeed, got errors:\n{String.intercalate "\n" errs.toList}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 2 "Expected two lowered equations"
    LeanTest.assertEqual (jaxpr.invars.map (fun v => v.id)) #[0]
      "Input inference should keep only used-but-never-defined ids"
    LeanTest.assertEqual (jaxpr.outvars.map (fun v => v.id)) #[2]
      "Output inference should keep only defined-but-never-used ids"

@[test]
def testFromKStmtsExtendedStructuralSubset : IO Unit := do
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let v3 : Tyr.GPU.Codegen.VarId := { idx := 3 }
  let v4 : Tyr.GPU.Codegen.VarId := { idx := 4 }
  let stmts := #[
    KStmt.broadcast .Row v1 v0,
    KStmt.reduce .Sum .Col v2 v1,
    KStmt.transpose v3 v2,
    KStmt.concatCols v4 v3 v0
  ]
  match fromKStmts stmts with
  | .error errs =>
    LeanTest.fail s!"fromKStmts should support broadcast/reduce/transpose/concatCols, got errors:\n{String.intercalate "\n" errs.toList}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 4 "Expected four lowered equations"
    let atom := fun s => Name.str .anonymous s
    LeanTest.assertEqual (jaxpr.eqns.map (fun e => e.params.findValue? .kind))
      #[
        some (.name (atom "broadcast")),
        some (.name (atom "reduce")),
        some (.name (atom "transpose")),
        some (.name (atom "concatCols"))
      ]
      "Lowered equation kinds should preserve structural op metadata"
    LeanTest.assertEqual (jaxpr.invars.map (fun v => v.id)) #[0]
      "Input inference should keep source-only vars"
    LeanTest.assertEqual (jaxpr.outvars.map (fun v => v.id)) #[4]
      "Output inference should keep terminal vars"

@[test]
def testFromKStmtsDeclarationMetadataPropagation : IO Unit := do
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let stmts := #[
    KStmt.declRV v0 .Float32 4,
    KStmt.declRV v1 .Float32 4,
    KStmt.unary .Exp v1 v0
  ]
  match fromKStmts stmts with
  | .error errs =>
    LeanTest.fail s!"fromKStmts should accept declarations and propagate metadata, got errors:\n{String.intercalate "\n" errs.toList}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 1 "Expected one lowered equation after declaration filtering"
    let eqn := jaxpr.eqns[0]!
    LeanTest.assertEqual eqn.invars[0]!.metaInfo.shape (some #[4])
      "Declaration shape metadata should flow to unary input var."
    LeanTest.assertEqual eqn.outvars[0]!.metaInfo.shape (some #[4])
      "Declaration/inferred shape metadata should flow to unary output var."
    LeanTest.assertEqual eqn.outvars[0]!.metaInfo.dtype (some "Float32")
      "Declaration dtype metadata should propagate to lowered vars."

@[test]
def testFromKStmtsMMLoweringAndMMAArity : IO Unit := do
  let a : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let b : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let c : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let mmOut : Tyr.GPU.Codegen.VarId := { idx := 3 }
  let mmaOut : Tyr.GPU.Codegen.VarId := { idx := 4 }
  let stmts := #[
    KStmt.declRT a .Float16 8 16 .Row,
    KStmt.declRT b .Float16 16 4 .Col,
    KStmt.declRT c .Float16 8 4 .Row,
    KStmt.declRT mmOut .Float16 8 4 .Row,
    KStmt.declRT mmaOut .Float16 8 4 .Row,
    KStmt.mm .AB mmOut a b,
    KStmt.mma .AB mmaOut a b c
  ]
  match fromKStmts stmts with
  | .error errs =>
    LeanTest.fail s!"fromKStmts should support mm/mma lowering, got errors:\n{String.intercalate "\n" errs.toList}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 2 "Expected two lowered equations (`mm`, `mma`)"
    let mmEqn := jaxpr.eqns[0]!
    let mmaEqn := jaxpr.eqns[1]!
    LeanTest.assertEqual mmEqn.op kstmtDotGeneralOpName
      "`mm` should lower through dot-general op name."
    LeanTest.assertEqual (mmEqn.params.findValue? .kind) (some (.name (Name.mkSimple "mm")))
      "`mm` lowering should preserve kind metadata."
    LeanTest.assertEqual (mmEqn.params.findNats? .lhsContract) (some #[1])
      "`mm.AB` should encode lhs contract axis `[1]`."
    LeanTest.assertEqual (mmEqn.params.findNats? .rhsContract) (some #[0])
      "`mm.AB` should encode rhs contract axis `[0]`."
    LeanTest.assertEqual mmaEqn.op (kstmtMmaOpName .AB)
      "`mma` should lower to dedicated kstmt mma op name."
    LeanTest.assertEqual mmaEqn.invars.size 3 "`mma` should carry three inputs (a,b,c)."
    LeanTest.assertEqual mmaEqn.outvars[0]!.metaInfo.shape (some #[8, 4])
      "`mma` output should preserve propagated matrix shape metadata."

@[test]
def testFromKStmtsUnsupported : IO Unit := do
  let stmts := #[KStmt.sync 0]
  match fromKStmts stmts with
  | .ok _ =>
    LeanTest.fail "fromKStmts should fail on unsupported statements"
  | .error errs =>
    LeanTest.assertTrue (!errs.isEmpty) "Expected at least one unsupported-statement diagnostic"

@[test]
def testLowerKStmtWrapperParityWithFromKStmts : IO Unit := do
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let stmts := #[
    KStmt.unary .Exp v1 v0,
    KStmt.binary .Add v2 v1 v0
  ]
  let direct := fromKStmts stmts
  let wrapped := LowerKStmt.run { stmts := stmts }
  match direct, wrapped with
  | .ok lhs, .ok rhs =>
    LeanTest.assertEqual (jaxprFingerprint rhs) (jaxprFingerprint lhs)
      "LowerKStmt wrapper should match fromKStmts output."
  | .error lhsErrs, .error rhsErr =>
    LeanTest.assertEqual (toString rhsErr) (String.intercalate "\n" lhsErrs.toList)
      "LowerKStmt wrapper should preserve conversion diagnostics."
  | .ok _, .error rhsErr =>
    LeanTest.fail s!"LowerKStmt wrapper failed unexpectedly: {toString rhsErr}"
  | .error lhsErrs, .ok _ =>
    LeanTest.fail s!"LowerKStmt wrapper succeeded unexpectedly:\n{String.intercalate "\n" lhsErrs.toList}"

@[test]
def testFromFnBodySimpleFap : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap `test.simple #[Arg.var x]) (
      .ret (.var y)
    )
  match fromFnBody `test.simple #[p] body with
  | .error msg =>
    LeanTest.fail s!"fromFnBody should succeed on simple fap body, got: {msg}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 1 "Expected one lowered equation"
    LeanTest.assertEqual (jaxpr.invars.map (fun v => v.id)) #[1]
      "Expected normalized input var IDs (Lean-style fresh binder numbering)"
    LeanTest.assertEqual (jaxpr.outvars.map (fun v => v.id)) #[2]
      "Expected normalized return var IDs (Lean-style fresh binder numbering)"

@[test]
def testLowerFnBodyWrapperParityWithFromFnBody : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap `test.lower_wrapper #[Arg.var x]) (
      .ret (.var y)
    )
  let declName := `test.lower_wrapper
  let direct := fromFnBody declName #[p] body
  let wrapped := LowerFnBody.run { declName := declName, params := #[p], body := body }
  match direct, wrapped with
  | .ok lhs, .ok rhs =>
    LeanTest.assertEqual (jaxprFingerprint rhs) (jaxprFingerprint lhs)
      "LowerFnBody wrapper should match fromFnBody output."
  | .error lhsErr, .error rhsErr =>
    LeanTest.assertEqual (toString rhsErr) lhsErr
      "LowerFnBody wrapper should preserve conversion diagnostics."
  | .ok _, .error rhsErr =>
    LeanTest.fail s!"LowerFnBody wrapper failed unexpectedly: {toString rhsErr}"
  | .error lhsErr, .ok _ =>
    LeanTest.fail s!"LowerFnBody wrapper succeeded unexpectedly: {lhsErr}"

@[test]
def testFromFnBodySpecialPrimitiveLoweringKinds : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let z : Lean.IR.VarId := { idx := 2 }
  let w : Lean.IR.VarId := { idx := 3 }
  let q : Lean.IR.VarId := { idx := 4 }
  let s : Lean.IR.VarId := { idx := 5 }
  let t : Lean.IR.VarId := { idx := 6 }
  let r : Lean.IR.VarId := { idx := 7 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap stopGradientOpName #[Arg.var x]) (
      .vdecl z IRType.object (Expr.fap iotaOpName #[]) (
        .vdecl w IRType.object (Expr.fap devicePutOpName #[Arg.var y]) (
          .vdecl q IRType.object (Expr.fap pjitOpName #[Arg.var w]) (
            .vdecl s IRType.object (Expr.fap allGatherOpName #[Arg.var q]) (
              .vdecl t IRType.object (Expr.fap reshapeAliasOpName #[Arg.var s]) (
                .vdecl r IRType.object (Expr.fap kstmtDotGeneralOpName #[Arg.var t, Arg.var z]) (
                  .ret (.var r)
                )
              )
            )
          )
        )
      )
    )
  match fromFnBody `test.special_kinds #[p] body with
  | .error msg =>
    LeanTest.fail s!"fromFnBody should lower special primitives, got: {msg}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 7 "Expected seven lowered equations"
    LeanTest.assertEqual
      (jaxpr.eqns.map (fun e => e.params.findValue? .loweringKind))
      #[
        some (.name `fnbody.vdecl.fap.stop_gradient),
        some (.name `fnbody.vdecl.fap.iota),
        some (.name `fnbody.vdecl.fap.device_put),
        some (.name `fnbody.vdecl.fap.pjit),
        some (.name `fnbody.vdecl.fap.communication),
        some (.name `fnbody.vdecl.fap.structural),
        some (.name `fnbody.vdecl.fap.dot_general)
      ]
      "Special primitive lowering kinds should be tagged explicitly."

@[test]
def testFromFnBodySpecialPrimitiveArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap stopGradientOpName #[]) (
      .ret (.var y)
    )
  match fromFnBody `test.special_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when special primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 1, got 0")
      s!"Expected arity diagnostic for stop_gradient, got: {msg}"

@[test]
def testFromFnBodyDevicePutArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap devicePutOpName #[]) (
      .ret (.var y)
    )
  match fromFnBody `test.device_put_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when device_put arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 1, got 0")
      s!"Expected arity diagnostic for device_put, got: {msg}"

@[test]
def testFromFnBodyCommunicationArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap allGatherOpName #[]) (
      .ret (.var y)
    )
  match fromFnBody `test.communication_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when communication primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 1, got 0")
      s!"Expected arity diagnostic for communication primitive, got: {msg}"

@[test]
def testFromFnBodyStructuralArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap reshapeAliasOpName #[]) (
      .ret (.var y)
    )
  match fromFnBody `test.structural_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when structural primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 1, got 0")
      s!"Expected arity diagnostic for structural primitive, got: {msg}"

@[test]
def testFromFnBodyReductionArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap reduceSumAliasOpName #[]) (
      .ret (.var y)
    )
  match fromFnBody `test.reduction_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when reduction primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 1, got 0")
      s!"Expected arity diagnostic for reduction primitive, got: {msg}"

@[test]
def testFromFnBodyPadSelectDynamicUpdateIndexLoweringKinds : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let s : Lean.IR.VarId := { idx := 2 }
  let p0 : Param := { x := x, borrow := false, ty := IRType.object }
  let p1 : Param := { x := y, borrow := false, ty := IRType.object }
  let p2 : Param := { x := s, borrow := false, ty := IRType.object }
  let a : Lean.IR.VarId := { idx := 3 }
  let b : Lean.IR.VarId := { idx := 4 }
  let c : Lean.IR.VarId := { idx := 5 }
  let body : FnBody :=
    .vdecl a IRType.object (Expr.fap padAliasOpName #[Arg.var x, Arg.var y]) (
      .vdecl b IRType.object (Expr.fap selectAliasOpName #[Arg.var s, Arg.var a, Arg.var x]) (
        .vdecl c IRType.object (Expr.fap dynamicUpdateIndexInDimAliasOpName #[Arg.var b, Arg.var y, Arg.var s]) (
          .ret (.var c)
        )
      )
    )
  match fromFnBody `test.pad_select_dynamic_update_index_kinds #[p0, p1, p2] body with
  | .error msg =>
    LeanTest.fail s!"fromFnBody should lower pad/select/dynamic_update_index_in_dim, got: {msg}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 3 "Expected three lowered equations"
    LeanTest.assertEqual
      (jaxpr.eqns.map (fun e => e.params.findValue? .loweringKind))
      #[
        some (.name `fnbody.vdecl.fap.structural),
        some (.name `fnbody.vdecl.fap.structural),
        some (.name `fnbody.vdecl.fap.structural)
      ]
      "pad/select/dynamic_update_index_in_dim should lower under structural kind."

@[test]
def testFromFnBodyPadArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let p : Param := { x := x, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl y IRType.object (Expr.fap padAliasOpName #[Arg.var x]) (
      .ret (.var y)
    )
  match fromFnBody `test.pad_arity #[p] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when pad primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 2, got 1")
      s!"Expected arity diagnostic for pad primitive, got: {msg}"

@[test]
def testFromFnBodySelectArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let z : Lean.IR.VarId := { idx := 2 }
  let p0 : Param := { x := x, borrow := false, ty := IRType.object }
  let p1 : Param := { x := y, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl z IRType.object (Expr.fap selectAliasOpName #[Arg.var x, Arg.var y]) (
      .ret (.var z)
    )
  match fromFnBody `test.select_arity #[p0, p1] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when select primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 3, got 2")
      s!"Expected arity diagnostic for select primitive, got: {msg}"

@[test]
def testFromFnBodyDynamicUpdateIndexInDimArityDiagnostics : IO Unit := do
  let x : Lean.IR.VarId := { idx := 0 }
  let y : Lean.IR.VarId := { idx := 1 }
  let z : Lean.IR.VarId := { idx := 2 }
  let p0 : Param := { x := x, borrow := false, ty := IRType.object }
  let p1 : Param := { x := y, borrow := false, ty := IRType.object }
  let body : FnBody :=
    .vdecl z IRType.object (Expr.fap dynamicUpdateIndexInDimAliasOpName #[Arg.var x, Arg.var y]) (
      .ret (.var z)
    )
  match fromFnBody `test.dynamic_update_index_in_dim_arity #[p0, p1] body with
  | .ok _ =>
    LeanTest.fail "fromFnBody should fail when dynamic_update_index_in_dim primitive arity is invalid"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "expects arity 3, got 2")
      s!"Expected arity diagnostic for dynamic_update_index_in_dim primitive, got: {msg}"

@[test]
def testBuildFromDeclRequiresCoverage : IO Unit := do
  let result ← runCoreM (do
    let x : Lean.IR.VarId := { idx := 0 }
    let y : Lean.IR.VarId := { idx := 1 }
    let p : Param := { x := x, borrow := false, ty := IRType.object }
    let body : FnBody :=
      .vdecl y IRType.object (Expr.fap `test.no_rule #[Arg.var x]) (.ret (.var y))
    let decl : Decl := .fdecl `test.no_rule_decl #[p] IRType.object body {}
    buildFromDecl {} decl
  )
  match result with
  | .ok _ =>
    LeanTest.fail "buildFromDecl should fail coverage when no local-Jacobian rule is registered"
  | .error (.coverage msgs) =>
    LeanTest.assertTrue (!msgs.isEmpty) "Expected non-empty coverage diagnostics"
  | .error err =>
    LeanTest.fail s!"Expected coverage error, got: {buildErrorToString err}"

@[test]
def testBuildFromDeclCoveragePassesWithRule : IO Unit := do
  let result ← runCoreM (do
    let x : Lean.IR.VarId := { idx := 0 }
    let y : Lean.IR.VarId := { idx := 1 }
    let p : Param := { x := x, borrow := false, ty := IRType.object }
    let body : FnBody :=
      .vdecl y IRType.object (Expr.fap `test.with_rule #[Arg.var x]) (.ret (.var y))
    let decl : Decl := .fdecl `test.with_rule_decl #[p] IRType.object body {}
    registerLocalJacRule `test.with_rule (passthroughUnaryRule `test.with_rule)
    buildFromDecl {} decl
  )
  match result with
  | .error err =>
    LeanTest.fail s!"buildFromDecl should pass after registering rule, got: {buildErrorToString err}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 1 "Expected one equation in lowered jaxpr"

@[test]
def testVertexOrderHelpers : IO Unit := do
  LeanTest.assertEqual (forwardVertexOrder 4) #[1, 2, 3, 4]
    "Forward vertex order should be deterministic and 1-based"
  LeanTest.assertEqual (reverseVertexOrder 4) #[4, 3, 2, 1]
    "Reverse vertex order should be deterministic"
  LeanTest.assertTrue (eqnIdx0OfVertexId? 0 == none)
    "Vertex 0 should be invalid in 1-based space"
  LeanTest.assertTrue (eqnIdx0OfVertexId? 3 == some 2)
    "Inverse mapping should convert 1-based vertex IDs to 0-based eqn indices"
  match validateCustomVertexOrderAgainstEqnCount 3 #[1, 3, 2] with
  | .ok () => pure ()
  | .error msg => LeanTest.fail s!"Expected valid custom order, got: {msg}"
  match validateCustomVertexOrderAgainstEqnCount 3 #[1, 1, 2] with
  | .ok () => LeanTest.fail "Duplicate custom order should fail validation"
  | .error _ => pure ()

@[test]
def testValidateTopologicalFailure : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }]
    eqns := #[
      { op := `test.eqn0, invars := #[{ id := 2 }], outvars := #[{ id := 3 }] },
      { op := `test.eqn1, invars := #[{ id := 0 }], outvars := #[{ id := 2 }] }
    ]
    outvars := #[{ id := 3 }]
  }
  match validate jaxpr with
  | .ok () => LeanTest.fail "Topological validation should fail for forward reference input"
  | .error errs =>
    LeanTest.assertTrue (!errs.isEmpty) "Expected non-empty validation errors"
    LeanTest.assertTrue ((errs[0]!).contains "references unavailable variable ID 2")
      s!"Unexpected topological error: {errs[0]!}"

@[test]
def testValidateOutvarAvailabilityFailure : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }]
    eqns := #[{ op := `test.eqn0, invars := #[{ id := 0 }], outvars := #[{ id := 1 }] }]
    outvars := #[{ id := 99 }]
  }
  match validate jaxpr with
  | .ok () => LeanTest.fail "Output availability validation should fail for unknown outvar"
  | .error errs =>
    LeanTest.assertTrue (!errs.isEmpty) "Expected non-empty validation errors"
    let hasOutErr := errs.any (fun e => e.contains "output 0 references unavailable variable ID 99")
    LeanTest.assertTrue hasOutErr s!"Expected unavailable-output diagnostic, got: {errs}"

@[test]
def testFromFnBodyCanonicalDotGeneralAndControlMetadata : IO Unit := do
  let x0 : Lean.IR.VarId := { idx := 0 }
  let x1 : Lean.IR.VarId := { idx := 1 }
  let pred : Lean.IR.VarId := { idx := 2 }
  let vDot : Lean.IR.VarId := { idx := 3 }
  let vScan : Lean.IR.VarId := { idx := 4 }
  let vCond : Lean.IR.VarId := { idx := 5 }
  let params : Array Param := #[
    { x := x0, borrow := false, ty := IRType.object },
    { x := x1, borrow := false, ty := IRType.object },
    { x := pred, borrow := false, ty := IRType.object }
  ]
  let body : FnBody :=
    .vdecl vDot IRType.object (Expr.fap `jax.lax.dot_general #[Arg.var x0, Arg.var x1]) (
      .vdecl vScan IRType.object (Expr.fap scanAliasOpName #[Arg.var x0, Arg.var x1, Arg.erased]) (
        .vdecl vCond IRType.object (Expr.fap condAliasOpName #[Arg.var pred, Arg.var vScan, Arg.var vDot, Arg.erased, Arg.erased]) (
          .ret (.var vCond)
        )
      )
    )
  match fromFnBody `test.canonical_dot_general_and_control_metadata params body with
  | .error msg =>
    LeanTest.fail s!"fromFnBody should lower canonical dot_general/control metadata, got: {msg}"
  | .ok jaxpr =>
    LeanTest.assertEqual jaxpr.eqns.size 3 "Expected three lowered equations"
    let dotEqn := jaxpr.eqns[0]!
    let scanEqn := jaxpr.eqns[1]!
    let condEqn := jaxpr.eqns[2]!
    LeanTest.assertEqual dotEqn.op kstmtDotGeneralOpName
      "dot_general aliases should canonicalize to `kstmtDotGeneralOpName`."
    LeanTest.assertEqual (dotEqn.params.findName? .variant) (some `jax.lax.dot_general)
      "dot_general alias lowering should preserve raw op name in `.variant`."
    LeanTest.assertEqual (dotEqn.params.findNats? .lhsContract) (some #[])
      "dot_general default lhs contract should be explicit."
    LeanTest.assertEqual (dotEqn.params.findNats? .rhsContract) (some #[])
      "dot_general default rhs contract should be explicit."
    LeanTest.assertEqual (dotEqn.params.findNats? .lhsBatch) (some #[])
      "dot_general default lhs batch should be explicit."
    LeanTest.assertEqual (dotEqn.params.findNats? .rhsBatch) (some #[])
      "dot_general default rhs batch should be explicit."

    LeanTest.assertEqual scanEqn.op scanAliasOpName
      "scan aliases should canonicalize to `scanAliasOpName`."
    LeanTest.assertEqual (scanEqn.params.findNat? .controlStaticArgCount) (some 1)
      "scan lowering should record erased/static control-arg count."
    LeanTest.assertEqual (scanEqn.params.findNat? .scanCarryInputCount) (some 1)
      "scan lowering should record carry-input count."
    LeanTest.assertEqual (scanEqn.params.findNat? .scanDataInputCount) (some 1)
      "scan lowering should record data-input count."
    LeanTest.assertEqual (scanEqn.params.findNat? .scanCarryOutputCount) (some 1)
      "scan lowering should record carry-output count."

    LeanTest.assertEqual condEqn.op condAliasOpName
      "cond aliases should canonicalize to `condAliasOpName`."
    LeanTest.assertEqual (condEqn.params.findNat? .controlStaticArgCount) (some 2)
      "cond lowering should record erased/static control-arg count."
    LeanTest.assertEqual (condEqn.params.findNat? .condPredicateCount) (some 1)
      "cond lowering should record predicate-input count."
    LeanTest.assertEqual (condEqn.params.findNat? .condDataInputCount) (some 2)
      "cond lowering should record data-input count."

end Tests.ADJaxprLike
