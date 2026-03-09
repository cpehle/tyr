import Lean.CoreM
import LeanTest
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADGraphaxAlphaGradParity

open Lean
open LeanTest
open Tyr.AD.JaxprLike
open Tyr.AD.Sparse
open Tyr.GPU.Codegen

private def approx (a b : Float) (tol : Float := 1e-9) : Bool :=
  Float.abs (a - b) < tol

private def findEdge? (edges : Array LocalJacEdge) (src dst : Nat) : Option LocalJacEdge :=
  edges.find? (fun e => e.src = src && e.dst = dst)

private def sameEntries (lhs rhs : Array SparseEntry) : Bool :=
  lhs.size = rhs.size &&
    lhs.all (fun e =>
      rhs.any (fun f => f.src = e.src && f.dst = e.dst && approx f.weight e.weight))

private def assertSameSparsePayload
    (label : String)
    (lhs rhs : Tyr.AD.Sparse.SparseLinearMap) :
    IO Unit := do
  LeanTest.assertEqual lhs.inDim? rhs.inDim?
    s!"{label} should agree on input dimension."
  LeanTest.assertEqual lhs.outDim? rhs.outDim?
    s!"{label} should agree on output dimension."
  LeanTest.assertTrue (sameEntries lhs.entries rhs.entries)
    s!"{label} should agree on sparse entries, lhs={reprStr lhs.entries}, rhs={reprStr rhs.entries}"

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

private def overlapUnaryOps : Array UnaryOp := #[
  .Neg, .Abs, .Exp, .Log, .Sqrt, .Square, .Sin, .Cos, .Tanh, .Sigmoid
]

private def overlapBinaryOps : Array BinaryOp := #[
  .Add, .Sub, .Mul, .Div, .Max, .Min
]

private def buildOverlapKStmts : Array KStmt := Id.run do
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }

  let mut cur := x
  let mut nextIdx := 2
  let mut out : Array KStmt := #[]

  for op in overlapUnaryOps do
    let dst : VarId := { idx := nextIdx }
    out := out.push (KStmt.unary op dst cur)
    cur := dst
    nextIdx := nextIdx + 1

  for op in overlapBinaryOps do
    let dst : VarId := { idx := nextIdx }
    out := out.push (KStmt.binary op dst cur y)
    cur := dst
    nextIdx := nextIdx + 1

  out

@[test]
def testGraphaxAlphaGradOverlapCoverageFromKStmt : IO Unit := do
  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    buildAndExtractFromKStmts {} buildOverlapKStmts
  )
  match res with
  | .error err =>
    LeanTest.fail s!"Graphax/AlphaGrad overlap KStmt path should extract successfully, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    let expected := overlapUnaryOps.size + (2 * overlapBinaryOps.size)
    LeanTest.assertEqual edges.size expected
      s!"Expected one edge per unary op and two per binary op over overlap set; expected={expected}, got={edges.size}"
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic _ => true
        | _ => false))
      "Overlap primitive local Jacobians should carry structured semantic tags (no placeholders)."

@[test]
def testGraphaxAliasCoverageNoGradAndDotGeneral : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }]
    eqns := #[
      {
        op := `Graphax.stop_gradient
        invars := #[{ id := 0 }]
        outvars := #[{ id := 2 }]
        source := { decl := `test.graphax_aliases, line? := some 10 }
      },
      {
        op := `Graphax.iota
        invars := #[]
        outvars := #[{ id := 3 }]
        source := { decl := `test.graphax_aliases, line? := some 11 }
      },
      {
        op := `Graphax.device_put
        invars := #[{ id := 2 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_aliases, line? := some 12 }
      },
      {
        op := `Graphax.pjit
        invars := #[{ id := 4 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_aliases, line? := some 13 }
      },
      {
        op := `Graphax.dot_general
        invars := #[{ id := 5 }, { id := 3 }]
        outvars := #[{ id := 6 }]
        params := #[
          OpParam.mkName .variant `matmul,
          OpParam.mkNats .lhsContract #[1],
          OpParam.mkNats .rhsContract #[0],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
        source := { decl := `test.graphax_aliases, line? := some 14 }
      }
    ]
    outvars := #[{ id := 6 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 2
      "stop_gradient/iota/device_put/pjit should produce no edges; dot_general should produce lhs/rhs edges."
    let hasStopGradEdge := edges.any (fun e => e.src == 0 && e.dst == 2)
    LeanTest.assertTrue (!hasStopGradEdge)
      "stop_gradient alias should emit no local-Jac edge."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 2 && e.dst == 4)))
      "device_put alias should emit no local-Jac edge."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 4 && e.dst == 5)))
      "pjit alias should emit no local-Jac edge."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic (.dotGeneral _) => true
        | _ => false))
      "Graphax dot_general alias should use structured dot_general semantic tags."

@[test]
def testGraphaxJaxAliasCoverageExtraAndCommunication : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }]
    eqns := #[
      {
        op := `jax.lax.log1p_p
        invars := #[{ id := 0 }]
        outvars := #[{ id := 2 }]
        source := { decl := `test.graphax_jax_aliases, line? := some 20 }
      },
      {
        op := `jax.lax.atan2_p
        invars := #[{ id := 2 }, { id := 1 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.graphax_jax_aliases, line? := some 21 }
      },
      {
        op := allGatherOpName
        invars := #[{ id := 3 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_jax_aliases, line? := some 22 }
      },
      {
        op := `jax.lax.eq
        invars := #[{ id := 4 }, { id := 1 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_jax_aliases, line? := some 23 }
      },
      {
        op := `jax.lax.add
        invars := #[{ id := 4 }, { id := 1 }]
        outvars := #[{ id := 6 }]
        source := { decl := `test.graphax_jax_aliases, line? := some 24 }
      }
    ]
    outvars := #[{ id := 5 }, { id := 6 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 6
      "log1p(1) + atan2(2) + all_gather(1) + eq(0) + add(2) should produce six edges."
    LeanTest.assertTrue ((edges.any (fun e => e.src == 3 && e.dst == 4)))
      "Communication alias should preserve unary dependency edge."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 4 && e.dst == 5)))
      "eq alias should not emit a lhs local-Jac edge."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 1 && e.dst == 5)))
      "eq alias should not emit a rhs local-Jac edge."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic _ => true
        | _ => false))
      "Alias/communication rules should emit structured semantic tags."

@[test]
def testGraphaxJaxRsqrtAliasCoverage : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }]
    eqns := #[
      {
        op := `jax.lax.rsqrt_p
        invars := #[{ id := 0 }]
        outvars := #[{ id := 1 }]
        source := { decl := `test.graphax_jax_rsqrt_alias, line? := some 25 }
      }
    ]
    outvars := #[{ id := 1 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX rsqrt alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 1 "rsqrt should emit one unary local-Jac edge."
    LeanTest.assertTrue (edges.any (fun e => e.src == 0 && e.dst == 1))
      "rsqrt alias should preserve unary dependency edge."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic _ => true
        | _ => false))
      "rsqrt alias rule should emit structured semantic tags."

@[test]
def testGraphaxJaxStructuralAliasCoverage : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }, { id := 2 }]
    eqns := #[
      {
        op := reshapeAliasOpName
        invars := #[{ id := 0 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 30 }
      },
      {
        op := squeezeAliasOpName
        invars := #[{ id := 3 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 31 }
      },
      {
        op := broadcastInDimAliasOpName
        invars := #[{ id := 4 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 32 }
      },
      {
        op := sliceAliasOpName
        invars := #[{ id := 5 }]
        outvars := #[{ id := 6 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 33 }
      },
      {
        op := convertElementTypeAliasOpName
        invars := #[{ id := 6 }]
        outvars := #[{ id := 7 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 34 }
      },
      {
        op := transposeAliasOpName
        invars := #[{ id := 7 }]
        outvars := #[{ id := 8 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 35 }
      },
      {
        op := concatenateAliasOpName
        invars := #[{ id := 8 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 9 }]
        source := { decl := `test.graphax_structural_aliases, line? := some 36 }
      }
    ]
    outvars := #[{ id := 9 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX structural alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 9
      "Six unary structural aliases plus 3-input concatenate should produce nine edges."
    LeanTest.assertTrue ((edges.any (fun e => e.src == 8 && e.dst == 9)))
      "concatenate alias should preserve edge from transformed stream input."
    LeanTest.assertTrue ((edges.any (fun e => e.src == 1 && e.dst == 9)))
      "concatenate alias should preserve edge from second input."
    LeanTest.assertTrue ((edges.any (fun e => e.src == 2 && e.dst == 9)))
      "concatenate alias should preserve edge from third input."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic _ => true
        | _ => false))
      "Structural alias rules should emit structured semantic tags."

@[test]
def testGraphaxJaxReductionAndSelectAliasCoverage : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }, { id := 2 }]
    eqns := #[
      {
        op := reduceSumAliasOpName
        invars := #[{ id := 0 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.graphax_reduce_select_aliases, line? := some 40 }
      },
      {
        op := selectNAliasOpName
        invars := #[{ id := 1 }, { id := 3 }, { id := 2 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_reduce_select_aliases, line? := some 41 }
      },
      {
        op := `Graphax.reduce_max
        invars := #[{ id := 4 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_reduce_select_aliases, line? := some 42 }
      }
    ]
    outvars := #[{ id := 5 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX reduction/select alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 4
      "reduce_sum(1) + select_n(data only=2) + reduce_max(1) should produce four edges."
    LeanTest.assertTrue (edges.any (fun e => e.src == 0 && e.dst == 3))
      "reduce_sum alias should emit unary edge."
    LeanTest.assertTrue (edges.any (fun e => e.src == 3 && e.dst == 4))
      "select_n alias should emit edge for first data branch."
    LeanTest.assertTrue (edges.any (fun e => e.src == 2 && e.dst == 4))
      "select_n alias should emit edge for second data branch."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 1 && e.dst == 4)))
      "select_n alias should not emit edge for selector input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 4 && e.dst == 5))
      "reduce_max alias should emit unary edge."

@[test]
def testGraphaxJaxDynamicAliasCoverage : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }, { id := 2 }]
    eqns := #[
      {
        op := dynamicSliceAliasOpName
        invars := #[{ id := 0 }, { id := 2 }]
        outvars := #[{ id := 3 }]
        source := { decl := `test.graphax_dynamic_aliases, line? := some 50 }
      },
      {
        op := gatherAliasOpName
        invars := #[{ id := 3 }, { id := 2 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_dynamic_aliases, line? := some 51 }
      },
      {
        op := dynamicUpdateSliceAliasOpName
        invars := #[{ id := 4 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_dynamic_aliases, line? := some 52 }
      },
      {
        op := scatterAliasOpName
        invars := #[{ id := 5 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 6 }]
        source := { decl := `test.graphax_dynamic_aliases, line? := some 53 }
      }
    ]
    outvars := #[{ id := 6 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX dynamic alias coverage should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 6
      "dynamic_slice(1) + gather(1) + dynamic_update_slice(2) + scatter(2) should produce six edges."
    LeanTest.assertTrue (edges.any (fun e => e.src == 0 && e.dst == 3))
      "dynamic_slice should emit edge from base operand."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 2 && e.dst == 3)))
      "dynamic_slice should not emit edge from index operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 3 && e.dst == 4))
      "gather should emit edge from base operand."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 2 && e.dst == 4)))
      "gather should not emit edge from index operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 4 && e.dst == 5))
      "dynamic_update_slice should emit edge from base operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 1 && e.dst == 5))
      "dynamic_update_slice should emit edge from update operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 5 && e.dst == 6))
      "scatter should emit edge from base operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 1 && e.dst == 6))
      "scatter should emit edge from update operand."

@[test]
def testGraphaxJaxPadSelectAndDynamicUpdateIndexAliasCoverage : IO Unit := do
  let jaxpr : LeanJaxpr := {
    invars := #[{ id := 0 }, { id := 1 }, { id := 2 }, { id := 3 }]
    eqns := #[
      {
        op := `jax.lax.pad_p
        invars := #[{ id := 0 }, { id := 1 }]
        outvars := #[{ id := 4 }]
        source := { decl := `test.graphax_pad_select_dynamic_update_index_aliases, line? := some 60 }
      },
      {
        op := `jax.lax.select_p
        invars := #[{ id := 2 }, { id := 4 }, { id := 3 }]
        outvars := #[{ id := 5 }]
        source := { decl := `test.graphax_pad_select_dynamic_update_index_aliases, line? := some 61 }
      },
      {
        op := `jax.lax.dynamic_update_index_in_dim_p
        invars := #[{ id := 5 }, { id := 1 }, { id := 2 }]
        outvars := #[{ id := 6 }]
        source := { decl := `test.graphax_pad_select_dynamic_update_index_aliases, line? := some 62 }
      }
    ]
    outvars := #[{ id := 6 }]
  }

  let res ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  match res with
  | .error errs =>
    LeanTest.fail s!"Graphax/JAX pad/select/dynamic_update_index aliases should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | .ok edges =>
    LeanTest.assertEqual edges.size 6
      "pad(2) + select(data-only=2) + dynamic_update_index_in_dim(2) should produce six edges."
    LeanTest.assertTrue (edges.any (fun e => e.src == 0 && e.dst == 4))
      "pad should emit edge from base operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 1 && e.dst == 4))
      "pad should emit edge from padding-value operand."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 2 && e.dst == 5)))
      "select should not emit edge from selector input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 4 && e.dst == 5))
      "select should emit edge from true/data input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 3 && e.dst == 5))
      "select should emit edge from false/data input."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 2 && e.dst == 6)))
      "dynamic_update_index_in_dim should not emit edge from index input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 5 && e.dst == 6))
      "dynamic_update_index_in_dim should emit edge from base operand."
    LeanTest.assertTrue (edges.any (fun e => e.src == 1 && e.dst == 6))
      "dynamic_update_index_in_dim should emit edge from update operand."

@[test]
def testGraphaxParityScanAndCondExtractionFromFnBody : IO Unit := do
  let scanCarry : Lean.IR.VarId := { idx := 0 }
  let scanData : Lean.IR.VarId := { idx := 1 }
  let scanOut : Lean.IR.VarId := { idx := 2 }
  let scanParams : Array Lean.IR.Param := #[
    { x := scanCarry, borrow := false, ty := Lean.IR.IRType.object },
    { x := scanData, borrow := false, ty := Lean.IR.IRType.object }
  ]
  let scanBody : Lean.IR.FnBody :=
    .vdecl scanOut Lean.IR.IRType.object
      (Lean.IR.Expr.fap `Graphax.scan #[Lean.IR.Arg.var scanCarry, Lean.IR.Arg.var scanData]) (
      .ret (.var scanOut)
    )

  let scanRes ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    buildAndExtractFromFnBody {} `test.graphax_scan_supported scanParams scanBody
  )
  match scanRes with
  | .error err =>
    LeanTest.fail s!"Graphax-parity path should extract successfully for `lax.scan`, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 2
      "scan should propagate all inputs (carry/data) to its output."
    LeanTest.assertTrue (edges.any (fun e => e.src == 1 && e.dst == 3))
      "scan should propagate carry input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 2 && e.dst == 3))
      "scan should propagate data input."
    let carryEdge? := edges.find? (fun e => e.src == 1 && e.dst == 3)
    let dataEdge? := edges.find? (fun e => e.src == 2 && e.dst == 3)
    match carryEdge? with
    | none =>
      LeanTest.fail "Missing carry edge for scan."
    | some e =>
      LeanTest.assertTrue
        (match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic (.unary op .x .carry) => op == `Graphax.scan
        | _ => false)
        "scan carry input should emit a carry-tagged semantic map for the frontend primitive."
    match dataEdge? with
    | none =>
      LeanTest.fail "Missing data edge for scan."
    | some e =>
      LeanTest.assertTrue
        (match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic (.unary op .x .projection) => op == `Graphax.scan
        | _ => false)
        "scan data input should emit a projection-tagged semantic map for the frontend primitive."

  let condPred : Lean.IR.VarId := { idx := 10 }
  let condTrue : Lean.IR.VarId := { idx := 11 }
  let condFalse : Lean.IR.VarId := { idx := 12 }
  let condOut : Lean.IR.VarId := { idx := 13 }
  let condParams : Array Lean.IR.Param := #[
    { x := condPred, borrow := false, ty := Lean.IR.IRType.object },
    { x := condTrue, borrow := false, ty := Lean.IR.IRType.object },
    { x := condFalse, borrow := false, ty := Lean.IR.IRType.object }
  ]
  let condBody : Lean.IR.FnBody :=
    .vdecl condOut Lean.IR.IRType.object
      (Lean.IR.Expr.fap `Graphax.cond #[Lean.IR.Arg.var condPred, Lean.IR.Arg.var condTrue, Lean.IR.Arg.var condFalse]) (
      .ret (.var condOut)
    )

  let condRes ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    buildAndExtractFromFnBody {} `test.graphax_cond_supported condParams condBody
  )
  match condRes with
  | .error err =>
    LeanTest.fail s!"Graphax-parity path should extract successfully for `lax.cond`, got: {buildExtractErrorToString err}"
  | .ok (_jaxpr, edges) =>
    LeanTest.assertEqual edges.size 2
      "cond should ignore predicate and propagate only data inputs."
    LeanTest.assertTrue (!(edges.any (fun e => e.src == 1 && e.dst == 4)))
      "cond should not emit an edge for predicate input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 2 && e.dst == 4))
      "cond should propagate true/data input."
    LeanTest.assertTrue (edges.any (fun e => e.src == 3 && e.dst == 4))
      "cond should propagate false/data input."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic (.unary op .x .projection) => op == `Graphax.cond
        | _ => false))
      "cond alias rule should emit projection-tagged semantic maps for the frontend primitive."

@[test]
def testGraphaxStructuralAliasPayloadParityWithKStmtSubset : IO Unit := do
  let x : JVar := { id := 0, metaInfo := { shape := some #[3] } }
  let bcast : JVar := { id := 1, metaInfo := { shape := some #[2, 3] } }
  let transposed : JVar := { id := 2, metaInfo := { shape := some #[3, 2] } }
  let sliced : JVar := { id := 3, metaInfo := { shape := some #[3, 1] } }
  let y : JVar := { id := 4, metaInfo := { shape := some #[3, 1] } }
  let out : JVar := { id := 5, metaInfo := { shape := some #[3, 2] } }
  let jaxpr : LeanJaxpr := {
    invars := #[x, y]
    eqns := #[
      {
        op := broadcastInDimAliasOpName
        invars := #[x]
        outvars := #[bcast]
        source := { decl := `test.graphax_structural_payload_parity, line? := some 70 }
      },
      {
        op := transposeAliasOpName
        invars := #[bcast]
        outvars := #[transposed]
        source := { decl := `test.graphax_structural_payload_parity, line? := some 71 }
      },
      {
        op := `jax.lax.slice_in_dim_p
        invars := #[transposed]
        outvars := #[sliced]
        params := #[
          OpParam.mkNat .startCol 1,
          OpParam.mkNat .numCols 1
        ]
        source := { decl := `test.graphax_structural_payload_parity, line? := some 72 }
      },
      {
        op := concatenateAliasOpName
        invars := #[sliced, y]
        outvars := #[out]
        source := { decl := `test.graphax_structural_payload_parity, line? := some 73 }
      }
    ]
    outvars := #[out]
  }

  let jaxEdges ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractLocalJacEdges jaxpr
  )
  let kstmtEdgesRes ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    let v0 : VarId := { idx := 0 }
    let v1 : VarId := { idx := 1 }
    let v2 : VarId := { idx := 2 }
    let v3 : VarId := { idx := 3 }
    let v4 : VarId := { idx := 4 }
    let v5 : VarId := { idx := 5 }
    buildAndExtractFromKStmts {} #[
      KStmt.declRV v0 .Float32 3,
      KStmt.declRT v1 .Float32 2 3 .Row,
      KStmt.broadcast .Row v1 v0,
      KStmt.declRT v2 .Float32 3 2 .Row,
      KStmt.transpose v2 v1,
      KStmt.declRT v3 .Float32 3 1 .Row,
      KStmt.sliceCols v3 v2 1 1,
      KStmt.declRT v4 .Float32 3 1 .Row,
      KStmt.declRT v5 .Float32 3 2 .Row,
      KStmt.concatCols v5 v3 v4
    ]
  )

  match jaxEdges, kstmtEdgesRes with
  | .error errs, _ =>
    LeanTest.fail s!"Graphax structural alias parity fixture should extract successfully, got: {errs.map ruleExecutionErrorToString}"
  | _, .error err =>
    LeanTest.fail s!"KStmt structural parity fixture should lower/extract successfully, got: {buildExtractErrorToString err}"
  | .ok jEdges, .ok (_jaxpr, kEdges) =>
    LeanTest.assertEqual jEdges.size 5
      "Graphax structural parity fixture should emit one edge per supported exact transform/input."
    LeanTest.assertEqual kEdges.size 5
      "KStmt structural parity fixture should emit one edge per supported exact transform/input."

    match findEdge? jEdges 0 1, findEdge? kEdges 0 1 with
    | some jEdge, some kEdge =>
      assertSameSparsePayload "broadcast payload parity" jEdge.map kEdge.map
    | _, _ =>
      LeanTest.fail "Missing broadcast parity edge."

    match findEdge? jEdges 1 2, findEdge? kEdges 1 2 with
    | some jEdge, some kEdge =>
      assertSameSparsePayload "transpose payload parity" jEdge.map kEdge.map
    | _, _ =>
      LeanTest.fail "Missing transpose parity edge."

    match findEdge? jEdges 2 3, findEdge? kEdges 2 3 with
    | some jEdge, some kEdge =>
      assertSameSparsePayload "slice payload parity" jEdge.map kEdge.map
    | _, _ =>
      LeanTest.fail "Missing slice parity edge."

    match findEdge? jEdges 3 5, findEdge? kEdges 3 5 with
    | some jEdge, some kEdge =>
      assertSameSparsePayload "concat-left payload parity" jEdge.map kEdge.map
    | _, _ =>
      LeanTest.fail "Missing concat-left parity edge."

    match findEdge? jEdges 4 5, findEdge? kEdges 4 5 with
    | some jEdge, some kEdge =>
      assertSameSparsePayload "concat-right payload parity" jEdge.map kEdge.map
    | _, _ =>
      LeanTest.fail "Missing concat-right parity edge."

end Tests.ADGraphaxAlphaGradParity
