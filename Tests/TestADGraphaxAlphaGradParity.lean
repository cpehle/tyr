import Lean.CoreM
import LeanTest
import Tyr.AD.JaxprLike
import Tyr.GPU.Codegen.IR

namespace Tests.ADGraphaxAlphaGradParity

open Lean
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
        op := `Graphax.dot_general
        invars := #[{ id := 2 }, { id := 3 }]
        outvars := #[{ id := 4 }]
        params := #[
          OpParam.mkName .variant `matmul,
          OpParam.mkNats .lhsContract #[1],
          OpParam.mkNats .rhsContract #[0],
          OpParam.mkNats .lhsBatch #[],
          OpParam.mkNats .rhsBatch #[]
        ]
        source := { decl := `test.graphax_aliases, line? := some 12 }
      }
    ]
    outvars := #[{ id := 4 }]
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
      "stop_gradient/iota should produce no edges; dot_general should produce lhs/rhs edges."
    let hasStopGradEdge := edges.any (fun e => e.src == 0 && e.dst == 2)
    LeanTest.assertTrue (!hasStopGradEdge)
      "stop_gradient alias should emit no local-Jac edge."
    LeanTest.assertTrue
      (edges.all (fun e =>
        match e.map.repr with
        | Tyr.AD.Sparse.SparseMapTag.semantic (.dotGeneral _) => true
        | _ => false))
      "Graphax dot_general alias should use structured dot_general semantic tags."

end Tests.ADGraphaxAlphaGradParity
