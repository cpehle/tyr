import Tyr.GPU.AutoGrad
import Tyr.GPU.Codegen.IR
import LeanTest
import Lean.CoreM

/-!
# `Tests.TestGPUAutoGrad`

GPU autodiff tests for trace generation, linearization, transpose, and VJP behavior.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

open Tyr.GPU.Codegen
open Tyr.GPU.AD
open Tyr.AD
open Lean

namespace Tests.GPU.AutoGrad

open LeanTest

-- Helper to run CoreM in IO
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

-- Helper to run TraceM
def runTraceM (x : TraceM α) : IO (α × Tyr.GPU.AD.TraceState) := do
  runCoreM (do
    let (res, _) ← (x.run {}).run {}
    return res)

-- Helper to run Transpose
def runTranspose (trace : Array LinearInst) : IO (Array KStmt) := do
  runCoreM (do
    let (res, _) ← (transposeTrace trace).run {}
    return res)

-- Test Linearization (JVP) for Add
@[test]
def testJVPAdd : IO Unit := do
  -- v2 = v0 + v1
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let stmt := KStmt.binary .Add v2 v0 v1
  
  -- Run Linearize
  let (_, state) ← runTraceM (linearizeStmt stmt)
  
  -- Check Primal
  let primalStmts := state.primalStmts
  LeanTest.assertTrue (primalStmts.size == 1) s!"Expected 1 primal stmt, got {primalStmts.size}"
  match primalStmts[0]? with
  | some (KStmt.binary .Add _ _ _) => pure ()
  | _ => throw (IO.userError s!"Expected primal Add")

  -- Check Linear Trace
  let linearTrace := state.linearTrace
  LeanTest.assertTrue (linearTrace.size == 1) s!"Expected 1 linear inst, got {linearTrace.size}"
  match linearTrace[0]? with
  | some (LinearInst.add _ _ _) => pure ()
  | _ => throw (IO.userError s!"Expected linear Add")

-- Test Transposition (VJP) for Mul
@[test]
def testVJPMul : IO Unit := do
  -- v2 = v0 * v1
  let v0 : Tyr.GPU.Codegen.VarId := { idx := 0 }
  let v1 : Tyr.GPU.Codegen.VarId := { idx := 1 }
  let v2 : Tyr.GPU.Codegen.VarId := { idx := 2 }
  let stmt := KStmt.binary .Mul v2 v0 v1
  
  -- 1. Linearize
  let (_, state) ← runTraceM (linearizeStmt stmt)
  
  -- Verify Linear Trace is Mul
  let linearTrace := state.linearTrace
  match linearTrace[0]? with
  | some (LinearInst.mul _ _ _ _ _) => pure ()
  | _ => throw (IO.userError s!"Expected linear Mul")

  -- 2. Transpose
  let vjpStmts ← runTranspose linearTrace
  
  -- Expected VJP stmts for Mul:
  -- da += dout * b
  -- db += dout * a
  -- Each += is decomposed into Mul then Add. So 2 stmts per input. Total 4.
  
  LeanTest.assertTrue (vjpStmts.size == 4) s!"Expected 4 VJP instructions, got {vjpStmts.size}"
  
  match vjpStmts[0]?, vjpStmts[1]? with
  | some (KStmt.binary .Mul _ _ _), some (KStmt.binary .Add _ _ _) => pure ()
  | _, _ => throw (IO.userError s!"Expected Mul then Add for first input gradient")

/-! ## Unary Operations Tests -/

-- Test Linearization (JVP) for Exp
@[test]
def testJVPExp : IO Unit := do
  -- y = exp(x)
  -- dy = exp(x) * dx = y * dx
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let stmt := KStmt.unary .Exp y x

  let (_, state) ← runTraceM (linearizeStmt stmt)

  -- Check primal
  LeanTest.assertTrue (state.primalStmts.size == 1) s!"Expected 1 primal stmt"
  match state.primalStmts[0]? with
  | some (KStmt.unary .Exp _ _) => pure ()
  | _ => throw (IO.userError "Expected primal Exp")

  -- Check linear trace contains exp instruction with correct primal reference
  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.exp _dout _din primalY) =>
    -- primalY should reference the output y for chain rule: dy = dx * y
    LeanTest.assertTrue (primalY.idx == y.idx) s!"primalY should be y, got {primalY.idx}"
  | _ => throw (IO.userError "Expected linear Exp")

-- Test Transposition (VJP) for Exp
@[test]
def testVJPExp : IO Unit := do
  -- Backward: dx += dy * exp(x) = dy * y
  let trace := #[LinearInst.exp { idx := 2 } { idx := 0 } { idx := 1 }]
  let vjpStmts ← runTranspose trace

  -- Expected: t1 = dout * y; din += t1
  LeanTest.assertTrue (vjpStmts.size == 2) s!"Expected 2 VJP stmts for exp, got {vjpStmts.size}"

  match vjpStmts[0]?, vjpStmts[1]? with
  | some (KStmt.binary .Mul _ _ _), some (KStmt.binary .Add _ _ _) => pure ()
  | _, _ => throw (IO.userError "Expected Mul then Add for exp backward")

-- Test Linearization (JVP) for Log
@[test]
def testJVPLog : IO Unit := do
  -- y = log(x)
  -- dy = dx / x
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let stmt := KStmt.unary .Log y x

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.log _dout _din primalX) =>
    -- primalX should reference the input x for chain rule: dy = dx / x
    LeanTest.assertTrue (primalX.idx == x.idx) s!"primalX should be x"
  | _ => throw (IO.userError "Expected linear Log")

-- Test Transposition (VJP) for Log
@[test]
def testVJPLog : IO Unit := do
  -- Backward: dx += dy / x
  let trace := #[LinearInst.log { idx := 2 } { idx := 0 } { idx := 1 }]
  let vjpStmts ← runTranspose trace

  -- Expected: t1 = dout / x; din += t1
  LeanTest.assertTrue (vjpStmts.size == 2) s!"Expected 2 VJP stmts for log, got {vjpStmts.size}"

-- Test Linearization (JVP) for Tanh
@[test]
def testJVPTanh : IO Unit := do
  -- y = tanh(x)
  -- dy = (1 - y²) * dx
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let stmt := KStmt.unary .Tanh y x

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.tanh _ _ _) => pure ()
  | _ => throw (IO.userError "Expected linear Tanh")

-- Test Transposition (VJP) for Tanh
@[test]
def testVJPTanh : IO Unit := do
  -- Backward: dx += dy * (1 - y²)
  let trace := #[LinearInst.tanh { idx := 2 } { idx := 0 } { idx := 1 }]
  let vjpStmts ← runTranspose trace

  -- Should have: t1 = y*y; t1 *= -1; t1 += 1; t2 = dout*t1; din += t2
  LeanTest.assertTrue (vjpStmts.size == 5) s!"Expected 5 VJP stmts for tanh, got {vjpStmts.size}"

/-! ## Binary Operations Tests -/

-- Test Linearization (JVP) for Sub
@[test]
def testJVPSub : IO Unit := do
  -- z = x - y
  -- dz = dx - dy
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmt := KStmt.binary .Sub v2 v0 v1

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.sub _ _ _) => pure ()
  | _ => throw (IO.userError "Expected linear Sub")

-- Test Transposition (VJP) for Sub
@[test]
def testVJPSub : IO Unit := do
  -- Backward: dx += dz; dy -= dz (implemented as dy += -dz via Sub)
  let trace := #[LinearInst.sub { idx := 2 } { idx := 0 } { idx := 1 }]
  let vjpStmts ← runTranspose trace

  -- Expected: din1 += dout; din2 = din2 - dout
  LeanTest.assertTrue (vjpStmts.size == 2) s!"Expected 2 VJP stmts for sub, got {vjpStmts.size}"

  match vjpStmts[0]?, vjpStmts[1]? with
  | some (KStmt.binary .Add _ _ _), some (KStmt.binary .Sub _ _ _) => pure ()
  | _, _ => throw (IO.userError "Expected Add then Sub for sub backward")

-- Test Linearization (JVP) for Div
@[test]
def testJVPDiv : IO Unit := do
  -- z = x / y
  let v0 : VarId := { idx := 0 }
  let v1 : VarId := { idx := 1 }
  let v2 : VarId := { idx := 2 }
  let stmt := KStmt.binary .Div v2 v0 v1

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.div _ _ _ _ _) => pure ()
  | _ => throw (IO.userError "Expected linear Div")

-- Test Transposition (VJP) for Div
@[test]
def testVJPDiv : IO Unit := do
  -- Backward:
  -- da += dout * (1 / b)
  -- db += dout * (-a / b^2)
  let trace := #[LinearInst.div { idx := 2 } { idx := 0 } { idx := 1 } { idx := 3 } { idx := 4 }]
  let vjpStmts ← runTranspose trace

  -- recip + mul + add + mul + scalarMul + mul + mul + add
  LeanTest.assertTrue (vjpStmts.size == 8) s!"Expected 8 VJP stmts for div, got {vjpStmts.size}"

  match vjpStmts[0]?, vjpStmts[1]?, vjpStmts[2]?, vjpStmts[7]? with
  | some (KStmt.unary .Recip _ _), some (KStmt.binary .Mul _ _ _),
    some (KStmt.binary .Add _ _ _), some (KStmt.binary .Add _ _ _) => pure ()
  | _, _, _, _ => throw (IO.userError "Expected recip/mul/add pattern for div backward")

/-! ## Broadcast/Reduce Tests -/

-- Test Linearization (JVP) for Broadcast Row
@[test]
def testJVPBroadcastRow : IO Unit := do
  -- dst[i,j] = vec[i] (broadcast row vector to matrix)
  let vec : VarId := { idx := 0 }
  let dst : VarId := { idx := 1 }
  let stmt := KStmt.broadcast .Row dst vec

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.broadcast .Row _ _) => pure ()
  | _ => throw (IO.userError "Expected linear Broadcast with Row axis")

-- Test Transposition (VJP) for Broadcast Row
@[test]
def testVJPBroadcastRow : IO Unit := do
  -- Backward: reduce sum along columns (opposite axis)
  let trace := #[LinearInst.broadcast .Row { idx := 1 } { idx := 0 }]
  let vjpStmts ← runTranspose trace

  -- Should produce reduceAccum with Col axis (opposite of Row)
  LeanTest.assertTrue (vjpStmts.size == 1) s!"Expected 1 VJP stmt for broadcast, got {vjpStmts.size}"
  match vjpStmts[0]? with
  | some (KStmt.reduceAccum .Sum .Col _ _ _) => pure ()
  | some other => throw (IO.userError s!"Expected reduceAccum with Col axis, got {repr other}")
  | none => throw (IO.userError "No VJP stmt produced")

-- Test Transposition (VJP) for Sum Row
@[test]
def testVJPSumRow : IO Unit := do
  -- Forward row reduction is reversed by column broadcast.
  let trace := #[LinearInst.sum .Row { idx := 1 } { idx := 0 }]
  let vjpStmts ← runTranspose trace

  LeanTest.assertTrue (vjpStmts.size == 2) s!"Expected 2 VJP stmts for sum row, got {vjpStmts.size}"
  match vjpStmts[0]?, vjpStmts[1]? with
  | some (KStmt.broadcast .Col _ _), some (KStmt.binary .Add _ _ _) => pure ()
  | _, _ => throw (IO.userError "Expected broadcast Col then Add for sum row backward")

-- Test Transposition (VJP) for MMA transpose variants
@[test]
def testVJPMMATransposes : IO Unit := do
  let cases : Array (MMATranspose × MMATranspose × MMATranspose) := #[
    (.AB, .ABt, .AtB),
    (.ABt, .AB, .AtB),
    (.AtB, .ABt, .AB),
    (.AtBt, .AtBt, .AtBt)
  ]

  for case in cases do
    let (fwdTrans, daTrans, dbTrans) := case
    let trace := #[LinearInst.mma fwdTrans { idx := 9 } { idx := 0 } { idx := 1 } { idx := 2 } { idx := 3 } { idx := 4 }]
    let vjpStmts ← runTranspose trace

    LeanTest.assertTrue (vjpStmts.size == 3) s!"Expected 3 VJP stmts for mma {fwdTrans}, got {vjpStmts.size}"

    match vjpStmts[0]? with
    | some (KStmt.binary .Add _ _ _) => pure ()
    | _ => throw (IO.userError s!"Expected dC accumulation for mma {fwdTrans}")

    match vjpStmts[1]? with
    | some (KStmt.mma trans _ _ _ _) =>
      LeanTest.assertTrue (trans == daTrans) s!"Unexpected dA transpose for {fwdTrans}: got {trans}, expected {daTrans}"
    | _ => throw (IO.userError s!"Expected mma for dA in {fwdTrans}")

    match vjpStmts[2]? with
    | some (KStmt.mma trans _ _ _ _) =>
      LeanTest.assertTrue (trans == dbTrans) s!"Unexpected dB transpose for {fwdTrans}: got {trans}, expected {dbTrans}"
    | _ => throw (IO.userError s!"Expected mma for dB in {fwdTrans}")

/-! ## Memory Operations Tests -/

-- Test Linearization (JVP) for Load
@[test]
def testJVPLoad : IO Unit := do
  let src : VarId := { idx := 0 }
  let dst : VarId := { idx := 1 }
  let stmt := KStmt.load dst src

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.id _ _) => pure ()  -- Load is identity for gradients
  | _ => throw (IO.userError "Expected linear id for load")

-- Test Linearization (JVP) for Store
@[test]
def testJVPStore : IO Unit := do
  let src : VarId := { idx := 0 }
  let dst : VarId := { idx := 1 }
  let stmt := KStmt.store dst src

  let (_, state) ← runTraceM (linearizeStmt stmt)

  LeanTest.assertTrue (state.linearTrace.size == 1) s!"Expected 1 linear inst"
  match state.linearTrace[0]? with
  | some (LinearInst.id _ _) => pure ()  -- Store is identity for gradients
  | _ => throw (IO.userError "Expected linear id for store")

-- Test Transposition (VJP) for Identity
@[test]
def testVJPId : IO Unit := do
  -- Identity backward: din += dout
  let trace := #[LinearInst.id { idx := 1 } { idx := 0 }]
  let vjpStmts ← runTranspose trace

  LeanTest.assertTrue (vjpStmts.size == 1) s!"Expected 1 VJP stmt for id, got {vjpStmts.size}"
  match vjpStmts[0]? with
  | some (KStmt.binary .Add _ _ _) => pure ()
  | _ => throw (IO.userError "Expected Add for id backward")

/-! ## Integration Tests -/

-- Test chain rule: z = exp(x * y)
@[test]
def testChainRule : IO Unit := do
  -- z = exp(x * y)
  -- dz/dx = exp(x*y) * y
  -- dz/dy = exp(x*y) * x
  let v0 : VarId := { idx := 0 }  -- x
  let v1 : VarId := { idx := 1 }  -- y
  let v2 : VarId := { idx := 2 }  -- x * y
  let v3 : VarId := { idx := 3 }  -- exp(x * y)

  let stmts := #[
    KStmt.binary .Mul v2 v0 v1,  -- v2 = x * y
    KStmt.unary .Exp v3 v2       -- v3 = exp(v2)
  ]

  let (_, state) ← runTraceM (stmts.forM linearizeStmt)

  -- Should have 2 primal stmts
  LeanTest.assertTrue (state.primalStmts.size == 2) s!"Expected 2 primal stmts, got {state.primalStmts.size}"

  -- Should have 2 linear instructions
  LeanTest.assertTrue (state.linearTrace.size == 2) s!"Expected 2 linear insts, got {state.linearTrace.size}"

  -- Transpose and verify chain rule application
  let vjpStmts ← runTranspose state.linearTrace

  -- exp backward (2 stmts) + mul backward (4 stmts) = 6 total
  LeanTest.assertTrue (vjpStmts.size == 6) s!"Expected 6 VJP stmts for chain, got {vjpStmts.size}"

-- Test nested computation: z = (x + y) * (x - y)
@[test]
def testNestedComputation : IO Unit := do
  let x : VarId := { idx := 0 }
  let y : VarId := { idx := 1 }
  let sum : VarId := { idx := 2 }   -- x + y
  let diff : VarId := { idx := 3 }  -- x - y
  let z : VarId := { idx := 4 }     -- (x+y) * (x-y)

  let stmts := #[
    KStmt.binary .Add sum x y,
    KStmt.binary .Sub diff x y,
    KStmt.binary .Mul z sum diff
  ]

  let (_, state) ← runTraceM (stmts.forM linearizeStmt)

  -- Should have 3 primal and 3 linear
  LeanTest.assertTrue (state.primalStmts.size == 3) s!"Expected 3 primal stmts"
  LeanTest.assertTrue (state.linearTrace.size == 3) s!"Expected 3 linear insts"

  -- Transpose
  let vjpStmts ← runTranspose state.linearTrace

  -- mul backward (4) + sub backward (2) + add backward (2) = 8 total
  LeanTest.assertTrue (vjpStmts.size == 8) s!"Expected 8 VJP stmts, got {vjpStmts.size}"

end Tests.GPU.AutoGrad
