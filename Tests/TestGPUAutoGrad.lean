import Tyr.GPU.AutoGrad
import Tyr.GPU.Codegen.IR
import LeanTest
import Lean.CoreM

open Tyr.GPU.Codegen
open Tyr.GPU.AD
open Tyr.AD
open Lean

namespace Tests.GPU.AutoGrad

open LeanTest

-- Helper to run CoreM in IO
def runCoreM (x : CoreM α) : IO α := do
  let env ← mkEmptyEnvironment
  let ctx : Core.Context := { fileName := "<test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let (res, _) ← EIO.toIO (fun _ => IO.userError "CoreM Error") eio
  return res

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

end Tests.GPU.AutoGrad
