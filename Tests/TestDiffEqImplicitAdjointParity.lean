/-
  Tests/TestDiffEqImplicitAdjointParity.lean

  Gradient correctness for the DIRK discrete adjoint (implicit solvers).
  The reverse sweep solves the transposed stage equations by fixed-point
  iteration using only the term's VJP; these tests validate the resulting
  gradients against central finite differences on a nonlinear problem, and
  cross-check an implicit solver's gradient against an explicit one.
-/
import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Torch

open torch
open torch.DiffEq

namespace Tests.DiffEqImplicitAdjointParity

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

/-- Nonlinear scalar problem dy/dt = a·y − y³ (state-dependent Jacobian, so
    the transposed fixed-point solve is genuinely exercised). -/
private def nonlinearTerm : ODETerm (T #[]) (T #[]) :=
  { vectorField := fun _t y a => sub (mul a y) (mul (mul y y) y) }

private def solveFinal
    (solver : AbstractSolver (ODETerm (T #[]) (T #[])) (T #[]) (T #[]) Time (T #[]))
    (y0 a : T #[]) (dt0 : Float) : Float :=
  let sol :=
    diffeqsolve
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[]) (VF := T #[]) (Control := Time) (Args := T #[])
      (Controller := ConstantStepSize)
      nonlinearTerm solver 0.0 1.0 (some dt0) y0 a (saveat := { t1 := true })
  match sol.ys with
  | some ys => nn.item (ys.getD (ys.size - 1) y0)
  | none => 0.0

private def checkAgainstFiniteDifferences
    (label : String)
    (solver : AbstractSolver (ODETerm (T #[]) (T #[])) (T #[]) (T #[]) Time (T #[]))
    (dt0 : Float) (tol : Float) : IO Unit := do
  let y0 := full #[] 0.8
  let a := full #[] 0.5
  let adjY1 := ones #[]
  let (_, adjOpt) :=
    diffeqsolveDirectAdjoint nonlinearTerm solver 0.0 1.0 (some dt0) y0 a adjY1
      (saveat := { t1 := true })
  match adjOpt with
  | none => LeanTest.fail s!"{label}: expected direct adjoint result"
  | some adj => do
      let eps := 1.0e-4
      let fdY0 :=
        (solveFinal solver (add_scalar y0 eps) a dt0 -
         solveFinal solver (add_scalar y0 (-eps)) a dt0) / (2.0 * eps)
      let fdA :=
        (solveFinal solver y0 (add_scalar a eps) dt0 -
         solveFinal solver y0 (add_scalar a (-eps)) dt0) / (2.0 * eps)
      let adjY0 := nn.item adj.adjY0
      let adjA := nn.item adj.adjArgs
      LeanTest.assertTrue (approx adjY0 fdY0 tol)
        s!"{label}: d/dy0 expected {fdY0}, got {adjY0}"
      LeanTest.assertTrue (approx adjA fdA tol)
        s!"{label}: d/da expected {fdA}, got {adjA}"

@[test] def testImplicitEulerDirectAdjointFiniteDiff : IO Unit := do
  checkAgainstFiniteDifferences "implicitEuler"
    (ImplicitEuler.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[]) (VF := T #[]) (Args := T #[]))
    0.01 1.0e-2

@[test] def testKvaerno3DirectAdjointFiniteDiff : IO Unit := do
  checkAgainstFiniteDifferences "kvaerno3"
    (Kvaerno3.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[]) (VF := T #[]) (Args := T #[]))
    0.05 1.0e-2

@[test] def testKvaerno5MatchesDopri5Adjoint : IO Unit := do
  -- Cross-solver agreement: a 5th-order implicit and a 5th-order explicit
  -- method must agree on the gradients of a smooth problem.
  let y0 := full #[] 0.8
  let a := full #[] 0.5
  let adjY1 := ones #[]
  let dt0 := some 0.05
  let implicitSolver :=
    Kvaerno5.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[]) (VF := T #[]) (Args := T #[])
  let explicitSolver :=
    Dopri5.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[]) (VF := T #[]) (Args := T #[])
  let (_, adjI) :=
    diffeqsolveDirectAdjoint nonlinearTerm implicitSolver 0.0 1.0 dt0 y0 a adjY1
      (saveat := { t1 := true })
  let (_, adjE) :=
    diffeqsolveDirectAdjoint nonlinearTerm explicitSolver 0.0 1.0 dt0 y0 a adjY1
      (saveat := { t1 := true })
  match adjI, adjE with
  | some i, some e => do
      LeanTest.assertTrue (approx (nn.item i.adjY0) (nn.item e.adjY0) 1.0e-4)
        s!"kvaerno5 d/dy0 {nn.item i.adjY0} vs dopri5 {nn.item e.adjY0}"
      LeanTest.assertTrue (approx (nn.item i.adjArgs) (nn.item e.adjArgs) 1.0e-4)
        s!"kvaerno5 d/da {nn.item i.adjArgs} vs dopri5 {nn.item e.adjArgs}"
  | _, _ => LeanTest.fail "expected adjoint results from both solvers"

end Tests.DiffEqImplicitAdjointParity
