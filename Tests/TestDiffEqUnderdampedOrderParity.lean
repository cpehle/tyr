import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqUnderdampedOrderParity

open LeanTest
open torch
open torch.DiffEq

private def finalSavedPair {S C : Type}
    (label : String) (sol : Solution (Float × Float) S C) : IO (Float × Float) := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure (0.0, 0.0)
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure (0.0, 0.0)

private def pairL2 (a b : Float × Float) : Float :=
  let dx := a.1 - b.1
  let dv := a.2 - b.2
  Float.sqrt (dx * dx + dv * dv)

private def vector2 (a b : Float) : Vector 2 Float :=
  {
    data := #[a, b]
    size_eq := by simp
  }

private def vector2L2 (a b : Vector 2 Float) : Float :=
  let d0 := a.get ⟨0, by decide⟩ - b.get ⟨0, by decide⟩
  let d1 := a.get ⟨1, by decide⟩ - b.get ⟨1, by decide⟩
  Float.sqrt (d0 * d0 + d1 * d1)

private def finalSavedVec2Pair {S C : Type}
    (label : String) (sol : Solution (Vector 2 Float × Vector 2 Float) S C) :
    IO (Vector 2 Float × Vector 2 Float) := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure (vector2 0.0 0.0, vector2 0.0 0.0)
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure (vector2 0.0 0.0, vector2 0.0 0.0)

private def assertFiniteVector2 (label : String) (v : Vector 2 Float) : IO Unit := do
  let x0 := v.get ⟨0, by decide⟩
  let x1 := v.get ⟨1, by decide⟩
  LeanTest.assertTrue (Float.isFinite x0 && Float.isFinite x1)
    s!"{label}: expected finite vector entries, got {x0}, {x1}"

private def mkUnderdampedSingleStepTerms :
    MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
      (UnderdampedLangevinDiffusionTerm Float Unit) :=
  let path := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (1.0 : Float)
  let drift : UnderdampedLangevinDriftTerm Float Unit := {
    gradPotential := fun _t x _ => x
    gamma := fun _t _x _v _ => 0.3
    u := fun _t _x _v _ => 0.4
  }
  let diffusion : UnderdampedLangevinDiffusionTerm Float Unit :=
    UnderdampedLangevinDiffusionTerm.ofPath path
      (gamma := fun _t _x _v _ => 0.3)
      (u := fun _t _x _v _ => 0.4)
  { term1 := drift, term2 := diffusion }

private def solveUnderdampedSingleStep
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Float × Float)
      ((Float × Float) × Scalar)
      (Time × Float)
      Unit) : IO (Float × Float) := do
  let terms := mkUnderdampedSingleStepTerms
  let sol :=
    diffeqsolve
      (Term := MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Y := (Float × Float))
      (VF := ((Float × Float) × Scalar))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) ((1.0, 0.5) : Float × Float) ()
      (saveat := { t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    s!"{label}: single-step solve should succeed"
  finalSavedPair label sol

private def assertThresholdSwitchChangesEndpoint
    (label : String)
    (solverDirect solverTaylor : AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Float × Float)
      ((Float × Float) × Scalar)
      (Time × Float)
      Unit)
    (minDelta : Float := 1.0e-4) : IO Unit := do
  let yDirect ← solveUnderdampedSingleStep s!"{label} direct" solverDirect
  let yTaylor ← solveUnderdampedSingleStep s!"{label} taylor" solverTaylor
  let delta := pairL2 yDirect yTaylor
  LeanTest.assertTrue (delta > minDelta)
    s!"{label}: expected endpoint delta > {minDelta} between forced direct/taylor branches, got {delta}"

private def mkUnderdampedVectorTerms (seed : UInt64) :
    MultiTerm (UnderdampedLangevinDriftTerm (Vector 2 Float) Unit)
      (UnderdampedLangevinDiffusionTerm (Vector 2 Float) Unit) :=
  let tLo := 0.3
  let tHi := 1.0
  let bm : VirtualBrownianTree (Vector 2 Float) := {
    t0 := tLo
    t1 := tHi
    tol := 1.0e-3
    maxDepth := 24
    seed := seed
    shape := vector2 0.0 0.0
  }
  let bmPath := (VirtualBrownianTree.toAbstract bm).toPath
  let gammaConst := 0.3
  let uConst := 0.4
  let drift : UnderdampedLangevinDriftTerm (Vector 2 Float) Unit := {
    gradPotential := fun _t x _ => DiffEqSpace.scale 0.2 x
    gamma := fun _t _x _v _ => gammaConst
    u := fun _t _x _v _ => uConst
  }
  let diffusion : UnderdampedLangevinDiffusionTerm (Vector 2 Float) Unit :=
    UnderdampedLangevinDiffusionTerm.ofPath bmPath
      (gamma := fun _t _x _v _ => gammaConst)
      (u := fun _t _x _v _ => uConst)
  { term1 := drift, term2 := diffusion }

private def solveUnderdampedVectorShape
    (solver : AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm (Vector 2 Float) Unit)
        (UnderdampedLangevinDiffusionTerm (Vector 2 Float) Unit))
      (Vector 2 Float × Vector 2 Float)
      (((Vector 2 Float) × (Vector 2 Float)) × Scalar)
      (Time × Vector 2 Float)
      Unit) :
    Solution (Vector 2 Float × Vector 2 Float) solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) := by
  let terms := mkUnderdampedVectorTerms 20260307
  let ts : Array Time := #[0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
  let x0 := vector2 1.0 (-0.5)
  let v0 := vector2 0.2 0.0
  exact
    diffeqsolve
      (Term := MultiTerm (UnderdampedLangevinDriftTerm (Vector 2 Float) Unit)
        (UnderdampedLangevinDiffusionTerm (Vector 2 Float) Unit))
      (Y := (Vector 2 Float × Vector 2 Float))
      (VF := (((Vector 2 Float) × (Vector 2 Float)) × Scalar))
      (Control := (Time × Vector 2 Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.3 1.0 (some 0.1) (x0, v0) ()
      (saveat := { ts := some ts, t1 := false })

private def mkUnderdampedTerms (seed : UInt64) :
    MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
      (UnderdampedLangevinDiffusionTerm Float Unit) :=
  let tLo := 0.0
  let tHi := 1.0
  let bm : VirtualBrownianTree Float := {
    t0 := tLo
    t1 := tHi
    tol := 5.0e-4
    maxDepth := 24
    seed := seed
    shape := 0.0
  }
  let bmPath := (VirtualBrownianTree.toAbstract bm).toPath
  let gammaConst := 1.3
  let uConst := 0.9
  let drift : UnderdampedLangevinDriftTerm Float Unit := {
    gradPotential := fun _t x _ => 0.75 * x
    gamma := fun _t _x _v _ => gammaConst
    u := fun _t _x _v _ => uConst
  }
  let diffusion : UnderdampedLangevinDiffusionTerm Float Unit :=
    UnderdampedLangevinDiffusionTerm.ofPath bmPath
      (gamma := fun _t _x _v _ => gammaConst)
      (u := fun _t _x _v _ => uConst)
  { term1 := drift, term2 := diffusion }

private def solveUnderdampedEndpoint
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Float × Float)
      ((Float × Float) × Scalar)
      (Time × Float)
      Unit)
    (dt : Float) : IO (Float × Float) := do
  let terms := mkUnderdampedTerms 20260306
  let sol :=
    diffeqsolve
      (Term := MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Y := (Float × Float))
      (VF := ((Float × Float) × Scalar))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some dt) ((0.2, -0.3) : Float × Float) ()
      (saveat := { t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    s!"{label}: solve should succeed"
  finalSavedPair label sol

private def assertSelfConvergence
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm Float Unit)
        (UnderdampedLangevinDiffusionTerm Float Unit))
      (Float × Float)
      ((Float × Float) × Scalar)
      (Time × Float)
      Unit)
    (minRatio : Float := 1.2)
    (fineTol : Float := 0.12) : IO Unit := do
  let yRef ← solveUnderdampedEndpoint s!"{label} ref" solver 0.0125
  let yCoarse ← solveUnderdampedEndpoint s!"{label} coarse" solver 0.2
  let yMedium ← solveUnderdampedEndpoint s!"{label} medium" solver 0.1
  let yFine ← solveUnderdampedEndpoint s!"{label} fine" solver 0.05

  let errCoarse := pairL2 yCoarse yRef
  let errMedium := pairL2 yMedium yRef
  let errFine := pairL2 yFine yRef

  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"{label}: expected coarse/medium/fine endpoint error decrease, got {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > minRatio && ratio2 > minRatio)
    s!"{label}: self-convergence ratios too weak, got {ratio1}, {ratio2}"
  LeanTest.assertTrue (errFine < fineTol)
    s!"{label}: fine-grid endpoint error too large: {errFine}"

/--
Deterministic underdamped-Langevin self-convergence checks guided by:
- `../diffrax/test/test_underdamped_langevin.py::test_uld_strong_order`
-/
@[test] def testALIGNUnderdampedTaylorThresholdParity : IO Unit := do
  assertThresholdSwitchChangesEndpoint
    "ALIGN underdamped taylor-threshold parity"
    (ALIGN.solver { taylorThreshold := 0.0 })
    (ALIGN.solver { taylorThreshold := 100.0 })

@[test] def testShOULDUnderdampedTaylorThresholdParity : IO Unit := do
  assertThresholdSwitchChangesEndpoint
    "ShOULD underdamped taylor-threshold parity"
    (ShOULD.solver { taylorThreshold := 0.0 })
    (ShOULD.solver { taylorThreshold := 100.0 })

@[test] def testQUICSORTUnderdampedTaylorThresholdParity : IO Unit := do
  assertThresholdSwitchChangesEndpoint
    "QUICSORT underdamped taylor-threshold parity"
    (QUICSORT.solver { taylorThreshold := 0.0 })
    (QUICSORT.solver { taylorThreshold := 100.0 })

/--
Structured-state underdamped shape/branch parity inspired by:
- `../diffrax/test/test_underdamped_langevin.py::test_shape`
-/
@[test] def testUnderdampedVectorStateShapeParity : IO Unit := do
  let solAlignDirect := solveUnderdampedVectorShape (ALIGN.solver { taylorThreshold := 0.0 })
  let solAlignTaylor := solveUnderdampedVectorShape (ALIGN.solver { taylorThreshold := 100.0 })
  let solShouldDirect := solveUnderdampedVectorShape (ShOULD.solver { taylorThreshold := 0.0 })
  let solShouldTaylor := solveUnderdampedVectorShape (ShOULD.solver { taylorThreshold := 100.0 })
  let solQuicDirect := solveUnderdampedVectorShape (QUICSORT.solver { taylorThreshold := 0.0 })
  let solQuicTaylor := solveUnderdampedVectorShape (QUICSORT.solver { taylorThreshold := 100.0 })

  let all := #[
    ("ALIGN direct", solAlignDirect),
    ("ALIGN taylor", solAlignTaylor),
    ("ShOULD direct", solShouldDirect),
    ("ShOULD taylor", solShouldTaylor),
    ("QUICSORT direct", solQuicDirect),
    ("QUICSORT taylor", solQuicTaylor)
  ]
  for (label, sol) in all do
    LeanTest.assertTrue (sol.result == Result.successful)
      s!"{label}: solve should succeed for vector/PyTree-like state"
    match sol.ts, sol.ys with
    | some ts, some ys =>
        LeanTest.assertTrue (ts.size == 6 && ys.size == 6)
          s!"{label}: expected six saved outputs, got ts={ts.size}, ys={ys.size}"
    | _, _ =>
        LeanTest.fail s!"{label}: expected ts and ys outputs"

    let yLast ← finalSavedVec2Pair label sol
    assertFiniteVector2 s!"{label} x_last" yLast.1
    assertFiniteVector2 s!"{label} v_last" yLast.2

  let yAlignDirect ← finalSavedVec2Pair "ALIGN direct final" solAlignDirect
  let yAlignTaylor ← finalSavedVec2Pair "ALIGN taylor final" solAlignTaylor
  let yShouldDirect ← finalSavedVec2Pair "ShOULD direct final" solShouldDirect
  let yShouldTaylor ← finalSavedVec2Pair "ShOULD taylor final" solShouldTaylor
  let yQuicDirect ← finalSavedVec2Pair "QUICSORT direct final" solQuicDirect
  let yQuicTaylor ← finalSavedVec2Pair "QUICSORT taylor final" solQuicTaylor

  let deltaAlign := vector2L2 yAlignDirect.1 yAlignTaylor.1 + vector2L2 yAlignDirect.2 yAlignTaylor.2
  let deltaShould := vector2L2 yShouldDirect.1 yShouldTaylor.1 + vector2L2 yShouldDirect.2 yShouldTaylor.2
  let deltaQuic := vector2L2 yQuicDirect.1 yQuicTaylor.1 + vector2L2 yQuicDirect.2 yQuicTaylor.2
  LeanTest.assertTrue (deltaAlign > 1.0e-8 && deltaShould > 1.0e-8 && deltaQuic > 1.0e-8)
    s!"Expected direct/taylor branch endpoints to differ for vector-state solves: ALIGN={deltaAlign}, ShOULD={deltaShould}, QUICSORT={deltaQuic}"

@[test] def testALIGNUnderdampedSelfConvergence : IO Unit := do
  assertSelfConvergence "ALIGN underdamped strong-order trend" ALIGN.solver 1.1 0.2

@[test] def testShOULDUnderdampedSelfConvergence : IO Unit := do
  assertSelfConvergence "ShOULD underdamped strong-order trend" ShOULD.solver 1.2 0.12

@[test] def testQUICSORTUnderdampedSelfConvergence : IO Unit := do
  assertSelfConvergence "QUICSORT underdamped strong-order trend" QUICSORT.solver 1.2 0.12

def run : IO Unit := do
  testALIGNUnderdampedTaylorThresholdParity
  testShOULDUnderdampedTaylorThresholdParity
  testQUICSORTUnderdampedTaylorThresholdParity
  testUnderdampedVectorStateShapeParity
  testALIGNUnderdampedSelfConvergence
  testShOULDUnderdampedSelfConvergence
  testQUICSORTUnderdampedSelfConvergence

end Tests.DiffEqUnderdampedOrderParity

unsafe def main : IO Unit := do
  Tests.DiffEqUnderdampedOrderParity.run
  IO.println "TestDiffEqUnderdampedOrderParity: ok"
