import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqSDEOrderParity

open LeanTest
open torch
open torch.DiffEq

private def finalSaved {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure 0.0
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure 0.0

private def deterministicSeeds (count : Nat) : Array UInt64 :=
  Id.run do
    let mut seeds := #[]
    for i in [:count] do
      seeds := seeds.push (UInt64.ofNat (810001 + i * 7919))
    pure seeds

private def solveAdditiveNoiseEndpoint
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (ODETerm Float Unit)
        (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      Float
      (Float × Float)
      (Time × SpaceTimeLevyArea Time Float)
      Unit)
    (dt : Float)
    (seed : UInt64) : IO Float := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -0.4 * y }
  let bm : VirtualBrownianTree Float := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-6
    maxDepth := 24
    seed := seed
    shape := 0.0
  }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath
      (fun _t _y _ => 0.2)
      bmPath
      (fun vf control => vf * control.W)
  let terms :
      MultiTerm (ODETerm Float Unit)
        (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) := {
    term1 := drift
    term2 := diffusion
  }
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some dt) (1.0 : Float) ()
      (saveat := { t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    s!"{label}: solve should succeed"
  finalSaved label sol

private def meanErrorVsRef
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (ODETerm Float Unit)
        (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      Float
      (Float × Float)
      (Time × SpaceTimeLevyArea Time Float)
      Unit)
    (dt dtRef : Float)
    (seeds : Array UInt64) : IO Float := do
  let mut sum := 0.0
  for seed in seeds do
    let yDt ← solveAdditiveNoiseEndpoint s!"{label} dt={dt}" solver dt seed
    let yRef ← solveAdditiveNoiseEndpoint s!"{label} ref={dtRef}" solver dtRef seed
    sum := sum + Float.abs (yDt - yRef)
  let n := Float.ofNat seeds.size
  pure (sum / n)

private def assertStrongSelfConvergence
    (label : String)
    (solver : AbstractSolver
      (MultiTerm (ODETerm Float Unit)
        (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      Float
      (Float × Float)
      (Time × SpaceTimeLevyArea Time Float)
      Unit)
    (minRatio : Float := 1.1) : IO Unit := do
  let seeds := deterministicSeeds 24
  let dtRef := 0.0078125
  let errCoarse ← meanErrorVsRef label solver 0.25 dtRef seeds
  let errMedium ← meanErrorVsRef label solver 0.125 dtRef seeds
  let errFine ← meanErrorVsRef label solver 0.0625 dtRef seeds

  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"{label}: expected coarse/medium/fine mean strong errors to decrease, got {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > minRatio && ratio2 > minRatio)
    s!"{label}: strong-order trend too weak, got ratios {ratio1}, {ratio2}"

/--
Strong-order self-convergence regressions inspired by:
- `../diffrax/test/test_sde1.py::test_sde_strong_order_new`
-/
@[test] def testSPaRKStrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "SPaRK additive-noise strong trend" SPaRK.solver 1.1

@[test] def testGeneralShARKStrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "GeneralShARK additive-noise strong trend" GeneralShARK.solver 1.1

@[test] def testSlowRKStrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "SlowRK additive-noise strong trend" SlowRK.solver 1.1

@[test] def testShARKStrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "ShARK additive-noise strong trend" ShARK.solver 1.1

@[test] def testSRA1StrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "SRA1 additive-noise strong trend" SRA1.solver 1.1

@[test] def testSEAStrongSelfConvergenceAdditive : IO Unit := do
  assertStrongSelfConvergence "SEA additive-noise strong trend" SEA.solver 1.1

def run : IO Unit := do
  testSPaRKStrongSelfConvergenceAdditive
  testGeneralShARKStrongSelfConvergenceAdditive
  testSlowRKStrongSelfConvergenceAdditive
  testShARKStrongSelfConvergenceAdditive
  testSRA1StrongSelfConvergenceAdditive
  testSEAStrongSelfConvergenceAdditive

end Tests.DiffEqSDEOrderParity
