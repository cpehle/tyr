import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqWeakStatsParity2

open LeanTest
open torch
open torch.DiffEq

private def finalSaved {Y S C : Type} [Inhabited Y]
    (label : String) (sol : Solution Y S C) : IO Y := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure default
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure default

private def deterministicWeakSeeds (count : Nat) : Array UInt64 :=
  Id.run do
    let mut seeds := #[]
    for i in [:count] do
      seeds := seeds.push (UInt64.ofNat (1100003 + i * 7919))
    pure seeds

private def gbmExactMeanVariance (mu sigma tFinal x0 : Float) : Float × Float :=
  let mean := x0 * Float.exp (mu * tFinal)
  let variance :=
    (x0 * x0) * Float.exp ((2.0 * mu) * tFinal) * (Float.exp ((sigma * sigma) * tFinal) - 1.0)
  (mean, variance)

private def eulerMaruyamaGBMWeakMeanVariance
    (mu sigma x0 tFinal dt : Float) (seeds : Array UInt64) : IO (Float × Float) := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => mu * y }
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let mut sum := 0.0
  let mut sumSq := 0.0
  for seed in seeds do
    let bm : VirtualBrownianTree Float := {
      t0 := 0.0
      t1 := tFinal
      tol := 1.0e-6
      maxDepth := 22
      seed := seed
      shape := 0.0
    }
    let bmPath := (VirtualBrownianTree.toAbstract bm).toPath
    let diffusion : ControlTerm Float Float Float Unit :=
      ControlTerm.ofPath (fun _t y _ => sigma * y) bmPath (fun vf control => vf * control)
    let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) :=
      { term1 := drift, term2 := diffusion }
    let sol :=
      diffeqsolve
        (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
        (Y := Float)
        (VF := (Float × Float))
        (Control := (Time × Float))
        (Args := Unit)
        (Controller := ConstantStepSize)
        terms solver 0.0 tFinal (some dt) x0 () (saveat := { t1 := true })
    let y1 ← finalSaved "weak GBM Euler-Maruyama" sol
    sum := sum + y1
    sumSq := sumSq + (y1 * y1)
  let n := Float.ofNat seeds.size
  let mean := sum / n
  let varRaw := (sumSq / n) - (mean * mean)
  let variance := if varRaw < 0.0 then 0.0 else varRaw
  pure (mean, variance)

private def exactTimeInhomogeneousOUSecondMoment
    (a x0 tFinal : Float) : Float :=
  let mean := x0 * Float.exp (-a * tFinal)
  let k := 2.0 * a
  let invK := 1.0 / k
  let invK2 := invK * invK
  let invK3 := invK2 * invK
  let variance :=
    (tFinal * tFinal) * invK
      - (2.0 * tFinal) * invK2
      + 2.0 * invK3
      - (2.0 * Float.exp (-k * tFinal)) * invK3
  variance + (mean * mean)

private def eulerMaruyamaTimeInhomogeneousOUWeakSecondMoment
    (a x0 tFinal dt : Float) (seeds : Array UInt64) : IO Float := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -a * y }
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let mut sumM2 := 0.0
  for seed in seeds do
    let bm : VirtualBrownianTree Float := {
      t0 := 0.0
      t1 := tFinal
      tol := 1.0e-6
      maxDepth := 22
      seed := seed
      shape := 0.0
    }
    let bmPath := (VirtualBrownianTree.toAbstract bm).toPath
    let diffusion : ControlTerm Float Float Float Unit :=
      ControlTerm.ofPath (fun t _y _ => t) bmPath (fun vf control => vf * control)
    let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) :=
      { term1 := drift, term2 := diffusion }
    let sol :=
      diffeqsolve
        (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
        (Y := Float)
        (VF := (Float × Float))
        (Control := (Time × Float))
        (Args := Unit)
        (Controller := ConstantStepSize)
        terms solver 0.0 tFinal (some dt) x0 () (saveat := { t1 := true })
    let y1 ← finalSaved "weak time-inhomogeneous OU Euler-Maruyama" sol
    sumM2 := sumM2 + (y1 * y1)
  let n := Float.ofNat seeds.size
  pure (sumM2 / n)

/--
Weak-stat parity regression for geometric Brownian motion under Euler-Maruyama.
Guidance sources:
- `../diffrax/test/test_sde1.py` (coarse/medium/fine discretization trend checks)
- `../diffrax/test/test_sde2.py` (multiplicative-noise `ControlTerm` wiring shape)
-/
@[test] def testSDEWeakGBMMeanVarianceConvergenceEulerMaruyama : IO Unit := do
  let mu := 0.2
  let sigma := 0.3
  let tFinal := 1.0
  let x0 := 1.0
  let seeds := deterministicWeakSeeds 640

  let dtCoarse := 0.25
  let dtMedium := 0.125
  let dtFine := 0.0625
  let dtRef := 0.0078125

  let (meanRef, varRef) ←
    eulerMaruyamaGBMWeakMeanVariance mu sigma x0 tFinal dtRef seeds

  let (meanCoarse, varCoarse) ←
    eulerMaruyamaGBMWeakMeanVariance mu sigma x0 tFinal dtCoarse seeds
  let (meanMedium, varMedium) ←
    eulerMaruyamaGBMWeakMeanVariance mu sigma x0 tFinal dtMedium seeds
  let (meanFine, varFine) ←
    eulerMaruyamaGBMWeakMeanVariance mu sigma x0 tFinal dtFine seeds

  let m2Ref := varRef + (meanRef * meanRef)
  let m2Coarse := varCoarse + (meanCoarse * meanCoarse)
  let m2Medium := varMedium + (meanMedium * meanMedium)
  let m2Fine := varFine + (meanFine * meanFine)

  let meanErrCoarse := Float.abs (meanCoarse - meanRef)
  let meanErrMedium := Float.abs (meanMedium - meanRef)
  let meanErrFine := Float.abs (meanFine - meanRef)
  let m2ErrCoarse := Float.abs (m2Coarse - m2Ref)
  let m2ErrMedium := Float.abs (m2Medium - m2Ref)
  let m2ErrFine := Float.abs (m2Fine - m2Ref)
  let varErrCoarse := Float.abs (varCoarse - varRef)
  let varErrMedium := Float.abs (varMedium - varRef)
  let varErrFine := Float.abs (varFine - varRef)

  LeanTest.assertTrue (meanErrCoarse > meanErrMedium && meanErrMedium > meanErrFine)
    s!"Weak GBM mean-to-reference errors should decrease with dt: {meanErrCoarse}, {meanErrMedium}, {meanErrFine}"
  LeanTest.assertTrue (m2ErrCoarse > m2ErrMedium && m2ErrMedium > m2ErrFine)
    s!"Weak GBM second-moment-to-reference errors should decrease with dt: {m2ErrCoarse}, {m2ErrMedium}, {m2ErrFine}"
  LeanTest.assertTrue (varErrCoarse > varErrMedium && varErrCoarse > varErrFine)
    s!"Weak GBM variance-to-reference should improve from coarse to finer dt: {varErrCoarse}, {varErrMedium}, {varErrFine}"

  let meanRatio1 := meanErrCoarse / meanErrMedium
  let meanRatio2 := meanErrMedium / meanErrFine
  let m2Ratio1 := m2ErrCoarse / m2ErrMedium
  let m2Ratio2 := m2ErrMedium / m2ErrFine
  let varImprove1 := varErrCoarse / varErrMedium
  let varImprove2 := varErrCoarse / varErrFine

  LeanTest.assertTrue (meanRatio1 > 1.5 && meanRatio2 > 1.5)
    s!"Weak GBM mean trend too weak: ratios {meanRatio1}, {meanRatio2}"
  LeanTest.assertTrue (m2Ratio1 > 1.5 && m2Ratio2 > 1.3)
    s!"Weak GBM second-moment trend too weak: ratios {m2Ratio1}, {m2Ratio2}"
  LeanTest.assertTrue (varImprove1 > 2.0 && varImprove2 > 2.0)
    s!"Weak GBM variance coarse-to-fine improvement too weak: {varImprove1}, {varImprove2}"

  let (exactMean, exactVariance) := gbmExactMeanVariance mu sigma tFinal x0
  LeanTest.assertTrue (Float.abs (meanRef - exactMean) < 0.2)
    s!"Weak GBM reference mean unexpectedly far from exact: {meanRef} vs {exactMean}"
  LeanTest.assertTrue (Float.abs (varRef - exactVariance) < 6.0e-2)
    s!"Weak GBM reference variance unexpectedly far from exact: {varRef} vs {exactVariance}"

/--
Weak-stat parity regression for a time-inhomogeneous additive-noise OU family:
`dY = -aY dt + t dW`.
Guidance sources:
- `../diffrax/test/test_sde2.py` (time-dependent diffusion coefficients)
- `../diffrax/test/test_sde1.py` (coarse/medium/fine-to-reference convergence structure)
-/
@[test] def testSDEWeakTimeInhomogeneousOUM2ConvergenceEulerMaruyama : IO Unit := do
  let a := 0.7
  let x0 := 1.0
  let tFinal := 1.0
  let seeds := deterministicWeakSeeds 640

  let dtCoarse := 0.5
  let dtMedium := 0.25
  let dtFine := 0.125

  let m2Coarse ←
    eulerMaruyamaTimeInhomogeneousOUWeakSecondMoment a x0 tFinal dtCoarse seeds
  let m2Medium ←
    eulerMaruyamaTimeInhomogeneousOUWeakSecondMoment a x0 tFinal dtMedium seeds
  let m2Fine ←
    eulerMaruyamaTimeInhomogeneousOUWeakSecondMoment a x0 tFinal dtFine seeds

  let exactM2 := exactTimeInhomogeneousOUSecondMoment a x0 tFinal
  let errCoarse := Float.abs (m2Coarse - exactM2)
  let errMedium := Float.abs (m2Medium - exactM2)
  let errFine := Float.abs (m2Fine - exactM2)

  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"Weak time-inhomogeneous OU M2-to-exact errors should decrease with dt: {errCoarse}, {errMedium}, {errFine}"

  let ratio1 := errCoarse / errMedium
  let ratio2 := errMedium / errFine
  LeanTest.assertTrue (ratio1 > 1.4 && ratio2 > 1.4)
    s!"Weak time-inhomogeneous OU M2 trend too weak: ratios {ratio1}, {ratio2}"

  LeanTest.assertTrue (errFine < 1.0e-2)
    s!"Weak time-inhomogeneous OU fine-step M2 unexpectedly far from exact: {m2Fine} vs {exactM2}"

def run : IO Unit := do
  testSDEWeakGBMMeanVarianceConvergenceEulerMaruyama
  testSDEWeakTimeInhomogeneousOUM2ConvergenceEulerMaruyama

end Tests.DiffEqWeakStatsParity2
