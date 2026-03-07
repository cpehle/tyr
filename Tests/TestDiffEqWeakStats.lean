import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqWeakStats

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
      seeds := seeds.push (UInt64.ofNat (930001 + i * 7919))
    pure seeds

private def eulerMaruyamaOUWeakHighMoments
    (dt : Float) (seeds : Array UInt64) : IO (Float × Float) := do
  let theta := 1.0
  let sigma := 0.1
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -theta * y }
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let mut sumM3 := 0.0
  let mut sumM4 := 0.0
  for seed in seeds do
    let bm : VirtualBrownianTree Float := {
      t0 := 0.0
      t1 := 1.0
      tol := 1.0e-6
      maxDepth := 22
      seed := seed
      shape := 0.0
    }
    let bmPath := (VirtualBrownianTree.toAbstract bm).toPath
    let diffusion : ControlTerm Float Float Float Unit :=
      ControlTerm.ofPath (fun _t _y _ => sigma) bmPath (fun vf control => vf * control)
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
        terms solver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
    let y1 ← finalSaved "weak OU higher moments Euler-Maruyama" sol
    let y2 := y1 * y1
    sumM3 := sumM3 + (y2 * y1)
    sumM4 := sumM4 + (y2 * y2)
  let n := Float.ofNat seeds.size
  pure (sumM3 / n, sumM4 / n)

private def exactOUThirdFourthMoments
    (theta sigma tFinal x0 : Float) : Float × Float :=
  let mean := x0 * Float.exp (-theta * tFinal)
  let variance := ((sigma * sigma) / (2.0 * theta)) * (1.0 - Float.exp (-2.0 * theta * tFinal))
  let m3 := (mean * mean * mean) + 3.0 * mean * variance
  let m4 := (mean * mean * mean * mean) + 6.0 * (mean * mean) * variance + 3.0 * (variance * variance)
  (m3, m4)

/--
Weak-order regression for a second observable family (third and fourth moments).
Guidance source: `../diffrax/docs/devdocs/SDE_solver_table.md` lists weak order 1.0
for Euler on Itô SDEs, and Euler-Maruyama in Tyr is the Itô Euler scheme.
-/
@[test] def testSDEWeakHigherMomentConvergenceEulerMaruyama : IO Unit := do
  let theta := 1.0
  let sigma := 0.1
  let tFinal := 1.0
  let x0 := 1.0
  let seeds := deterministicWeakSeeds 192

  let (m3Coarse, m4Coarse) ← eulerMaruyamaOUWeakHighMoments 0.2 seeds
  let (m3Medium, m4Medium) ← eulerMaruyamaOUWeakHighMoments 0.1 seeds
  let (m3Fine, m4Fine) ← eulerMaruyamaOUWeakHighMoments 0.05 seeds

  let (exactM3, exactM4) := exactOUThirdFourthMoments theta sigma tFinal x0
  let m3ErrCoarse := Float.abs (m3Coarse - exactM3)
  let m3ErrMedium := Float.abs (m3Medium - exactM3)
  let m3ErrFine := Float.abs (m3Fine - exactM3)
  let m4ErrCoarse := Float.abs (m4Coarse - exactM4)
  let m4ErrMedium := Float.abs (m4Medium - exactM4)
  let m4ErrFine := Float.abs (m4Fine - exactM4)

  LeanTest.assertTrue (m3ErrCoarse > m3ErrMedium && m3ErrMedium > m3ErrFine)
    s!"Weak third-moment errors should decrease with dt: {m3ErrCoarse}, {m3ErrMedium}, {m3ErrFine}"
  LeanTest.assertTrue (m4ErrCoarse > m4ErrMedium && m4ErrMedium > m4ErrFine)
    s!"Weak fourth-moment errors should decrease with dt: {m4ErrCoarse}, {m4ErrMedium}, {m4ErrFine}"

  let m3Ratio1 := if m3ErrMedium <= 1.0e-16 then 0.0 else m3ErrCoarse / m3ErrMedium
  let m3Ratio2 := if m3ErrFine <= 1.0e-16 then 0.0 else m3ErrMedium / m3ErrFine
  let m4Ratio1 := if m4ErrMedium <= 1.0e-16 then 0.0 else m4ErrCoarse / m4ErrMedium
  let m4Ratio2 := if m4ErrFine <= 1.0e-16 then 0.0 else m4ErrMedium / m4ErrFine

  LeanTest.assertTrue (m3Ratio1 > 1.3 && m3Ratio2 > 1.3)
    s!"Weak third-moment trend too weak: ratios {m3Ratio1}, {m3Ratio2}"
  LeanTest.assertTrue (m4Ratio1 > 1.3 && m4Ratio2 > 1.3)
    s!"Weak fourth-moment trend too weak: ratios {m4Ratio1}, {m4Ratio2}"

  LeanTest.assertTrue (m3ErrFine < 4.0e-3)
    s!"Weak third-moment fine-grid error too large: {m3ErrFine}"
  LeanTest.assertTrue (m4ErrFine < 2.0e-3)
    s!"Weak fourth-moment fine-grid error too large: {m4ErrFine}"

def run : IO Unit := do
  testSDEWeakHigherMomentConvergenceEulerMaruyama

end Tests.DiffEqWeakStats
