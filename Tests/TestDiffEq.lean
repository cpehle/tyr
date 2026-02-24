import LeanTest
import Tyr.DiffEq

/-!
# `Tests.TestDiffEq`

Differential-equation solver tests covering ODE and SDE solve paths and diffusion-related losses.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tests.DiffEq

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def getStat (name : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun kv => kv.fst == name) with
  | some (_, v) => v
  | none => 0

/-- Basic ODE test inspired by Diffrax test_integrate.py::test_basic. -/
@[test] def testHeunODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Heun.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Control := Time) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 0.05)
          s!"Heun ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for ODE solve"

@[test] def testEulerODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Euler solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.pow (1.0 - 0.1) 10.0
        LeanTest.assertTrue (approx y1 expected 1e-5)
          s!"Euler ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Euler solve"

@[test] def testMidpointODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Midpoint.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Midpoint solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 2e-3)
          s!"Midpoint ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Midpoint solve"

@[test] def testRalstonODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Ralston.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Ralston solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 2e-3)
          s!"Ralston ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Ralston solve"

@[test] def testBosh3ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Bosh3.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Bosh3 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"Bosh3 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Bosh3 solve"

/-- SDE Euler-Heun step test based on Diffrax EulerHeun formula. -/
@[test] def testEulerHeunSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 12345 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let diffusion : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    EulerHeun.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let dW := (ScalarBrownianPath.increment bm 0.0 1.0).W
        let f0 := -2.0
        let g0 := 2.0 * dW
        let yPrime := 2.0 + g0
        let gPrime := yPrime * dW
        let expected := 2.0 + f0 + 0.5 * (g0 + gPrime)
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"EulerHeun SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testEulerMaruyamaSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 54321 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let diffusion : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let dW := (ScalarBrownianPath.increment bm 0.0 1.0).W
        let f0 := -2.0
        let g0 := 2.0 * dW
        let expected := 2.0 + f0 + g0
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"EulerMaruyama SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testBrownianBridgeAdditivity : IO Unit := do
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 24680, shape := (0.0 : Float) }
  let inc01 := (VirtualBrownianTree.increment bm 0.0 1.0).W
  let inc05 := (VirtualBrownianTree.increment bm 0.0 0.5).W
  let inc51 := (VirtualBrownianTree.increment bm 0.5 1.0).W
  let summed := inc05 + inc51
  LeanTest.assertTrue (approx inc01 summed 1e-6)
    s!"Brownian bridge increments inconsistent: {inc01} vs {summed}"

@[test] def testSpaceTimeLevyChen : IO Unit := do
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 97531, shape := (0.0 : Float) }
  let inc01 := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
  let inc05 := VirtualBrownianTree.incrementSpaceTime bm 0.0 0.5
  let inc51 := VirtualBrownianTree.incrementSpaceTime bm 0.5 1.0
  let barH01 := inc01.H * inc01.dt
  let barH05 := inc05.H * inc05.dt
  let barH51 := inc51.H * inc51.dt
  let rhs := barH05 + barH51 + 0.5 * (inc51.dt * inc05.W - inc05.dt * inc51.W)
  LeanTest.assertTrue (approx barH01 rhs 1e-6)
    s!"Space-time Levy Chen relation failed: {barH01} vs {rhs}"

@[test] def testSpaceTimeTimeSign : IO Unit := do
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 86420, shape := (0.0 : Float) }
  let inc01 := VirtualBrownianTree.incrementSpaceTimeTime bm 0.0 1.0
  let inc10 := VirtualBrownianTree.incrementSpaceTimeTime bm 1.0 0.0
  LeanTest.assertTrue (approx inc01.dt (-inc10.dt) 1e-12)
    s!"Space-time-time dt sign failed: {inc01.dt} vs {-inc10.dt}"
  LeanTest.assertTrue (approx inc01.W (-inc10.W) 1e-6)
    s!"Space-time-time W sign failed: {inc01.W} vs {-inc10.W}"
  LeanTest.assertTrue (approx inc01.H (-inc10.H) 1e-6)
    s!"Space-time-time H sign failed: {inc01.H} vs {-inc10.H}"
  LeanTest.assertTrue (approx inc01.K (-inc10.K) 1e-6)
    s!"Space-time-time K sign failed: {inc01.K} vs {-inc10.K}"

@[test] def testMilsteinSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 424242 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let diffusion : DiffusionTerm Float Float Float Unit :=
    DiffusionTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control)
      (fun _t y _ => y)
  let terms : MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := DiffusionTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let dW := (ScalarBrownianPath.increment bm 0.0 1.0).W
        let f0 := -2.0
        let g0 := 2.0 * dW
        let gg0 := 2.0
        let corr := 0.5 * gg0 * (dW * dW - 1.0)
        let expected := 2.0 + f0 + g0 + corr
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"Milstein SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testStratonovichMilsteinSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 20240202 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let diffusion : DiffusionTerm Float Float Float Unit :=
    DiffusionTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control)
      (fun _t y _ => y)
  let terms : MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    StratonovichMilstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := DiffusionTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let dW := (ScalarBrownianPath.increment bm 0.0 1.0).W
        let f0 := -2.0
        let g0 := 2.0 * dW
        let gg0 := 2.0
        let corr := 0.5 * gg0 * (dW * dW)
        let expected := 2.0 + f0 + g0 + corr
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"Stratonovich Milstein SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testSRA1SDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 24601, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t _y _ => 1.0) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    SRA1.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let inc := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
        let w := inc.W
        let h := inc.H
        let h_kf0 := -2.0
        let z1 := 2.0 + 0.75 * h_kf0 + 0.75 * w + 1.5 * h
        let h_kf1 := -z1
        let drift_result := (1.0 / 3.0) * h_kf0 + (2.0 / 3.0) * h_kf1
        let expected := 2.0 + drift_result + w
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"SRA1 SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testShARKSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 13579, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t _y _ => 1.0) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    ShARK.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let inc := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
        let w := inc.W
        let h := inc.H
        let z0 := 2.0 + h
        let kf0 := -z0
        let z1 := 2.0 + (5.0 / 6.0) * kf0 + (5.0 / 6.0) * w + h
        let kf1 := -z1
        let drift_result := 0.4 * kf0 + 0.6 * kf1
        let expected := 2.0 + drift_result + w
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"ShARK SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testSEASDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 42430, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t _y _ => 1.0) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    SEA.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let inc := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
        let w := inc.W
        let h := inc.H
        let z0 := 2.0 + 0.5 * w + h
        let expected := 2.0 + (-z0) + w
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"SEA SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testGeneralShARKSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 70707, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    GeneralShARK.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SDE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let inc := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
        let w := inc.W
        let h := inc.H
        let y0 := 2.0
        let z0 := y0
        let hkg0 := z0 * h
        let z1 := y0 + hkg0
        let kf1 := -z1
        let wkg1 := z1 * w
        let hkg1 := z1 * h
        let z2 :=
          y0 + (5.0 / 6.0) * kf1 + (5.0 / 6.0) * wkg1 + hkg0
        let kf2 := -z2
        let wkg2 := z2 * w
        let hkg2 := z2 * h
        let driftResult := 0.4 * kf1 + 0.6 * kf2
        let wResult := 0.4 * wkg1 + 0.6 * wkg2
        let hResult := 1.2 * hkg1 - 1.2 * hkg2
        let expected := y0 + driftResult + wResult + hResult
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"GeneralShARK SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SDE solve"

@[test] def testRK4ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    RK4.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-5)
          s!"RK4 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for ODE solve"

@[test] def testDopri5ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"Dopri5 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for ODE solve"

@[test] def testTsit5ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Tsit5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"Tsit5 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for ODE solve"

@[test] def testDopri8ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri8.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-8)
          s!"Dopri8 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for ODE solve"

@[test] def testStepTo : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ctrl : StepTo := { ts := #[0.0, 0.5, 1.0] }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := StepTo)
      term solver 0.0 1.0 none (1.0 : Float) () (saveat := { steps := true })
      (controller := ctrl)
  match sol.ts with
  | none => LeanTest.fail "Expected ts for StepTo solve"
  | some ts =>
      if ts.size == 3 then
        let ok :=
          approx ts[0]! 0.0 1e-12 && approx ts[1]! 0.5 1e-12 && approx ts[2]! 1.0 1e-12
        LeanTest.assertTrue ok s!"StepTo ts mismatch: {ts}"
      else
        LeanTest.fail s!"Unexpected ts size for StepTo: {ts.size}"

@[test] def testReverseTimeODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 1.0 0.0 (some 0.05) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"Reverse time ts size {ts.size}"
      LeanTest.assertTrue (approx ts[0]! 0.0 1e-12)
        s!"Reverse time ts[0] expected 0.0, got {ts[0]!}"
      let y1 := ys[0]!
      let expected := Float.exp 1.0
      LeanTest.assertTrue (approx y1 expected 1e-3)
        s!"Reverse time expected {expected}, got {y1}"
  | _, _ => LeanTest.fail "Expected ts/ys for reverse-time solve"
  LeanTest.assertTrue (sol.result == Result.successful) "Reverse time result should be successful"

@[test] def testSaveAtT0 : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -0.5 * y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let saveat : SaveAt := { t0 := true, t1 := false }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (2.1 : Float) () (saveat := saveat)
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"SaveAt(t0) ts size {ts.size}"
      LeanTest.assertTrue (ys.size == 1) s!"SaveAt(t0) ys size {ys.size}"
      LeanTest.assertTrue (approx ts[0]! 0.0 1e-12)
        s!"SaveAt(t0) ts[0] expected 0.0, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 2.1 1e-12)
        s!"SaveAt(t0) ys[0] expected 2.1, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected ts/ys for SaveAt(t0)"
  LeanTest.assertTrue sol.solverState.isNone "SaveAt(t0) solverState should be none"
  LeanTest.assertTrue sol.controllerState.isNone "SaveAt(t0) controllerState should be none"
  LeanTest.assertTrue (getStat "num_steps" sol.stats > 0) "SaveAt(t0) num_steps should be > 0"
  LeanTest.assertTrue (sol.result == Result.successful) "SaveAt(t0) result should be successful"
  LeanTest.assertTrue sol.interpolation.isNone "SaveAt(t0) should not save dense interpolation"

@[test] def testSaveAtT1 : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -0.5 * y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let saveat : SaveAt := { t1 := true, solverState := true, controllerState := true }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.01) (2.1 : Float) () (saveat := saveat)
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"SaveAt(t1) ts size {ts.size}"
      LeanTest.assertTrue (ys.size == 1) s!"SaveAt(t1) ys size {ys.size}"
      LeanTest.assertTrue (approx ts[0]! 1.0 1e-12)
        s!"SaveAt(t1) ts[0] expected 1.0, got {ts[0]!}"
      let expected := 2.1 * Float.exp (-0.5)
      LeanTest.assertTrue (approx ys[0]! expected 1e-4)
        s!"SaveAt(t1) ys[0] expected {expected}, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected ts/ys for SaveAt(t1)"
  LeanTest.assertTrue sol.solverState.isSome "SaveAt(t1) solverState should be some"
  LeanTest.assertTrue sol.controllerState.isSome "SaveAt(t1) controllerState should be some"
  LeanTest.assertTrue (getStat "num_steps" sol.stats > 0) "SaveAt(t1) num_steps should be > 0"
  LeanTest.assertTrue (sol.result == Result.successful) "SaveAt(t1) result should be successful"
  LeanTest.assertTrue sol.interpolation.isNone "SaveAt(t1) should not save dense interpolation"

@[test] def testSaveAtTsOutOfRange : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let badSaveAt : SaveAt := { ts := some #[-0.1, 0.5] }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := badSaveAt)
  LeanTest.assertTrue (sol.result == Result.internalError) "SaveAt(ts) out-of-range should error"

@[test] def testSaveAtTs : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -0.5 * y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ts : Array Time := #[0.25, 0.75]
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 1.0e-3) (2.0 : Float) () (saveat := { ts := some ts, t1 := false })
  match sol.ts, sol.ys with
  | some outTs, some ys =>
      LeanTest.assertTrue (outTs.size == ts.size) s!"SaveAt(ts) ts size {outTs.size}"
      LeanTest.assertTrue (ys.size == ts.size) s!"SaveAt(ts) ys size {ys.size}"
      let expected0 := 2.0 * Float.exp (-0.5 * ts[0]!)
      let expected1 := 2.0 * Float.exp (-0.5 * ts[1]!)
      LeanTest.assertTrue (approx outTs[0]! ts[0]! 1e-12)
        s!"SaveAt(ts) ts[0] expected {ts[0]!}, got {outTs[0]!}"
      LeanTest.assertTrue (approx outTs[1]! ts[1]! 1e-12)
        s!"SaveAt(ts) ts[1] expected {ts[1]!}, got {outTs[1]!}"
      LeanTest.assertTrue (approx ys[0]! expected0 1e-4)
        s!"SaveAt(ts) ys[0] expected {expected0}, got {ys[0]!}"
      LeanTest.assertTrue (approx ys[1]! expected1 1e-4)
        s!"SaveAt(ts) ys[1] expected {expected1}, got {ys[1]!}"
  | _, _ => LeanTest.fail "Expected ts/ys for SaveAt(ts)"

@[test] def testSaveAtTsWithEndpoints : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 2.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ts : Array Time := #[0.25, 0.75]
  let saveat : SaveAt := { t0 := true, t1 := true, ts := some ts }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.25) (1.0 : Float) () (saveat := saveat)
  match sol.ts, sol.ys with
  | some outTs, some ys =>
      LeanTest.assertTrue (outTs.size == 4) s!"SaveAt(ts,t0,t1) ts size {outTs.size}"
      LeanTest.assertTrue (ys.size == 4) s!"SaveAt(ts,t0,t1) ys size {ys.size}"
      LeanTest.assertTrue (approx outTs[0]! 0.0 1e-12)
        s!"SaveAt(ts,t0,t1) ts[0] expected 0.0, got {outTs[0]!}"
      LeanTest.assertTrue (approx outTs[1]! 0.25 1e-12)
        s!"SaveAt(ts,t0,t1) ts[1] expected 0.25, got {outTs[1]!}"
      LeanTest.assertTrue (approx outTs[2]! 0.75 1e-12)
        s!"SaveAt(ts,t0,t1) ts[2] expected 0.75, got {outTs[2]!}"
      LeanTest.assertTrue (approx outTs[3]! 1.0 1e-12)
        s!"SaveAt(ts,t0,t1) ts[3] expected 1.0, got {outTs[3]!}"
      let expected := #[1.0, 1.5, 2.5, 3.0]
      let ok :=
        approx ys[0]! expected[0]! 1e-12 &&
        approx ys[1]! expected[1]! 1e-12 &&
        approx ys[2]! expected[2]! 1e-12 &&
        approx ys[3]! expected[3]! 1e-12
      LeanTest.assertTrue ok s!"SaveAt(ts,t0,t1) ys mismatch: {ys}"
  | _, _ => LeanTest.fail "Expected ts/ys for SaveAt(ts,t0,t1)"
  LeanTest.assertTrue sol.interpolation.isSome "SaveAt(ts,t0,t1) should save interpolation"

@[test] def testSaveAtSteps : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.2) (1.0 : Float) () (saveat := { steps := true })
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == ys.size) s!"SaveAt(steps) size mismatch {ts.size}/{ys.size}"
      LeanTest.assertTrue (ts.size == getStat "num_steps" sol.stats + 1)
        s!"SaveAt(steps) expected ts size {getStat "num_steps" sol.stats + 1}, got {ts.size}"
      LeanTest.assertTrue (approx ts[0]! 0.0 1e-12)
        s!"SaveAt(steps) ts[0] expected 0.0, got {ts[0]!}"
      let last := ts[ts.size - 1]!
      LeanTest.assertTrue (approx last 1.0 1e-12)
        s!"SaveAt(steps) ts[end] expected 1.0, got {last}"
  | _, _ => LeanTest.fail "Expected ts/ys for SaveAt(steps)"

@[test] def testSaveAtDenseLinear : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 2.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let saveat : SaveAt := { dense := true, t1 := false }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.25) (1.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue sol.ts.isNone "SaveAt(dense) ts should be none"
  LeanTest.assertTrue sol.ys.isNone "SaveAt(dense) ys should be none"
  LeanTest.assertTrue sol.interpolation.isSome "SaveAt(dense) should save interpolation"
  let y025 := sol.evaluate 0.25
  let y075 := sol.evaluate 0.75
  LeanTest.assertTrue (approx y025 1.5 1e-12) s!"Dense evaluate(0.25) expected 1.5, got {y025}"
  LeanTest.assertTrue (approx y075 2.5 1e-12) s!"Dense evaluate(0.75) expected 2.5, got {y075}"
  let diff := sol.evaluate 0.25 (t1 := some 0.75)
  LeanTest.assertTrue (approx diff (y075 - y025) 1e-12)
    s!"Dense evaluate diff expected {y075 - y025}, got {diff}"
  let dy := sol.derivative 0.25
  LeanTest.assertTrue (approx dy 2.0 1e-12) s!"Dense derivative expected 2.0, got {dy}"

@[test] def testSaveAtDenseTrivial : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let saveat : SaveAt := { dense := true, t1 := false }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := PIDController)
      term solver 2.0 2.0 none (1.5 : Float) () (saveat := saveat)
  LeanTest.assertTrue sol.interpolation.isSome "SaveAt(dense) should save interpolation"
  let y := sol.evaluate 2.0
  LeanTest.assertTrue (approx y 1.5 1e-12) s!"Dense trivial evaluate expected 1.5, got {y}"
  let dy := sol.derivative 2.0
  LeanTest.assertTrue (approx dy 0.0 1e-12) s!"Dense trivial derivative expected 0.0, got {dy}"

@[test] def testDt0Zero : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.0) (1.0 : Float) () (saveat := { t1 := true })
  LeanTest.assertTrue (sol.result == Result.dtMinReached) "dt0=0 should return dtMinReached"

@[test] def testPIDAdaptiveODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let controller : PIDController := { rtol := 1.0e-4, atol := 1.0e-6 }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := PIDController)
      term solver 0.0 1.0 none (1.0 : Float) () (saveat := { t1 := true })
      (controller := controller)
  match sol.ys with
  | none => LeanTest.fail "Expected ys for adaptive ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"PID adaptive ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for adaptive ODE solve"

@[test] def testImplicitEulerODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    ImplicitEuler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for implicit Euler solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.pow (1.0 / 1.1) 10.0
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"Implicit Euler expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for implicit Euler solve"

@[test] def testKvaerno3ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Kvaerno3.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Kvaerno3 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-4)
          s!"Kvaerno3 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Kvaerno3 solve"

@[test] def testKvaerno4ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Kvaerno4.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Kvaerno4 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-4)
          s!"Kvaerno4 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Kvaerno4 solve"

@[test] def testKvaerno5ODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Kvaerno5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for Kvaerno5 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-4)
          s!"Kvaerno5 ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for Kvaerno5 solve"

@[test] def testReversibleHeunODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    ReversibleHeun.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Control := Time)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for reversible Heun solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"Reversible Heun ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for reversible Heun solve"

@[test] def testLeapfrogMidpointODE : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    LeapfrogMidpoint.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Control := Time)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for leapfrog midpoint solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 2e-2)
          s!"Leapfrog midpoint ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for leapfrog midpoint solve"

@[test] def testKencarp3IMEX : IO Unit := do
  let expTerm : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let impTerm : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) :=
    { term1 := expTerm, term2 := impTerm }
  let solver :=
    Kencarp3.solver
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for KenCarp3 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"KenCarp3 IMEX expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for KenCarp3 solve"

@[test] def testKencarp4IMEX : IO Unit := do
  let expTerm : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let impTerm : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) :=
    { term1 := expTerm, term2 := impTerm }
  let solver :=
    Kencarp4.solver
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for KenCarp4 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"KenCarp4 IMEX expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for KenCarp4 solve"

@[test] def testKencarp5IMEX : IO Unit := do
  let expTerm : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let impTerm : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) :=
    { term1 := expTerm, term2 := impTerm }
  let solver :=
    Kencarp5.solver
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for KenCarp5 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"KenCarp5 IMEX expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for KenCarp5 solve"

@[test] def testMultiTermODE : IO Unit := do
  let term1 : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let term2 : ODETerm Float Unit := { vectorField := fun _t y _ => 0.5 * y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) :=
    { term1 := term1, term2 := term2 }
  let solver :=
    Euler.solver
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.1) (2.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for MultiTerm ODE solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := 2.0 * Float.pow (1.0 - 0.05) 10.0
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"MultiTerm ODE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for MultiTerm ODE solve"

@[test] def testSIL3IMEX : IO Unit := do
  let expTerm : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let impTerm : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) :=
    { term1 := expTerm, term2 := impTerm }
  let solver :=
    SIL3.solver
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  match sol.ys with
  | none => LeanTest.fail "Expected ys for SIL3 solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let expected := Float.exp (-1.0)
        LeanTest.assertTrue (approx y1 expected 1e-3)
          s!"SIL3 IMEX expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SIL3 solve"

def run : IO Unit := do
  testHeunODE
  testEulerODE
  testMidpointODE
  testRalstonODE
  testBosh3ODE
  testEulerHeunSDE
  testEulerMaruyamaSDE
  testBrownianBridgeAdditivity
  testSpaceTimeLevyChen
  testSpaceTimeTimeSign
  testMilsteinSDE
  testStratonovichMilsteinSDE
  testSRA1SDE
  testShARKSDE
  testSEASDE
  testGeneralShARKSDE
  testRK4ODE
  testDopri5ODE
  testTsit5ODE
  testDopri8ODE
  testStepTo
  testReverseTimeODE
  testSaveAtT0
  testSaveAtT1
  testSaveAtTsOutOfRange
  testSaveAtTs
  testSaveAtTsWithEndpoints
  testSaveAtSteps
  testSaveAtDenseLinear
  testSaveAtDenseTrivial
  testDt0Zero
  testPIDAdaptiveODE
  testImplicitEulerODE
  testKvaerno3ODE
  testKvaerno4ODE
  testKvaerno5ODE
  testReversibleHeunODE
  testLeapfrogMidpointODE
  testKencarp3IMEX
  testKencarp4IMEX
  testKencarp5IMEX
  testMultiTermODE
  testSIL3IMEX

end Tests.DiffEq
