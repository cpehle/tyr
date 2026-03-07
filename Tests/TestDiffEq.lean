import LeanTest
import Tyr.DiffEq

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

private def evaluateDenseFloat {S C : Type}
    (label : String) (sol : Solution Float S C) (t : Time) : IO Float := do
  match sol.interpolation with
  | some interp => pure (interp.evaluate t none true)
  | none =>
      LeanTest.fail s!"{label}: expected dense interpolation"
      pure 0.0

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

@[test] def testSlowRKSDE : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 88123, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t y _ => y + 1.0) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    SlowRK.solver
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
  | none => LeanTest.fail "Expected ys for SlowRK solve"
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        let inc := VirtualBrownianTree.incrementSpaceTime bm 0.0 1.0
        let w := inc.W
        let h := inc.H
        let y0 := 2.0
        let h_kf0 := -y0
        let z1 := y0 + 0.5 * h_kf0
        let w_kg1 := (z1 + 1.0) * w
        let z2 := y0 + (0.5 * h_kf0 + 0.5 * w_kg1)
        let w_kg2 := (z2 + 1.0) * w
        let z3 := y0 + (0.5 * h_kf0 + 0.5 * w_kg2)
        let w_kg3 := (z3 + 1.0) * w
        let h_kg3 := (z3 + 1.0) * h
        let z4 := y0 + (0.5 * h_kf0 + w_kg3)
        let w_kg4 := (z4 + 1.0) * w
        let z5 := y0 + (0.75 * h_kf0 + (0.75 * w_kg3 + 1.5 * h_kg3))
        let h_kf5 := -z5
        let z6 := y0 + (h_kf0 + 0.5 * w_kg2)
        let h_kg6 := (z6 + 1.0) * h
        let driftResult := (1.0 / 3.0) * h_kf0 + (2.0 / 3.0) * h_kf5
        let wResult := (1.0 / 6.0) * w_kg1 + (1.0 / 3.0) * w_kg2 +
          (1.0 / 3.0) * w_kg3 + (1.0 / 6.0) * w_kg4
        let hResult := 2.0 * h_kg3 - 2.0 * h_kg6
        let expected := y0 + driftResult + (wResult + hResult)
        LeanTest.assertTrue (approx y1 expected 1e-6)
          s!"SlowRK SDE expected {expected}, got {y1}"
      else
        LeanTest.fail "Empty ys for SlowRK solve"

@[test] def testSlowRKDiffersFromGeneralShARK : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -0.7 * y + 0.3 }
  let bm : VirtualBrownianTree Float :=
    { t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 99123, shape := (0.0 : Float) }
  let bmPath := (VirtualBrownianTree.toAbstractSpaceTime bm).toPath
  let diffusion : ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit :=
    ControlTerm.ofPath (fun _t y _ => y * y + 0.5) bmPath (fun vf control => vf * control.W)
  let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit) :=
    { term1 := drift, term2 := diffusion }
  let slowSolver :=
    SlowRK.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let gsSolver :=
    GeneralShARK.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit)
      (Y := Float)
      (VFg := Float)
      (Control := SpaceTimeLevyArea Time Float)
      (Args := Unit)
  let slowSol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms slowSolver 0.0 1.0 (some 1.0) (1.25 : Float) () (saveat := { t1 := true })
  let gsSol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (SpaceTimeLevyArea Time Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × SpaceTimeLevyArea Time Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms gsSolver 0.0 1.0 (some 1.0) (1.25 : Float) () (saveat := { t1 := true })
  let ySlow ← finalSaved "SlowRK" slowSol
  let yGS ← finalSaved "GeneralShARK" gsSol
  let delta := Float.abs (ySlow - yGS)
  LeanTest.assertTrue (delta > 1e-6)
    s!"SlowRK should differ from GeneralShARK on this case: slow={ySlow}, gshark={yGS}, |Δ|={delta}"

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

@[test] def testStepToReverseTime : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ctrl : StepTo := { ts := #[1.0, 0.6, 0.0] }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := StepTo)
      term solver 1.0 0.0 none (1.0 : Float) () (saveat := { steps := true })
      (controller := ctrl)
  match sol.ts with
  | none => LeanTest.fail "Expected ts for reverse StepTo solve"
  | some ts =>
      if ts.size == 3 then
        let ok :=
          approx ts[0]! 1.0 1e-12 && approx ts[1]! 0.6 1e-12 && approx ts[2]! 0.0 1e-12
        LeanTest.assertTrue ok s!"Reverse StepTo ts mismatch: {ts}"
      else
        LeanTest.fail s!"Unexpected ts size for reverse StepTo: {ts.size}"

@[test] def testStepToRejectsEndpointMismatch : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ctrl : StepTo := { ts := #[0.1, 0.5, 1.0] }
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
  LeanTest.assertTrue (sol.result == Result.internalError)
    "StepTo endpoint mismatch should return internalError"

@[test] def testStepToRejectsNonMonotoneTs : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ctrl : StepTo := { ts := #[0.0, 0.5, 0.5, 1.0] }
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
  LeanTest.assertTrue (sol.result == Result.internalError)
    "StepTo non-monotone ts should return internalError"

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

@[test] def testSaveAtTsDirectionForward : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let badSaveAt : SaveAt := { ts := some #[0.8, 0.2], t1 := false }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := badSaveAt)
  LeanTest.assertTrue (sol.result == Result.internalError)
    "Forward solve with non-monotone SaveAt.ts should fail"

@[test] def testSaveAtTsDirectionReverse : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let badSaveAt : SaveAt := { ts := some #[0.2, 0.8], t1 := false }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 1.0 0.0 (some 0.1) (1.0 : Float) () (saveat := badSaveAt)
  LeanTest.assertTrue (sol.result == Result.internalError)
    "Reverse solve with non-monotone SaveAt.ts should fail"

@[test] def testNestedSubSaveAtPayloadFlags : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Dopri5.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let nested : SubSaveAt := {
    t1 := true
    solverState := true
    controllerState := true
    madeJump := true
  }
  let saveat : SaveAt := {
    t0 := false
    t1 := false
    solverState := false
    controllerState := false
    madeJump := false
    subs := #[nested]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (2.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful) "Nested SaveAt solve should succeed"
  LeanTest.assertTrue sol.solverState.isSome "Nested SubSaveAt should request solverState"
  LeanTest.assertTrue sol.controllerState.isSome "Nested SubSaveAt should request controllerState"
  LeanTest.assertTrue sol.madeJump.isSome "Nested SubSaveAt should request madeJump"
  match sol.ts with
  | some ts =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one saved t1 value, got {ts.size}"
      LeanTest.assertTrue (approx ts[0]! 1.0 1e-12) s!"Expected t1=1.0, got {ts[0]!}"
  | none =>
      LeanTest.fail "Nested SubSaveAt(t1=True) should save endpoint"

@[test] def testBooleanEventTerminate : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.35)
    terminate := true
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (event := some ev)
  LeanTest.assertTrue (sol.result == Result.eventOccurred) "Terminating event should stop solve"
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one output time, got {ts.size}"
      LeanTest.assertTrue (ys.size == 1) s!"Expected one output state, got {ys.size}"
      LeanTest.assertTrue (approx ts[0]! 0.4 1e-12) s!"Expected event time 0.4, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 0.4 1e-12) s!"Expected event state 0.4, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected endpoint save for terminating event"
  match sol.eventMask with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected one event mask entry, got {mask.size}"
      LeanTest.assertTrue mask[0]! "Event mask should record the triggered event"
  | none => LeanTest.fail "Event mask should be present when events are configured"
  match sol.eventMaskLast with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected one last-event mask entry, got {mask.size}"
      LeanTest.assertTrue mask[0]! "Last-event mask should record terminating event"
  | none => LeanTest.fail "Last-event mask should be present for triggered event"

@[test] def testBooleanEventNonTerminating : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.35)
    terminate := false
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (event := some ev)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Non-terminating event should allow solve completion"
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one output time, got {ts.size}"
      LeanTest.assertTrue (approx ts[0]! 1.0 1e-12) s!"Expected final time 1.0, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 1.0 1e-12) s!"Expected final state 1.0, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected endpoint save for non-terminating event solve"
  match sol.eventMask with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected one event mask entry, got {mask.size}"
      LeanTest.assertTrue mask[0]! "Event mask should record non-terminating event hit"
  | none => LeanTest.fail "Event mask should be present when events are configured"
  match sol.eventMaskLast with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected one last-event mask entry, got {mask.size}"
      LeanTest.assertTrue mask[0]! "Last-event mask should track latest event time hits"
  | none => LeanTest.fail "Last-event mask should be present after an event hit"

@[test] def testRealEventDirection : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let up : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.5)
    direction := some true
    terminate := true
  }
  let down : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.5)
    direction := some false
    terminate := true
  }
  let solUp :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.2) (0.0 : Float) () (saveat := { t1 := true })
      (event := some up)
  LeanTest.assertTrue (solUp.result == Result.eventOccurred)
    "Upward real event should trigger"
  match solUp.ts with
  | some ts =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one output time, got {ts.size}"
      LeanTest.assertTrue (approx ts[0]! 0.5 1.0e-4) s!"Expected localized root near 0.5, got {ts[0]!}"
  | none => LeanTest.fail "Expected saved root time for upward event"

  let solDown :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.2) (0.0 : Float) () (saveat := { t1 := true })
      (event := some down)
  LeanTest.assertTrue (solDown.result == Result.successful)
    "Downward-only event should not trigger on increasing trajectory"
  match solDown.eventMask with
  | some mask =>
      LeanTest.assertTrue (mask.size == 1) s!"Expected one event mask entry, got {mask.size}"
      LeanTest.assertTrue (!mask[0]!) "Downward event mask should remain false"
  | none => LeanTest.fail "Event mask should be present when events are configured"
  LeanTest.assertTrue solDown.eventMaskLast.isNone
    "Last-event mask should remain none when no event fired"

@[test] def testBooleanEventDirectionDown : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => -1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .boolean (fun _t y _ => y > 0.5)
    direction := some false
    terminate := true
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.2) (1.0 : Float) () (saveat := { t1 := true })
      (event := some ev)
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Downward boolean direction should trigger on true->false edge (not at t0)."
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one output time, got {ts.size}"
      LeanTest.assertTrue (ys.size == 1) s!"Expected one output state, got {ys.size}"
      LeanTest.assertTrue (approx ts[0]! 0.6 1e-12)
        s!"Expected downward boolean edge at t=0.6, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 0.4 1e-12)
        s!"Expected state y=0.4 at downward edge, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected endpoint save for downward boolean event"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 1 && lastMask.size == 1)
        s!"Expected single-event masks, got sizes {mask.size} and {lastMask.size}"
      LeanTest.assertTrue (mask[0]! && lastMask[0]!)
        "Downward boolean event should be recorded in eventMask and eventMaskLast"
  | _, _ => LeanTest.fail "Expected event masks for downward boolean direction event"

@[test] def testEventTiePrefersTerminating : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let evKeep : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.5)
    terminate := false
  }
  let evStop : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.5)
    terminate := true
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (events := #[evKeep, evStop])
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Terminating event should win on tie-time event hits"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 2) s!"Expected two event mask entries, got {mask.size}"
      LeanTest.assertTrue (lastMask.size == 2)
        s!"Expected two last-event mask entries, got {lastMask.size}"
      LeanTest.assertTrue (mask[0]! && mask[1]!) "Both events should be marked as hit"
      LeanTest.assertTrue (lastMask[0]! && lastMask[1]!)
        "Both events at chosen event time should be reflected in eventMaskLast"
  | _, _ =>
      LeanTest.fail "Expected event masks for multi-event tie case"

@[test] def testEventMaskExcludesLaterSameStepRoots : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let evEarly : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.2)
    terminate := true
  }
  let evLate : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.8)
    terminate := false
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 1.0) (0.0 : Float) () (saveat := { t1 := true })
      (events := #[evEarly, evLate])
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Earliest terminating root should stop solve."
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1) s!"Expected one output time, got {ts.size}"
      LeanTest.assertTrue (ys.size == 1) s!"Expected one output state, got {ys.size}"
      LeanTest.assertTrue (approx ts[0]! 0.2 1.0e-4)
        s!"Expected early localized root near 0.2, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 0.2 1.0e-4)
        s!"Expected state near 0.2 at early root, got {ys[0]!}"
  | _, _ => LeanTest.fail "Expected endpoint save for terminating early-root event"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 2 && lastMask.size == 2)
        s!"Expected two-event masks, got sizes {mask.size} and {lastMask.size}"
      LeanTest.assertTrue mask[0]! "Early root should be marked as hit"
      LeanTest.assertTrue (!mask[1]!)
        "Later same-step root should not be marked hit when solve terminates at earlier root"
      LeanTest.assertTrue lastMask[0]! "Last-event mask should include early chosen root"
      LeanTest.assertTrue (!lastMask[1]!)
        "Last-event mask should exclude later same-step root"
  | _, _ => LeanTest.fail "Expected event masks for same-step multi-root case"

@[test] def testBrownianPairAndFinStructuredIncrements : IO Unit := do
  let pairTree : VirtualBrownianTree (Float × Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 11122
    shape := ((0.0 : Float), (0.0 : Float))
  }
  let p01 := VirtualBrownianTree.increment pairTree 0.0 1.0
  let p05 := VirtualBrownianTree.increment pairTree 0.0 0.5
  let p51 := VirtualBrownianTree.increment pairTree 0.5 1.0
  LeanTest.assertTrue (approx p01.W.1 (p05.W.1 + p51.W.1) 1e-6)
    s!"Pair Brownian component 0 not additive: {p01.W.1} vs {p05.W.1 + p51.W.1}"
  LeanTest.assertTrue (approx p01.W.2 (p05.W.2 + p51.W.2) 1e-6)
    s!"Pair Brownian component 1 not additive: {p01.W.2} vs {p05.W.2 + p51.W.2}"

  let finTree : VirtualBrownianTree (Fin 3 → Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 33344
    shape := fun _ => (0.0 : Float)
  }
  let f01 := VirtualBrownianTree.increment finTree 0.0 1.0
  let f05 := VirtualBrownianTree.increment finTree 0.0 0.5
  let f51 := VirtualBrownianTree.increment finTree 0.5 1.0
  let i0 : Fin 3 := ⟨0, by decide⟩
  let i1 : Fin 3 := ⟨1, by decide⟩
  let i2 : Fin 3 := ⟨2, by decide⟩
  LeanTest.assertTrue (approx (f01.W i0) ((f05.W i0) + (f51.W i0)) 1e-6)
    "Fin Brownian component 0 not additive"
  LeanTest.assertTrue (approx (f01.W i1) ((f05.W i1) + (f51.W i1)) 1e-6)
    "Fin Brownian component 1 not additive"
  LeanTest.assertTrue (approx (f01.W i2) ((f05.W i2) + (f51.W i2)) 1e-6)
    "Fin Brownian component 2 not additive"

@[test] def testMilsteinAutodiffJvpWrapper : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 909090 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath

  let baselineDiffusion : DiffusionTerm Float Float Float Unit :=
    DiffusionTerm.ofPath
      (fun _t y _ => y)
      bmPath
      (fun vf control => vf * control)
      (fun _t y _ => y)

  let wrappedBase : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) bmPath (fun vf control => vf * control)
  let wrappedDiffusion :
      AutodiffJvpJacobianDiffusion (ControlTerm Float Float Float Unit) Float Float Unit :=
    withAutodiffJvpJacobian
      wrappedBase
      (fun _t _y _args control tangent => tangent * control)

  let termsBaseline : MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit) := {
    term1 := drift
    term2 := baselineDiffusion
  }
  let termsWrapped :
      MultiTerm (ODETerm Float Unit)
        (AutodiffJvpJacobianDiffusion (ControlTerm Float Float Float Unit) Float Float Unit) := {
    term1 := drift
    term2 := wrappedDiffusion
  }

  let baselineSolver :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := DiffusionTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let wrappedSolver :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := AutodiffJvpJacobianDiffusion (ControlTerm Float Float Float Unit) Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)

  let solBaseline :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (DiffusionTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsBaseline baselineSolver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })
  let solWrapped :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (AutodiffJvpJacobianDiffusion (ControlTerm Float Float Float Unit) Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsWrapped wrappedSolver 0.0 1.0 (some 1.0) (2.0 : Float) () (saveat := { t1 := true })

  match solBaseline.ys, solWrapped.ys with
  | some ysBase, some ysWrap =>
      let yBase := ysBase[ysBase.size - 1]!
      let yWrap := ysWrap[ysWrap.size - 1]!
      LeanTest.assertTrue (approx yWrap yBase 1e-6)
        s!"Milstein JVP wrapper mismatch: expected {yBase}, got {yWrap}"
  | _, _ => LeanTest.fail "Expected ys from both Milstein solves"

@[test] def testControlTermToODEConversion : IO Unit := do
  let path := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (1.0 : Float)
  let term : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) path (fun vf control => vf * control)
  match term.toODE? with
  | none => LeanTest.fail "Differentiable control path should convert to ODE term"
  | some ode =>
      let vf := ode.vectorField 0.2 3.0 ()
      LeanTest.assertTrue (approx vf 3.0 1e-12)
        s!"ControlTerm.toODE produced wrong vf value: expected 3.0, got {vf}"

  let nondiffPath : AbstractPath Float :=
    AbstractPath.ofFunctions 0.0 1.0
      (fun t0 t1? _left =>
        let t1 := t1?.getD t0
        t1 - t0)
      none
  let nondiffTerm : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) nondiffPath (fun vf control => vf * control)
  LeanTest.assertTrue nondiffTerm.toODE?.isNone
    "ControlTerm.toODE? should be none when control derivative is unavailable"

@[test] def testAbstractPathComposeLinear : IO Unit := do
  let left := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (2.0 : Float)
  let right := AbstractPath.linearInterpolation 1.0 2.0 (2.0 : Float) (3.0 : Float)
  let path := AbstractPath.compose left right
  let forward := path.increment 0.25 1.5
  LeanTest.assertTrue (approx forward 2.0 1e-12)
    s!"Expected composed forward increment 2.0, got {forward}"
  let backward := path.increment 1.5 0.25
  LeanTest.assertTrue (approx backward (-2.0) 1e-12)
    s!"Expected composed backward increment -2.0, got {backward}"
  match path.derivative 1.0 true, path.derivative 1.0 false with
  | some leftDeriv, some rightDeriv =>
      LeanTest.assertTrue (approx leftDeriv 2.0 1e-12)
        s!"Expected left derivative 2.0 at split, got {leftDeriv}"
      LeanTest.assertTrue (approx rightDeriv 1.0 1e-12)
        s!"Expected right derivative 1.0 at split, got {rightDeriv}"
  | _, _ =>
      LeanTest.fail "Expected composed path derivatives on both sides of split"

@[test] def testAbstractPathMapAndRestrict : IO Unit := do
  let base := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let restricted := base.restrict 0.5 1.5
  let inc := restricted.increment 0.5 1.5
  LeanTest.assertTrue (approx inc 2.0 1e-12)
    s!"Expected restricted increment 2.0, got {inc}"
  let mapped := restricted.mapControl (fun x => 2.0 * x) (some (fun x => 3.0 * x))
  let mappedInc := mapped.increment 0.5 1.5
  LeanTest.assertTrue (approx mappedInc 4.0 1e-12)
    s!"Expected mapped increment 4.0, got {mappedInc}"
  match mapped.derivative 1.0 with
  | some deriv =>
      LeanTest.assertTrue (approx deriv 6.0 1e-12)
        s!"Expected mapped derivative 6.0, got {deriv}"
  | none =>
      LeanTest.fail "Expected mapped derivative when mapDerivative is provided"

@[test] def testUnderdampedLangevinTerms : IO Unit := do
  let drift : UnderdampedLangevinDriftTerm Float (Float × Float) := {
    gradPotential := fun _t x args => args.2 * x
    gamma := fun _t _x _v args => args.1
    u := fun _t _x _v _args => 0.5
    argsValid := fun _t _x _v args => args.1 >= 0.0 && args.2 >= 0.0
  }
  let y : Float × Float := (1.5, -0.25)
  let driftInst :
      TermLike (UnderdampedLangevinDriftTerm Float (Float × Float))
        (Float × Float) (Float × Float) Time (Float × Float) :=
    inferInstance
  let driftVf := driftInst.vf drift 0.0 y (0.4, 2.0)
  LeanTest.assertTrue (approx driftVf.1 (-0.25) 1e-12)
    s!"Underdamped drift position component mismatch: {driftVf.1}"
  LeanTest.assertTrue (approx driftVf.2 (-1.4) 1e-12)
    s!"Underdamped drift velocity component mismatch: {driftVf.2}"

  let diffusionPath := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (1.0 : Float)
  let diffusion : UnderdampedLangevinDiffusionTerm Float Unit :=
    UnderdampedLangevinDiffusionTerm.ofPath diffusionPath
      (gamma := fun _t _x _v _ => 0.5)
      (u := fun _t _x _v _ => 2.0)
  let diffusionInst :
      TermLike (UnderdampedLangevinDiffusionTerm Float Unit)
        (Float × Float) Float Float Unit :=
    inferInstance
  let sigma := diffusionInst.vf diffusion 0.25 y ()
  LeanTest.assertTrue (approx sigma (Float.sqrt 2.0) 1e-12)
    s!"Underdamped diffusion sigma mismatch: {sigma}"
  let dW := diffusionInst.contr diffusion 0.0 1.0
  let diffusionStep := diffusionInst.vf_prod diffusion 0.25 y () dW
  LeanTest.assertTrue (approx diffusionStep.1 0.0 1e-12)
    s!"Underdamped diffusion should not perturb position: {diffusionStep.1}"
  LeanTest.assertTrue (approx diffusionStep.2 sigma 1e-12)
    s!"Underdamped diffusion velocity mismatch: {diffusionStep.2}"

  let badDrift : UnderdampedLangevinDriftTerm Float Unit := {
    gradPotential := fun _t x _ => x
    gamma := fun _t _x _v _ => -0.1
  }
  LeanTest.assertTrue (badDrift.validate? 0.0 (1.0, 0.0) ()).isSome
    "Underdamped drift validation should reject negative gamma"

@[test] def testSaveFnTransformsOutputs : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => y }
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
      term solver 0.0 1.0 (some 0.5) (2.0 : Float) () (saveat := { t0 := true, t1 := true })
      (saveFn := some (fun t y _ => y + 10.0 * t))
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 2) s!"Expected 2 save times, got {ts.size}"
      LeanTest.assertTrue (ys.size == 2) s!"Expected 2 save values, got {ys.size}"
      LeanTest.assertTrue (approx ys[0]! 2.0 1e-12)
        s!"Expected transformed y(t0)=2.0, got {ys[0]!}"
      -- Euler with dt=0.5: y(1) = 2*(1+0.5)^2 = 4.5 ; saveFn adds +10 at t=1.
      LeanTest.assertTrue (approx ys[1]! 14.5 1e-12)
        s!"Expected transformed y(t1)=14.5, got {ys[1]!}"
  | _, _ => LeanTest.fail "Expected ts/ys for saveFn transform test"

@[test] def testFailureResultMessage : IO Unit := do
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
  LeanTest.assertTrue (sol.result == Result.dtMinReached)
    "dt0=0 should produce dtMinReached"
  LeanTest.assertTrue sol.result.isFailure "dt0=0 result should be marked as failure"
  LeanTest.assertTrue (sol.result.message.contains "minimum step size")
    s!"Expected failure message to mention minimum step size, got: {sol.result.message}"

@[test] def testMaxStepsReached : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
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
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true }) (maxSteps := 3)
  LeanTest.assertTrue (sol.result == Result.maxStepsReached)
    "Expected maxStepsReached when step budget is exhausted"
  LeanTest.assertTrue (getStat "num_steps" sol.stats == 3)
    s!"Expected num_steps=3, got {getStat "num_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_accepted_steps" sol.stats == 3)
    s!"Expected num_accepted_steps=3, got {getStat "num_accepted_steps" sol.stats}"
  LeanTest.assertTrue (getStat "num_rejected_steps" sol.stats == 0)
    s!"Expected num_rejected_steps=0, got {getStat "num_rejected_steps" sol.stats}"

@[test] def testMaxStepsNoneParitySuccess : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
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
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) ()
      (saveat := { t1 := true })
      (maxSteps := 3)
      (maxStepsOpt := none)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Expected successful solve with maxStepsOpt=none"
  let steps := getStat "num_steps" sol.stats
  LeanTest.assertTrue (steps > 3)
    s!"Expected unbounded mode to exceed finite maxSteps; got num_steps={steps}"
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        let y1 := ys[ys.size - 1]!
        LeanTest.assertTrue (approx y1 1.0 1e-6)
          s!"Expected y(t1)=1.0 in unbounded mode test, got {y1}"
      else
        LeanTest.fail "Empty ys for maxStepsOpt=none success test"
  | none => LeanTest.fail "Expected ys for maxStepsOpt=none success test"

@[test] def testMaxStepsNoneRejectsSaveatSteps : IO Unit := do
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
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) ()
      (saveat := { steps := (1 : Nat) })
      (maxStepsOpt := none)
  LeanTest.assertTrue (sol.result == Result.internalError)
    "Expected internalError when maxStepsOpt=none with saveat.steps"
  LeanTest.assertTrue sol.ts.isNone
    "Expected no ts output for incompatible maxStepsOpt=none + saveat.steps config"
  LeanTest.assertTrue sol.ys.isNone
    "Expected no ys output for incompatible maxStepsOpt=none + saveat.steps config"

@[test] def testMaxStepsNoneRejectsDenseSave : IO Unit := do
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
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) ()
      (saveat := { dense := true })
      (maxStepsOpt := none)
  LeanTest.assertTrue (sol.result == Result.internalError)
    "Expected internalError when maxStepsOpt=none with saveat.dense"
  LeanTest.assertTrue sol.interpolation.isNone
    "Expected no dense interpolation for incompatible maxStepsOpt=none + saveat.dense config"

@[test] def testUnsafeBrownianPathStructuredIncrements : IO Unit := do
  let pairPath : UnsafeBrownianPath (Float × Float) := {
    t0 := 0.0
    t1 := 1.0
    seed := 525252
    shape := ((0.0 : Float), (0.0 : Float))
  }
  let pA := UnsafeBrownianPath.increment pairPath 0.2 0.8
  let pB := UnsafeBrownianPath.increment pairPath 0.2 0.8
  LeanTest.assertTrue (approx pA.W.1 pB.W.1 1e-12)
    "UnsafeBrownianPath pair component 0 should be deterministic per interval/seed"
  LeanTest.assertTrue (approx pA.W.2 pB.W.2 1e-12)
    "UnsafeBrownianPath pair component 1 should be deterministic per interval/seed"
  LeanTest.assertTrue (Float.isFinite pA.W.1 && Float.isFinite pA.W.2)
    "UnsafeBrownianPath pair increments should be finite"

  let finPath : UnsafeBrownianPath (Fin 2 → Float) := {
    t0 := 0.0
    t1 := 1.0
    seed := 535353
    shape := fun _ => (0.0 : Float)
  }
  let fA := UnsafeBrownianPath.increment finPath 0.3 0.9
  let fB := UnsafeBrownianPath.increment finPath 0.3 0.9
  let i0 : Fin 2 := ⟨0, by decide⟩
  let i1 : Fin 2 := ⟨1, by decide⟩
  LeanTest.assertTrue (approx (fA.W i0) (fB.W i0) 1e-12)
    "UnsafeBrownianPath Fin component 0 should be deterministic per interval/seed"
  LeanTest.assertTrue (approx (fA.W i1) (fB.W i1) 1e-12)
    "UnsafeBrownianPath Fin component 1 should be deterministic per interval/seed"
  LeanTest.assertTrue (Float.isFinite (fA.W i0) && Float.isFinite (fA.W i1))
    "UnsafeBrownianPath Fin increments should be finite"

@[test] def testEulerPairStatePyTree : IO Unit := do
  let term : ODETerm (Float × Float) Unit := {
    vectorField := fun _t y _ => (-y.1, 2.0 * y.2)
  }
  let solver :=
    Euler.solver
      (Term := ODETerm (Float × Float) Unit)
      (Y := (Float × Float))
      (VF := (Float × Float))
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm (Float × Float) Unit)
      (Y := (Float × Float))
      (VF := (Float × Float))
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) ((1.0, 1.0) : Float × Float) ()
      (saveat := { t1 := true })
  let y1 ← finalSaved "pair Euler PyTree" sol
  let expected0 := Float.pow 0.9 10.0
  let expected1 := Float.pow 1.2 10.0
  LeanTest.assertTrue (approx y1.1 expected0 1e-6)
    s!"Pair Euler component 0 expected {expected0}, got {y1.1}"
  LeanTest.assertTrue (approx y1.2 expected1 1e-5)
    s!"Pair Euler component 1 expected {expected1}, got {y1.2}"

@[test] def testEulerFinStatePyTree : IO Unit := do
  let i0 : Fin 3 := ⟨0, by decide⟩
  let i1 : Fin 3 := ⟨1, by decide⟩
  let i2 : Fin 3 := ⟨2, by decide⟩
  let coeff : Fin 3 → Float := fun i =>
    if i.1 == 0 then
      -1.0
    else if i.1 == 1 then
      0.5
    else
      2.0
  let term : ODETerm (Fin 3 → Float) Unit := {
    vectorField := fun _t y _ => fun i => (coeff i) * (y i)
  }
  let solver :=
    Euler.solver
      (Term := ODETerm (Fin 3 → Float) Unit)
      (Y := (Fin 3 → Float))
      (VF := (Fin 3 → Float))
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm (Fin 3 → Float) Unit)
      (Y := (Fin 3 → Float))
      (VF := (Fin 3 → Float))
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (fun _ => (1.0 : Float)) () (saveat := { t1 := true })
  let y1 ← finalSaved "Fin Euler PyTree" sol
  let expected0 := Float.pow 0.9 10.0
  let expected1 := Float.pow 1.05 10.0
  let expected2 := Float.pow 1.2 10.0
  LeanTest.assertTrue (approx (y1 i0) expected0 1e-6)
    s!"Fin Euler component 0 expected {expected0}, got {y1 i0}"
  LeanTest.assertTrue (approx (y1 i1) expected1 1e-6)
    s!"Fin Euler component 1 expected {expected1}, got {y1 i1}"
  LeanTest.assertTrue (approx (y1 i2) expected2 1e-5)
    s!"Fin Euler component 2 expected {expected2}, got {y1 i2}"

@[test] def testEulerMaruyamaPairStatePyTree : IO Unit := do
  let drift : ODETerm (Float × Float) Unit := {
    vectorField := fun _t y _ => (-y.1, 0.5 * y.2)
  }
  let bm : ScalarBrownianPath := { t0 := 0.0, t1 := 1.0, seed := 24680 }
  let bmPath := (ScalarBrownianPath.toAbstract bm).toPath
  let diffusion : ControlTerm (Float × Float) (Float × Float) Float Unit :=
    ControlTerm.ofPath
      (fun _t y _ => (y.1, -2.0 * y.2))
      bmPath
      (fun vf control => (vf.1 * control, vf.2 * control))
  let terms : MultiTerm (ODETerm (Float × Float) Unit)
      (ControlTerm (Float × Float) (Float × Float) Float Unit) :=
    { term1 := drift, term2 := diffusion }
  let solver :=
    EulerMaruyama.solver
      (Drift := ODETerm (Float × Float) Unit)
      (Diffusion := ControlTerm (Float × Float) (Float × Float) Float Unit)
      (Y := (Float × Float))
      (VFd := (Float × Float))
      (VFg := (Float × Float))
      (Control := Float)
      (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := MultiTerm (ODETerm (Float × Float) Unit)
        (ControlTerm (Float × Float) (Float × Float) Float Unit))
      (Y := (Float × Float))
      (VF := ((Float × Float) × (Float × Float)))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 1.0) ((1.0, 2.0) : Float × Float) ()
      (saveat := { t1 := true })
  let y1 ← finalSaved "pair EulerMaruyama PyTree" sol
  let dW := (ScalarBrownianPath.increment bm 0.0 1.0).W
  let expected0 := dW
  let expected1 := 3.0 - 4.0 * dW
  LeanTest.assertTrue (approx y1.1 expected0 1e-6)
    s!"Pair EulerMaruyama component 0 expected {expected0}, got {y1.1}"
  LeanTest.assertTrue (approx y1.2 expected1 1e-6)
    s!"Pair EulerMaruyama component 1 expected {expected1}, got {y1.2}"

@[test] def testProgressMeterDefaultParity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let solDefault :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
  let solNone :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) () (saveat := { t1 := true })
      (progress_meter := .none)
  let yDefault ← finalSaved "progress_meter default solve" solDefault
  let yNone ← finalSaved "progress_meter none solve" solNone
  LeanTest.assertTrue (solDefault.result == Result.successful && solNone.result == Result.successful)
    "Default and explicit progress_meter:=.none should both succeed"
  LeanTest.assertTrue (approx yDefault yNone 1.0e-12)
    s!"Default and explicit progress_meter:=.none should match: {yDefault} vs {yNone}"
  LeanTest.assertTrue (getStat "num_steps" solDefault.stats == getStat "num_steps" solNone.stats)
    "Default and explicit progress_meter:=.none should preserve step counts"
  LeanTest.assertTrue (getStat "progress_meter_start" solDefault.stats == 0)
    "Default progress meter should not record lifecycle start stats"
  LeanTest.assertTrue (getStat "progress_meter_updates" solDefault.stats == 0)
    "Default progress meter should not record lifecycle update stats"
  LeanTest.assertTrue (getStat "progress_meter_close" solDefault.stats == 0)
    "Default progress meter should not record lifecycle close stats"

@[test] def testProgressMeterTextAndTqdmCompatibility : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let solText :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (progress_meter := .text)
  let solTqdm :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := { t1 := true })
      (progress_meter := .tqdm)
  let yText ← finalSaved "progress_meter text solve" solText
  let yTqdm ← finalSaved "progress_meter tqdm solve" solTqdm
  LeanTest.assertTrue (solText.result == Result.successful && solTqdm.result == Result.successful)
    "progress_meter text/tqdm should both succeed"
  LeanTest.assertTrue (approx yText yTqdm 1.0e-12)
    s!"progress_meter text and tqdm should produce the same output: {yText} vs {yTqdm}"
  LeanTest.assertTrue (getStat "progress_meter_start" solText.stats == 1)
    "progress_meter text should record one lifecycle start"
  LeanTest.assertTrue (getStat "progress_meter_close" solText.stats == 1)
    "progress_meter text should record one lifecycle close"
  LeanTest.assertTrue (getStat "progress_meter_updates" solText.stats == getStat "num_steps" solText.stats)
    "progress_meter text updates should track attempted steps"
  LeanTest.assertTrue (getStat "progress_meter_tqdm_alias" solText.stats == 0)
    "progress_meter text should not set tqdm alias stat"
  LeanTest.assertTrue (getStat "progress_meter_start" solTqdm.stats == 1)
    "progress_meter tqdm should record one lifecycle start"
  LeanTest.assertTrue (getStat "progress_meter_close" solTqdm.stats == 1)
    "progress_meter tqdm should record one lifecycle close"
  LeanTest.assertTrue (getStat "progress_meter_updates" solTqdm.stats == getStat "num_steps" solTqdm.stats)
    "progress_meter tqdm updates should track attempted steps"
  LeanTest.assertTrue (getStat "progress_meter_tqdm_alias" solTqdm.stats == 1)
    "progress_meter tqdm should set alias compatibility stat"

@[test] def testEulerGlobalOrderTrend : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let solve := fun (dt : Float) =>
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
  let yCoarse ← finalSaved "Euler order coarse" (solve 0.2)
  let yMedium ← finalSaved "Euler order medium" (solve 0.1)
  let yFine ← finalSaved "Euler order fine" (solve 0.05)
  let exact := Float.exp (-1.0)
  let errCoarse := Float.abs (yCoarse - exact)
  let errMedium := Float.abs (yMedium - exact)
  let errFine := Float.abs (yFine - exact)
  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"Euler errors should decrease with dt: {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > 1.5 && ratio2 > 1.5)
    s!"Euler error ratios should show ~first-order trend: {ratio1}, {ratio2}"

@[test] def testRK4GlobalOrderTrend : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    RK4.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let solve := fun (dt : Float) =>
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
  let yCoarse ← finalSaved "RK4 order coarse" (solve 0.25)
  let yMedium ← finalSaved "RK4 order medium" (solve 0.125)
  let yFine ← finalSaved "RK4 order fine" (solve 0.0625)
  let exact := Float.exp (-1.0)
  let errCoarse := Float.abs (yCoarse - exact)
  let errMedium := Float.abs (yMedium - exact)
  let errFine := Float.abs (yFine - exact)
  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"RK4 errors should decrease with dt: {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > 8.0 && ratio2 > 8.0)
    s!"RK4 error ratios should show high-order trend: {ratio1}, {ratio2}"

@[test] def testDenseInterpolationErrorTrend : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    RK4.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let solve := fun (dt : Float) =>
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { dense := true, t1 := false })
  let yCoarse ← evaluateDenseFloat "dense coarse" (solve 0.25) 0.37
  let yFine ← evaluateDenseFloat "dense fine" (solve 0.125) 0.37
  let exact := Float.exp (-0.37)
  let errCoarse := Float.abs (yCoarse - exact)
  let errFine := Float.abs (yFine - exact)
  LeanTest.assertTrue (errCoarse > errFine)
    s!"Dense interpolation error should decrease with dt: {errCoarse} vs {errFine}"
  LeanTest.assertTrue (errFine < 5.0e-4)
    s!"Dense interpolation fine error too large: {errFine}"

private def deterministicWeakSeeds (count : Nat) : Array UInt64 :=
  Id.run do
    let mut seeds := #[]
    for i in [:count] do
      seeds := seeds.push (UInt64.ofNat (700001 + i * 7919))
    pure seeds

private def eulerMaruyamaOUWeakStats (dt : Float) (seeds : Array UInt64) : IO (Float × Float) := do
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
  let mut sum := 0.0
  let mut sumSq := 0.0
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
    let y1 ← finalSaved "weak OU Euler-Maruyama" sol
    sum := sum + y1
    sumSq := sumSq + (y1 * y1)
  let n := Float.ofNat seeds.size
  pure (sum / n, sumSq / n)

@[test] def testSDEWeakMomentConvergenceEulerMaruyama : IO Unit := do
  let theta := 1.0
  let sigma := 0.1
  let tFinal := 1.0
  let x0 := 1.0
  let seeds := deterministicWeakSeeds 96

  let (meanCoarse, m2Coarse) ← eulerMaruyamaOUWeakStats 0.2 seeds
  let (meanMedium, m2Medium) ← eulerMaruyamaOUWeakStats 0.1 seeds
  let (meanFine, m2Fine) ← eulerMaruyamaOUWeakStats 0.05 seeds

  let exactMean := x0 * Float.exp (-theta * tFinal)
  let exactVariance := ((sigma * sigma) / (2.0 * theta)) * (1.0 - Float.exp (-2.0 * theta * tFinal))
  let exactM2 := exactMean * exactMean + exactVariance

  let meanErrCoarse := Float.abs (meanCoarse - exactMean)
  let meanErrMedium := Float.abs (meanMedium - exactMean)
  let meanErrFine := Float.abs (meanFine - exactMean)
  let m2ErrCoarse := Float.abs (m2Coarse - exactM2)
  let m2ErrMedium := Float.abs (m2Medium - exactM2)
  let m2ErrFine := Float.abs (m2Fine - exactM2)

  LeanTest.assertTrue (meanErrCoarse > meanErrMedium && meanErrMedium > meanErrFine)
    s!"Weak mean errors should decrease with dt: {meanErrCoarse}, {meanErrMedium}, {meanErrFine}"
  LeanTest.assertTrue (m2ErrCoarse > m2ErrMedium && m2ErrMedium > m2ErrFine)
    s!"Weak second-moment errors should decrease with dt: {m2ErrCoarse}, {m2ErrMedium}, {m2ErrFine}"

  let meanRatio1 := if meanErrMedium <= 1.0e-16 then 0.0 else meanErrCoarse / meanErrMedium
  let meanRatio2 := if meanErrFine <= 1.0e-16 then 0.0 else meanErrMedium / meanErrFine
  let m2Ratio1 := if m2ErrMedium <= 1.0e-16 then 0.0 else m2ErrCoarse / m2ErrMedium
  let m2Ratio2 := if m2ErrFine <= 1.0e-16 then 0.0 else m2ErrMedium / m2ErrFine

  LeanTest.assertTrue (meanRatio1 > 1.15 && meanRatio2 > 1.15)
    s!"Weak mean trend too weak: ratios {meanRatio1}, {meanRatio2}"
  LeanTest.assertTrue (m2Ratio1 > 1.15 && m2Ratio2 > 1.15)
    s!"Weak second-moment trend too weak: ratios {m2Ratio1}, {m2Ratio2}"

  LeanTest.assertTrue (meanErrFine < 2.5e-2)
    s!"Weak mean fine-grid error too large: {meanErrFine}"
  LeanTest.assertTrue (m2ErrFine < 2.5e-2)
    s!"Weak second-moment fine-grid error too large: {m2ErrFine}"

@[test] def testSDEStrongOrderTrendAndMilsteinAdvantage : IO Unit := do
  let mu := 0.2
  let sigma := 0.3
  let drift : ODETerm Float Unit := { vectorField := fun _t y _ => mu * y }
  let solverEM :=
    EulerMaruyama.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let solverMil :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let seeds : Array UInt64 := #[1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008]
  let mut sumErrCoarse := 0.0
  let mut sumErrMedium := 0.0
  let mut sumErrFine := 0.0
  let mut sumErrMilstein := 0.0
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
      ControlTerm.ofPath (fun _t y _ => sigma * y) bmPath (fun vf control => vf * control)
    let terms : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) :=
      { term1 := drift, term2 := diffusion }
    let solveEM := fun (dt : Float) =>
      diffeqsolve
        (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
        (Y := Float)
        (VF := (Float × Float))
        (Control := (Time × Float))
        (Args := Unit)
        (Controller := ConstantStepSize)
        terms solverEM 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
    let solveMil := fun (dt : Float) =>
      diffeqsolve
        (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
        (Y := Float)
        (VF := (Float × Float))
        (Control := (Time × Float))
        (Args := Unit)
        (Controller := ConstantStepSize)
        terms solverMil 0.0 1.0 (some dt) (1.0 : Float) () (saveat := { t1 := true })
    let yCoarse ← finalSaved "EM strong coarse" (solveEM 0.25)
    let yMedium ← finalSaved "EM strong medium" (solveEM 0.125)
    let yFine ← finalSaved "EM strong fine" (solveEM 0.0625)
    let yMilstein ← finalSaved "Milstein comparison" (solveMil 0.125)
    let wT := (VirtualBrownianTree.increment bm 0.0 1.0).W
    let exact := Float.exp ((mu - 0.5 * sigma * sigma) + sigma * wT)
    sumErrCoarse := sumErrCoarse + Float.abs (yCoarse - exact)
    sumErrMedium := sumErrMedium + Float.abs (yMedium - exact)
    sumErrFine := sumErrFine + Float.abs (yFine - exact)
    sumErrMilstein := sumErrMilstein + Float.abs (yMilstein - exact)
  let n := Float.ofNat seeds.size
  let errCoarse := sumErrCoarse / n
  let errMedium := sumErrMedium / n
  let errFine := sumErrFine / n
  let errMilstein := sumErrMilstein / n
  LeanTest.assertTrue (errCoarse > errMedium && errMedium > errFine)
    s!"Mean Euler-Maruyama errors should decrease with dt: {errCoarse}, {errMedium}, {errFine}"
  let ratio1 := if errMedium <= 1.0e-16 then 0.0 else errCoarse / errMedium
  let ratio2 := if errFine <= 1.0e-16 then 0.0 else errMedium / errFine
  LeanTest.assertTrue (ratio1 > 1.1 && ratio2 > 1.1)
    s!"Mean Euler-Maruyama strong-order trend too weak: ratios {ratio1}, {ratio2}"
  LeanTest.assertTrue (errMilstein < errMedium)
    s!"Mean Milstein error should improve over EM at same dt: {errMilstein} vs {errMedium}"

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
  testStepToReverseTime
  testStepToRejectsEndpointMismatch
  testStepToRejectsNonMonotoneTs
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
  testSaveAtTsDirectionForward
  testSaveAtTsDirectionReverse
  testNestedSubSaveAtPayloadFlags
  testBooleanEventTerminate
  testBooleanEventNonTerminating
  testRealEventDirection
  testBooleanEventDirectionDown
  testEventTiePrefersTerminating
  testEventMaskExcludesLaterSameStepRoots
  testBrownianPairAndFinStructuredIncrements
  testMilsteinAutodiffJvpWrapper
  testControlTermToODEConversion
  testAbstractPathComposeLinear
  testAbstractPathMapAndRestrict
  testUnderdampedLangevinTerms
  testSaveFnTransformsOutputs
  testFailureResultMessage
  testMaxStepsReached
  testUnsafeBrownianPathStructuredIncrements
  testEulerPairStatePyTree
  testEulerFinStatePyTree
  testEulerMaruyamaPairStatePyTree
  testProgressMeterDefaultParity
  testProgressMeterTextAndTqdmCompatibility
  testEulerGlobalOrderTrend
  testRK4GlobalOrderTrend
  testDenseInterpolationErrorTrend
  testSDEWeakMomentConvergenceEulerMaruyama
  testSDEStrongOrderTrendAndMilsteinAdvantage

end Tests.DiffEq
