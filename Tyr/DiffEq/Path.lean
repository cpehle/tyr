import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## Path Abstractions

Paths provide control increments over time intervals.
-/

structure AbstractPath (Control : Type) where
  t0 : Time
  t1 : Time
  evaluate : Time → Option Time → Bool → Control
  derivativeFn? : Option (Time → Bool → Control) := none

namespace AbstractPath

def derivative (path : AbstractPath Control) (t : Time) (left : Bool := true) : Option Control :=
  match path.derivativeFn? with
  | some derivativeFn => some (derivativeFn t left)
  | none => none

def hasDerivative (path : AbstractPath Control) : Bool :=
  path.derivativeFn?.isSome

def derivativeFnOrElse (path : AbstractPath Control)
    (fallback : Time → Bool → Control) : Time → Bool → Control :=
  match path.derivativeFn? with
  | some derivativeFn => derivativeFn
  | none => fallback

def ofFunctions (t0 t1 : Time)
    (evaluate : Time → Option Time → Bool → Control)
    (derivativeFn? : Option (Time → Bool → Control) := none) : AbstractPath Control :=
  {
    t0 := t0
    t1 := t1
    evaluate := evaluate
    derivativeFn? := derivativeFn?
  }

def ofPosition [DiffEqSpace Control]
    (t0 t1 : Time)
    (position : Time → Control)
    (derivativeFn? : Option (Time → Bool → Control) := none) : AbstractPath Control :=
  ofFunctions t0 t1
    (fun tStart tEnd? _left =>
      let tEnd := tEnd?.getD tStart
      DiffEqSpace.sub (position tEnd) (position tStart))
    derivativeFn?

def ofDifferentiablePosition [DiffEqSpace Control]
    (t0 t1 : Time)
    (position : Time → Control)
    (derivativeFn : Time → Bool → Control) : AbstractPath Control :=
  ofPosition t0 t1 position (some derivativeFn)

def linearInterpolation [DiffEqSpace Control]
    (t0 t1 : Time) (x0 x1 : Control) : AbstractPath Control :=
  let dt := t1 - t0
  let delta := DiffEqSpace.sub x1 x0
  let slope :=
    if dt == 0.0 then
      DiffEqSpace.scale 0.0 delta
    else
      DiffEqSpace.scale (1.0 / dt) delta
  ofFunctions t0 t1
    (fun tStart tEnd? _left =>
      let tEnd := tEnd?.getD tStart
      DiffEqSpace.scale (tEnd - tStart) slope)
    (some (fun _ _ => slope))

def cubicHermiteInterpolation [DiffEqSpace Control]
    (t0 t1 : Time)
    (x0 x1 : Control)
    (dx0 dx1 : Control) : AbstractPath Control :=
  let dt := t1 - t0
  if dt == 0.0 then
    ofFunctions t0 t1
      (fun tStart tEnd? _left =>
        let tEnd := tEnd?.getD tStart
        DiffEqSpace.scale (tEnd - tStart) dx0)
      (some (fun _ _ => dx0))
  else
    let position := fun t =>
      let u := (t - t0) / dt
      let u2 := u * u
      let u3 := u2 * u
      let h00 := 2.0 * u3 - 3.0 * u2 + 1.0
      let h10 := u3 - 2.0 * u2 + u
      let h01 := -2.0 * u3 + 3.0 * u2
      let h11 := u3 - u2
      let p0 := DiffEqSpace.scale h00 x0
      let p1 := DiffEqSpace.scale (h10 * dt) dx0
      let p2 := DiffEqSpace.scale h01 x1
      let p3 := DiffEqSpace.scale (h11 * dt) dx1
      DiffEqSpace.add (DiffEqSpace.add p0 p1) (DiffEqSpace.add p2 p3)
    let derivativeFn := fun t _left =>
      let u := (t - t0) / dt
      let u2 := u * u
      let dh00 := (6.0 * u2 - 6.0 * u) / dt
      let dh10 := 3.0 * u2 - 4.0 * u + 1.0
      let dh01 := (-6.0 * u2 + 6.0 * u) / dt
      let dh11 := 3.0 * u2 - 2.0 * u
      let p0 := DiffEqSpace.scale dh00 x0
      let p1 := DiffEqSpace.scale dh10 dx0
      let p2 := DiffEqSpace.scale dh01 x1
      let p3 := DiffEqSpace.scale dh11 dx1
      DiffEqSpace.add (DiffEqSpace.add p0 p1) (DiffEqSpace.add p2 p3)
    ofPosition t0 t1 position (some derivativeFn)

def withDerivative (path : AbstractPath Control) (derivativeFn : Time → Bool → Control) :
    AbstractPath Control :=
  { path with derivativeFn? := some derivativeFn }

def clearDerivative (path : AbstractPath Control) : AbstractPath Control :=
  { path with derivativeFn? := none }

def mapControl (path : AbstractPath Control)
    (mapEval : Control → Control₂)
    (mapDerivative : Option (Control → Control₂) := none) :
    AbstractPath Control₂ :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun tStart tEnd left => mapEval (path.evaluate tStart tEnd left)
    derivativeFn? :=
      match path.derivativeFn? with
      | some derivativeFn =>
          let mapDeriv := mapDerivative.getD mapEval
          some (fun t left => mapDeriv (derivativeFn t left))
      | none => none
  }

def restrict (path : AbstractPath Control) (t0 t1 : Time) : AbstractPath Control :=
  {
    t0 := t0
    t1 := t1
    evaluate := path.evaluate
    derivativeFn? := path.derivativeFn?
  }

end AbstractPath

end DiffEq
end torch
