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

/-- Reusable linear interpolation data with constant derivative. -/
structure LinearPath (Control : Type) where
  t0 : Time
  t1 : Time
  x0 : Control
  x1 : Control

namespace LinearPath

def dt (path : LinearPath Control) : Time :=
  path.t1 - path.t0

def delta [DiffEqSpace Control] (path : LinearPath Control) : Control :=
  DiffEqSpace.sub path.x1 path.x0

def slope [DiffEqSpace Control] (path : LinearPath Control) : Control :=
  let dt := path.dt
  let delta := path.delta
  if dt == 0.0 then
    DiffEqSpace.scale 0.0 delta
  else
    DiffEqSpace.scale (1.0 / dt) delta

def position [DiffEqSpace Control] (path : LinearPath Control) (t : Time) : Control :=
  DiffEqSpace.add path.x0 (DiffEqSpace.scale (t - path.t0) path.slope)

def derivative [DiffEqSpace Control] (path : LinearPath Control) (_t : Time) (_left : Bool := true) :
    Control :=
  path.slope

def increment [DiffEqSpace Control] (path : LinearPath Control) (tStart tEnd : Time) : Control :=
  DiffEqSpace.scale (tEnd - tStart) path.slope

def evaluate [DiffEqSpace Control] (path : LinearPath Control)
    (tStart : Time) (tEnd? : Option Time) (_left : Bool := true) : Control :=
  let tEnd := tEnd?.getD tStart
  path.increment tStart tEnd

def toAbstract [DiffEqSpace Control] (path : LinearPath Control) : AbstractPath Control :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun tStart tEnd? left => path.evaluate tStart tEnd? left
    derivativeFn? := some (fun t left => path.derivative t left)
  }

end LinearPath

/-- Reusable cubic Hermite interpolation data with endpoint derivatives. -/
structure CubicHermitePath (Control : Type) where
  t0 : Time
  t1 : Time
  x0 : Control
  x1 : Control
  dx0 : Control
  dx1 : Control

namespace CubicHermitePath

def dt (path : CubicHermitePath Control) : Time :=
  path.t1 - path.t0

def position [DiffEqSpace Control] (path : CubicHermitePath Control) (t : Time) : Control :=
  let dt := path.dt
  if dt == 0.0 then
    path.x0
  else
    let u := (t - path.t0) / dt
    let u2 := u * u
    let u3 := u2 * u
    let h00 := 2.0 * u3 - 3.0 * u2 + 1.0
    let h10 := u3 - 2.0 * u2 + u
    let h01 := -2.0 * u3 + 3.0 * u2
    let h11 := u3 - u2
    let p0 := DiffEqSpace.scale h00 path.x0
    let p1 := DiffEqSpace.scale (h10 * dt) path.dx0
    let p2 := DiffEqSpace.scale h01 path.x1
    let p3 := DiffEqSpace.scale (h11 * dt) path.dx1
    DiffEqSpace.add (DiffEqSpace.add p0 p1) (DiffEqSpace.add p2 p3)

def derivative [DiffEqSpace Control]
    (path : CubicHermitePath Control) (t : Time) (_left : Bool := true) : Control :=
  let dt := path.dt
  if dt == 0.0 then
    path.dx0
  else
    let u := (t - path.t0) / dt
    let u2 := u * u
    let dh00 := (6.0 * u2 - 6.0 * u) / dt
    let dh10 := 3.0 * u2 - 4.0 * u + 1.0
    let dh01 := (-6.0 * u2 + 6.0 * u) / dt
    let dh11 := 3.0 * u2 - 2.0 * u
    let p0 := DiffEqSpace.scale dh00 path.x0
    let p1 := DiffEqSpace.scale dh10 path.dx0
    let p2 := DiffEqSpace.scale dh01 path.x1
    let p3 := DiffEqSpace.scale dh11 path.dx1
    DiffEqSpace.add (DiffEqSpace.add p0 p1) (DiffEqSpace.add p2 p3)

def increment [DiffEqSpace Control] (path : CubicHermitePath Control) (tStart tEnd : Time) : Control :=
  if path.dt == 0.0 then
    DiffEqSpace.scale (tEnd - tStart) path.dx0
  else
    DiffEqSpace.sub (path.position tEnd) (path.position tStart)

def evaluate [DiffEqSpace Control] (path : CubicHermitePath Control)
    (tStart : Time) (tEnd? : Option Time) (_left : Bool := true) : Control :=
  let tEnd := tEnd?.getD tStart
  path.increment tStart tEnd

def toAbstract [DiffEqSpace Control] (path : CubicHermitePath Control) : AbstractPath Control :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun tStart tEnd? left => path.evaluate tStart tEnd? left
    derivativeFn? := some (fun t left => path.derivative t left)
  }

end CubicHermitePath

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

def derivativeOrElse (path : AbstractPath Control)
    (fallback : Time → Bool → Control) (t : Time) (left : Bool := true) : Control :=
  (path.derivativeFnOrElse fallback) t left

def increment (path : AbstractPath Control) (tStart tEnd : Time) (left : Bool := true) : Control :=
  path.evaluate tStart (some tEnd) left

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
  LinearPath.toAbstract
    ({ t0 := t0, t1 := t1, x0 := x0, x1 := x1 } : LinearPath Control)

def cubicHermiteInterpolation [DiffEqSpace Control]
    (t0 t1 : Time)
    (x0 x1 : Control)
    (dx0 dx1 : Control) : AbstractPath Control :=
  CubicHermitePath.toAbstract
    ({ t0 := t0, t1 := t1, x0 := x0, x1 := x1, dx0 := dx0, dx1 := dx1 } :
      CubicHermitePath Control)

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
