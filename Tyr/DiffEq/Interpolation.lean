import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## Interpolation

Dense interpolation interfaces used by solutions.
-/

structure DenseInterpolation (Y : Type) where
  evaluate : Time → Option Time → Bool → Y
  derivative : Time → Bool → Y

/-! ## Local Linear Interpolation

Used by low-order solvers for per-step dense output.
-/

structure LocalLinearDenseInfo (Y : Type) where
  t0 : Time
  t1 : Time
  y0 : Y
  y1 : Y

namespace LocalLinearDenseInfo

def toInterpolation [DiffEqSpace Y] (info : LocalLinearDenseInfo Y) :
    DenseInterpolation Y := by
  let evalAt := fun (t : Time) =>
    if info.t0 == info.t1 then
      info.y0
    else
      let theta := (t - info.t0) / (info.t1 - info.t0)
      DiffEqSpace.add info.y0 (DiffEqSpace.scale theta (DiffEqSpace.sub info.y1 info.y0))
  exact {
    evaluate := fun t0 t1 _left =>
      match t1 with
      | none => evalAt t0
      | some t1 => DiffEqSpace.sub (evalAt t1) (evalAt t0)
    derivative := fun _t _left =>
      if info.t0 == info.t1 then
        DiffEqSpace.scale 0.0 (DiffEqSpace.sub info.y1 info.y0)
      else
        let inv := 1.0 / (info.t1 - info.t0)
        DiffEqSpace.scale inv (DiffEqSpace.sub info.y1 info.y0)
  }

end LocalLinearDenseInfo

/-- Piecewise-linear interpolation over saved step points. -/
structure LinearInterpolation (Y : Type) where
  ts : Array Time
  ys : Array Y

namespace LinearInterpolation

private def clampIndex (n : Nat) (i : Nat) : Nat :=
  if i < n then i else if n == 0 then 0 else n - 1

private def findBracket (ts : Array Time) (t : Time) : Nat :=
  let n := ts.size
  if n <= 1 then
    0
  else
    let rec go (i : Nat) : Nat :=
      if h : i + 1 < n then
        let t1 := ts[i + 1]!
        if t <= t1 then i else go (i + 1)
      else
        n - 2
    go 0

def toDense [DiffEqSpace Y] [Inhabited Y] (interp : LinearInterpolation Y) : DenseInterpolation Y := by
  let evalAt := fun (t : Time) =>
    if interp.ts.size == 0 then
      panic! "LinearInterpolation requires at least one point."
    else
      let i := findBracket interp.ts t
      let i0 := clampIndex interp.ts.size i
      let i1 := clampIndex interp.ts.size (i0 + 1)
      let t0 := interp.ts.getD i0 0.0
      let t1 := interp.ts.getD i1 0.0
      let y0 := interp.ys.getD i0 default
      let y1 := interp.ys.getD i1 default
      if t0 == t1 then
        y0
      else
        let theta := (t - t0) / (t1 - t0)
        DiffEqSpace.add y0 (DiffEqSpace.scale theta (DiffEqSpace.sub y1 y0))
  exact {
    evaluate := fun t0 t1 _left =>
      match t1 with
      | none => evalAt t0
      | some t1 =>
          DiffEqSpace.sub (evalAt t1) (evalAt t0)
    derivative := fun t _left =>
      if interp.ts.size <= 1 then
        panic! "LinearInterpolation derivative requires at least two points."
      else
        let i := findBracket interp.ts t
        let i0 := clampIndex interp.ts.size i
        let i1 := clampIndex interp.ts.size (i0 + 1)
        let t0 := interp.ts.getD i0 0.0
        let t1 := interp.ts.getD i1 0.0
        let y0 := interp.ys.getD i0 default
        let y1 := interp.ys.getD i1 default
        if t0 == t1 then
          DiffEqSpace.scale 0.0 (DiffEqSpace.sub y1 y0)
        else
          let inv := 1.0 / (t1 - t0)
          DiffEqSpace.scale inv (DiffEqSpace.sub y1 y0)
  }

end LinearInterpolation


end DiffEq
end torch
