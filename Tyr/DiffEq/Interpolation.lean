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

/-- Per-step cubic Hermite dense output with endpoint tangents in normalized step time. -/
structure LocalHermiteDenseInfo (Y : Type) where
  t0 : Time
  t1 : Time
  y0 : Y
  y1 : Y
  m0 : Y
  m1 : Y

namespace LocalHermiteDenseInfo

private def evalTheta [DiffEqSpace Y] (info : LocalHermiteDenseInfo Y) (theta : Time) : Y :=
  let theta2 := theta * theta
  let theta3 := theta2 * theta
  let h00 := 2.0 * theta3 - 3.0 * theta2 + 1.0
  let h10 := theta3 - 2.0 * theta2 + theta
  let h01 := -2.0 * theta3 + 3.0 * theta2
  let h11 := theta3 - theta2
  let a0 := DiffEqSpace.scale h00 info.y0
  let a1 := DiffEqSpace.scale h10 info.m0
  let a2 := DiffEqSpace.scale h01 info.y1
  let a3 := DiffEqSpace.scale h11 info.m1
  DiffEqSpace.add (DiffEqSpace.add a0 a1) (DiffEqSpace.add a2 a3)

private def derivTheta [DiffEqSpace Y] (info : LocalHermiteDenseInfo Y) (theta : Time) : Y :=
  let theta2 := theta * theta
  let dh00 := 6.0 * theta2 - 6.0 * theta
  let dh10 := 3.0 * theta2 - 4.0 * theta + 1.0
  let dh01 := -6.0 * theta2 + 6.0 * theta
  let dh11 := 3.0 * theta2 - 2.0 * theta
  let a0 := DiffEqSpace.scale dh00 info.y0
  let a1 := DiffEqSpace.scale dh10 info.m0
  let a2 := DiffEqSpace.scale dh01 info.y1
  let a3 := DiffEqSpace.scale dh11 info.m1
  DiffEqSpace.add (DiffEqSpace.add a0 a1) (DiffEqSpace.add a2 a3)

def toInterpolation [DiffEqSpace Y] (info : LocalHermiteDenseInfo Y) :
    DenseInterpolation Y := by
  let evalAt := fun (t : Time) =>
    if info.t0 == info.t1 then
      info.y0
    else
      let theta := (t - info.t0) / (info.t1 - info.t0)
      evalTheta info theta
  exact {
    evaluate := fun t0 t1 _left =>
      match t1 with
      | none => evalAt t0
      | some t1 => DiffEqSpace.sub (evalAt t1) (evalAt t0)
    derivative := fun t _left =>
      if info.t0 == info.t1 then
        DiffEqSpace.scale 0.0 info.m0
      else
        let h := info.t1 - info.t0
        let theta := (t - info.t0) / h
        let dTheta := derivTheta info theta
        DiffEqSpace.scale (1.0 / h) dTheta
  }

end LocalHermiteDenseInfo

/-- Piecewise dense interpolation assembled from per-step solver interpolation data. -/
structure PiecewiseDenseInterpolation (Y : Type) where
  ts : Array Time
  segments : Array (DenseInterpolation Y)

namespace PiecewiseDenseInterpolation

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

private def evalAt [DiffEqSpace Y] [Inhabited Y]
    (interp : PiecewiseDenseInterpolation Y) (t : Time) (left : Bool) : Y :=
  if hzero : interp.segments.size = 0 then
    panic! "PiecewiseDenseInterpolation requires at least one segment."
  else
    let hpos : 0 < interp.segments.size := Nat.pos_of_ne_zero hzero
    let defaultSeg := interp.segments[0]'hpos
    let idx :=
      if interp.ts.size <= 1 then
        0
      else
        clampIndex interp.segments.size (findBracket interp.ts t)
    let seg := interp.segments.getD idx defaultSeg
    seg.evaluate t none left

def toDense [DiffEqSpace Y] [Inhabited Y]
    (interp : PiecewiseDenseInterpolation Y) : DenseInterpolation Y := by
  exact {
    evaluate := fun t0 t1 left =>
      let y0 := evalAt interp t0 left
      match t1 with
      | none => y0
      | some t1 =>
          let y1 := evalAt interp t1 left
          DiffEqSpace.sub y1 y0
    derivative := fun t left =>
      if hzero : interp.segments.size = 0 then
        panic! "PiecewiseDenseInterpolation requires at least one segment."
      else
        let hpos : 0 < interp.segments.size := Nat.pos_of_ne_zero hzero
        let defaultSeg := interp.segments[0]'hpos
        let idx :=
          if interp.ts.size <= 1 then
            0
          else
            clampIndex interp.segments.size (findBracket interp.ts t)
        let seg := interp.segments.getD idx defaultSeg
        seg.derivative t left
  }

end PiecewiseDenseInterpolation

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
