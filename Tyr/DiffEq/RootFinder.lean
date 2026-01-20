import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## Root Finders -/

structure RootFindResult (Y : Type) where
  value : Y
  converged : Bool
  iterations : Nat

structure FixedPoint where
  rtol : Float := 1.0e-6
  atol : Float := 1.0e-6
  maxIters : Nat := 20
  deriving Inhabited

class RootFinder (R : Type) where
  solve : {Y : Type} → [DiffEqSpace Y] → [DiffEqSeminorm Y] → [DiffEqElem Y] →
    R → (Y → Y) → Y → RootFindResult Y

private def scaledError {Y : Type} [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rtol atol : Float) (yPrev yNext : Y) : Float :=
  let diff := DiffEqSpace.sub yNext yPrev
  let scale :=
    DiffEqElem.addScalar atol (DiffEqSpace.scale rtol (DiffEqElem.max (DiffEqElem.abs yPrev) (DiffEqElem.abs yNext)))
  DiffEqSeminorm.rms (DiffEqElem.div diff scale)

instance : RootFinder FixedPoint where
  solve {Y} _ _ _ cfg step y0 :=
    let rec loop (i : Nat) (yPrev : Y) : RootFindResult Y :=
      if i >= cfg.maxIters then
        { value := yPrev, converged := false, iterations := i }
      else
        let yNext := step yPrev
        let err := scaledError cfg.rtol cfg.atol yPrev yNext
        if err < 1.0 then
          { value := yNext, converged := true, iterations := i + 1 }
        else
          loop (i + 1) yNext
    loop 0 y0

end DiffEq
end torch
