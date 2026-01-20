import Tyr.DiffEq.RootFinder
import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Implicit Runge--Kutta Infrastructure -/

structure ImplicitRK (s : Nat) where
  tableau : ButcherTableau s
  rootFinder : FixedPoint := {}

structure IMEXRK (s : Nat) where
  explicit : ButcherTableau s
  implicit : ButcherTableau s
  rootFinder : FixedPoint := {}

namespace ImplicitRK

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  DiffEqSpace.scale 0.0 y0

private def weightedSum {s : Nat} [DiffEqSpace Y] (coeffs : Vector s Time)
    (ks : Array Y) (y0 : Y) : Y :=
  let coeffArr := coeffs.toArray
  (List.range coeffArr.size).foldl
    (fun acc j =>
      let a := coeffArr.getD j 0.0
      let kj := ks.getD j (zeroLike y0)
      DiffEqSpace.add acc (DiffEqSpace.scale a kj))
    (zeroLike y0)

def solver {s : Nat} {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rk : ImplicitRK s) : AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => rk.tableau.order
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    let dt := inst.contr term t0 t1
    let zero := zeroLike y0
    let rows := rk.tableau.a.toArray
    let cs := rk.tableau.c.toArray
    let (ks, ok) :=
      (List.range rows.size).foldl
        (fun acc i =>
          let (ks, ok) := acc
          let row := rows.getD i #[]
          let sum :=
            (List.range i).foldl
              (fun acc j =>
                let aij := row.getD j 0.0
                let kj := ks.getD j zero
                DiffEqSpace.add acc (DiffEqSpace.scale aij kj))
              zero
          let ti := t0 + cs.getD i 0.0 * dt
          let base := DiffEqSpace.add y0 sum
          let aii := row.getD i 0.0
          let (yi, ok) :=
            if aii == 0.0 then
              (base, ok)
            else
              let stepFn := fun y =>
                DiffEqSpace.add base (DiffEqSpace.scale aii (inst.vf_prod term ti y args dt))
              let sol := RootFinder.solve rk.rootFinder stepFn base
              (sol.value, ok && sol.converged)
          let ki := inst.vf_prod term ti yi args dt
          (ks.push ki, ok))
        (#[], true)
    let y1 := DiffEqSpace.add y0 (weightedSum rk.tableau.b ks y0)
    let yErr :=
      match rk.tableau.bErr with
      | none => none
      | some bErr =>
          let high := weightedSum rk.tableau.b ks y0
          let low := weightedSum bErr ks y0
          some (DiffEqSpace.sub high low)
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := yErr
      denseInfo := dense
      solverState := state
      result := if ok then Result.successful else Result.internalError
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

end ImplicitRK

namespace IMEXRK

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  DiffEqSpace.scale 0.0 y0

private def weightedSum {s : Nat} [DiffEqSpace Y] (coeffs : Vector s Time)
    (ks : Array Y) (y0 : Y) : Y :=
  let coeffArr := coeffs.toArray
  (List.range coeffArr.size).foldl
    (fun acc j =>
      let a := coeffArr.getD j 0.0
      let kj := ks.getD j (zeroLike y0)
      DiffEqSpace.add acc (DiffEqSpace.scale a kj))
    (zeroLike y0)

def solver {s : Nat} {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rk : IMEXRK s) :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.pair
  order := fun _ => rk.explicit.order
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let explicit := terms.term1
    let implicit := terms.term2
    let expInst := (inferInstance : TermLike ExplicitTerm Y VFe Time Args)
    let impInst := (inferInstance : TermLike ImplicitTerm Y VFi Time Args)
    let dt := expInst.contr explicit t0 t1
    let zero := zeroLike y0
    let rowsE := rk.explicit.a.toArray
    let rowsI := rk.implicit.a.toArray
    let cs := rk.explicit.c.toArray
    let (ksExp, ksImp, ok) :=
      (List.range rowsE.size).foldl
        (fun acc i =>
          let (ksExp, ksImp, ok) := acc
          let rowE := rowsE.getD i #[]
          let rowI := rowsI.getD i #[]
          let sum :=
            (List.range i).foldl
              (fun acc j =>
                let aE := rowE.getD j 0.0
                let aI := rowI.getD j 0.0
                let kE := ksExp.getD j zero
                let kI := ksImp.getD j zero
                let sumE := DiffEqSpace.scale aE kE
                let sumI := DiffEqSpace.scale aI kI
                DiffEqSpace.add acc (DiffEqSpace.add sumE sumI))
              zero
          let ti := t0 + cs.getD i 0.0 * dt
          let base := DiffEqSpace.add y0 sum
          let aii := rowI.getD i 0.0
          let (yi, ok) :=
            if aii == 0.0 then
              (base, ok)
            else
              let stepFn := fun y =>
                DiffEqSpace.add base (DiffEqSpace.scale aii (impInst.vf_prod implicit ti y args dt))
              let sol := RootFinder.solve rk.rootFinder stepFn base
              (sol.value, ok && sol.converged)
          let kE := expInst.vf_prod explicit ti yi args dt
          let kI := impInst.vf_prod implicit ti yi args dt
          (ksExp.push kE, ksImp.push kI, ok))
        (#[], #[], true)
    let yHigh :=
      DiffEqSpace.add y0
        (DiffEqSpace.add (weightedSum rk.explicit.b ksExp y0) (weightedSum rk.implicit.b ksImp y0))
    let yErr :=
      match rk.explicit.bErr, rk.implicit.bErr with
      | some bErrE, some bErrI =>
          let high :=
            DiffEqSpace.add (weightedSum rk.explicit.b ksExp y0) (weightedSum rk.implicit.b ksImp y0)
          let low :=
            DiffEqSpace.add (weightedSum bErrE ksExp y0) (weightedSum bErrI ksImp y0)
          some (DiffEqSpace.sub high low)
      | _, _ => none
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := yHigh }
    {
      y1 := yHigh
      yError := yErr
      denseInfo := dense
      solverState := state
      result := if ok then Result.successful else Result.internalError
    }
  func := fun terms t y args =>
    let explicit := terms.term1
    let implicit := terms.term2
    let expInst := (inferInstance : TermLike ExplicitTerm Y VFe Time Args)
    let impInst := (inferInstance : TermLike ImplicitTerm Y VFi Time Args)
    (expInst.vf explicit t y args, impInst.vf implicit t y args)
  interpolation := fun info => info.toInterpolation
}

end IMEXRK

end DiffEq
end torch
