import Tyr.DiffEq.RootFinder
import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Implicit Runge--Kutta Infrastructure -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

inductive ImplicitRKDenseKind where
  | hermite
  | splitAtStage (stage : Nat) (tag : String)
  | kencarp3Poly2
  deriving Repr, BEq, Inhabited

structure ImplicitRK (s : Nat) where
  tableau : ButcherTableau s
  rootFinder : FixedPoint := {}
  rootMethod : Option RootFindMethod := none
  rootRtol : Option Float := none
  rootAtol : Option Float := none
  rootMaxIters : Option Nat := none
  denseKind : ImplicitRKDenseKind := .hermite

structure IMEXRK (s : Nat) where
  explicit : ButcherTableau s
  implicit : ButcherTableau s
  rootFinder : FixedPoint := {}
  rootMethod : Option RootFindMethod := none
  rootRtol : Option Float := none
  rootAtol : Option Float := none
  rootMaxIters : Option Nat := none
  denseKind : ImplicitRKDenseKind := .hermite

inductive ImplicitDenseInfo (Y : Type) where
  | hermite (info : LocalHermiteDenseInfo Y)
  | poly4 (info : LocalPolynomial4DenseInfo Y)

namespace ImplicitDenseInfo

def toInterpolation [DiffEqSpace Y] (info : ImplicitDenseInfo Y) : DenseInterpolation Y :=
  match info with
  | .hermite info => info.toInterpolation
  | .poly4 info => info.toInterpolation

end ImplicitDenseInfo

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  0.0 * y0

private def weightedSumArray [DiffEqSpace Y] (coeffs : Array Time)
    (ks : Array Y) (y0 : Y) : Y := Id.run do
  let mut acc := zeroLike y0
  for j in [:coeffs.size] do
    let a := coeffs.getD j 0.0
    let kj := ks.getD j (zeroLike y0)
    acc := acc + a * kj
  return acc

private def baseHermiteDenseInfo [DiffEqSpace Y]
    (t0 t1 : Time) (y0 y1 : Y) (m0 m1 : Y) : LocalHermiteDenseInfo Y := {
  t0 := t0
  t1 := t1
  y0 := y0
  y1 := y1
  m0 := m0
  m1 := m1
}

private def kencarp3Poly2DenseInfo [DiffEqSpace Y]
    (t0 t1 : Time) (y0 : Y) (stageMs : Array Y) (fallback : LocalHermiteDenseInfo Y) :
    ImplicitDenseInfo Y :=
  if stageMs.size == 4 then
    -- Diffrax KenCarp3 interpolation: y(theta) = y0 + Σ_i (a_i * theta^2 + b_i * theta) k_i.
    -- We encode this exactly into Tyr's quartic power-basis container (with c4 = c3 = 0).
    let quadWeights : Array Time := #[
      -215264564351.0 / 13552729205753.0,
      17870216137069.0 / 13817060693119.0,
      -28141676662227.0 / 17317692491321.0,
      2508943948391.0 / 7218656332882.0
    ]
    let linearWeights : Array Time := #[
      4655552711362.0 / 22874653954995.0,
      -18682724506714.0 / 9892148508045.0,
      34259539580243.0 / 13192909600954.0,
      584795268549.0 / 6622622206610.0
    ]
    .poly4 {
      t0 := t0
      t1 := t1
      c4 := zeroLike y0
      c3 := zeroLike y0
      c2 := weightedSumArray quadWeights stageMs y0
      c1 := weightedSumArray linearWeights stageMs y0
      c0 := y0
    }
  else
    .hermite fallback

private def denseInfo [DiffEqSpace Y]
    (kind : ImplicitRKDenseKind)
    (cs : Array Time)
    (t0 t1 dt : Time)
    (y0 y1 : Y)
    (m0 m1 : Y)
    (stageYs stageMs : Array Y) :
    ImplicitDenseInfo Y :=
  let base := baseHermiteDenseInfo t0 t1 y0 y1 m0 m1
  match kind with
  | .hermite =>
      .hermite base
  | .splitAtStage stage tag =>
      if hStage : stage < cs.size then
        let c := cs[stage]'hStage
        if c <= 0.0 || c >= 1.0 then
          .hermite base
        else
          let tSplit := t0 + c * dt
          let ySplit := stageYs.getD stage y0
          let mSplit := stageMs.getD stage m0
          .hermite {
            base with split? := some (tSplit, ySplit, mSplit), splitKind? := some tag
          }
      else
        .hermite base
  | .kencarp3Poly2 =>
      kencarp3Poly2DenseInfo t0 t1 y0 stageMs base

namespace ImplicitRK

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  0.0 * y0

private def weightedSum {s : Nat} [DiffEqSpace Y] (coeffs : Vector s Time)
    (ks : Array Y) (y0 : Y) : Y :=
  let coeffArr := coeffs.toArray
  (List.range coeffArr.size).foldl
    (fun acc j =>
      let a := coeffArr.getD j 0.0
      let kj := ks.getD j (zeroLike y0)
      acc + a * kj)
    (zeroLike y0)

private def effectiveRootMethod {s : Nat} (rk : ImplicitRK s) : RootFindMethod :=
  let base : RootFindMethod := rk.rootMethod.getD (RootFindMethod.fixedPoint rk.rootFinder)
  RootFindMethod.withTolerances base rk.rootRtol rk.rootAtol rk.rootMaxIters

def solver {s : Nat} {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rk : ImplicitRK s) : AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := ImplicitDenseInfo Y
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
    let (ks, stageYs, ok) :=
      (List.range rows.size).foldl
        (fun acc i =>
          let (ks, stageYs, ok) := acc
          let row := rows.getD i #[]
          let sum :=
            (List.range i).foldl
              (fun acc j =>
                let aij := row.getD j 0.0
                let kj := ks.getD j zero
                acc + aij * kj)
              zero
          let ti := t0 + cs.getD i 0.0 * dt
          let base := y0 + sum
          let aii := row.getD i 0.0
          let (yi, ok) :=
            if aii == 0.0 then
              (base, ok)
            else
              let stepFn := fun y =>
                base + aii * (inst.vf_prod term ti y args dt)
              let rootMethod : RootFindMethod := effectiveRootMethod rk
              let sol := RootFinder.solve (R := RootFindMethod) rootMethod stepFn base
              (sol.value, ok && sol.converged)
          let ki := inst.vf_prod term ti yi args dt
          (ks.push ki, stageYs.push yi, ok))
        (#[], #[], true)
    let y1 := y0 + weightedSum rk.tableau.b ks y0
    let yErr :=
      match rk.tableau.bErr with
      | none => none
      | some bErr =>
          let high := weightedSum rk.tableau.b ks y0
          let low := weightedSum bErr ks y0
          some (high - low)
    let m0 := inst.vf_prod term t0 y0 args dt
    let m1 := inst.vf_prod term t1 y1 args dt
    let dense := torch.DiffEq.denseInfo rk.denseKind cs t0 t1 dt y0 y1 m0 m1 stageYs ks
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
  interpolation := fun info => ImplicitDenseInfo.toInterpolation info
}

end ImplicitRK

namespace IMEXRK

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  0.0 * y0

private def weightedSum {s : Nat} [DiffEqSpace Y] (coeffs : Vector s Time)
    (ks : Array Y) (y0 : Y) : Y :=
  let coeffArr := coeffs.toArray
  (List.range coeffArr.size).foldl
    (fun acc j =>
      let a := coeffArr.getD j 0.0
      let kj := ks.getD j (zeroLike y0)
      acc + a * kj)
    (zeroLike y0)

private def effectiveRootMethod {s : Nat} (rk : IMEXRK s) : RootFindMethod :=
  let base : RootFindMethod := rk.rootMethod.getD (RootFindMethod.fixedPoint rk.rootFinder)
  RootFindMethod.withTolerances base rk.rootRtol rk.rootAtol rk.rootMaxIters

def solver {s : Nat} {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (rk : IMEXRK s) :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args := {
  SolverState := Unit
  DenseInfo := ImplicitDenseInfo Y
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
    let (ksExp, ksImp, stageYs, ok) :=
      (List.range rowsE.size).foldl
        (fun acc i =>
          let (ksExp, ksImp, stageYs, ok) := acc
          let rowE := rowsE.getD i #[]
          let rowI := rowsI.getD i #[]
          let sum :=
            (List.range i).foldl
              (fun acc j =>
                let aE := rowE.getD j 0.0
                let aI := rowI.getD j 0.0
                let kE := ksExp.getD j zero
                let kI := ksImp.getD j zero
                let sumE := aE * kE
                let sumI := aI * kI
                acc + (sumE + sumI))
              zero
          let ti := t0 + cs.getD i 0.0 * dt
          let base := y0 + sum
          let aii := rowI.getD i 0.0
          let (yi, ok) :=
            if aii == 0.0 then
              (base, ok)
            else
              let stepFn := fun y =>
                base + aii * (impInst.vf_prod implicit ti y args dt)
              let rootMethod : RootFindMethod := effectiveRootMethod rk
              let sol := RootFinder.solve (R := RootFindMethod) rootMethod stepFn base
              (sol.value, ok && sol.converged)
          let kE := expInst.vf_prod explicit ti yi args dt
          let kI := impInst.vf_prod implicit ti yi args dt
          (ksExp.push kE, ksImp.push kI, stageYs.push yi, ok))
        (#[], #[], #[], true)
    let yHigh :=
      y0 + ((weightedSum rk.explicit.b ksExp y0) + (weightedSum rk.implicit.b ksImp y0))
    let yErr :=
      match rk.explicit.bErr, rk.implicit.bErr with
      | some bErrE, some bErrI =>
          let high :=
            (weightedSum rk.explicit.b ksExp y0) + (weightedSum rk.implicit.b ksImp y0)
          let low :=
            (weightedSum bErrE ksExp y0) + (weightedSum bErrI ksImp y0)
          some (high - low)
      | _, _ => none
    let m0 :=
      (expInst.vf_prod explicit t0 y0 args dt) + (impInst.vf_prod implicit t0 y0 args dt)
    let m1 :=
      (expInst.vf_prod explicit t1 yHigh args dt) + (impInst.vf_prod implicit t1 yHigh args dt)
    let stageMs : Array Y := Id.run do
      let mut ms : Array Y := #[]
      for i in [:cs.size] do
        ms := ms.push ((ksExp.getD i zero) + (ksImp.getD i zero))
      return ms
    let dense := torch.DiffEq.denseInfo rk.denseKind cs t0 t1 dt y0 yHigh m0 m1 stageYs stageMs
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
  interpolation := fun info => ImplicitDenseInfo.toInterpolation info
}

end IMEXRK

end DiffEq
end torch
