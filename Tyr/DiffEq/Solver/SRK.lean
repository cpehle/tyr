import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## Stochastic Runge-Kutta (Additive Noise)

Minimal SRK infrastructure for additive-noise SDEs using space-time Levy areas.
-/

structure AdditiveCoeffs (s : Nat) where
  a : Vector s Time
  bSol : Time
  bErr : Option Time := none

structure GeneralCoeffs (s : Nat) where
  a : Vector s (Array Time)
  bSol : Vector s Time
  bErr : Option (Vector s Time) := none

inductive StochasticCoeffs (s : Nat) where
  | additive : AdditiveCoeffs s → StochasticCoeffs s
  | general : GeneralCoeffs s → StochasticCoeffs s

structure StochasticButcherTableau (s : Nat) where
  a : Vector s (Array Time)
  bSol : Vector s Time
  bErr : Option (Vector s Time) := none
  c : Vector s Time
  coeffsW : StochasticCoeffs s
  coeffsH : Option (StochasticCoeffs s) := none
  order : Nat := 1
  strongOrder : Float := 0.0

structure StochasticRK (s : Nat) where
  tableau : StochasticButcherTableau s

namespace StochasticRK

private def zeroLike [DiffEqSpace Y] (y0 : Y) : Y :=
  DiffEqSpace.scale 0.0 y0

private def weightedSum {s : Nat} [DiffEqSpace Y] (coeffs : Vector s Time)
    (ks : Array Y) (y0 : Y) : Y := Id.run do
  let mut acc := zeroLike y0
  let coeffArr := coeffs.toArray
  for j in [:coeffArr.size] do
    let a := coeffArr.getD j 0.0
    let kj := ks.getD j (zeroLike y0)
    acc := DiffEqSpace.add acc (DiffEqSpace.scale a kj)
  return acc

private def weightedSumArray [DiffEqSpace Y] (coeffs : Array Time)
    (ks : Array Y) (zero : Y) : Y := Id.run do
  let mut acc := zero
  for j in [:coeffs.size] do
    let a := coeffs.getD j 0.0
    let kj := ks.getD j zero
    acc := DiffEqSpace.add acc (DiffEqSpace.scale a kj)
  return acc

def solver {s : Nat} {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] (rk : StochasticRK s) :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.multi
  order := fun _ => rk.tableau.order
  strongOrder := fun _ => rk.tableau.strongOrder
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y Y Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    let ctrlInst := (inferInstance : SpaceTimeLevyAreaLike Control Float)
    let buildInst := (inferInstance : SpaceTimeLevyAreaBuild Control Float)
    let dt := driftInst.contr drift t0 t1
    let control := diffInst.contr diffusion t0 t1
    let w := ctrlInst.W control
    let h := ctrlInst.H control
    let dtControl := ctrlInst.dt control
    let zero := zeroLike y0
    let rows := rk.tableau.a.toArray
    let cs := rk.tableau.c.toArray
    let (y1, yError) :=
      match rk.tableau.coeffsW with
      | StochasticCoeffs.additive coeffsW =>
          let coeffsH :=
            match rk.tableau.coeffsH with
            | some (StochasticCoeffs.additive coeffsH) => some coeffsH
            | _ => none
          let ctrlW := buildInst.build dtControl w 0.0
          let ctrlH := buildInst.build dtControl h 0.0
          let g0 := diffInst.vf diffusion t0 y0 args
          let g1 := diffInst.vf diffusion t1 y0 args
          let gDelta := DiffEqSpace.scale 0.5 (DiffEqSpace.sub g1 g0)
          let wKg := diffInst.prod diffusion g0 ctrlW
          let hKg :=
            match coeffsH with
            | some _ => diffInst.prod diffusion g0 ctrlH
            | none => DiffEqSpace.scale 0.0 wKg
          let coeffsWArr := coeffsW.a.toArray
          let ks := Id.run do
            let mut ks : Array Y := #[]
            for j in [:rows.size] do
              let row := rows.getD j #[]
              let mut driftSum := zero
              for i in [:row.size] do
                let aij := row.getD i 0.0
                let ki := ks.getD i zero
                driftSum := DiffEqSpace.add driftSum (DiffEqSpace.scale aij ki)
              let aW := coeffsWArr.getD j 0.0
              let mut diffSum := DiffEqSpace.scale aW wKg
              match coeffsH with
              | some coeffsH =>
                  let aH := coeffsH.a.toArray.getD j 0.0
                  diffSum := DiffEqSpace.add diffSum (DiffEqSpace.scale aH hKg)
              | none => pure ()
              let ti := t0 + cs.getD j 0.0 * dt
              let zi := DiffEqSpace.add y0 (DiffEqSpace.add driftSum diffSum)
              let ki := driftInst.vf_prod drift ti zi args dt
              ks := ks.push ki
            return ks
          let driftResult := weightedSum rk.tableau.bSol ks y0
          let diffResult :=
            match coeffsH with
            | some coeffsH =>
                let base := DiffEqSpace.add
                  (DiffEqSpace.scale coeffsW.bSol wKg)
                  (DiffEqSpace.scale coeffsH.bSol hKg)
                let ctrlTime := buildInst.build dtControl (w - 2.0 * h) 0.0
                let timeVarTerm := diffInst.prod diffusion gDelta ctrlTime
                DiffEqSpace.add base timeVarTerm
            | none =>
                let base := DiffEqSpace.scale coeffsW.bSol wKg
                let timeVarTerm := diffInst.prod diffusion gDelta ctrlW
                DiffEqSpace.add base timeVarTerm
          let y1 := DiffEqSpace.add y0 (DiffEqSpace.add driftResult diffResult)
          (y1, none)
      | StochasticCoeffs.general coeffsW =>
          let coeffsH :=
            match rk.tableau.coeffsH with
            | some (StochasticCoeffs.general coeffsH) => some coeffsH
            | _ => none
          let ctrlW := buildInst.build dtControl w 0.0
          let ctrlH := buildInst.build dtControl h 0.0
          let coeffsWArr := coeffsW.a.toArray
          let coeffsHArr :=
            match coeffsH with
            | some coeffsH => coeffsH.a.toArray
            | none => #[]
          let (ksF, ksW, ksH) := Id.run do
            let mut ksF : Array Y := #[]
            let mut ksW : Array Y := #[]
            let mut ksH : Array Y := #[]
            for j in [:rows.size] do
              let row := rows.getD j #[]
              let driftSum := weightedSumArray row ksF zero
              let aWRow := coeffsWArr.getD j #[]
              let wSum := weightedSumArray aWRow ksW zero
              let hSum :=
                match coeffsH with
                | some _ =>
                    let aHRow := coeffsHArr.getD j #[]
                    weightedSumArray aHRow ksH zero
                | none => zero
              let ti := t0 + cs.getD j 0.0 * dt
              let zi := DiffEqSpace.add y0 (DiffEqSpace.add driftSum (DiffEqSpace.add wSum hSum))
              let kf := driftInst.vf_prod drift ti zi args dt
              let kgW := diffInst.vf_prod diffusion ti zi args ctrlW
              let kgH :=
                match coeffsH with
                | some _ => diffInst.vf_prod diffusion ti zi args ctrlH
                | none => zero
              ksF := ksF.push kf
              ksW := ksW.push kgW
              ksH := ksH.push kgH
            return (ksF, ksW, ksH)
          let driftResult := weightedSum rk.tableau.bSol ksF y0
          let wResult := weightedSum coeffsW.bSol ksW y0
          let hResult :=
            match coeffsH with
            | some coeffsH => weightedSum coeffsH.bSol ksH y0
            | none => zero
          let diffResult := DiffEqSpace.add wResult hResult
          let y1 := DiffEqSpace.add y0 (DiffEqSpace.add driftResult diffResult)
          let yError :=
            match rk.tableau.bErr with
            | none => none
            | some bErr =>
                let driftErr := weightedSum bErr ksF y0
                let wErr :=
                  match coeffsW.bErr with
                  | some bWErr => weightedSum bWErr ksW y0
                  | none => zero
                let hErr :=
                  match coeffsH with
                  | some coeffsH =>
                      match coeffsH.bErr with
                      | some bHErr => weightedSum bHErr ksH y0
                      | none => zero
                  | none => zero
                let diffErr := DiffEqSpace.add wErr hErr
                some (DiffEqSpace.add driftErr diffErr)
          (y1, yError)
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := yError
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun terms t y args =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y Y Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    (driftInst.vf drift t y args, diffInst.vf diffusion t y args)
  interpolation := fun info => info.toInterpolation
}

end StochasticRK

def vec1 {α : Type} (a : α) : Vector 1 α := ⟨#[a], by simp⟩

end DiffEq
end torch
