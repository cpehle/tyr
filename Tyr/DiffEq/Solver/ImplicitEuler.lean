import Tyr.DiffEq.RootFinder
import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Implicit Euler Solver -/

structure ImplicitEuler where
  rootFinder : FixedPoint := {}
  deriving Inhabited

def ImplicitEuler.solver (cfg : ImplicitEuler := {}) {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => 1
  errorOrder := fun _ => 2.0
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    let dt := inst.contr term t0 t1
    let k0 := inst.vf_prod term t0 y0 args dt
    let stepFn := fun k =>
      inst.vf_prod term t1 (DiffEqSpace.add y0 k) args dt
    let sol := RootFinder.solve cfg.rootFinder stepFn k0
    let k1 := sol.value
    let y1 := DiffEqSpace.add y0 k1
    let yErr := DiffEqSpace.scale 0.5 (DiffEqSpace.sub k1 k0)
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := some yErr
      denseInfo := dense
      solverState := state
      result := if sol.converged then Result.successful else Result.internalError
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ImplicitSolver ImplicitEuler := ⟨True.intro⟩
instance : AdaptiveSolver ImplicitEuler := ⟨True.intro⟩

end DiffEq
end torch
