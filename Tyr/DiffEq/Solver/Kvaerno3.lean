import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## Kvaerno 3/2 Solver -/

structure Kvaerno3 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def gamma : Float := 0.43586652150
private def a21 : Float := gamma
private def a31 : Float := (-4.0 * gamma * gamma + 6.0 * gamma - 1.0) / (4.0 * gamma)
private def a32 : Float := (-2.0 * gamma + 1.0) / (4.0 * gamma)
private def a41 : Float := (6.0 * gamma - 1.0) / (12.0 * gamma)
private def a42 : Float := -1.0 / ((24.0 * gamma - 12.0) * gamma)
private def a43 : Float := (-6.0 * gamma * gamma + 6.0 * gamma - 1.0) / (6.0 * gamma - 3.0)

def kvaerno3Tableau : ButcherTableau 4 := {
  a := vec4
    #[]
    #[a21, gamma]
    #[a31, a32, gamma]
    #[a41, a42, a43, gamma]
  b := vec4 a41 a42 a43 gamma
  c := vec4 0.0 (2.0 * gamma) 1.0 1.0
  bErr := some (vec4 (a41 - a31) (a42 - a32) (a43 - gamma) gamma)
  order := 3
}

def Kvaerno3.solver (cfg : Kvaerno3 := {}) {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver Term Y VF Time Args :=
  ImplicitRK.solver { tableau := kvaerno3Tableau, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kvaerno3 := ⟨True.intro⟩
instance : AdaptiveSolver Kvaerno3 := ⟨True.intro⟩

end DiffEq
end torch
