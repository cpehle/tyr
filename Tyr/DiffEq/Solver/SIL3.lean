import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## SIL3 IMEX Solver -/

structure SIL3 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def sil3Explicit : ButcherTableau 4 := {
  a := vec4
    #[]
    #[1.0 / 3.0]
    #[1.0 / 6.0, 0.5]
    #[0.5, -0.5, 1.0]
  b := vec4 0.5 (-0.5) 1.0 0.0
  c := vec4 0.0 (1.0 / 3.0) (2.0 / 3.0) 1.0
  bErr := some (vec4 0.0 0.5 (-1.0) 0.5)
  order := 2
}

private def sil3Implicit : ButcherTableau 4 := {
  a := vec4
    #[]
    #[1.0 / 6.0, 1.0 / 6.0]
    #[1.0 / 3.0, 0.0, 1.0 / 3.0]
    #[3.0 / 8.0, 0.0, 3.0 / 8.0, 1.0 / 4.0]
  b := vec4 (3.0 / 8.0) 0.0 (3.0 / 8.0) (1.0 / 4.0)
  c := vec4 0.0 (1.0 / 3.0) (2.0 / 3.0) 1.0
  bErr := some (vec4 (1.0 / 8.0) 0.0 (-3.0 / 8.0) (1.0 / 4.0))
  order := 2
}

def SIL3.solver (cfg : SIL3 := {}) {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args :=
  IMEXRK.solver { explicit := sil3Explicit, implicit := sil3Implicit, rootFinder := cfg.rootFinder }

instance : ImplicitSolver SIL3 := ⟨True.intro⟩
instance : AdaptiveSolver SIL3 := ⟨True.intro⟩

end DiffEq
end torch
