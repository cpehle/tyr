import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## KenCarp 3/2 IMEX Solver -/

structure Kencarp3 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def gamma : Float := 1767732205903.0 / 4055673282236.0
private def b1 : Float := 1471266399579.0 / 7840856788654.0
private def b2 : Float := -4482444167858.0 / 7529755066697.0
private def b3 : Float := 11266239266428.0 / 11593286722821.0
private def b1Emb : Float := 2756255671327.0 / 12835298489170.0
private def b2Emb : Float := -10771552573575.0 / 22201958757719.0
private def b3Emb : Float := 9247589265047.0 / 10645013368117.0
private def b4Emb : Float := 2193209047091.0 / 5459859503100.0
private def b1Err : Float := b1 - b1Emb
private def b2Err : Float := b2 - b2Emb
private def b3Err : Float := b3 - b3Emb
private def b4Err : Float := gamma - b4Emb

private def kencarp3Explicit : ButcherTableau 4 := {
  a := vec4
    #[]
    #[2.0 * gamma]
    #[5535828885825.0 / 10492691773637.0, 788022342437.0 / 10882634858940.0]
    #[6485989280629.0 / 16251701735622.0, -4246266847089.0 / 9704473918619.0,
      10755448449292.0 / 10357097424841.0]
  b := vec4 b1 b2 b3 gamma
  c := vec4 0.0 (2.0 * gamma) (3.0 / 5.0) 1.0
  bErr := some (vec4 b1Err b2Err b3Err b4Err)
  order := 3
}

private def kencarp3Implicit : ButcherTableau 4 := {
  a := vec4
    #[]
    #[gamma, gamma]
    #[2746238789719.0 / 10658868560708.0, -640167445237.0 / 6845629431997.0, gamma]
    #[b1, b2, b3, gamma]
  b := vec4 b1 b2 b3 gamma
  c := vec4 0.0 (2.0 * gamma) (3.0 / 5.0) 1.0
  bErr := some (vec4 b1Err b2Err b3Err b4Err)
  order := 3
}

def Kencarp3.solver (cfg : Kencarp3 := {}) {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args :=
  IMEXRK.solver { explicit := kencarp3Explicit, implicit := kencarp3Implicit, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kencarp3 := ⟨True.intro⟩
instance : AdaptiveSolver Kencarp3 := ⟨True.intro⟩

end DiffEq
end torch
