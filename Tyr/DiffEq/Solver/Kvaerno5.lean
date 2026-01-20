import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## Kvaerno 5/4 Solver -/

structure Kvaerno5 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def gamma : Float := 0.26
private def a21 : Float := gamma
private def a31 : Float := 0.13
private def a32 : Float := 0.84033320996790809
private def a41 : Float := 0.22371961478320505
private def a42 : Float := 0.47675532319799699
private def a43 : Float := -0.06470895363112615
private def a51 : Float := 0.16648564323248321
private def a52 : Float := 0.10450018841591720
private def a53 : Float := 0.03631482272098715
private def a54 : Float := -0.13090704451073998
private def a61 : Float := 0.13855640231268224
private def a62 : Float := 0.0
private def a63 : Float := -0.04245337201752043
private def a64 : Float := 0.02446657898003141
private def a65 : Float := 0.61943039072480676
private def a71 : Float := 0.13659751177640291
private def a72 : Float := 0.0
private def a73 : Float := -0.05496908796538376
private def a74 : Float := -0.04118626728321046
private def a75 : Float := 0.62993304899016403
private def a76 : Float := 0.06962479448202728

def kvaerno5Tableau : ButcherTableau 7 := {
  a := vec7
    #[]
    #[a21, gamma]
    #[a31, a32, gamma]
    #[a41, a42, a43, gamma]
    #[a51, a52, a53, a54, gamma]
    #[a61, a62, a63, a64, a65, gamma]
    #[a71, a72, a73, a74, a75, a76, gamma]
  b := vec7 a71 a72 a73 a74 a75 a76 gamma
  c := vec7 0.0 0.52 1.230333209967908 0.8957659843500759 0.43639360985864756 1.0 1.0
  bErr := some (vec7 (a71 - a61) (a72 - a62) (a73 - a63) (a74 - a64) (a75 - a65) (a76 - gamma) gamma)
  order := 5
}

def Kvaerno5.solver (cfg : Kvaerno5 := {}) {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver Term Y VF Time Args :=
  ImplicitRK.solver { tableau := kvaerno5Tableau, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kvaerno5 := ⟨True.intro⟩
instance : AdaptiveSolver Kvaerno5 := ⟨True.intro⟩

end DiffEq
end torch
