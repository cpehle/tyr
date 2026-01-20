import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## Kvaerno 4/3 Solver -/

structure Kvaerno4 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def poly (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc c => acc * x + c) 0.0

private def gamma : Float := 0.5728160625
private def p12 : Float := poly [12.0, -6.0, 1.0] gamma
private def p3 : Float := poly [3.0, -1.0] gamma
private def p1292 : Float := poly [12.0, -9.0, 2.0] gamma
private def p66 : Float := poly [6.0, -6.0, 1.0] gamma
private def a21 : Float := gamma
private def a31 : Float := (poly [144.0, -180.0, 81.0, -15.0, 1.0] gamma * gamma) / (p12 * p12)
private def a32 : Float := (poly [-36.0, 39.0, -15.0, 2.0] gamma * gamma) / (p12 * p12)
private def a41 : Float := poly [-144.0, 396.0, -330.0, 117.0, -18.0, 1.0] gamma / (12.0 * gamma * gamma * p1292)
private def a42 : Float := poly [72.0, -126.0, 69.0, -15.0, 1.0] gamma / (12.0 * gamma * gamma * p3)
private def a43 : Float :=
  (poly [-6.0, 6.0, -1.0] gamma * p12 * p12) / (12.0 * gamma * gamma * p1292 * p3)
private def a51 : Float := poly [288.0, -312.0, 120.0, -18.0, 1.0] gamma / (48.0 * gamma * gamma * p1292)
private def a52 : Float := poly [24.0, -12.0, 1.0] gamma / (48.0 * gamma * gamma * p3)
private def a53 : Float :=
  -(p12 * p12 * p12) / (48.0 * gamma * gamma * p3 * p1292 * p66)
private def a54 : Float := poly [-24.0, 36.0, -12.0, 1.0] gamma / poly [24.0, -24.0, 4.0] gamma
private def c2 : Float := gamma + a21
private def c3 : Float := gamma + a31 + a32

def kvaerno4Tableau : ButcherTableau 5 := {
  a := vec5
    #[]
    #[a21, gamma]
    #[a31, a32, gamma]
    #[a41, a42, a43, gamma]
    #[a51, a52, a53, a54, gamma]
  b := vec5 a51 a52 a53 a54 gamma
  c := vec5 0.0 c2 c3 1.0 1.0
  bErr := some (vec5 (a51 - a41) (a52 - a42) (a53 - a43) (a54 - gamma) gamma)
  order := 4
}

def Kvaerno4.solver (cfg : Kvaerno4 := {}) {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver Term Y VF Time Args :=
  ImplicitRK.solver { tableau := kvaerno4Tableau, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kvaerno4 := ⟨True.intro⟩
instance : AdaptiveSolver Kvaerno4 := ⟨True.intro⟩

end DiffEq
end torch
