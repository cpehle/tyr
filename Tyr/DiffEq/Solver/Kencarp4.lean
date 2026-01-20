import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## KenCarp 4/3 IMEX Solver -/

structure Kencarp4 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def gamma : Float := 0.25
private def b1 : Float := 82889.0 / 524892.0
private def b2 : Float := 0.0
private def b3 : Float := 15625.0 / 83664.0
private def b4 : Float := 69875.0 / 102672.0
private def b5 : Float := -2260.0 / 8211.0
private def b1Emb : Float := 4586570599.0 / 29645900160.0
private def b2Emb : Float := 0.0
private def b3Emb : Float := 178811875.0 / 945068544.0
private def b4Emb : Float := 814220225.0 / 1159782912.0
private def b5Emb : Float := -3700637.0 / 11593932.0
private def b6Emb : Float := 61727.0 / 225920.0
private def b1Err : Float := b1 - b1Emb
private def b2Err : Float := b2 - b2Emb
private def b3Err : Float := b3 - b3Emb
private def b4Err : Float := b4 - b4Emb
private def b5Err : Float := b5 - b5Emb
private def b6Err : Float := gamma - b6Emb

private def kencarp4Explicit : ButcherTableau 6 := {
  a := vec6
    #[]
    #[0.5]
    #[13861.0 / 62500.0, 6889.0 / 62500.0]
    #[-116923316275.0 / 2393684061468.0, -2731218467317.0 / 15368042101831.0,
      9408046702089.0 / 11113171139209.0]
    #[-451086348788.0 / 2902428689909.0, -2682348792572.0 / 7519795681897.0,
      12662868775082.0 / 11960479115383.0, 3355817975965.0 / 11060851509271.0]
    #[647845179188.0 / 3216320057751.0, 73281519250.0 / 8382639484533.0,
      552539513391.0 / 3454668386233.0, 3354512671639.0 / 8306763924573.0,
      4040.0 / 17871.0]
  b := vec6 b1 b2 b3 b4 b5 gamma
  c := vec6 0.0 0.5 (83.0 / 250.0) (31.0 / 50.0) (17.0 / 20.0) 1.0
  bErr := some (vec6 b1Err b2Err b3Err b4Err b5Err b6Err)
  order := 4
}

private def kencarp4Implicit : ButcherTableau 6 := {
  a := vec6
    #[]
    #[gamma, gamma]
    #[8611.0 / 62500.0, -1743.0 / 31250.0, gamma]
    #[5012029.0 / 34652500.0, -654441.0 / 2922500.0, 174375.0 / 388108.0, gamma]
    #[15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
      730878875.0 / 902184768.0, 2285395.0 / 8070912.0, gamma]
    #[b1, b2, b3, b4, b5, gamma]
  b := vec6 b1 b2 b3 b4 b5 gamma
  c := vec6 0.0 0.5 (83.0 / 250.0) (31.0 / 50.0) (17.0 / 20.0) 1.0
  bErr := some (vec6 b1Err b2Err b3Err b4Err b5Err b6Err)
  order := 4
}

def Kencarp4.solver (cfg : Kencarp4 := {}) {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args :=
  IMEXRK.solver { explicit := kencarp4Explicit, implicit := kencarp4Implicit, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kencarp4 := ⟨True.intro⟩
instance : AdaptiveSolver Kencarp4 := ⟨True.intro⟩

end DiffEq
end torch
