import Tyr.DiffEq.Solver.ImplicitRungeKutta

namespace torch
namespace DiffEq

/-! ## KenCarp 5/4 IMEX Solver -/

structure Kencarp5 where
  rootFinder : FixedPoint := {}
  deriving Inhabited

private def gamma : Float := 41.0 / 200.0
private def b1 : Float := -872700587467.0 / 9133579230613.0
private def b2 : Float := 0.0
private def b3 : Float := 0.0
private def b4 : Float := 22348218063261.0 / 9555858737531.0
private def b5 : Float := -1143369518992.0 / 8141816002931.0
private def b6 : Float := -39379526789629.0 / 19018526304540.0
private def b7 : Float := 32727382324388.0 / 42900044865799.0
private def b1Emb : Float := -975461918565.0 / 9796059967033.0
private def b2Emb : Float := 0.0
private def b3Emb : Float := 0.0
private def b4Emb : Float := 78070527104295.0 / 32432590147079.0
private def b5Emb : Float := -548382580838.0 / 3424219808633.0
private def b6Emb : Float := -33438840321285.0 / 15594753105479.0
private def b7Emb : Float := 3629800801594.0 / 4656183773603.0
private def b8Emb : Float := 4035322873751.0 / 18575991585200.0
private def b1Err : Float := b1 - b1Emb
private def b2Err : Float := b2 - b2Emb
private def b3Err : Float := b3 - b3Emb
private def b4Err : Float := b4 - b4Emb
private def b5Err : Float := b5 - b5Emb
private def b6Err : Float := b6 - b6Emb
private def b7Err : Float := b7 - b7Emb
private def b8Err : Float := gamma - b8Emb

private def kencarp5Explicit : ButcherTableau 8 := {
  a := vec8
    #[]
    #[41.0 / 100.0]
    #[367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0]
    #[1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0]
    #[14463281900351.0 / 6315353703477.0, 0.0,
      66114435211212.0 / 5879490589093.0, -54053170152839.0 / 4284798021562.0]
    #[14090043504691.0 / 34967701212078.0, 0.0,
      15191511035443.0 / 11219624916014.0, -18461159152457.0 / 12425892160975.0,
      -281667163811.0 / 9011619295870.0]
    #[19230459214898.0 / 13134317526959.0, 0.0,
      21275331358303.0 / 2942455364971.0, -38145345988419.0 / 4862620318723.0,
      -1.0 / 8.0, -1.0 / 8.0]
    #[-19977161125411.0 / 11928030595625.0, 0.0,
      -40795976796054.0 / 6384907823539.0, 177454434618887.0 / 12078138498510.0,
      782672205425.0 / 8267701900261.0, -69563011059811.0 / 9646580694205.0,
      7356628210526.0 / 4942186776405.0]
  b := vec8 b1 b2 b3 b4 b5 b6 b7 gamma
  c := vec8 0.0 (41.0 / 100.0) (2935347310677.0 / 11292855782101.0)
    (1426016391358.0 / 7196633302097.0) (92.0 / 100.0) (24.0 / 100.0) (3.0 / 5.0) 1.0
  bErr := some (vec8 b1Err b2Err b3Err b4Err b5Err b6Err b7Err b8Err)
  order := 5
}

private def kencarp5Implicit : ButcherTableau 8 := {
  a := vec8
    #[]
    #[gamma, gamma]
    #[41.0 / 400.0, -567603406766.0 / 11931857230679.0, gamma]
    #[683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, gamma]
    #[3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0,
      -22760509404356.0 / 11113319521817.0, gamma]
    #[218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0,
      -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, gamma]
    #[1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0,
      -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0,
      -4269925059573.0 / 7827059040719.0, gamma]
    #[b1, b2, b3, b4, b5, b6, b7, gamma]
  b := vec8 b1 b2 b3 b4 b5 b6 b7 gamma
  c := vec8 0.0 (41.0 / 100.0) (2935347310677.0 / 11292855782101.0)
    (1426016391358.0 / 7196633302097.0) (92.0 / 100.0) (24.0 / 100.0) (3.0 / 5.0) 1.0
  bErr := some (vec8 b1Err b2Err b3Err b4Err b5Err b6Err b7Err b8Err)
  order := 5
}

def Kencarp5.solver (cfg : Kencarp5 := {}) {ExplicitTerm ImplicitTerm Y VFe VFi Args : Type}
    [TermLike ExplicitTerm Y VFe Time Args]
    [TermLike ImplicitTerm Y VFi Time Args]
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    AbstractSolver (MultiTerm ExplicitTerm ImplicitTerm) Y (VFe × VFi) (Time × Time) Args :=
  IMEXRK.solver { explicit := kencarp5Explicit, implicit := kencarp5Implicit, rootFinder := cfg.rootFinder }

instance : ImplicitSolver Kencarp5 := ⟨True.intro⟩
instance : AdaptiveSolver Kencarp5 := ⟨True.intro⟩

end DiffEq
end torch
