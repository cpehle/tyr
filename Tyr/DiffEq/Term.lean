import Tyr.DiffEq.Types
import Tyr.DiffEq.Path

namespace torch
namespace DiffEq

/-! ## Term System

Terms capture vector fields, controls, and their bilinear interaction.
-/

structure AbstractTerm (Y VF Control Args : Type) where
  vf : Time → Y → Args → VF
  contr : Time → Time → Control
  prod : VF → Control → Y
  vf_prod : Time → Y → Args → Control → Y
  is_vf_expensive : Time → Time → Y → Args → Bool

/-- Typeclass view of terms for solver genericity. -/
class TermLike (τ : Type) (Y VF Control Args : Type) where
  vf : τ → Time → Y → Args → VF
  contr : τ → Time → Time → Control
  prod : τ → VF → Control → Y
  vf_prod : τ → Time → Y → Args → Control → Y
  is_vf_expensive : τ → Time → Time → Y → Args → Bool

/-- Additional interface for diffusion terms that can provide a Jacobian-vector product. -/
class DiffusionTermLike (τ : Type) (Y VF Control Args : Type) where
  jacobian_prod : τ → Time → Y → Args → Y

instance : TermLike (AbstractTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vf
  contr term := term.contr
  prod term := term.prod
  vf_prod term := term.vf_prod
  is_vf_expensive term := term.is_vf_expensive

namespace AbstractTerm

def ofTermLike {τ Y VF Control Args : Type} [TermLike τ Y VF Control Args] (term : τ) :
    AbstractTerm Y VF Control Args := {
  vf := (inferInstance : TermLike τ Y VF Control Args).vf term
  contr := (inferInstance : TermLike τ Y VF Control Args).contr term
  prod := (inferInstance : TermLike τ Y VF Control Args).prod term
  vf_prod := (inferInstance : TermLike τ Y VF Control Args).vf_prod term
  is_vf_expensive := (inferInstance : TermLike τ Y VF Control Args).is_vf_expensive term
}

end AbstractTerm

/-- ODE term: vector field with time as the control. -/
structure ODETerm (Y Args : Type) where
  vectorField : Time → Y → Args → Y

instance [DiffEqSpace Y] : TermLike (ODETerm Y Args) Y Y Time Args where
  vf term := term.vectorField
  contr _ t0 t1 := t1 - t0
  prod _ vf control := DiffEqSpace.scale control vf
  vf_prod term t y args control :=
    DiffEqSpace.scale control (term.vectorField t y args)
  is_vf_expensive _ _ _ _ _ := false

namespace ODETerm

def toAbstract [DiffEqSpace Y] (term : ODETerm Y Args) : AbstractTerm Y Y Time Args :=
  AbstractTerm.ofTermLike term

end ODETerm

/-- Control term: general control with a user-provided product. -/
structure ControlTerm (Y VF Control Args : Type) where
  vectorField : Time → Y → Args → VF
  control : Time → Time → Control
  prod : VF → Control → Y

instance : TermLike (ControlTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vectorField
  contr term := term.control
  prod term := term.prod
  vf_prod term t y args control := term.prod (term.vectorField t y args) control
  is_vf_expensive _ _ _ _ _ := false

namespace ControlTerm

def toAbstract (term : ControlTerm Y VF Control Args) : AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

def ofPath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) : ControlTerm Y VF Control Args :=
  {
    vectorField := vectorField
    control := fun t0 t1 => path.evaluate t0 (some t1) true
    prod := prod
  }

end ControlTerm

/-- Diffusion term with a Jacobian-vector product for Milstein-like solvers. -/
structure DiffusionTerm (Y VF Control Args : Type) where
  vectorField : Time → Y → Args → VF
  control : Time → Time → Control
  prod : VF → Control → Y
  jacobianProd : Time → Y → Args → Y

instance : TermLike (DiffusionTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vectorField
  contr term := term.control
  prod term := term.prod
  vf_prod term t y args control := term.prod (term.vectorField t y args) control
  is_vf_expensive _ _ _ _ _ := false

instance : DiffusionTermLike (DiffusionTerm Y VF Control Args) Y VF Control Args where
  jacobian_prod term := term.jacobianProd

namespace DiffusionTerm

def toAbstract (term : DiffusionTerm Y VF Control Args) : AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

def ofPath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (jacobianProd : Time → Y → Args → Y) :
    DiffusionTerm Y VF Control Args :=
  {
    vectorField := vectorField
    control := fun t0 t1 => path.evaluate t0 (some t1) true
    prod := prod
    jacobianProd := jacobianProd
  }

end DiffusionTerm

/-- Simple wrapper for forwarding term behavior. -/
structure WrapTerm (Term : Type) where
  term : Term

instance {Term Y VF Control Args : Type} [TermLike Term Y VF Control Args] :
    TermLike (WrapTerm Term) Y VF Control Args where
  vf t := (inferInstance : TermLike Term Y VF Control Args).vf t.term
  contr t := (inferInstance : TermLike Term Y VF Control Args).contr t.term
  prod t := (inferInstance : TermLike Term Y VF Control Args).prod t.term
  vf_prod t := (inferInstance : TermLike Term Y VF Control Args).vf_prod t.term
  is_vf_expensive t := (inferInstance : TermLike Term Y VF Control Args).is_vf_expensive t.term

instance {Term Y VF Control Args : Type} [DiffusionTermLike Term Y VF Control Args] :
    DiffusionTermLike (WrapTerm Term) Y VF Control Args where
  jacobian_prod t := (inferInstance : DiffusionTermLike Term Y VF Control Args).jacobian_prod t.term

namespace WrapTerm

def toAbstract {Term Y VF Control Args : Type} [TermLike Term Y VF Control Args]
    (term : WrapTerm Term) :
    AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

end WrapTerm

/-- Additive combination of two terms (pair version). -/
structure MultiTerm (T1 T2 : Type) where
  term1 : T1
  term2 : T2

instance {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args]
    [DiffEqSpace Y] :
    TermLike (MultiTerm T1 T2) Y (VF1 × VF2) (C1 × C2) Args where
  vf term t y args :=
    ((inferInstance : TermLike T1 Y VF1 C1 Args).vf term.term1 t y args,
     (inferInstance : TermLike T2 Y VF2 C2 Args).vf term.term2 t y args)
  contr term t0 t1 :=
    ((inferInstance : TermLike T1 Y VF1 C1 Args).contr term.term1 t0 t1,
     (inferInstance : TermLike T2 Y VF2 C2 Args).contr term.term2 t0 t1)
  prod term vf control :=
    let y1 := (inferInstance : TermLike T1 Y VF1 C1 Args).prod term.term1 vf.1 control.1
    let y2 := (inferInstance : TermLike T2 Y VF2 C2 Args).prod term.term2 vf.2 control.2
    DiffEqSpace.add y1 y2
  vf_prod term t y args control :=
    let y1 := (inferInstance : TermLike T1 Y VF1 C1 Args).vf_prod term.term1 t y args control.1
    let y2 := (inferInstance : TermLike T2 Y VF2 C2 Args).vf_prod term.term2 t y args control.2
    DiffEqSpace.add y1 y2
  is_vf_expensive term t0 t1 y args :=
    (inferInstance : TermLike T1 Y VF1 C1 Args).is_vf_expensive term.term1 t0 t1 y args ||
    (inferInstance : TermLike T2 Y VF2 C2 Args).is_vf_expensive term.term2 t0 t1 y args

namespace MultiTerm

def toAbstract {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args] [DiffEqSpace Y]
    (term : MultiTerm T1 T2) :
    AbstractTerm Y (VF1 × VF2) (C1 × C2) Args :=
  AbstractTerm.ofTermLike term

end MultiTerm

end DiffEq
end torch
