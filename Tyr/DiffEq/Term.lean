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

/-- Optional structural metadata for terms (arity/layout hints). -/
class TermShape (τ : Type) where
  arity? : τ → Option Nat
  layoutTag? : τ → Option String := fun _ => none

namespace TermShape

def combineArity (lhs rhs : Option Nat) : Option Nat :=
  match lhs, rhs with
  | some n1, some n2 => some (n1 + n2)
  | _, _ => none

end TermShape

instance : TermLike (AbstractTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vf
  contr term := term.contr
  prod term := term.prod
  vf_prod term := term.vf_prod
  is_vf_expensive term := term.is_vf_expensive

instance : TermShape (AbstractTerm Y VF Control Args) where
  arity? _ := some 1
  layoutTag? _ := some "single"

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

instance : TermShape (ODETerm Y Args) where
  arity? _ := some 1
  layoutTag? _ := some "single"

namespace ODETerm

def toAbstract [DiffEqSpace Y] (term : ODETerm Y Args) : AbstractTerm Y Y Time Args :=
  AbstractTerm.ofTermLike term

end ODETerm

/-- Control term: general control with a user-provided product. -/
structure ControlTerm (Y VF Control Args : Type) where
  vectorField : Time → Y → Args → VF
  control : Time → Time → Control
  prod : VF → Control → Y
  controlDerivative? : Option (Time → Bool → Control) := none

instance : TermLike (ControlTerm Y VF Control Args) Y VF Control Args where
  vf term := term.vectorField
  contr term := term.control
  prod term := term.prod
  vf_prod term t y args control := term.prod (term.vectorField t y args) control
  is_vf_expensive _ _ _ _ _ := false

instance : TermShape (ControlTerm Y VF Control Args) where
  arity? _ := some 1
  layoutTag? _ := some "single"

namespace ControlTerm

def toAbstract (term : ControlTerm Y VF Control Args) : AbstractTerm Y VF Control Args :=
  AbstractTerm.ofTermLike term

def withControlDerivative (term : ControlTerm Y VF Control Args)
    (controlDerivative : Time → Bool → Control) : ControlTerm Y VF Control Args :=
  { term with controlDerivative? := some controlDerivative }

def clearControlDerivative (term : ControlTerm Y VF Control Args) : ControlTerm Y VF Control Args :=
  { term with controlDerivative? := none }

def toODEWithDerivative (term : ControlTerm Y VF Control Args)
    (controlDerivative : Time → Bool → Control) (left : Bool := true) :
    ODETerm Y Args :=
  {
    vectorField := fun t y args =>
      term.prod (term.vectorField t y args) (controlDerivative t left)
  }

def toODE? (term : ControlTerm Y VF Control Args) (left : Bool := true) :
    Option (ODETerm Y Args) :=
  match term.controlDerivative? with
  | some controlDerivative => some (toODEWithDerivative term controlDerivative left)
  | none => none

def ofPath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) : ControlTerm Y VF Control Args :=
  {
    vectorField := vectorField
    control := fun t0 t1 => path.evaluate t0 (some t1) true
    prod := prod
    controlDerivative? := path.derivativeFn?
  }

def ofDifferentiablePath (vectorField : Time → Y → Args → VF) (path : AbstractPath Control)
    (prod : VF → Control → Y) (controlDerivative : Time → Bool → Control) :
    ControlTerm Y VF Control Args :=
  withControlDerivative (ofPath vectorField path prod) controlDerivative

def derivativeAware? (term : ControlTerm Y VF Control Args) : Bool :=
  term.controlDerivative?.isSome

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

instance : TermShape (DiffusionTerm Y VF Control Args) where
  arity? _ := some 1
  layoutTag? _ := some "single"

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

instance {Term : Type} [TermShape Term] : TermShape (WrapTerm Term) where
  arity? t := (inferInstance : TermShape Term).arity? t.term
  layoutTag? t := (inferInstance : TermShape Term).layoutTag? t.term

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

abbrev MultiTerm3 (T1 T2 T3 : Type) : Type :=
  MultiTerm (MultiTerm T1 T2) T3

abbrev MultiTerm4 (T1 T2 T3 T4 : Type) : Type :=
  MultiTerm (MultiTerm3 T1 T2 T3) T4

abbrev MultiTerm5 (T1 T2 T3 T4 T5 : Type) : Type :=
  MultiTerm (MultiTerm4 T1 T2 T3 T4) T5

abbrev MultiTerm6 (T1 T2 T3 T4 T5 T6 : Type) : Type :=
  MultiTerm (MultiTerm5 T1 T2 T3 T4 T5) T6

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

instance {T1 T2 : Type} [TermShape T1] [TermShape T2] : TermShape (MultiTerm T1 T2) where
  arity? term :=
    TermShape.combineArity
      ((inferInstance : TermShape T1).arity? term.term1)
      ((inferInstance : TermShape T2).arity? term.term2)
  layoutTag? _ := some "pair"

namespace MultiTerm

def toAbstract {T1 T2 Y VF1 VF2 C1 C2 Args : Type}
    [TermLike T1 Y VF1 C1 Args] [TermLike T2 Y VF2 C2 Args] [DiffEqSpace Y]
    (term : MultiTerm T1 T2) :
    AbstractTerm Y (VF1 × VF2) (C1 × C2) Args :=
  AbstractTerm.ofTermLike term

def append (term : MultiTerm T1 T2) (term3 : T3) : MultiTerm3 T1 T2 T3 :=
  { term1 := term, term2 := term3 }

def prepend (term0 : T0) (term : MultiTerm T1 T2) : MultiTerm3 T0 T1 T2 :=
  { term1 := { term1 := term0, term2 := term.term1 }, term2 := term.term2 }

def of3 (term1 : T1) (term2 : T2) (term3 : T3) : MultiTerm3 T1 T2 T3 :=
  append { term1 := term1, term2 := term2 } term3

def of4 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) : MultiTerm4 T1 T2 T3 T4 :=
  append (of3 term1 term2 term3) term4

def of5 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) (term5 : T5) :
    MultiTerm5 T1 T2 T3 T4 T5 :=
  append (of4 term1 term2 term3 term4) term5

def of6 (term1 : T1) (term2 : T2) (term3 : T3) (term4 : T4) (term5 : T5) (term6 : T6) :
    MultiTerm6 T1 T2 T3 T4 T5 T6 :=
  append (of5 term1 term2 term3 term4 term5) term6

def reassocRight (term : MultiTerm (MultiTerm T1 T2) T3) : MultiTerm T1 (MultiTerm T2 T3) :=
  {
    term1 := term.term1.term1
    term2 := {
      term1 := term.term1.term2
      term2 := term.term2
    }
  }

def reassocLeft (term : MultiTerm T1 (MultiTerm T2 T3)) : MultiTerm (MultiTerm T1 T2) T3 :=
  {
    term1 := {
      term1 := term.term1
      term2 := term.term2.term1
    }
    term2 := term.term2.term2
  }

end MultiTerm

/-- Homogeneous additive combination of 1+ terms (array-like version). -/
structure MultiTermArray (Term : Type) where
  head : Term
  tail : Array Term := #[]

instance {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args] [DiffEqSpace Y]
    [Inhabited Y] :
    TermLike (MultiTermArray Term) Y (Array VF) (Array Control) Args where
  vf terms t y args :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    (#[terms.head] ++ terms.tail).map (fun term => termLike.vf term t y args)
  contr terms t0 t1 :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    (#[terms.head] ++ terms.tail).map (fun term => termLike.contr term t0 t1)
  prod terms vf control :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    match vf[0]?, control[0]? with
    | some vf0, some control0 =>
        let y0 := termLike.prod terms.head vf0 control0
        Id.run do
          let mut acc := y0
          for offset in [:terms.tail.size] do
            let i := offset + 1
            match terms.tail[offset]?, vf[i]?, control[i]? with
            | some term, some vf_i, some control_i =>
                acc := DiffEqSpace.add acc (termLike.prod term vf_i control_i)
            | _, _, _ => ()
          acc
    | _, _ => default
  vf_prod terms t y args control :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    match control[0]? with
    | some control0 =>
        let y0 := termLike.vf_prod terms.head t y args control0
        Id.run do
          let mut acc := y0
          for offset in [:terms.tail.size] do
            let i := offset + 1
            match terms.tail[offset]?, control[i]? with
            | some term, some control_i =>
                acc := DiffEqSpace.add acc (termLike.vf_prod term t y args control_i)
            | _, _ => ()
          acc
    | none => default
  is_vf_expensive terms t0 t1 y args :=
    let termLike := (inferInstance : TermLike Term Y VF Control Args)
    Id.run do
      let mut expensive := termLike.is_vf_expensive terms.head t0 t1 y args
      for offset in [:terms.tail.size] do
        match terms.tail[offset]? with
        | some term => expensive := expensive || termLike.is_vf_expensive term t0 t1 y args
        | none => ()
      expensive

instance {Term : Type} : TermShape (MultiTermArray Term) where
  arity? terms := some (terms.tail.size + 1)
  layoutTag? _ := some "array"

namespace MultiTermArray

def singleton (term : Term) : MultiTermArray Term :=
  { head := term }

def toArray (terms : MultiTermArray Term) : Array Term :=
  #[terms.head] ++ terms.tail

def size (terms : MultiTermArray Term) : Nat :=
  terms.tail.size + 1

def push (terms : MultiTermArray Term) (term : Term) : MultiTermArray Term :=
  { terms with tail := terms.tail.push term }

def appendArray (terms : MultiTermArray Term) (extraTerms : Array Term) : MultiTermArray Term :=
  { terms with tail := terms.tail ++ extraTerms }

def ofArray? (terms : Array Term) : Option (MultiTermArray Term) :=
  match terms[0]? with
  | none => none
  | some head =>
      some {
        head := head
        tail := terms.extract 1 terms.size
      }

def mapTerms (terms : MultiTermArray Term) (f : Term → Term₂) : MultiTermArray Term₂ :=
  {
    head := f terms.head
    tail := terms.tail.map f
  }

def toAbstract {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args] [DiffEqSpace Y]
    [Inhabited Y]
    (terms : MultiTermArray Term) :
    AbstractTerm Y (Array VF) (Array Control) Args :=
  AbstractTerm.ofTermLike terms

end MultiTermArray

end DiffEq
end torch
