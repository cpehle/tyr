import Tyr.Torch
import Tyr.TensorStruct

/-!
  Minimal Flowfusion-style abstractions.

  This module provides lightweight wrappers used by BranchingFlows-like code:
  - MaskedState: state with conditioning/loss masks
  - Guide: wrapper for alternative prediction targets

  These are intentionally minimal and avoid committing to a full Flowfusion port.
-/

namespace torch.flowfusion

/-- Wrap a state with conditioning and loss masks. Masks are stored as `Static`
    so they are skipped by tensor traversal utilities. -/
structure MaskedState (α : Type) where
  state : α
  cmask : Static (T #[])
  lmask : Static (T #[])
  deriving Repr

instance [TensorStruct α] : TensorStruct (MaskedState α) where
  map f x := { x with state := TensorStruct.map f x.state }
  mapM f x := do
    let state' ← TensorStruct.mapM f x.state
    pure { x with state := state' }
  zipWith f x y := { x with state := TensorStruct.zipWith f x.state y.state }
  fold f init x := TensorStruct.fold f init x.state

namespace MaskedState

/-- Convenience constructor from raw masks. -/
def ofMasks (state : α) (cmask lmask : T #[]) : MaskedState α :=
  { state, cmask := (cmask : Static (T #[])), lmask := (lmask : Static (T #[])) }

def getCmask (m : MaskedState α) : T #[] := m.cmask.val
def getLmask (m : MaskedState α) : T #[] := m.lmask.val

end MaskedState

/-- Wrapper for alternative prediction targets (e.g., tangent vectors).
    Masks are optional and stored as `Static`. -/
structure Guide (α : Type) where
  value : α
  cmask : Option (Static (T #[])) := none
  lmask : Option (Static (T #[])) := none
  deriving Repr

instance [TensorStruct α] : TensorStruct (Guide α) where
  map f x := { x with value := TensorStruct.map f x.value }
  mapM f x := do
    let value' ← TensorStruct.mapM f x.value
    pure { x with value := value' }
  zipWith f x y := { x with value := TensorStruct.zipWith f x.value y.value }
  fold f init x := TensorStruct.fold f init x.value

namespace Guide

def getCmask (g : Guide α) : Option (T #[]) := g.cmask.map (·.val)
def getLmask (g : Guide α) : Option (T #[]) := g.lmask.map (·.val)

end Guide

/-- Unwrap a masked state to its underlying state. -/
def unmask (m : MaskedState α) : α := m.state

/-- Wrap a state with the masks from another masked state. -/
def maskLike (x : α) (m : MaskedState α) : MaskedState α :=
  MaskedState.ofMasks x (MaskedState.getCmask m) (MaskedState.getLmask m)

end torch.flowfusion
