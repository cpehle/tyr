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
  deriving Repr, TensorStruct

namespace MaskedState

/-- Convenience constructor from raw masks. -/
def mk (state : α) (cmask lmask : T #[]) : MaskedState α :=
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
  deriving Repr, TensorStruct

namespace Guide

def getCmask (g : Guide α) : Option (T #[]) := g.cmask.map (·.val)
def getLmask (g : Guide α) : Option (T #[]) := g.lmask.map (·.val)

end Guide

/-- Unwrap a masked state to its underlying state. -/
def unmask (m : MaskedState α) : α := m.state

/-- Wrap a state with the masks from another masked state. -/
def maskLike (x : α) (m : MaskedState α) : MaskedState α :=
  MaskedState.mk x (MaskedState.getCmask m) (MaskedState.getLmask m)

end torch.flowfusion
