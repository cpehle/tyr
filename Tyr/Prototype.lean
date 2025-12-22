import Tyr.Torch

namespace prototype

open torch

/-- A typeclass for structures that contain tensors.
    Allows generic traversal and transformation of all tensors in the structure. -/
class TensorContainer (α : Type) where
  -- Map a function over all tensors
  map : (∀ {s}, T s → T s) → α → α
  -- Monadic map (e.g., for initialization with IO)
  mapM {m : Type → Type} [Monad m] : (∀ {s}, T s → m (T s)) → α → m α
  -- Zip two containers with a function
  zipWith : (∀ {s}, T s → T s → T s) → α → α → α

/-- A simple Linear layer -/
structure Linear (in_f out_f : UInt64) where
  weight : T #[out_f, in_f]
  bias : T #[out_f]
  deriving Repr

/-- Implementation of TensorContainer for Linear -/
instance {i o} : TensorContainer (Linear i o) where
  map f l := { weight := f l.weight, bias := f l.bias }
  mapM f l := do
    let w ← f l.weight
    let b ← f l.bias
    return { weight := w, bias := b }
  zipWith f l1 l2 := { weight := f l1.weight l2.weight, bias := f l1.bias l2.bias }

/-- Mock Optimizer State -/
structure AdamState (α : Type) [TensorContainer α] where
  moments : α
  velocities : α
  step : Nat

/-- Generic initialization -/
def init_random [TensorContainer α] (model : α) : IO α :=
  TensorContainer.mapM (fun {s} _ => torch.randn s) model

/-- Generic optimizer step (conceptual) -/
def step [TensorContainer α] (model : α) (grads : α) (state : AdamState α) : α × AdamState α :=
  let new_moments := TensorContainer.zipWith (fun m g => m + g) state.moments grads -- simplified
  let new_model := TensorContainer.zipWith (fun p m => p - m) model new_moments   -- simplified
  (new_model, { state with moments := new_moments })

end prototype
