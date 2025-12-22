import Tyr.Torch

namespace torch

/-! ## Static and Frozen Wrappers

These wrapper types control how fields are handled during tensor tree traversal:
- `Static α`: Non-tensor data that should be completely skipped (e.g., config, hyperparameters)
- `Frozen s`: Tensor that participates in forward pass but not in gradient updates
-/

/-- Non-tensor static data that should be skipped during traversal.
    Use this for configuration, hyperparameters, or any non-tensor metadata. -/
structure Static (α : Type) where
  val : α
  deriving Repr, BEq, Hashable

instance [Inhabited α] : Inhabited (Static α) := ⟨⟨default⟩⟩

/-- Coercion from α to Static α -/
instance : Coe α (Static α) := ⟨Static.mk⟩

/-- A frozen (non-trainable) tensor parameter.
    Participates in forward pass but gradients are not tracked for optimization. -/
structure Frozen (s : Shape) where
  tensor : T s

namespace Frozen

def map {s : Shape} (f : T s → T s) (fr : Frozen s) : Frozen s :=
  { tensor := f fr.tensor }

def get {s : Shape} (fr : Frozen s) : T s := fr.tensor

end Frozen

/-- Coercion from T s to Frozen s -/
instance {s : Shape} : Coe (T s) (Frozen s) := ⟨Frozen.mk⟩

/-! ## TensorStruct Typeclass

A typeclass for structures that contain tensors, enabling generic traversal
and transformation of all tensors in the structure. This is similar to
JAX's PyTree concept or Equinox's filtered transformations.
-/

/-- A typeclass for structures that contain tensors.
    Allows generic traversal and transformation of all tensors in the structure. -/
class TensorStruct (α : Type) where
  -- Map a function over all tensors
  map : (∀ {s}, T s → T s) → α → α
  -- Monadic map (e.g., for initialization with IO)
  mapM {m : Type → Type} [Monad m] : (∀ {s}, T s → m (T s)) → α → m α
  -- Zip two containers with a function
  zipWith : (∀ {s}, T s → T s → T s) → α → α → α
  -- Fold over tensors
  fold {β : Type} : (∀ {s}, T s → β → β) → β → α → β

instance {α : Type} [TensorStruct α] : TensorStruct (Array α) where
  map f arr := arr.map (TensorStruct.map f)
  mapM f arr := arr.mapM (TensorStruct.mapM f)
  zipWith f arr1 arr2 :=
    if arr1.size == arr2.size then
      Array.zipWith (TensorStruct.zipWith f) arr1 arr2
    else
      panic! "TensorStruct.zipWith: Array size mismatch"
  fold f init arr := arr.foldl (fun acc x => TensorStruct.fold f acc x) init

instance {α : Type} [TensorStruct α] : TensorStruct (List α) where
  map f l := l.map (TensorStruct.map f)
  mapM f l := l.mapM (TensorStruct.mapM f)
  zipWith f l1 l2 :=
    if l1.length == l2.length then
      List.zipWith (TensorStruct.zipWith f) l1 l2
    else
      panic! "TensorStruct.zipWith: List length mismatch"
  fold f init l := l.foldl (fun acc x => TensorStruct.fold f acc x) init

instance {α : Type} [TensorStruct α] : TensorStruct (Option α) where
  map f opt := opt.map (TensorStruct.map f)
  mapM f opt := match opt with
    | some x => do let x' ← TensorStruct.mapM f x; return some x'
    | none => return none
  zipWith f opt1 opt2 := match opt1, opt2 with
    | some x, some y => some (TensorStruct.zipWith f x y)
    | _, _ => none
  fold f init opt := match opt with
    | some x => TensorStruct.fold f init x
    | none => init

/-- Static values are completely skipped during TensorStruct traversal -/
instance {α : Type} [Inhabited α] : TensorStruct (Static α) where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ init _ := init

/-- Frozen tensors are traversed (for forward pass) but can be filtered out for gradients -/
instance {s : Shape} : TensorStruct (Frozen s) where
  map f fr := { tensor := f fr.tensor }
  mapM f fr := do pure { tensor := ← f fr.tensor }
  zipWith f fr1 fr2 := { tensor := f fr1.tensor fr2.tensor }
  fold f init fr := f fr.tensor init

/-! ## TensorStruct Instances for Basic Types

Basic types contain no tensors, so they are passed through unchanged.
This allows structures with mixed tensor and non-tensor fields to derive TensorStruct.
-/

instance : TensorStruct Bool where
  map _ b := b
  mapM _ b := pure b
  zipWith _ b _ := b
  fold _ init _ := init

instance : TensorStruct Float where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct UInt8 where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct UInt64 where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct Nat where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct Int where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

instance : TensorStruct String where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ init _ := init

/-! ## TensorStruct Utility Methods

Convenient operations for common tensor tree manipulations.
-/

namespace TensorStruct

/-- Count the number of tensor leaves in the structure -/
def count [TensorStruct α] (model : α) : Nat :=
  fold (fun _ n => n + 1) 0 model

/-- Get gradients for all tensors in the structure -/
def grads [TensorStruct α] (model : α) : α :=
  map autograd.grad_of model

/-- Zero all gradients in the structure -/
def zeroGrads [TensorStruct α] (model : α) : α :=
  map autograd.zero_grad model

/-- Detach all tensors from the computation graph -/
def detach [TensorStruct α] (model : α) : α :=
  map autograd.detach model

/-- Set requires_grad on all tensors -/
def requiresGrad [TensorStruct α] (model : α) (b : Bool) : α :=
  map (fun t => autograd.set_requires_grad t b) model

/-- Make a tensor a trainable leaf parameter (detach and set requires_grad) -/
private def _makeLeafParam {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (autograd.detach t) true

/-- Make all tensors trainable leaf parameters -/
def makeLeafParams [TensorStruct α] (model : α) : α :=
  map _makeLeafParam model

/-- Apply a scalar multiplication to all tensors -/
def scale [TensorStruct α] (model : α) (s : Float) : α :=
  map (fun t => mul_scalar t s) model

/-- Element-wise addition of two structures -/
def add [TensorStruct α] (a b : α) : α :=
  zipWith torch.add a b

/-- Element-wise subtraction of two structures -/
def sub [TensorStruct α] (a b : α) : α :=
  zipWith torch.sub a b

end TensorStruct

end torch