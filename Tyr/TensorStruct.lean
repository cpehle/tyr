import Tyr.Torch

namespace torch

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

namespace optim

/-- Generic Optimizer State for any model structure α -/
structure GenericAdamWState (α : Type) where
  m : α       -- First moments
  v : α       -- Second moments
  step : Nat
  deriving Inhabited

/-- Initialize generic optimizer state -/
def GenericAdamWState.init [TensorStruct α] (model : α) : GenericAdamWState α :=
  {
    m := TensorStruct.map (fun _ => torch.zeros _) model
    v := TensorStruct.map (fun _ => torch.zeros _) model
    step := 0
  }

/-- Generic AdamW step
    Updates the model parameters in-place (functionally) and returns new model + new state. -/
def generic_adamw [TensorStruct α] (config : AdamWConfig)
    (params : α) (grads : α) (state : GenericAdamWState α) : IO (α × GenericAdamWState α) := do
  let step := state.step + 1
  let { lr, beta1, beta2, eps, weight_decay } := config
  let bc1 := 1.0 - Float.pow beta1 step.toFloat
  let bc2 := 1.0 - Float.pow beta2 step.toFloat

  -- 1. Update first moment m
  -- m_new = beta1 * m + (1 - beta1) * grad
  let m_new := TensorStruct.zipWith (fun m g =>
    let g_d := autograd.detach g
    add (mul_scalar m beta1) (mul_scalar g_d (1.0 - beta1))
  ) state.m grads

  -- 2. Update second moment v
  -- v_new = beta2 * v + (1 - beta2) * (grad * grad)
  let v_new := TensorStruct.zipWith (fun v g =>
    let g_d := autograd.detach g
    add (mul_scalar v beta2) (mul_scalar (mul g_d g_d) (1.0 - beta2))
  ) state.v grads

  -- 3. Compute step direction
  -- update = (m_new / bc1) / (sqrt(v_new / bc2) + eps)
  let step_dir := TensorStruct.zipWith (fun m v =>
    let m_hat := div_scalar m bc1
    let v_hat := div_scalar v bc2
    nn.div m_hat (add_scalar (nn.sqrt v_hat) eps)
  ) m_new v_new

  -- 4. Apply update to parameters
  -- p_new = p - lr * (step_dir + weight_decay * p)
  let params_new := TensorStruct.zipWith (fun p d =>
    let p_d := autograd.detach p
    let p_decayed := mul_scalar p_d (1.0 - lr * weight_decay)
    let p_new := sub p_decayed (mul_scalar d lr)
    autograd.set_requires_grad p_new true
  ) params step_dir

  return (params_new, { m := m_new, v := v_new, step := step })

end optim

end torch