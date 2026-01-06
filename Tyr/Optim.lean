/-
  Tyr/Optim.lean

  Optax-style gradient transformation library for composable optimizers.
  Based on https://optax.readthedocs.io/en/latest/

  Key concepts:
  - GradientTransformation: transforms gradients, returns (updates, new_state)
  - Composable via `chain`: stack multiple transformations
  - Stateless functions: all state passed explicitly
-/
import Tyr.TensorStruct

namespace torch.Optim

open torch

/-! ## Core Types -/

/-- Empty state for stateless transformations -/
structure EmptyState where
  deriving Inhabited

/-- Adam state: step count + first/second moments -/
structure ScaleByAdamState (α : Type) where
  count : Nat
  mu : α      -- First moment (mean of gradients)
  nu : α      -- Second moment (mean of squared gradients)

/-- Trace state for momentum -/
structure TraceState (α : Type) where
  trace : α

/-- Chained state for composed transformations -/
structure ChainState (S1 S2 : Type) where
  fst : S1
  snd : S2

/-! ## TensorStruct instances for optimizer states -/

instance : TensorStruct EmptyState where
  map _ s := s
  mapM _ s := pure s
  zipWith _ s _ := s
  fold _ acc _ := acc

instance [TensorStruct α] : TensorStruct (ScaleByAdamState α) where
  map f s := { count := s.count, mu := TensorStruct.map f s.mu, nu := TensorStruct.map f s.nu }
  mapM f s := do
    let mu' ← TensorStruct.mapM f s.mu
    let nu' ← TensorStruct.mapM f s.nu
    pure { count := s.count, mu := mu', nu := nu' }
  zipWith f s1 s2 := {
    count := s1.count
    mu := TensorStruct.zipWith f s1.mu s2.mu
    nu := TensorStruct.zipWith f s1.nu s2.nu
  }
  fold f acc s :=
    let acc' := TensorStruct.fold f acc s.mu
    TensorStruct.fold f acc' s.nu

instance [TensorStruct α] : TensorStruct (TraceState α) where
  map f s := { trace := TensorStruct.map f s.trace }
  mapM f s := do
    let trace' ← TensorStruct.mapM f s.trace
    pure { trace := trace' }
  zipWith f s1 s2 := { trace := TensorStruct.zipWith f s1.trace s2.trace }
  fold f acc s := TensorStruct.fold f acc s.trace

instance [TensorStruct S1] [TensorStruct S2] : TensorStruct (ChainState S1 S2) where
  map f s := { fst := TensorStruct.map f s.fst, snd := TensorStruct.map f s.snd }
  mapM f s := do
    let fst' ← TensorStruct.mapM f s.fst
    let snd' ← TensorStruct.mapM f s.snd
    pure { fst := fst', snd := snd' }
  zipWith f s1 s2 := {
    fst := TensorStruct.zipWith f s1.fst s2.fst
    snd := TensorStruct.zipWith f s1.snd s2.snd
  }
  fold f acc s :=
    let acc' := TensorStruct.fold f acc s.fst
    TensorStruct.fold f acc' s.snd

/-! ## Gradient Transformation

A gradient transformation consists of:
- `init`: Create initial optimizer state from model structure
- `update`: Transform gradients, return (updates, new_state)

Unlike traditional optimizers, `update` returns transformed gradients,
not updated parameters. This enables composition.
-/

/-- A gradient transformation with state type S -/
structure GradientTransformation (α : Type) (S : Type) where
  /-- Initialize optimizer state from model structure -/
  init : α → S
  /-- Transform gradients: (params, grads, state) → (updates, new_state) -/
  update : α → α → S → (α × S)

/-! ## Basic Transformations -/

/-- Scale gradients by a constant factor -/
def scale [TensorStruct α] (stepSize : Float) : GradientTransformation α EmptyState where
  init _ := {}
  update _params grads state :=
    (TensorStruct.map (fun t => mul_scalar t stepSize) grads, state)

/-- Scale by Adam: EMA of gradients and squared gradients with bias correction -/
def scale_by_adam [TensorStruct α] (b1 : Float := 0.9) (b2 : Float := 0.999) (eps : Float := 1e-8)
    : GradientTransformation α (ScaleByAdamState α) where
  init model := {
    count := 0
    -- Use zeros_like to create state tensors on same device as model
    mu := TensorStruct.map torch.zeros_like model
    nu := TensorStruct.map torch.zeros_like model
  }
  update _params grads state :=
    let count := state.count + 1
    -- Update biased first moment estimate: mu = b1 * mu + (1 - b1) * g
    let mu := TensorStruct.zipWith (fun m g =>
      let g_d := autograd.detach g
      add (mul_scalar m b1) (mul_scalar g_d (1.0 - b1))
    ) state.mu grads
    -- Update biased second moment estimate: nu = b2 * nu + (1 - b2) * g^2
    let nu := TensorStruct.zipWith (fun v g =>
      let g_d := autograd.detach g
      add (mul_scalar v b2) (mul_scalar (mul g_d g_d) (1.0 - b2))
    ) state.nu grads
    -- Bias correction
    let bc1 := 1.0 - Float.pow b1 count.toFloat
    let bc2 := 1.0 - Float.pow b2 count.toFloat
    -- Compute updates: mu_hat / (sqrt(nu_hat) + eps)
    let updates := TensorStruct.zipWith (fun m v =>
      let m_hat := div_scalar m bc1
      let v_hat := div_scalar v bc2
      nn.div m_hat (add_scalar (nn.sqrt v_hat) eps)
    ) mu nu
    (updates, { count, mu, nu })

/-- Add weight decay to gradients: g' = g + decay * p -/
def add_decayed_weights [TensorStruct α] (decay : Float) : GradientTransformation α EmptyState where
  init _ := {}
  update params grads state :=
    let updates := TensorStruct.zipWith (fun p g =>
      let p_d := autograd.detach p
      add g (mul_scalar p_d decay)
    ) params grads
    (updates, state)

/-- Accumulate momentum trace: trace = decay * trace + g -/
def trace [TensorStruct α] (decay : Float) : GradientTransformation α (TraceState α) where
  init model := { trace := TensorStruct.map torch.zeros_like model }
  update _params grads state :=
    let new_trace := TensorStruct.zipWith (fun t g =>
      add (mul_scalar t decay) g
    ) state.trace grads
    (new_trace, { trace := new_trace })

/-! ## Composition -/

/-- Chain two gradient transformations: apply t1 then t2 -/
def chain [TensorStruct α] (t1 : GradientTransformation α S1) (t2 : GradientTransformation α S2)
    : GradientTransformation α (ChainState S1 S2) where
  init model := { fst := t1.init model, snd := t2.init model }
  update params grads state :=
    let (updates1, newState1) := t1.update params grads state.fst
    let (updates2, newState2) := t2.update params updates1 state.snd
    (updates2, { fst := newState1, snd := newState2 })

/-! ## High-Level Optimizers -/

/-- AdamW optimizer state type -/
abbrev AdamWState (α : Type) := ChainState (ScaleByAdamState α) (ChainState EmptyState EmptyState)

/-- AdamW optimizer: Adam + decoupled weight decay

    Equivalent to: scale_by_adam → add_decayed_weights → scale(-lr)
-/
def adamw [TensorStruct α] (lr : Float := 1e-3) (b1 : Float := 0.9) (b2 : Float := 0.999)
    (eps : Float := 1e-8) (weight_decay : Float := 0.01) : GradientTransformation α (AdamWState α) :=
  chain (scale_by_adam b1 b2 eps)
    (chain (add_decayed_weights weight_decay)
           (scale (-lr)))

/-- SGD state type (with optional momentum) -/
abbrev SGDState (α : Type) := ChainState (TraceState α) EmptyState

/-- SGD with momentum -/
def sgd_momentum [TensorStruct α] (lr : Float) (momentum : Float := 0.9)
    : GradientTransformation α (SGDState α) :=
  chain (trace momentum) (scale (-lr))

/-- Simple SGD (no momentum) -/
def sgd [TensorStruct α] (lr : Float) : GradientTransformation α EmptyState :=
  scale (-lr)

/-! ## Applying Updates -/

/-- Apply gradient updates to parameters: p' = p + updates

    Handles detaching and re-enabling requires_grad for training. -/
def apply_updates [TensorStruct α] (params : α) (updates : α) : α :=
  TensorStruct.zipWith (fun p u =>
    let p_d := autograd.detach p
    let p_new := add p_d u
    autograd.set_requires_grad p_new true
  ) params updates

/-- Full optimizer step: transform gradients and apply to parameters

    Returns (new_params, new_state) -/
def step [TensorStruct α] (opt : GradientTransformation α S) (params grads : α) (state : S)
    : (α × S) :=
  let (updates, newState) := opt.update params grads state
  (apply_updates params updates, newState)

/-! ## Gradient Clipping (TODO)

def clip_by_global_norm [TensorStruct α] (max_norm : Float) : GradientTransformation α EmptyState
def clip_by_value [TensorStruct α] (min_val max_val : Float) : GradientTransformation α EmptyState
-/

end torch.Optim
