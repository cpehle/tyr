/-
  Examples/TypedTensor/MiniMLP.lean

  Demonstration of a minimal MLP using the typed `Tensor m s` API
  end-to-end. Every intermediate has its TensorMeta and shape pinned
  in the type, so any wiring mistake becomes a Lean type error rather
  than a runtime PyTorch crash or silent dtype/device coercion.

  Uses:
    - typed Module instances (`m |> x` infix notation)
    - typed RMSNorm + Linear forwards
    - typed activations
    - typed reduction (sumDim) for a final scalar loss-like output

  Compare with `Tyr/Module/Linear.lean` etc., which carry both legacy
  `T s` and typed `Tensor m s` instances; this file uses only the
  typed forms.
-/
import Tyr

open torch

namespace examples.typed

/-- Two-layer feedforward block with RMSNorm pre-norm and SiLU.
    Input/output shape pinned at `[batch, seq, hidden]`. Metadata `m`
    threaded through every intermediate. -/
structure MiniMLP (hidden : UInt64) (mid : UInt64) where
  pre  : RMSNorm hidden
  up   : Linear hidden mid
  gate : Linear hidden mid
  down : Linear mid hidden

/-- Pre-norm SwiGLU forward.
    `up(x) * silu(gate(x))` then project back via `down`. -/
def MiniMLP.forward {tm : TensorMeta} {hidden mid batch seq : UInt64}
    (mlp : MiniMLP hidden mid)
    (x : Tensor tm #[batch, seq, hidden])
    : Tensor tm #[batch, seq, hidden] :=
  let n : Tensor tm #[batch, seq, hidden] := mlp.pre |> x
  let u : Tensor tm #[batch, seq, mid] := mlp.up |> n
  let g : Tensor tm #[batch, seq, mid] := mlp.gate |> n
  let activated : Tensor tm #[batch, seq, mid] := u * (Tensor.silu g)
  mlp.down |> activated

/-- Initialize all parameters on the given metadata via legacy
    constructors, then unsafely re-tag — Linear/RMSNorm internally
    store `T #[…]` weights, but the runtime tensors are placed on
    `m.device` (and Linear init produces fp32 weights). -/
def MiniMLP.init (hidden mid : UInt64) (_m : TensorMeta) : IO (MiniMLP hidden mid) := do
  let pre  := torch.RMSNorm.init hidden 1e-6
  let up   ← torch.Linear.init hidden mid (withBias := true)
  let gate ← torch.Linear.init hidden mid (withBias := true)
  let down ← torch.Linear.init mid hidden (withBias := true)
  return { pre, up, gate, down }

/-- Smoke-test the typed MLP end-to-end. Returns the sum of the
    output's squared values — useful as a "did we get a finite
    number" check. -/
def MiniMLP.run (hidden mid batch seq : UInt64) (m : TensorMeta) : IO Float := do
  let mlp ← MiniMLP.init hidden mid m
  let x : Tensor m #[batch, seq, hidden] := Tensor.zeros #[batch, seq, hidden] m
  let y : Tensor m #[batch, seq, hidden] := MiniMLP.forward mlp x
  let y2 : Tensor m #[batch, seq, hidden] := y * y
  -- Reduce all dimensions to a scalar by repeated sumDim.
  let s1 := Tensor.sumDim y2 (s := #[batch, seq, hidden]) 0 false
  let s2 := Tensor.sumDim s1 0 false
  let s3 := Tensor.sumDim s2 0 false
  pure (nn.item (Tensor.toT s3))

end examples.typed
