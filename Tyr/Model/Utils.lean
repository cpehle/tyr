/-
  Tyr/Model/Utils.lean

  Shared model utility functions used across multiple model families.
  These are higher-level than Tyr.Torch primitives and are specific to
  neural network initialization and token-level inference helpers.

  Note: `logicalOr`, `falseMask`, `zerosOn`, `onesOn`, `castLike`,
  and `restoreInputDType` are already provided by `Tyr.Torch`.
-/
import Tyr.Torch
import Tyr.Tensor

namespace torch.Model

open torch

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn). -/
def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

/-! ## Typed initialization helpers

Produce `Tensor m s` directly from random init routines so callers
can keep parameter dtype/device pinned at construction. -/

/-- Typed Kaiming-like init. The result tensor lives on `m.device` with
    `m.dtype` claimed at the type level. The randn/scaling is performed
    in the legacy float path then transferred. -/
def initWeightT (shape : Shape) (fanIn : UInt64) (m : TensorMeta) : IO (Tensor m shape) := do
  let w ← initWeight shape fanIn
  let w := T.to w m.device
  return Tensor.unsafeOfT m w

/-- Typed bias init — zero tensor with requires_grad=true on `m.device`. -/
def initBiasT (shape : Shape) (m : TensorMeta) : Tensor m shape :=
  Tensor.unsafeOfT m (T.to (initBias shape) m.device)

end torch.Model
