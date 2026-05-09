/-
  Tyr/Tensor.lean

  Typed tensor API. Wraps the legacy `T s` ops in `Tyr/Torch.lean` to expose
  `Tensor m s` versions that thread `TensorMeta` (device + dtype) at the
  type level.

  This is the foundation for migrating model code from `T s` to `Tensor m s`.
  Both representations share the same opaque FFI handle (see `Tyr/Basic.lean`),
  so wrapping is zero-cost: `@[inline]` defs bridge the two views without
  any runtime work.

  Migration policy:
    - new code should `import Tyr.Tensor` and use `Tensor m s` throughout
    - legacy `T s` ops in `Tyr.Torch` remain available for incremental
      migration; wrappers below delegate to them by `Tensor.toT` / `Tensor.unsafeOfT`
    - see `dev/type_safe_tensor_plan.md` for the migration plan
-/
import Tyr.Torch

namespace torch
namespace Tensor

/-! ## Construction

`Tensor` constructors take a `TensorMeta` so the resulting tensor's
device + dtype are reflected in the type. The runtime arg drives both the
type-level index and the actual placement. -/

/-- Allocate a zero tensor with the given shape on the given metadata. -/
@[inline] def zeros (s : Shape) (m : TensorMeta) (requires_grad : Bool := false) : Tensor m s :=
  Tensor.unsafeOfT m (torch.zeros s requires_grad m.device)

/-- Allocate a ones tensor with the given shape on the given metadata. -/
@[inline] def ones (s : Shape) (m : TensorMeta) (requires_grad : Bool := false) : Tensor m s :=
  Tensor.unsafeOfT m (torch.ones s requires_grad m.device)

/-- Allocate a constant-filled tensor. -/
@[inline] def full (s : Shape) (m : TensorMeta) (value : Float) (requires_grad : Bool := false) : Tensor m s :=
  Tensor.unsafeOfT m (torch.full s value requires_grad m.device)

/-- Allocate a uniform-random tensor on the given metadata. -/
@[inline] def rand (s : Shape) (m : TensorMeta) (requires_grad : Bool := false) : IO (Tensor m s) := do
  let raw ← torch.rand s requires_grad m.device
  return Tensor.unsafeOfT m raw

/-- Allocate a normal-random tensor on the given metadata. -/
@[inline] def randn (s : Shape) (m : TensorMeta) (requires_grad : Bool := false) : IO (Tensor m s) := do
  let raw ← torch.randn s requires_grad m.device
  return Tensor.unsafeOfT m raw

/-! ## Conversion

Conversion ops produce a tensor whose `TensorMeta` is a record-update of
the input's. The runtime target arg drives both the type-level result
metadata and the underlying tensor conversion. -/

/-- Move a tensor to a different device, preserving dtype. -/
@[inline] def toDevice {m : TensorMeta} {s : Shape} (t : Tensor m s) (target : Device)
    : Tensor { m with device := target } s :=
  Tensor.unsafeOfT _ ((Tensor.toT t).to target)

/-- Cast a tensor to a different dtype, preserving device. The runtime
    cast must be expressible by an op in `Tyr.Torch`; for now we cover
    the most common targets (`Float32`, `BFloat16`). Extend as needed. -/
@[inline] def toFloat32 {m : TensorMeta} {s : Shape} (t : Tensor m s)
    : Tensor { m with dtype := .Float32 } s :=
  Tensor.unsafeOfT _ (toFloat' (Tensor.toT t))

@[inline] def toBFloat16 {m : TensorMeta} {s : Shape} (t : Tensor m s)
    : Tensor { m with dtype := .BFloat16 } s :=
  Tensor.unsafeOfT _ (toBFloat16' (Tensor.toT t))

/-- Assert a runtime tensor has the claimed metadata; used at FFI
    boundaries where the caller already knows the device/dtype. Does not
    verify at runtime — caller is responsible. -/
@[inline] def viewAs {s : Shape} (m : TensorMeta) (t : T s) : Tensor m s :=
  Tensor.unsafeOfT m t

/-! ## Arithmetic

Same-meta in, same-meta out. Lean enforces metadata equality across
operands. -/

@[inline] def add {m : TensorMeta} {s : Shape} (a b : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.add (Tensor.toT a) (Tensor.toT b))

@[inline] def sub {m : TensorMeta} {s : Shape} (a b : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.sub (Tensor.toT a) (Tensor.toT b))

@[inline] def mul {m : TensorMeta} {s : Shape} (a b : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.mul (Tensor.toT a) (Tensor.toT b))

@[inline] def addScalar {m : TensorMeta} {s : Shape} (t : Tensor m s) (k : Float) : Tensor m s :=
  Tensor.unsafeOfT m (torch.add_scalar (Tensor.toT t) k)

@[inline] def subScalar {m : TensorMeta} {s : Shape} (t : Tensor m s) (k : Float) : Tensor m s :=
  Tensor.unsafeOfT m (torch.sub_scalar (Tensor.toT t) k)

@[inline] def mulScalar {m : TensorMeta} {s : Shape} (t : Tensor m s) (k : Float) : Tensor m s :=
  Tensor.unsafeOfT m (torch.mul_scalar (Tensor.toT t) k)

@[inline] def divScalar {m : TensorMeta} {s : Shape} (t : Tensor m s) (k : Float) : Tensor m s :=
  Tensor.unsafeOfT m (torch.div_scalar (Tensor.toT t) k)

instance {m : TensorMeta} {s : Shape} : Add (Tensor m s) := ⟨add⟩
instance {m : TensorMeta} {s : Shape} : Sub (Tensor m s) := ⟨sub⟩
instance {m : TensorMeta} {s : Shape} : Mul (Tensor m s) := ⟨mul⟩
instance {m : TensorMeta} {s : Shape} : HMul (Tensor m s) Float (Tensor m s) := ⟨mulScalar⟩
instance {m : TensorMeta} {s : Shape} : HDiv (Tensor m s) Float (Tensor m s) := ⟨divScalar⟩

/-! ## Activations / elementwise

Elementwise ops over a single tensor — shape and metadata preserved. -/

@[inline] def sigmoid {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.sigmoid (Tensor.toT t))

@[inline] def silu {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.silu (Tensor.toT t))

@[inline] def softmax {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : Int32 := -1) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.softmax (Tensor.toT t) dim)

@[inline] def rsqrt {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.rsqrt (Tensor.toT t))

/-! ## Shape transforms

Shape-changing, metadata-preserving. Output shape is determined by
`Shape`-pure helpers in `Tyr.Basic`. -/

@[inline] def reshape {m : TensorMeta} {s : Shape} (t : Tensor m s) (newShape : Shape) : Tensor m newShape :=
  Tensor.unsafeOfT m (torch.reshape (Tensor.toT t) newShape)

@[inline] def transpose {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim0 dim1 : UInt64)
    : Tensor m (transposeShape s dim0.toNat dim1.toNat) :=
  Tensor.unsafeOfT m (torch.nn.transpose (Tensor.toT t) dim0 dim1)

@[inline] def expand {m : TensorMeta} {s : Shape} (t : Tensor m s) (target : Shape) : Tensor m target :=
  Tensor.unsafeOfT m (torch.nn.expand (Tensor.toT t) target)

@[inline] def cat {m : TensorMeta} {s1 s2 : Shape}
    (a : Tensor m s1) (b : Tensor m s2) (dim : Nat)
    : Tensor m (torch.nn.catShape s1 s2 dim) :=
  Tensor.unsafeOfT m (torch.nn.cat (Tensor.toT a) (Tensor.toT b) dim)

/-! ## Linear algebra

Matrix multiplication (rank-2) and dependent batched/3D variants for
the typed `Linear` module. All preserve metadata since they don't
cross device or dtype boundaries. -/

@[inline] def matmul {m : TensorMeta} {s1 s2 : Shape}
    (a : Tensor m s1) (b : Tensor m s2)
    : Tensor m (matmulShape s1 s2) :=
  Tensor.unsafeOfT m (torch.nn.matmul (Tensor.toT a) (Tensor.toT b))

@[inline] def linear2d {m : TensorMeta} {batch in_dim out_dim : UInt64}
    (x : Tensor m #[batch, in_dim]) (w : Tensor m #[out_dim, in_dim])
    : Tensor m #[batch, out_dim] :=
  Tensor.unsafeOfT m (torch.linear (Tensor.toT x) (Tensor.toT w))

/-! ## Inspection

Read-only queries that do not produce a new tensor — just plain values. -/

@[inline] def shape {m : TensorMeta} {s : Shape} (_ : Tensor m s) : Shape := s

@[inline] def tensorMeta {m : TensorMeta} {s : Shape} (_ : Tensor m s) : TensorMeta := m

end Tensor
end torch
