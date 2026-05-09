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

@[inline] def relu {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.relu (Tensor.toT t))

@[inline] def gelu {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.gelu (Tensor.toT t))

@[inline] def softplus {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.softplus (Tensor.toT t))

@[inline] def tanh {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.tanh (Tensor.toT t))

@[inline] def exp {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.exp (Tensor.toT t))

@[inline] def log {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.log (Tensor.toT t))

@[inline] def abs {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.abs (Tensor.toT t))

@[inline] def sqrt {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.sqrt (Tensor.toT t))

/-! ## Reductions

Reduce along a dimension, optionally keeping it as size 1.
Output shape via `reduceShape`. -/

@[inline] def sumDim {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : Nat) (keepdim : Bool := false)
    : Tensor m (reduceShape s dim keepdim) :=
  Tensor.unsafeOfT m (torch.nn.sumDim (Tensor.toT t) dim keepdim)

@[inline] def meanDim {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : Nat) (keepdim : Bool := false)
    : Tensor m (reduceShape s dim keepdim) :=
  Tensor.unsafeOfT m (torch.nn.meanDim (Tensor.toT t) dim keepdim)

/-- `argmax` returns indices, dtype shifts to `.Int64` regardless of input. -/
@[inline] def argmax {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : UInt64)
    : Tensor { m with dtype := .Int64 } (reduceShape s dim.toNat false) :=
  Tensor.unsafeOfT _ (torch.nn.argmax (Tensor.toT t) dim)

/-! ## Logical / comparison

Boolean-valued ops that semantically produce a Bool-dtype tensor.
`eq_scalar` returns a same-dtype tensor in PyTorch (via cast-back), so
metadata-preserving here. -/

@[inline] def eqScalar {m : TensorMeta} {s : Shape} (t : Tensor m s) (scalar : Int64)
    : Tensor m s :=
  Tensor.unsafeOfT m (torch.eq_scalar (Tensor.toT t) scalar)

@[inline] def whereSelect {m : TensorMeta} {s : Shape}
    (cond : Tensor m s) (x y : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.where_ (Tensor.toT cond) (Tensor.toT x) (Tensor.toT y))

@[inline] def logicalNot {m : TensorMeta} {s : Shape} (t : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.logical_not (Tensor.toT t))

@[inline] def logicalOr {m : TensorMeta} {s : Shape} (a b : Tensor m s) : Tensor m s :=
  Tensor.unsafeOfT m (torch.logicalOr (Tensor.toT a) (Tensor.toT b))

@[inline] def anyB {m : TensorMeta} {s : Shape} (t : Tensor m s) : Bool :=
  torch.any (Tensor.toT t)

/-- Tensor-of-Int64 constant, useful for token-id constants alongside
    embedding/argmax outputs. The result claims `.Int64` dtype. -/
@[inline] def fullInt (s : Shape) (m : TensorMeta) (value : Int64)
    : Tensor { m with dtype := .Int64 } s :=
  Tensor.unsafeOfT _ ((torch.full_int s value).to m.device)

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

@[inline] def unsqueeze {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : Nat)
    : Tensor m (unsqueezeShape s dim) :=
  Tensor.unsafeOfT m (torch.nn.unsqueeze (Tensor.toT t) dim)

@[inline] def squeeze {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : Nat)
    : Tensor m (squeezeShape s dim) :=
  Tensor.unsafeOfT m (torch.nn.squeeze (Tensor.toT t) dim)

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

@[inline] def linear3d {m : TensorMeta} {batch seq in_dim out_dim : UInt64}
    (x : Tensor m #[batch, seq, in_dim]) (w : Tensor m #[out_dim, in_dim])
    : Tensor m #[batch, seq, out_dim] :=
  Tensor.unsafeOfT m (torch.linear3d (Tensor.toT x) (Tensor.toT w))

@[inline] def affine3d {m : TensorMeta} {batch seq in_dim out_dim : UInt64}
    (x : Tensor m #[batch, seq, in_dim])
    (w : Tensor m #[out_dim, in_dim])
    (b : Tensor m #[out_dim])
    : Tensor m #[batch, seq, out_dim] :=
  Tensor.unsafeOfT m (torch.affine3d (Tensor.toT x) (Tensor.toT w) (Tensor.toT b))

/-! ## Slicing

Metadata-preserving, shape-changing. -/

@[inline] def slice {m : TensorMeta} {s : Shape} (t : Tensor m s) (dim : UInt64) (start len : UInt64)
    : Tensor m (torch.data.sliceShape s dim len) :=
  Tensor.unsafeOfT m (torch.data.slice (Tensor.toT t) dim start len)

/-- Replace a slice of `data` along `dim` (starting at `start`) with `src`.
    Output shape matches `data`; metadata must match between `data` and `src`. -/
@[inline] def sliceScatter {m : TensorMeta} {s src : Shape}
    (data : Tensor m s) (dim : UInt64) (start : UInt64) (src : Tensor m src)
    : Tensor m s :=
  Tensor.unsafeOfT m (torch.data.sliceScatter (Tensor.toT data) dim start (Tensor.toT src))

/-! ## Attention helpers

Reshape to / from canonical attention layout
[batch, seq, n_head, head_dim] ↔ [batch, n_head, seq, head_dim].
Metadata-preserving. -/

@[inline] def transposeForAttention {m : TensorMeta} {batch seq n_head head_dim : UInt64}
    (x : Tensor m #[batch, seq, n_head, head_dim])
    : Tensor m #[batch, n_head, seq, head_dim] :=
  Tensor.unsafeOfT m (torch.nn.transpose_for_attention (Tensor.toT x))

@[inline] def transposeFromAttention {m : TensorMeta} {batch n_head seq head_dim : UInt64}
    (x : Tensor m #[batch, n_head, seq, head_dim])
    : Tensor m #[batch, seq, n_head, head_dim] :=
  Tensor.unsafeOfT m (torch.nn.transpose_from_attention (Tensor.toT x))

/-! ## Embedding

Token-ID lookup: `ids` must be Int64; the result inherits the
embedding weight's metadata. The Int64 constraint is enforced at the
type level via a dtype-pinned input parameter. -/

@[inline] def embedding {m : TensorMeta} {batch seq vocab embed : UInt64}
    (weight : Tensor m #[vocab, embed])
    (ids : Tensor { m with dtype := .Int64 } #[batch, seq])
    : Tensor m #[batch, seq, embed] :=
  Tensor.unsafeOfT m (torch.nn.embedding (Tensor.toT ids) (Tensor.toT weight))

/-! ## Scaled Dot-Product Attention

GQA-aware SDPA variants. All preserve metadata across Q/K/V/out. -/

/-- SDPA-GQA with Q/K/V sharing kv_seq = q_seq. -/
@[inline] def sdpaGQA {m : TensorMeta} {batch n_head n_kv_head seq head_dim : UInt64}
    (query : Tensor m #[batch, n_head, seq, head_dim])
    (key : Tensor m #[batch, n_kv_head, seq, head_dim])
    (value : Tensor m #[batch, n_kv_head, seq, head_dim])
    (dropout_p : Float := 0.0) (is_causal : Bool := true) (enable_gqa : Bool := false)
    : Tensor m #[batch, n_head, seq, head_dim] :=
  Tensor.unsafeOfT m
    (torch.nn.scaledDotProductAttentionGQA (Tensor.toT query) (Tensor.toT key) (Tensor.toT value)
      dropout_p is_causal enable_gqa)

/-- SDPA-GQA with separate q_seq vs kv_seq (KV-cache decode). -/
@[inline] def sdpaGQAQKV {m : TensorMeta} {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64}
    (query : Tensor m #[batch, n_head, q_seq, head_dim])
    (key : Tensor m #[batch, n_kv_head, kv_seq, head_dim])
    (value : Tensor m #[batch, n_kv_head, kv_seq, head_dim])
    (dropout_p : Float := 0.0) (is_causal : Bool := false) (enable_gqa : Bool := false)
    : Tensor m #[batch, n_head, q_seq, head_dim] :=
  Tensor.unsafeOfT m
    (torch.nn.scaledDotProductAttentionGQAQKV (Tensor.toT query) (Tensor.toT key) (Tensor.toT value)
      dropout_p is_causal enable_gqa)

/-- Tyr/TK FlashAttention dispatcher. C++ side (`select_route`) routes
    Hopper + BF16 + qSeq=1 problems to the TK decode kernel and falls
    back to PyTorch SDPA otherwise. -/
@[inline] def tyrFlashAttn4d {m : TensorMeta}
    {batch n_head n_kv_head q_seq kv_seq head_dim : UInt64}
    (query : Tensor m #[batch, n_head, q_seq, head_dim])
    (key : Tensor m #[batch, n_kv_head, kv_seq, head_dim])
    (value : Tensor m #[batch, n_kv_head, kv_seq, head_dim])
    (attn_mask : Option (Tensor m #[batch, kv_seq]) := none)
    (dropout_p : Float := 0.0) (is_causal : Bool := false)
    (scale : Option Float := none) (enable_gqa : Bool := false)
    : Tensor m #[batch, n_head, q_seq, head_dim] :=
  Tensor.unsafeOfT m
    (torch.nn.tyrFlashAttn4d (Tensor.toT query) (Tensor.toT key) (Tensor.toT value)
      (attn_mask.map Tensor.toT) dropout_p is_causal scale enable_gqa)

/-! ## Module-level helpers (functional)

Direct functional-form access to module ops without the Module wrapper.
Useful when porting code that calls `nn.layer_norm x w b eps` directly. -/

@[inline] def rmsNormWeighted {m : TensorMeta} {s w : Shape}
    (x : Tensor m s) (weight : Tensor m w) (eps : Float := 1e-6) : Tensor m s :=
  Tensor.unsafeOfT m (torch.nn.rmsNormWeighted (Tensor.toT x) (Tensor.toT weight) eps)

@[inline] def layerNorm {m : TensorMeta} {batch seq n : UInt64}
    (x : Tensor m #[batch, seq, n]) (weight : Tensor m #[n]) (bias : Tensor m #[n])
    (eps : Float := 1e-5) : Tensor m #[batch, seq, n] :=
  Tensor.unsafeOfT m (torch.nn.layer_norm (Tensor.toT x) (Tensor.toT weight) (Tensor.toT bias) eps)

/-! ## SafeTensors / FFI ingress

Loading from a SafeTensors file. Two patterns:

  - **Caller-asserted metadata** (most common): `loadSafeTensor` takes
    the expected `TensorMeta` and asserts it on the loaded tensor. If
    the file has a different dtype/device than expected, the result
    will silently misclaim its type — only the runtime tensor knows.
    Use when you've already inspected the file header (or trust the
    schema) and want the typed view.

  - **Σ-typed auto-detect** (`loadSafeTensorAuto`): inspects the
    runtime tensor and returns a `(m : TensorMeta) × Tensor m s`. Use
    when the file's dtype is genuinely unknown at compile time. -/

/-- Load a tensor from a SafeTensors file, asserting `m`. The
    underlying tensor is moved to `m.device` after loading. Caller is
    responsible for matching `m.dtype` to the file's actual dtype. -/
def loadSafeTensor (path name : String) (s : Shape) (m : TensorMeta) : IO (Tensor m s) := do
  let t ← torch.safetensors.loadTensor path name s
  let t := T.to t m.device
  return Tensor.unsafeOfT m t

/-- Load a tensor from a SafeTensors file and recover its actual
    metadata at runtime, returned alongside the typed tensor. The
    caller can then destructure `let ⟨m, t⟩ ← ...` and proceed with a
    pinned-type tensor. -/
def loadSafeTensorAuto (path name : String) (s : Shape) (overrideDevice : Option Device := none)
    : IO ((m : TensorMeta) × Tensor m s) := do
  let t ← torch.safetensors.loadTensor path name s
  let t := match overrideDevice with
    | some d => T.to t d
    | none => t
  let m : TensorMeta := { device := t.device, dtype := t.dtype }
  return ⟨m, Tensor.unsafeOfT m t⟩

/-! ## Inspection

Read-only queries that do not produce a new tensor — just plain values. -/

@[inline] def shape {m : TensorMeta} {s : Shape} (_ : Tensor m s) : Shape := s

@[inline] def tensorMeta {m : TensorMeta} {s : Shape} (_ : Tensor m s) : TensorMeta := m

end Tensor
end torch
