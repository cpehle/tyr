/- Prototype: ONE struct index `StaticSpec` (shape, dtype, device) on the
   tensor type, reusing the existing TensorSpec algebra for result types. -/
import Tyr.Typed

namespace Proto

open torch

/-- The graded static spec. `dtype := .Unknown ""` plays the "dynamic"
    element (a real implementation would add a proper `.dynamic`). -/
structure StaticSpec where
  shape : Shape
  dtype : DType := .Unknown ""
  device : DevicePolicy := .any
  deriving Repr, BEq

namespace StaticSpec

/-- Result spec of a broadcasting pointwise op. -/
def pointwise (l r : StaticSpec) : StaticSpec :=
  { shape := TensorSpec.broadcastShape l.shape r.shape
    dtype := DType.promote l.dtype r.dtype
    device := l.device }

/-- Result spec of matmul (operands must share dtype; checked by the op). -/
def matmulSpec (l r : StaticSpec) : StaticSpec :=
  { shape := matmulShape l.shape r.shape, dtype := l.dtype, device := l.device }

def withShape (σ : StaticSpec) (shape : Shape) : StaticSpec := { σ with shape }
def withDType (σ : StaticSpec) (dtype : DType) : StaticSpec := { σ with dtype }

def sumSpec (σ : StaticSpec) : StaticSpec :=
  { σ with shape := #[], dtype := σ.dtype.sumResult }

end StaticSpec

/-- The graded tensor: phantom struct index over the same `T shape`. -/
structure Tensor (σ : StaticSpec) where
  raw : T σ.shape

namespace Tensor

def assumeSpec {σ : StaticSpec} (raw : T σ.shape) : Tensor σ := ⟨raw⟩

def ones (shape : Shape) : Tensor ⟨shape, .Float32, .any⟩ :=
  ⟨torch.ones shape⟩

/-- Broadcasting add, compile-checked. -/
def add {σ₁ σ₂ : StaticSpec} (a : Tensor σ₁) (b : Tensor σ₂)
    (_h : (TensorSpec.broadcastShapes? σ₁.shape σ₂.shape).isSome = true := by rfl) :
    Tensor (σ₁.pointwise σ₂) :=
  .assumeSpec (torch.reshape
    (torch.add (torch.reshape a.raw #[]) (torch.reshape b.raw #[]))
    (TensorSpec.broadcastShape σ₁.shape σ₂.shape))

/-- Homogeneous add (same σ), for generic code. -/
instance {σ : StaticSpec} : Add (Tensor σ) :=
  ⟨fun a b => ⟨torch.add a.raw b.raw⟩⟩

/-- Matmul: dtype and device agreement via shared spec fields. -/
def matmul {s1 s2 : Shape} {d : DType} {dev : DevicePolicy}
    (a : Tensor ⟨s1, d, dev⟩) (b : Tensor ⟨s2, d, dev⟩) :
    Tensor ⟨matmulShape s1 s2, d, dev⟩ :=
  .assumeSpec (nn.matmul a.raw b.raw)

def relu {σ : StaticSpec} (t : Tensor σ) : Tensor σ :=
  ⟨torch.relu t.raw⟩

def toFloat32 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .Float32) :=
  ⟨torch.toFloat' t.raw⟩

def toBFloat16 {σ : StaticSpec} (t : Tensor σ) : Tensor (σ.withDType .BFloat16) :=
  ⟨torch.toBFloat16' t.raw⟩

def sumAll {σ : StaticSpec} (t : Tensor σ) : Tensor σ.sumSpec :=
  ⟨nn.sumAll t.raw⟩

def rmsNormWeighted {σ : StaticSpec} {w : Shape}
    (x : Tensor σ) (weight : Tensor (σ.withShape w)) (eps : Float := 1e-6) :
    Tensor (σ.withDType (DType.promote .Float32 σ.dtype)) :=
  ⟨nn.rmsNormWeighted x.raw weight.raw eps⟩

def linear3d {batch seq inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩)
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  .assumeSpec (torch.linear3d x.raw weight.raw)

end Tensor

/-! Layers in the struct world. -/

structure Linear (inDim outDim : UInt64) (d : DType) (dev : DevicePolicy := .any) where
  weight : Tensor ⟨#[outDim, inDim], d, dev⟩

namespace Linear

def forward3d {inDim outDim batch seq : UInt64} {d : DType} {dev : DevicePolicy}
    (lin : Linear inDim outDim d dev)
    (x : Tensor ⟨#[batch, seq, inDim], d, dev⟩) :
    Tensor ⟨#[batch, seq, outDim], d, dev⟩ :=
  Tensor.linear3d x lin.weight

end Linear

structure RMSNorm (dim : UInt64) (d : DType) (dev : DevicePolicy := .any) where
  weight : Tensor ⟨#[dim], d, dev⟩
  eps : Float := 1e-6

namespace RMSNorm

def forward3d {dim batch seq : UInt64} {d : DType} {dev : DevicePolicy}
    (rn : RMSNorm dim d dev) (x : Tensor ⟨#[batch, seq, dim], d, dev⟩) :
    Tensor ⟨#[batch, seq, dim], DType.promote .Float32 d, dev⟩ :=
  Tensor.rmsNormWeighted x rn.weight rn.eps

end RMSNorm

/-! ============ PROBES ============ -/

-- P1: inferred type of a mixed-dtype broadcast add
#check fun (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3], .BFloat16, .any⟩) =>
  a.add b

-- P2: inferred type after a chain (matmul then cast)
#check fun (x : Tensor ⟨#[2, 3], .Float32, .any⟩) (w : Tensor ⟨#[3, 4], .Float32, .any⟩) =>
  (x.matmul w).toBFloat16

-- P3: generic dtype-preserving chain over a symbolic spec
def chain {σ : StaticSpec} (x : Tensor σ) : Tensor σ := (x.relu + x.relu).relu

-- P4: ascription against the computed spec (does defeq reduce?)
def ascribed (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3], .Float32, .any⟩) :
    Tensor ⟨#[2, 3], .Float32, .any⟩ :=
  a.add b

-- P5: device pinning — CUDA-only function rejects a CPU tensor
def cudaOnly (x : Tensor ⟨#[8], .Float32, .exact (.CUDA 0)⟩) : Tensor ⟨#[8], .Float32, .exact (.CUDA 0)⟩ :=
  x.relu

-- E1 (ERROR): incompatible broadcast
def bad1 (a : Tensor ⟨#[2, 4], .Float32, .any⟩) (b : Tensor ⟨#[3, 2], .Float32, .any⟩) :=
  a.add b

-- E2 (ERROR): mixed-dtype matmul
def bad2 (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3, 4], .BFloat16, .any⟩) :=
  a.matmul b

-- E3 (ERROR): device mismatch at compile time
def bad3 (x : Tensor ⟨#[8], .Float32, .exact .CPU⟩) := cudaOnly x

-- E4 (ERROR): wrong ascription
def bad4 (a : Tensor ⟨#[2, 3], .Float32, .any⟩) (b : Tensor ⟨#[3], .BFloat16, .any⟩) :
    Tensor ⟨#[2, 3], .BFloat16, .any⟩ :=
  a.add b

end Proto
