import Tyr.Torch

namespace torch

@[extern "lean_torch_invariant_rmsnorm"]
private opaque invariantRMSNormImpl (x weight : @& T #[]) (eps : Float) : IO (T #[])

/-- Opt-in CUDA zero-centered RMSNorm with fixed row-local Float32 reductions.
The scale is `1 + weight`; weight must be frozen. Contiguous Float32/BFloat16
inputs and weights on one device are supported, including empty row batches.
Epsilon must remain finite and positive in Float32. Output and first-order input
gradient have the input dtype. No weight or higher-order gradients are provided.
Unchanged rows/cotangents use the same reduction tree regardless of peer rows;
this is a primitive guarantee on one device/build, not model-wide determinism. -/
def invariantRMSNorm {rows width : UInt64} (x : T #[rows,width])
    (weight : T #[width]) (eps : Float := 1e-6) : IO (T #[rows,width]) := do
  let result ← invariantRMSNormImpl (nn.eraseShape x) (nn.eraseShape weight) eps
  return reshape result #[rows,width]

end torch
