import Tyr.Torch

namespace torch

@[extern "lean_torch_segmented_lora"]
private opaque segmentedLoRAImpl (x base a b : @& T #[])
    (offsets ranks : @& Array UInt32) (alphas : @& Array Float) : IO (T #[])

/-- Packed tenant segments share a frozen base weight. Each tenant uses the
first `ranks[t]` rows/columns of its padded factors and scale `alphas[t]/ranks[t]`.
Rank zero disables the adapter. The CUDA primitive supports Float32/BFloat16,
accumulates in Float32, and differentiates x/A/B without materializing adapted
weights. Metadata/shape/device errors return IO errors before kernel launch. -/
def segmentedLoRA {tokens input output tenants maxRank : UInt64}
    (x : T #[tokens, input]) (base : T #[output, input])
    (a : T #[tenants, maxRank, input]) (b : T #[tenants, output, maxRank])
    (offsets ranks : Array UInt32) (alphas : Array Float) : IO (T #[tokens, output]) := do
  let result ← segmentedLoRAImpl (nn.eraseShape x) (nn.eraseShape base)
    (nn.eraseShape a) (nn.eraseShape b) offsets ranks alphas
  return reshape result #[tokens, output]

end torch
