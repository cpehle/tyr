import Tyr.GPU.Kernels.FusedLayerNorm

/-!
# Tyr.GPU.Kernels.LayerNormResidual

Compatibility shim for the ThunderKittens-aligned fused residual + layernorm
kernel.

The canonical implementation now lives in
`Tyr.GPU.Kernels.tkFusedLayerNormResidual1024` in
`Tyr.GPU.Kernels.FusedLayerNorm`. Keep `tkLayerNorm` as an alias so existing
references continue to build while the API converges on the clearer name.
-/

namespace Tyr.GPU.Kernels

/-- Backwards-compatible alias for the canonical ThunderKittens-aligned fused
residual + layernorm kernel. Prefer `tkFusedLayerNormResidual1024`. -/
abbrev tkLayerNorm := tkFusedLayerNormResidual1024

end Tyr.GPU.Kernels
