import Tyr.GPU.Kernels.Activations
import Tyr.GPU.Kernels.Flux
import Tyr.GPU.Kernels.FusedLayerNorm

/-!
# Tyr.GPU.Kernels.Normalization

Normalization, activation, and small fusion kernels.

This family groups the canonical fused residual/layernorm port with the
normalization-adjacent activation/fusion kernels that are part of the public
GPU catalog.
-/
