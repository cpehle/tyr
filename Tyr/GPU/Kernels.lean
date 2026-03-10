import Tyr.GPU.Kernels.Activations
import Tyr.GPU.Kernels.Based
import Tyr.GPU.Kernels.Copy
import Tyr.GPU.Kernels.Distributed
import Tyr.GPU.Kernels.Examples
import Tyr.GPU.Kernels.FFTConv
import Tyr.GPU.Kernels.FlashAttn
import Tyr.GPU.Kernels.FlashAttn3
import Tyr.GPU.Kernels.FlashAttnBwd
import Tyr.GPU.Kernels.Flux
import Tyr.GPU.Kernels.FusedLayerNorm
import Tyr.GPU.Kernels.Hedgehog
import Tyr.GPU.Kernels.LayerNorm
import Tyr.GPU.Kernels.LayerNormBwd
import Tyr.GPU.Kernels.LayerNormResidual
import Tyr.GPU.Kernels.LinearAttn
import Tyr.GPU.Kernels.LinearAttnBwd
import Tyr.GPU.Kernels.MOE
import Tyr.GPU.Kernels.Mamba
import Tyr.GPU.Kernels.Mamba2
import Tyr.GPU.Kernels.MambaBwd
import Tyr.GPU.Kernels.MhaH100
import Tyr.GPU.Kernels.NvFp4Gemm
import Tyr.GPU.Kernels.PrecisionGemm
import Tyr.GPU.Kernels.RingAttn
import Tyr.GPU.Kernels.RingAttnBwd
import Tyr.GPU.Kernels.Rotary
import Tyr.GPU.Kernels.RotaryBwd
import Tyr.GPU.Kernels.TestNewStyle
import Tyr.GPU.Kernels.UlyssesAttn
import Tyr.GPU.Kernels.UlyssesAttnBwd

/-!
# Tyr.GPU.Kernels

Umbrella import for the built-in GPU kernel catalog.

Importing this module makes the concrete kernel declarations part of Tyr's normal
Lean build graph, which in turn allows documentation generation to discover them
without bespoke module lists.

Use `Tyr.GPU.Kernels.Examples` for the smallest curated entrypoint, and
`Tyr.GPU.Kernels` when you want the full catalog available for navigation,
build validation, and docs.
-/
