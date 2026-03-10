import Tyr.GPU.Kernels.Attention
import Tyr.GPU.Kernels.Experimental
import Tyr.GPU.Kernels.Gemm
import Tyr.GPU.Kernels.Normalization
import Tyr.GPU.Kernels.Parallel
import Tyr.GPU.Kernels.StateSpace

/-!
# Tyr.GPU.Kernels

Umbrella import for the built-in GPU kernel catalog.

Importing this module makes the concrete kernel declarations part of Tyr's normal
Lean build graph, which in turn allows documentation generation to discover them
without bespoke module lists.

The catalog is also grouped into family entrypoints:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

Use `Tyr.GPU.Kernels.Examples` for the smallest curated entrypoint, and
`Tyr.GPU.Kernels` when you want the full catalog available for navigation,
build validation, and docs.
-/
