import Tyr.GPU.Kernels.FFTConv
import Tyr.GPU.Kernels.Hedgehog
import Tyr.GPU.Kernels.Mamba
import Tyr.GPU.Kernels.Mamba2
import Tyr.GPU.Kernels.MambaBwd

/-!
# Tyr.GPU.Kernels.StateSpace

State-space, convolutional, and recurrent-sequence kernels in the GPU catalog.

This module groups the ThunderKittens-inspired sequence-model family so the
catalog has a clear home for:

- FFT-style sequence convolution,
- Hedgehog chunk/state kernels,
- Mamba/Mamba2 kernels and their educational sketches.
-/

namespace Tyr.GPU.Kernels

abbrev tkMamba2Fwd := Mamba2.mamba2Fwd

end Tyr.GPU.Kernels
