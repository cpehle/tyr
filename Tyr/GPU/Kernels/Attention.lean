import Tyr.GPU.Kernels.Based
import Tyr.GPU.Kernels.FlashAttn
import Tyr.GPU.Kernels.FlashAttn3
import Tyr.GPU.Kernels.FlashAttnBwd
import Tyr.GPU.Kernels.LinearAttn
import Tyr.GPU.Kernels.LinearAttnBwd
import Tyr.GPU.Kernels.MhaH100
import Tyr.GPU.Kernels.RingAttn
import Tyr.GPU.Kernels.RingAttnBwd
import Tyr.GPU.Kernels.Rotary
import Tyr.GPU.Kernels.RotaryBwd
import Tyr.GPU.Kernels.UlyssesAttn
import Tyr.GPU.Kernels.UlyssesAttnBwd

/-!
# Tyr.GPU.Kernels.Attention

Attention-family kernels in the built-in catalog.

This module groups:

- local attention kernels (`FlashAttn*`, `MhaH100`, `LinearAttn`, `Based`)
- rotary position kernels
- distributed attention transport/reduction families (`RingAttn*`, `UlyssesAttn*`)

The root `Tyr.GPU.Kernels` namespace exposes the canonical forward entrypoints
via abbreviations declared here, while the original family namespaces retain
their detailed phase helpers and compatibility shims.
-/

namespace Tyr.GPU.Kernels

abbrev tkBasedLinearAttnFwd := Based.tkBasedLinearAttnFwd
abbrev tkLinearAttnFwd := LinearAttn.tkLinearAttnFwd
abbrev tkRingAttnPartial := RingAttn.ringAttnPartial
abbrev tkRingAttnComm := RingAttn.ringAttnComm
abbrev tkRingAttnReduce := RingAttn.ringAttnReduce
abbrev tkUlyssesAllToAllFwd := UlyssesAttn.allToAllFwd
abbrev tkUlyssesQkvAllToAll := UlyssesAttn.ulyssesQkvAllToAll
abbrev tkUlyssesAttnReturn := UlyssesAttn.ulyssesAttnFwd

end Tyr.GPU.Kernels
