import Tyr.GPU.Kernels.Distributed
import Tyr.GPU.Kernels.MOE

/-!
# Tyr.GPU.Kernels.Parallel

Parallel collectives and distributed fused kernels.

This module groups:

- concrete collective transport kernels (`allGatherFwd`, `allReduceFwd`,
  `reduceScatterFwd`, all-to-all transport),
- communication+compute compositions (`agGemmFwd`, `gemmArFwd`, `gemmRsFwd`),
- the ThunderKittens-inspired MOE dispatch/grouped-GEMM pipeline.
-/

namespace Tyr.GPU.Kernels

abbrev tkAllGatherFwd := Distributed.allGatherFwd
abbrev tkAllReduceFwd := Distributed.allReduceFwd
abbrev tkReduceScatterFwd := Distributed.reduceScatterFwd
abbrev tkAllToAllHeadsToSeq := Distributed.allToAllHeadsToSeq
abbrev tkAllToAllSeqToHeads := Distributed.allToAllSeqToHeads
abbrev tkAgGemmFwd := Distributed.agGemmFwd
abbrev tkGemmArFwd := Distributed.gemmArFwd
abbrev tkGemmRsFwd := Distributed.gemmRsFwd
abbrev tkMoeDispatch := MOE.tkMoeDispatch
abbrev tkMoeGroupedGemm := MOE.tkMoeGroupedGemm
abbrev tkMoeDispatchGemm := MOE.tkMoeDispatchGemm

end Tyr.GPU.Kernels
