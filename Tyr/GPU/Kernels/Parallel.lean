import Tyr.GPU.Kernels.Distributed
import Tyr.GPU.Kernels.MOE

/-!
# Tyr.GPU.Kernels.Parallel

Parallel collectives and distributed fused kernels.

This module groups:

- concrete collective transport kernels (`allGatherFwd`, `allReduceFwd`,
  `reduceScatterFwd`, all-to-all transport),
- communication+compute compositions (`agGemmFwd`, `gemmArFwd`, `gemmRsFwd`),
- explicit ThunderKittens counterpart surfaces for the current H100/B200
  distributed GEMM variants and the educational all-reduce example,
- the ThunderKittens-inspired MOE dispatch/grouped-GEMM pipeline.
-/

namespace Tyr.GPU.Kernels

abbrev tkAllGatherFwd := Distributed.allGatherFwd
abbrev tkAllReduceFwd := Distributed.allReduceFwd
abbrev tkAllReduceEducationalCompatFwd := Distributed.allReduceEducationalCompatFwd
abbrev tkReduceScatterFwd := Distributed.reduceScatterFwd
abbrev tkAllToAllHeadsToSeq := Distributed.allToAllHeadsToSeq
abbrev tkAllToAllSeqToHeads := Distributed.allToAllSeqToHeads
abbrev tkAgGemmFwd := Distributed.agGemmFwd
abbrev tkAgGemmB200CompatFwd := Distributed.agGemmB200CompatFwd
abbrev tkAgGemmFp8B200CompatFwd := Distributed.agGemmFp8B200CompatFwd
abbrev tkGemmArFwd := Distributed.gemmArFwd
abbrev tkGemmArH100LcscCompatFwd := Distributed.gemmArH100LcscCompatFwd
abbrev tkGemmRsFwd := Distributed.gemmRsFwd
abbrev tkGemmRsB200CompatFwd := Distributed.gemmRsB200CompatFwd
abbrev tkGemmRsFp8B200CompatFwd := Distributed.gemmRsFp8B200CompatFwd
abbrev tkMoeDispatch := MOE.tkMoeDispatch
abbrev tkMoeGroupedGemm := MOE.tkMoeGroupedGemm
abbrev tkMoeDispatchGemm := MOE.tkMoeDispatchGemm

end Tyr.GPU.Kernels
