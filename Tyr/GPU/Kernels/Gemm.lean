import Tyr.GPU.Kernels.Copy
import Tyr.GPU.Kernels.NvFp4Gemm
import Tyr.GPU.Kernels.PrecisionGemm

/-!
# Tyr.GPU.Kernels.Gemm

Matrix multiplication and low-level data-movement kernels.

This family groups the H100 FP8, B200 NVFP4-compatibility, and small memory
movement kernels that back the higher-level model kernels.
-/

namespace Tyr.GPU.Kernels

abbrev tkH100Fp8E4M3GemmFwd := PrecisionGemm.tkH100Fp8E4M3GemmFwd
abbrev tkH100Fp8ScaledGemmFwd := PrecisionGemm.tkH100Fp8ScaledGemmFwd
abbrev tkB200NvFp4GemmCompatFwd := NvFp4Gemm.tkB200NvFp4GemmCompatFwd

end Tyr.GPU.Kernels
