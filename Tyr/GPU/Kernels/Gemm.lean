import Tyr.GPU.Kernels.Bf16Gemm
import Tyr.GPU.Kernels.Copy
import Tyr.GPU.Kernels.NvFp4Gemm
import Tyr.GPU.Kernels.PrecisionGemm

/-!
# Tyr.GPU.Kernels.Gemm

Matrix multiplication and low-level data-movement kernels.

This family groups the vendored ThunderKittens GEMM counterparts:

- H100 BF16 and FP8 surfaces
- B200 BF16, FP8, MXFP8, and NVFP4 surfaces
- small memory movement kernels that back the higher-level model kernels
-/

namespace Tyr.GPU.Kernels

abbrev tkH100Bf16GemmFwd := Bf16Gemm.tkH100Bf16GemmFwd
abbrev tkB200Bf16GemmFwd := Bf16Gemm.tkB200Bf16GemmFwd
abbrev tkH100Fp8E4M3GemmFwd := PrecisionGemm.tkH100Fp8E4M3GemmFwd
abbrev tkH100Fp8ScaledGemmFwd := PrecisionGemm.tkH100Fp8ScaledGemmFwd
abbrev tkB200Fp8E4M3Gemm1CtaFwd := PrecisionGemm.tkB200Fp8E4M3Gemm1CtaFwd
abbrev tkB200Fp8E4M3Gemm2CtaFwd := PrecisionGemm.tkB200Fp8E4M3Gemm2CtaFwd
abbrev tkB200MxFp8GemmFwd := PrecisionGemm.tkB200MxFp8GemmFwd
abbrev tkB200NvFp4GemmFwd := NvFp4Gemm.tkB200NvFp4GemmFwd

end Tyr.GPU.Kernels
