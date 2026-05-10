// tk_vendor_stubs.cpp - throwing stubs for the vendored TK MHA H100 entry points.
//
// The real implementation lives in `tk_vendor_mha_h100.cu`, which uses Hopper
// `wgmma` instructions and only compiles for sm_90a. On Blackwell-class
// targets (e.g. GB10/B200) the cu file is excluded from the build and these
// stubs satisfy the `tyr_tk_attention_*` references in `tyr_ops.cpp`. The
// runtime dispatcher's Hopper-arch gate ensures these stubs are never
// actually entered on non-Hopper hardware.
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

std::vector<at::Tensor> tyr_tk_attention_forward_nosync(
    at::Tensor /*q*/, at::Tensor /*k*/, at::Tensor /*v*/, bool /*causal*/) {
  throw std::runtime_error(
      "tyr::flash_attn: vendored TK forward unavailable in this build "
      "(Hopper-only kernel excluded for non-H100 targets)");
}

std::vector<at::Tensor> tyr_tk_attention_backward_nosync(
    at::Tensor /*q*/, at::Tensor /*k*/, at::Tensor /*v*/,
    at::Tensor /*o*/, at::Tensor /*l_vec*/, at::Tensor /*og*/, bool /*causal*/) {
  throw std::runtime_error(
      "tyr::flash_attn: vendored TK backward unavailable in this build "
      "(Hopper-only kernel excluded for non-H100 targets)");
}
