#pragma once
#include <ATen/ATen.h>
#include <vector>
namespace tyr {
// All rows of one tenant are contiguous; offsets include an end sentinel.
// Rank zero disables adaptation for a tenant at this projection.
std::vector<at::Tensor> segmented_lora_forward_cuda(
    const at::Tensor& x, const at::Tensor& base, const at::Tensor& a,
    const at::Tensor& b, const at::Tensor& offsets, const at::Tensor& ranks,
    const at::Tensor& scales);
std::vector<at::Tensor> segmented_lora_backward_cuda(
    const at::Tensor& dy, const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& h,
    const at::Tensor& offsets, const at::Tensor& ranks, const at::Tensor& scales);
at::Tensor segmented_lora(const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& offsets,
    const at::Tensor& ranks, const at::Tensor& alphas);
}
