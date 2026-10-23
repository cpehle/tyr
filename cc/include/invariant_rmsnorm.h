#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace tyr {
// Rank-2 CUDA input, frozen rank-1 zero-centered weight. First-order x gradient.
at::Tensor invariant_rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps);
std::vector<at::Tensor> invariant_rmsnorm_forward_cuda(
    const at::Tensor& x, const at::Tensor& weight_float, float eps);
at::Tensor invariant_rmsnorm_backward_cuda(const at::Tensor& dy,
    const at::Tensor& x, const at::Tensor& weight_float, const at::Tensor& inverse);
}
