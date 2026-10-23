#include "invariant_rmsnorm.h"
#include <torch/csrc/autograd/custom_function.h>
#include <cmath>
#include <limits>

namespace tyr {
namespace {
class InvariantRMSNorm : public torch::autograd::Function<InvariantRMSNorm> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
      at::Tensor x, at::Tensor weight, double eps) {
    auto weight_float = weight.to(at::kFloat);
    auto result = invariant_rmsnorm_forward_cuda(x, weight_float, static_cast<float>(eps));
    ctx->save_for_backward({x, weight_float, result[1]});
    return result[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list outputs) {
    auto saved = ctx->get_saved_variables();
    auto dx = invariant_rmsnorm_backward_cuda(outputs[0].contiguous(), saved[0], saved[1], saved[2]);
    return {dx, at::Tensor(), at::Tensor()};
  }
};
bool supported(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kBFloat16;
}
}

at::Tensor invariant_rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "invariant RMSNorm requires CUDA input");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "invariant RMSNorm input must be contiguous [rows,width]");
  TORCH_CHECK(supported(x.scalar_type()), "invariant RMSNorm input must be Float32 or BFloat16");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max() &&
      x.size(0) <= std::numeric_limits<int>::max() && x.numel() <= std::numeric_limits<int>::max(),
      "invariant RMSNorm dimensions exceed indexing limits or width is zero");
  TORCH_CHECK(weight.is_cuda() && weight.device() == x.device(),
      "invariant RMSNorm weight must share the input CUDA device");
  TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous() && weight.size(0) == x.size(1),
      "invariant RMSNorm weight must be contiguous [width]");
  TORCH_CHECK(supported(weight.scalar_type()), "invariant RMSNorm weight must be Float32 or BFloat16");
  TORCH_CHECK(!weight.requires_grad(), "invariant RMSNorm requires a frozen weight");
  const float eps_float = static_cast<float>(eps);
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(eps_float) && eps_float > 0,
      "invariant RMSNorm epsilon must be finite and positive in Float32");
  return InvariantRMSNorm::apply(x, weight, eps);
}
}
