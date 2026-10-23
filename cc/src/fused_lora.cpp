#include "fused_lora.h"
#include <torch/csrc/autograd/custom_function.h>
#include <cmath>
#include <limits>

namespace tyr {
namespace {
class SegmentedLoRA : public torch::autograd::Function<SegmentedLoRA> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
      at::Tensor x, at::Tensor base, at::Tensor a, at::Tensor b,
      at::Tensor offsets, at::Tensor ranks, at::Tensor scales, bool invariant) {
    auto result = segmented_lora_forward_cuda(x, base, a, b, offsets, ranks, scales, invariant);
    ctx->saved_data["invariant"] = invariant;
    ctx->save_for_backward({x, base, a, b, result[1], offsets, ranks, scales});
    return result[0];
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list outputs) {
    auto saved = ctx->get_saved_variables();
    auto grads = segmented_lora_backward_cuda(outputs[0].contiguous(), saved[0], saved[1],
        saved[2], saved[3], saved[4], saved[5], saved[6], saved[7],
        ctx->saved_data["invariant"].toBool());
    // The shared base and packing metadata never acquire gradients.
    return {grads[0], at::Tensor(), grads[1], grads[2], at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};
void check_tensor(const at::Tensor& tensor, const at::Tensor& x, int dims, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device(), name, " must share the input CUDA device");
  TORCH_CHECK(tensor.scalar_type() == x.scalar_type(), name, " dtype must match input");
  TORCH_CHECK(tensor.dim() == dims && tensor.is_contiguous(), name, " must have the expected rank and be contiguous");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name, " exceeds kernel indexing limit");
}
}
static at::Tensor segmented_lora_impl(const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& offsets,
    const at::Tensor& ranks, const at::Tensor& alphas, bool invariant) {
  TORCH_CHECK(x.is_cuda(), "segmented LoRA requires CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
      "segmented LoRA supports Float32 and BFloat16");
  check_tensor(x, x, 2, "input");
  check_tensor(base, x, 2, "base");
  check_tensor(a, x, 3, "A");
  check_tensor(b, x, 3, "B");
  auto tenants = a.size(0), max_rank = a.size(1);
  TORCH_CHECK(tenants > 0 && max_rank > 0 && max_rank <= 64, "invalid tenant count or maximum rank (1..64)");
  TORCH_CHECK(x.size(1) > 0 && base.size(0) > 0, "linear dimensions must be positive");
  TORCH_CHECK(base.size(1) == x.size(1) && a.size(2) == x.size(1), "input feature shape mismatch");
  TORCH_CHECK(b.size(0) == tenants && b.size(1) == base.size(0) && b.size(2) == max_rank,
      "B must have shape [tenants, output, maximum rank]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type() == at::kInt && offsets.dim() == 1 &&
      offsets.is_contiguous() && offsets.numel() == tenants + 1, "offsets must be CPU contiguous int32 [tenants+1]");
  TORCH_CHECK(ranks.device().is_cpu() && ranks.scalar_type() == at::kInt && ranks.dim() == 1 &&
      ranks.is_contiguous() && ranks.numel() == tenants, "ranks must be CPU contiguous int32 [tenants]");
  TORCH_CHECK(alphas.device().is_cpu() && alphas.scalar_type() == at::kDouble && alphas.dim() == 1 &&
      alphas.is_contiguous() && alphas.numel() == tenants, "alphas must be CPU contiguous float64 [tenants]");
  auto off = offsets.data_ptr<int>();
  auto rank = ranks.data_ptr<int>();
  auto alpha = alphas.data_ptr<double>();
  TORCH_CHECK(off[0] == 0 && off[tenants] == x.size(0), "offsets must cover all packed input rows");
  auto scales = at::empty({tenants}, at::TensorOptions().dtype(at::kFloat));
  auto scale = scales.data_ptr<float>();
  for (int64_t tenant = 0; tenant < tenants; ++tenant) {
    TORCH_CHECK(off[tenant] >= 0 && off[tenant] <= off[tenant + 1], "offsets must be nondecreasing");
    TORCH_CHECK(rank[tenant] >= 0 && rank[tenant] <= max_rank, "tenant rank exceeds packed rank capacity");
    TORCH_CHECK(std::isfinite(alpha[tenant]) && alpha[tenant] > 0, "tenant alpha must be finite and positive");
    double value = rank[tenant] == 0 ? 0 : alpha[tenant] / rank[tenant];
    TORCH_CHECK(value <= std::numeric_limits<float>::max(), "tenant LoRA scale overflows float32");
    scale[tenant] = float(value);
  }
  return SegmentedLoRA::apply(x, base.detach(), a, b, offsets.to(x.device()),
      ranks.to(x.device()), scales.to(x.device()), invariant);
}
at::Tensor segmented_lora(const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& offsets,
    const at::Tensor& ranks, const at::Tensor& alphas) {
  return segmented_lora_impl(x, base, a, b, offsets, ranks, alphas, false);
}
at::Tensor segmented_lora_invariant(const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& offsets,
    const at::Tensor& ranks, const at::Tensor& alphas) {
  return segmented_lora_impl(x, base, a, b, offsets, ranks, alphas, true);
}

}
