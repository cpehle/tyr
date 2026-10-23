#include "fused_lora.h"
#include <stdexcept>
namespace tyr {
at::Tensor segmented_lora(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {
  throw std::runtime_error("segmented LoRA requires a CUDA-enabled Tyr build");
}
at::Tensor segmented_lora_invariant(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {
  throw std::runtime_error("segmented LoRA requires a CUDA-enabled Tyr build");
}

}
