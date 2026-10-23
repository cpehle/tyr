#include "invariant_rmsnorm.h"
#include <stdexcept>
namespace tyr {
at::Tensor invariant_rmsnorm(const at::Tensor&, const at::Tensor&, double) {
  throw std::runtime_error("invariant RMSNorm requires a CUDA-enabled Tyr build");
}
}
