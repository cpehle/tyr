#include "fused_lora.h"
#include <lean/lean.h>
#include <limits>
#include <stdexcept>

at::Tensor borrowTensor(b_lean_obj_arg);
lean_object* giveTensor(at::Tensor);

static at::Tensor metadata_ints(b_lean_obj_arg values) {
  const auto n = lean_array_size(values);
  auto tensor = at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
  auto* data = tensor.data_ptr<int32_t>();
  for (size_t i = 0; i < n; ++i) {
    auto value = lean_unbox_uint32(lean_array_get_core(values, i));
    if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
      throw std::invalid_argument("segmented LoRA metadata exceeds int32 range");
    data[i] = static_cast<int32_t>(value);
  }
  return tensor;
}

static lean_obj_res segmented_lora_lean(
    b_lean_obj_arg x, b_lean_obj_arg base, b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg offsets, b_lean_obj_arg ranks, b_lean_obj_arg alphas,
    bool invariant) {
  try {
    auto scales = at::empty({static_cast<int64_t>(lean_array_size(alphas))},
        at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
    for (size_t i = 0; i < lean_array_size(alphas); ++i)
      scales.data_ptr<double>()[i] = lean_unbox_float(lean_array_get_core(alphas, i));
    auto operation = invariant ? tyr::segmented_lora_invariant : tyr::segmented_lora;
    auto result = operation(borrowTensor(x), borrowTensor(base),
        borrowTensor(a), borrowTensor(b), metadata_ints(offsets), metadata_ints(ranks), scales);
    return lean_io_result_mk_ok(giveTensor(result));
  } catch (const c10::Error& error) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
  } catch (const std::exception& error) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
  } catch (...) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("segmented LoRA failed with an unknown native exception")));
  }
}

// Lean 4.29 erases the IO world: generated extern calls have seven arguments.
// Do not add a world parameter: a tailcall can overwrite the caller's stack.
extern "C" lean_obj_res lean_torch_segmented_lora(
    b_lean_obj_arg x, b_lean_obj_arg base, b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg offsets, b_lean_obj_arg ranks, b_lean_obj_arg alphas) {
  return segmented_lora_lean(x, base, a, b, offsets, ranks, alphas, false);
}
extern "C" lean_obj_res lean_torch_segmented_lora_invariant(
    b_lean_obj_arg x, b_lean_obj_arg base, b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg offsets, b_lean_obj_arg ranks, b_lean_obj_arg alphas) {
  return segmented_lora_lean(x, base, a, b, offsets, ranks, alphas, true);
}
