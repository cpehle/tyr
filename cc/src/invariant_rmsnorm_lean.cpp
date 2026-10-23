#include "invariant_rmsnorm.h"
#include <lean/lean.h>
#include <exception>

at::Tensor borrowTensor(b_lean_obj_arg);
lean_object* giveTensor(at::Tensor);

// Lean 4.29 erases IO's world argument: match the generated three-argument ABI.
extern "C" lean_obj_res lean_torch_invariant_rmsnorm(
    b_lean_obj_arg x, b_lean_obj_arg weight, double eps) {
  try {
    return lean_io_result_mk_ok(giveTensor(tyr::invariant_rmsnorm(borrowTensor(x), borrowTensor(weight), eps)));
  } catch (const c10::Error& error) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
  } catch (const std::exception& error) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(error.what())));
  } catch (...) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("invariant RMSNorm failed with an unknown native exception")));
  }
}
