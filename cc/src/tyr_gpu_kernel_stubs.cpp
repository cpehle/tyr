#include <lean/lean.h>
#include <string>

static lean_object* gpuKernelUnavailable(const char* launcher) {
  std::string msg = "GPU kernel launcher unavailable in this build (missing NVCC/CUDA): ";
  msg += launcher;
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

extern "C" {

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_copy64x64(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_copy64x64");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_Rotary_rotaryFwd(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_Rotary_rotaryFwd");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd2Block(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd2Block");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd2BlockLse(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd2BlockLse");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd12Block(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd12Block");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd12BlockLse(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkFlashAttnFwd12BlockLse");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd2Block");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkMhaH100Fwd12Block");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkMhaH100BwdPrep2Block");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd2BlockPartials");
}

__attribute__((weak)) lean_object* lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials(...) {
  return gpuKernelUnavailable("lean_launch_Tyr_GPU_Kernels_tkMhaH100Bwd12BlockPartials");
}

} // extern "C"
