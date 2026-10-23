#include "invariant_rmsnorm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace tyr {
namespace {
constexpr int THREADS = 256;
__device__ float row_sum(float value) {
  __shared__ float partial[THREADS];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int step = THREADS / 2; step > 0; step /= 2) {
    if (threadIdx.x < step)
      partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + step]);
    __syncthreads();
  }
  return partial[0];
}

template <typename scalar_t>
__global__ void forward_rows(const scalar_t* x, const float* weight,
    scalar_t* y, float* inverse, int rows, int width, float eps) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += THREADS) {
      const float value = float(x[row * width + col]);
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sum = row_sum(sum);
    const float inv = rsqrtf(__fadd_rn(__fdiv_rn(sum, float(width)), eps));
    if (threadIdx.x == 0) inverse[row] = inv;
    for (int64_t col = threadIdx.x; col < width; col += THREADS) {
      const float value = float(x[row * width + col]);
      const float scale = __fadd_rn(weight[col], 1.0f);
      y[row * width + col] = scalar_t(__fmul_rn(__fmul_rn(value, inv), scale));
    }
    // Every thread finishes reading shared scratch before the next row reuses it.
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void backward_rows(const scalar_t* dy, const scalar_t* x,
    const float* weight, const float* inverse, scalar_t* dx, int rows, int width) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float inv = inverse[row];
    float dot = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += THREADS) {
      const float q = __fmul_rn(float(dy[row * width + col]), __fadd_rn(weight[col], 1.0f));
      const float z = __fmul_rn(float(x[row * width + col]), inv);
      dot = __fadd_rn(dot, __fmul_rn(q, z));
    }
    dot = row_sum(dot);
    // Normalized coordinates avoid forming inv^3 (overflow for tiny epsilon).
    const float correction = __fdiv_rn(dot, float(width));
    for (int64_t col = threadIdx.x; col < width; col += THREADS) {
      const float q = __fmul_rn(float(dy[row * width + col]), __fadd_rn(weight[col], 1.0f));
      const float z = __fmul_rn(float(x[row * width + col]), inv);
      dx[row * width + col] = scalar_t(__fmul_rn(inv,
          __fsub_rn(q, __fmul_rn(z, correction))));
    }
    __syncthreads();
  }
}
int row_blocks(int rows) { return rows < 65535 ? rows : 65535; }
}

std::vector<at::Tensor> invariant_rmsnorm_forward_cuda(
    const at::Tensor& x, const at::Tensor& weight_float, float eps) {
  c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  auto inverse = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
  if (x.size(0) == 0) return {y, inverse};
  const int rows = x.size(0), width = x.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "invariant_rmsnorm_forward", [&] {
    forward_rows<scalar_t><<<row_blocks(rows), THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), weight_float.data_ptr<float>(), y.data_ptr<scalar_t>(),
        inverse.data_ptr<float>(), rows, width, eps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, inverse};
}

at::Tensor invariant_rmsnorm_backward_cuda(const at::Tensor& dy,
    const at::Tensor& x, const at::Tensor& weight_float, const at::Tensor& inverse) {
  c10::cuda::CUDAGuard guard(x.device());
  auto dx = at::empty_like(x);
  if (x.size(0) == 0) return dx;
  const int rows = x.size(0), width = x.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "invariant_rmsnorm_backward", [&] {
    backward_rows<scalar_t><<<row_blocks(rows), THREADS, 0, stream>>>(
        dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), weight_float.data_ptr<float>(),
        inverse.data_ptr<float>(), dx.data_ptr<scalar_t>(), rows, width);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}
}
