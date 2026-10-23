#include "fused_lora.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace tyr {
namespace {
constexpr int THREADS = 256;
__device__ int tenant_for_row(int row, const int* offsets, int tenants) {
  int lo = 0, hi = tenants;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (offsets[mid + 1] <= row) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
__device__ float block_sum(float value) {
  __shared__ float sums[THREADS];
  sums[threadIdx.x] = value;
  __syncthreads();
  for (int step = THREADS / 2; step; step /= 2) {
    if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
    __syncthreads();
  }
  return sums[0];
}
// A fixed tree per output cell. Grid size only changes which independent
// cells a block visits, never their feature reduction order or dtype boundary.
// transpose_weight=false: x * W^T; true: dy * W.
template <typename scalar_t, bool transpose_weight>
__global__ void fixed_base_product(const scalar_t* x, const scalar_t* weight,
    scalar_t* y, int rows, int columns, int reduction) {
  const int64_t cells = int64_t(rows) * columns;
  for (int64_t cell = blockIdx.x; cell < cells; cell += gridDim.x) {
    const int64_t row = cell / columns;
    const int column = cell % columns;
    float sum = 0.0f;
    for (int64_t k = threadIdx.x; k < reduction; k += THREADS) {
      const int64_t wi = transpose_weight ? int64_t(k) * columns + column
                                         : int64_t(column) * reduction + k;
      sum += float(x[row * reduction + k]) * float(weight[wi]);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) y[cell] = scalar_t(sum);
    // Keep shared reduction scratch quiescent before the next grid-stride cell.
    __syncthreads();
  }
}
int fixed_blocks(int64_t cells) { return int(cells < 65535 ? cells : 65535); }

template <typename scalar_t>
__global__ void project_down(const scalar_t* x, const scalar_t* a, float* h,
    const int* offsets, const int* ranks, int tenants, int input, int max_rank) {
  int row = blockIdx.x, rank = blockIdx.y;
  int tenant = tenant_for_row(row, offsets, tenants);
  float sum = 0;
  if (rank < ranks[tenant]) {
    for (int i = threadIdx.x; i < input; i += blockDim.x)
      sum += float(x[int64_t(row) * input + i]) *
             float(a[(int64_t(tenant) * max_rank + rank) * input + i]);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) h[int64_t(row) * max_rank + rank] = sum;
}
template <typename scalar_t>
__global__ void project_up(const float* h, const scalar_t* b, scalar_t* y,
    const int* offsets, const int* ranks, const float* scales,
    int tenants, int rows, int output, int max_rank) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(rows) * output) return;
  int row = index / output, out = index % output;
  int tenant = tenant_for_row(row, offsets, tenants);
  float sum = 0;
  for (int r = 0; r < ranks[tenant]; ++r)
    sum += h[int64_t(row) * max_rank + r] *
           float(b[(int64_t(tenant) * output + out) * max_rank + r]);
  y[index] = scalar_t(float(y[index]) + sum * scales[tenant]);
}
template <typename scalar_t>
__global__ void backward_down(const scalar_t* dy, const scalar_t* b, float* dh,
    const int* offsets, const int* ranks, const float* scales,
    int tenants, int output, int max_rank) {
  int row = blockIdx.x, rank = blockIdx.y;
  int tenant = tenant_for_row(row, offsets, tenants);
  float sum = 0;
  if (rank < ranks[tenant]) {
    for (int out = threadIdx.x; out < output; out += blockDim.x)
      sum += float(dy[int64_t(row) * output + out]) *
             float(b[(int64_t(tenant) * output + out) * max_rank + rank]);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) dh[int64_t(row) * max_rank + rank] = sum * scales[tenant];
}
template <typename scalar_t>
__global__ void backward_input(const float* dh, const scalar_t* a, scalar_t* dx,
    const int* offsets, const int* ranks, int tenants, int rows, int input, int max_rank) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(rows) * input) return;
  int row = index / input, in = index % input;
  int tenant = tenant_for_row(row, offsets, tenants);
  float sum = 0;
  for (int r = 0; r < ranks[tenant]; ++r)
    sum += dh[int64_t(row) * max_rank + r] *
           float(a[(int64_t(tenant) * max_rank + r) * input + in]);
  // Match the input dtype boundary of x.to(Float32) in the low-rank
  // branch: its gradient is cast back before accumulation with the base
  // matmul gradient. Keep each reduction in FP32, including for BF16.
  dx[index] = scalar_t(float(dx[index]) + float(scalar_t(sum)));
}
// Each factor-gradient entry owns its full segment reduction: deterministic,
// no floating-point atomics, no cross-tenant gradient accumulation.
template <typename scalar_t>
__global__ void backward_a(const scalar_t* x, const float* dh, scalar_t* da,
    const int* offsets, const int* ranks, int tenants, int input, int max_rank) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(tenants) * max_rank * input) return;
  int in = index % input;
  int r = (index / input) % max_rank;
  int tenant = index / (int64_t(input) * max_rank);
  float sum = 0;
  if (r < ranks[tenant]) {
    for (int row = offsets[tenant]; row < offsets[tenant + 1]; ++row)
      sum += dh[int64_t(row) * max_rank + r] * float(x[int64_t(row) * input + in]);
  }
  da[index] = scalar_t(sum);
}
template <typename scalar_t>
__global__ void backward_b(const scalar_t* dy, const float* h, scalar_t* db,
    const int* offsets, const int* ranks, const float* scales,
    int tenants, int output, int max_rank) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(tenants) * output * max_rank) return;
  int r = index % max_rank;
  int out = (index / max_rank) % output;
  int tenant = index / (int64_t(output) * max_rank);
  float sum = 0;
  if (r < ranks[tenant]) {
    for (int row = offsets[tenant]; row < offsets[tenant + 1]; ++row)
      sum += float(dy[int64_t(row) * output + out]) * h[int64_t(row) * max_rank + r];
  }
  db[index] = scalar_t(sum * scales[tenant]);
}
int blocks(int64_t n) { return int((n + THREADS - 1) / THREADS); }
}
std::vector<at::Tensor> segmented_lora_forward_cuda(
    const at::Tensor& x, const at::Tensor& base, const at::Tensor& a,
    const at::Tensor& b, const at::Tensor& offsets, const at::Tensor& ranks,
    const at::Tensor& scales, bool invariant) {
  c10::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), input = x.size(1), output = base.size(0);
  int tenants = a.size(0), max_rank = a.size(1);
  auto y = invariant ? at::empty({rows, output}, x.options())
                     : at::matmul(x, base.transpose(0, 1));
  auto h = at::empty({rows, max_rank}, x.options().dtype(at::kFloat));
  if (rows == 0) return {y, h};
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "segmented_lora_forward", [&] {
    if (invariant) {
      fixed_base_product<scalar_t, false><<<fixed_blocks(int64_t(rows) * output), THREADS, 0, stream>>>(
          x.data_ptr<scalar_t>(), base.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), rows, output, input);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    project_down<scalar_t><<<dim3(rows, max_rank), THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), h.data_ptr<float>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), tenants, input, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    project_up<scalar_t><<<blocks(int64_t(rows) * output), THREADS, 0, stream>>>(
        h.data_ptr<float>(), b.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), scales.data_ptr<float>(),
        tenants, rows, output, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {y, h};
}
std::vector<at::Tensor> segmented_lora_backward_cuda(
    const at::Tensor& dy, const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& h,
    const at::Tensor& offsets, const at::Tensor& ranks, const at::Tensor& scales, bool invariant) {
  c10::cuda::CUDAGuard guard(x.device());
  int rows = x.size(0), input = x.size(1), output = base.size(0);
  int tenants = a.size(0), max_rank = a.size(1);
  auto dx = invariant ? at::empty_like(x) : at::matmul(dy, base);
  auto da = at::zeros_like(a), db = at::zeros_like(b);
  if (rows == 0) return {dx, da, db};
  auto dh = at::empty_like(h);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "segmented_lora_backward", [&] {
    if (invariant) {
      fixed_base_product<scalar_t, true><<<fixed_blocks(int64_t(rows) * input), THREADS, 0, stream>>>(
          dy.data_ptr<scalar_t>(), base.data_ptr<scalar_t>(), dx.data_ptr<scalar_t>(), rows, input, output);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    backward_down<scalar_t><<<dim3(rows, max_rank), THREADS, 0, stream>>>(
        dy.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), dh.data_ptr<float>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), scales.data_ptr<float>(),
        tenants, output, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    backward_input<scalar_t><<<blocks(int64_t(rows) * input), THREADS, 0, stream>>>(
        dh.data_ptr<float>(), a.data_ptr<scalar_t>(), dx.data_ptr<scalar_t>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), tenants, rows, input, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    backward_a<scalar_t><<<blocks(a.numel()), THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), dh.data_ptr<float>(), da.data_ptr<scalar_t>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), tenants, input, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    backward_b<scalar_t><<<blocks(b.numel()), THREADS, 0, stream>>>(
        dy.data_ptr<scalar_t>(), h.data_ptr<float>(), db.data_ptr<scalar_t>(),
        offsets.data_ptr<int>(), ranks.data_ptr<int>(), scales.data_ptr<float>(), tenants, output, max_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {dx, da, db};
}
std::vector<at::Tensor> segmented_lora_forward_cuda(
    const at::Tensor& x, const at::Tensor& base, const at::Tensor& a,
    const at::Tensor& b, const at::Tensor& offsets, const at::Tensor& ranks,
    const at::Tensor& scales) {
  return segmented_lora_forward_cuda(x, base, a, b, offsets, ranks, scales, false);
}
std::vector<at::Tensor> segmented_lora_backward_cuda(
    const at::Tensor& dy, const at::Tensor& x, const at::Tensor& base,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& h,
    const at::Tensor& offsets, const at::Tensor& ranks, const at::Tensor& scales) {
  return segmented_lora_backward_cuda(dy, x, base, a, b, h, offsets, ranks, scales, false);
}

}
