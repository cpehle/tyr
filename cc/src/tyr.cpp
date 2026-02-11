/*
 * tyr.cpp - Lean 4 ↔ LibTorch FFI Bindings
 *
 * ============================================================================
 * REFERENCE COUNTING CONVENTIONS
 * ============================================================================
 *
 * This file bridges Lean's reference counting with LibTorch's intrusive_ptr.
 * Understanding these conventions is critical for memory safety.
 *
 * ## Parameter Types
 *
 * - `lean_obj_arg` (owned): Caller receives ownership. MUST call `lean_dec()`
 *   after extracting data. Used for temporary parameters like shape arrays.
 *   Example:
 *     lean_object* lean_torch_randn(lean_obj_arg s, ...) {
 *       auto shape = getShape(s);
 *       lean_dec(s);  // REQUIRED: we own s, must release it
 *       ...
 *     }
 *
 * - `b_lean_obj_arg` (borrowed): Caller does NOT own the object. Must NOT
 *   call `lean_dec()`. Use `borrowTensor()` to get a C++ tensor with shared
 *   ownership that auto-decrements when it goes out of scope.
 *   Example:
 *     lean_object* lean_torch_add(b_lean_obj_arg x, b_lean_obj_arg y) {
 *       auto x_ = borrowTensor(x);  // Shared ownership, no manual cleanup
 *       auto y_ = borrowTensor(y);  // Same
 *       return giveTensor(x_ + y_); // Transfer new tensor to Lean
 *     }
 *
 * ## Key Functions
 *
 * - `borrowTensor(b_lean_obj_arg)`: Get C++ tensor with shared ownership.
 *   When the returned torch::Tensor goes out of scope, refcount is
 *   automatically decremented. No manual cleanup needed.
 *
 * - `giveTensor(torch::Tensor)`: Transfer ownership to Lean. Lean's finalizer
 *   will decrement the refcount when the Lean object is garbage collected.
 *   Increments g_live_lean_tensors counter.
 *
 * ## Memory Lifecycle
 *
 * 1. Tensor created in C++ → giveTensor() → Lean owns it
 * 2. Lean passes tensor to C++ → borrowTensor() → shared ownership
 * 3. Lean GC finalizes tensor → decrefFinalize() → releases C++ memory
 *
 * ## Debugging
 *
 * g_live_lean_tensors tracks outstanding tensors. If this grows unboundedly,
 * there's a memory leak. Query via lean_torch_get_live_tensors().
 *
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <atomic>
#include <lean/lean.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

// Global atomic counter for live tensors handed to Lean
std::atomic<int64_t> g_live_lean_tensors(0);

void trivialFinalize(void *p) { return; }
void trivialForeach(void *p, b_lean_obj_arg a) { return; }
static lean_external_class *getTrivialObjectClass() {
  static lean_external_class *c(
      lean_register_external_class(&trivialFinalize, &trivialForeach));
  return c;
}
template<typename T>
void deleteFinalize(void *p) { delete static_cast<T *>(p); }

template<typename T>
void decrefFinalize(void *p) { 
  auto ptr = c10::intrusive_ptr<T>::reclaim(static_cast<T *>(p));
  g_live_lean_tensors--; // Decrement when a tensor is finalized
}



template<typename T>
lean_external_class *registerDecRefClass() {
  return lean_register_external_class(&decrefFinalize<T>, &trivialForeach);
}

template<typename T>
lean_external_class *registerDeleteClass() {
  return lean_register_external_class(&deleteFinalize<T>, &trivialForeach);
}


static
lean_external_class* getTorchTensorImplClass() {
    // Use static thread to make this thread safe (hopefully).
    static lean_external_class* c = registerDecRefClass<torch::TensorImpl>();
    return c;
}

// Borrow a tensor from Lean with proper reference counting.
// Increments refcount so both Lean and C++ have shared ownership.
// When the returned tensor goes out of scope, refcount is decremented automatically.
// No manual unsafeReleaseTensorImpl() needed!
torch::Tensor borrowTensor(b_lean_obj_arg o) {
    auto impl = static_cast<torch::TensorImpl*>(lean_get_external_data(o));
    // unsafe_reclaim_from_nonowning increments the refcount
    return torch::Tensor(c10::intrusive_ptr<torch::TensorImpl>::unsafe_reclaim_from_nonowning(impl));
}

// Transfer ownership of a new tensor to Lean.
// The tensor's refcount is transferred to Lean's finalizer.
lean_object *giveTensor(torch::Tensor t) {
  g_live_lean_tensors++; // Increment when a new tensor is given to Lean
  return lean_alloc_external(getTorchTensorImplClass(), t.unsafeReleaseTensorImpl());
}

// Alias for backward compatibility
lean_object *fromTorchTensor(torch::Tensor t) {
  return giveTensor(t);
}

std::vector<int64_t> getShape(b_lean_obj_arg s) {
  std::vector<int64_t> shape;  
  for (size_t i = 0; i<lean_array_size(s); i++) {
    shape.push_back(lean_unbox_uint64(lean_array_get_core(s, i)));
  }
  return shape;
}

torch::Device getDevice(b_lean_obj_arg device) {
  auto tag = lean_obj_tag(device);
  if (tag == 0) {  // CUDA
    auto cuda_idx = lean_ctor_get_uint64(device, 0);
    return torch::Device(torch::kCUDA, cuda_idx);
  } else if (tag == 1) {  // CPU
    return torch::Device(torch::kCPU);
  } else {  // MPS (tag == 2)
#ifdef __APPLE__
    return torch::Device(torch::kMPS);
#else
    // MPS not available on non-Apple platforms, fallback to CPU
    return torch::Device(torch::kCPU);
#endif
  }
}

extern "C" {

lean_object* lean_torch_get_live_tensors(lean_object* /* w */) {
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_live_lean_tensors.load())));
}

// ============================================================================
// Tensor metadata extraction for visualization widgets
// ============================================================================

// Get runtime shape as Lean Array UInt64
lean_object* lean_torch_get_shape(lean_obj_arg /*s*/, b_lean_obj_arg t) {
    auto tensor = borrowTensor(t);
    auto sizes = tensor.sizes();

    lean_object* arr = lean_alloc_array(sizes.size(), sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
        lean_array_set_core(arr, i, lean_box_uint64(static_cast<uint64_t>(sizes[i])));
    }
    return arr;
}

// Get dtype as string
lean_object* lean_torch_get_dtype(lean_obj_arg /*s*/, b_lean_obj_arg t) {
    auto tensor = borrowTensor(t);
    std::string dtype_str;

    auto dtype = tensor.dtype();
    if (dtype == torch::kFloat32) dtype_str = "Float32";
    else if (dtype == torch::kFloat64) dtype_str = "Float64";
    else if (dtype == torch::kFloat16) dtype_str = "Float16";
    else if (dtype == torch::kBFloat16) dtype_str = "BFloat16";
    else if (dtype == torch::kInt64) dtype_str = "Int64";
    else if (dtype == torch::kInt32) dtype_str = "Int32";
    else if (dtype == torch::kInt16) dtype_str = "Int16";
    else if (dtype == torch::kInt8) dtype_str = "Int8";
    else if (dtype == torch::kUInt8) dtype_str = "UInt8";
    else if (dtype == torch::kBool) dtype_str = "Bool";
    else dtype_str = "Unknown";

    return lean_mk_string(dtype_str.c_str());
}

// Get device as string
lean_object* lean_torch_get_device(lean_obj_arg /*s*/, b_lean_obj_arg t) {
    auto tensor = borrowTensor(t);
    auto device = tensor.device();

    std::ostringstream stream;
    stream << device;
    return lean_mk_string(stream.str().c_str());
}

// Get device as Device enum (CUDA=0, CPU=1, MPS=2)
lean_object* lean_torch_get_device_enum(lean_obj_arg /*s*/, b_lean_obj_arg t) {
    auto tensor = borrowTensor(t);
    auto device = tensor.device();

    if (device.is_cuda()) {
        // CUDA tag=0, constructor with 1 field (device index)
        lean_object* obj = lean_alloc_ctor(0, 0, 8);
        lean_ctor_set_uint64(obj, 0, device.index());
        return obj;
    } else if (device.is_cpu()) {
        // CPU tag=1, scalar constructor
        return lean_box(1);
    } else {
        // MPS tag=2, scalar constructor
        return lean_box(2);
    }
}

// Get tensor values as flat Array Float (up to max_elements)
lean_object* lean_torch_get_values(lean_obj_arg /*s*/, b_lean_obj_arg t, uint64_t max_elements) {
    auto tensor = borrowTensor(t);

    // Move to CPU if needed and convert to float
    auto cpu_tensor = tensor.cpu().to(torch::kFloat32).contiguous();
    auto flat = cpu_tensor.flatten();

    size_t num_elements = std::min(static_cast<size_t>(flat.numel()),
                                   static_cast<size_t>(max_elements));

    // Use FloatArray for efficiency
    lean_object* arr = lean_alloc_sarray(sizeof(double), num_elements, num_elements);
    double* data = reinterpret_cast<double*>(lean_sarray_cptr(arr));

    auto accessor = flat.accessor<float, 1>();
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = static_cast<double>(accessor[i]);
    }

    return arr;
}

// Get tensor statistics as JSON string
lean_object* lean_torch_get_stats(lean_obj_arg /*s*/, b_lean_obj_arg t) {
    auto tensor = borrowTensor(t);
    auto cpu_tensor = tensor.cpu().to(torch::kFloat32);

    double min_val = cpu_tensor.min().item<double>();
    double max_val = cpu_tensor.max().item<double>();
    double mean_val = cpu_tensor.mean().item<double>();
    double std_val = cpu_tensor.std().item<double>();

    std::ostringstream stream;
    stream << "{\"min\":" << min_val
           << ",\"max\":" << max_val
           << ",\"mean\":" << mean_val
           << ",\"std\":" << std_val << "}";

    return lean_mk_string(stream.str().c_str());
}


// --
// tensor creation api
lean_object* lean_torch_randn(lean_obj_arg s, int requires_grad, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  auto t = torch::randn(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}
lean_object* lean_torch_rand(lean_obj_arg s, int requires_grad, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  auto t = torch::rand(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_randint(int64_t low, int64_t high, lean_obj_arg s, int /*requires_grad*/, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  // randint always returns Long (int64) dtype - integral types don't support requires_grad
  auto t = torch::randint(low, high, getShape(s), torch::TensorOptions().dtype(torch::kLong).device(device_));
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_full(lean_obj_arg s, double value, int requires_grad, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  auto t = torch::full(getShape(s), value, torch::TensorOptions().requires_grad(requires_grad).device(device_));
  lean_dec(s);
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros(lean_obj_arg s, int requires_grad, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  auto t = torch::zeros(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  lean_dec(s);
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones(lean_obj_arg s, int requires_grad, lean_obj_arg device) {
  auto device_ = getDevice(device);
  lean_dec(device);
  auto t = torch::ones(getShape(s), torch::TensorOptions().requires_grad(requires_grad).device(device_));
  lean_dec(s);
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros_like(lean_obj_arg /*s*/, b_lean_obj_arg self) {
  auto self_ = borrowTensor(self);
  auto t = torch::zeros_like(self_);
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones_like(lean_obj_arg /*s*/, b_lean_obj_arg self) {
  auto self_ = borrowTensor(self);
  auto t = torch::ones_like(self_);
  return fromTorchTensor(t);
}

lean_object* lean_torch_arange(int start, int stop, int step) {
  auto t = torch::arange(start, stop, step, torch::kLong);
  return fromTorchTensor(t);
}

lean_object* lean_torch_eye(int n, int requires_grad) {
  auto t = torch::eye(n, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_linspace(double start, double stop, int steps, int requires_grad) {
  auto t = torch::linspace(start, stop, steps, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_logspace(double start, double stop, int steps, double base, int requires_grad) {
  auto t = torch::logspace(start, stop, steps, base, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}


bool lean_torch_requires_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto res_ = x_.requires_grad();
    return res_;
}

//

lean_object* lean_torch_reshape(lean_obj_arg /*s*/, b_lean_obj_arg self, b_lean_obj_arg shape) {
  auto shape_ = getShape(shape);
  auto self_ = borrowTensor(self);
  auto res = torch::reshape(self_, shape_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_permute(lean_obj_arg /*s*/, b_lean_obj_arg self, b_lean_obj_arg permutation) {
  auto permutation_ = getShape(permutation);
  auto self_ = borrowTensor(self);
  auto res = torch::permute(self_, permutation_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_get(lean_obj_arg /*s*/, b_lean_obj_arg self, int idx) {
  auto self_ = borrowTensor(self);
  auto res = self_.index({idx});
  return fromTorchTensor(res);
}

lean_object* lean_torch_to(lean_obj_arg /*s*/, b_lean_obj_arg self, lean_obj_arg device) {
  auto self_ = borrowTensor(self);
  auto device_ = getDevice(device);
  lean_dec(device);
  auto res = self_.to(device_);
  return fromTorchTensor(res);
}

lean_object* lean_torch_slice(lean_obj_arg /*s*/,
  b_lean_obj_arg self,
  b_lean_obj_arg dim,
  int64_t start,
  int64_t stop,
  int64_t step) {

  auto dim_ = lean_uint64_of_nat(dim);
  auto self_ = borrowTensor(self);
  auto res = self_.slice(dim_, start, stop, step);
  return fromTorchTensor(res);
}

#define BINOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) { \
  auto a_ = borrowTensor(a); \
  auto b_ = borrowTensor(b); \
  auto c_ = torch::F(a_, b_); \
  return fromTorchTensor(c_); \
}

BINOP_FUN(add)
BINOP_FUN(sub)
BINOP_FUN(mul)
#undef BINOP_FUN

#define UNOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a) { \
  auto a_ = borrowTensor(a); \
  auto c_ = torch::F(a_); \
  return fromTorchTensor(c_); \
}

UNOP_FUN(celu)
UNOP_FUN(elu)
UNOP_FUN(gelu)
UNOP_FUN(hardtanh)
UNOP_FUN(leaky_relu)
UNOP_FUN(relu)
UNOP_FUN(relu6)
UNOP_FUN(rrelu)
UNOP_FUN(selu)
UNOP_FUN(silu)
UNOP_FUN(sigmoid)
UNOP_FUN(sin)
UNOP_FUN(cos)
UNOP_FUN(atan)
UNOP_FUN(tanh)
UNOP_FUN(exp)
UNOP_FUN(log)
#undef UNOP_FUN

lean_object* lean_torch_tensor_grad(lean_obj_arg /* s */, lean_obj_arg /* s' */, b_lean_obj_arg output, b_lean_obj_arg input, b_lean_obj_arg grad_output) {
  auto output_ = borrowTensor(output);
  auto input_ = borrowTensor(input);
  auto grad_output_ = borrowTensor(grad_output);

  torch::autograd::variable_list out_v({output_});
  torch::autograd::variable_list in_v({input_});
  torch::autograd::variable_list grad_out_v({grad_output_});    
  // Keep the graph for callers that request multiple grads from the same output
  // (e.g. structured VJP over many leaves).
  auto grad_in_v = torch::autograd::grad(
      out_v, in_v, grad_out_v,
      /*retain_graph=*/true,
      /*create_graph=*/false,
      /*allow_unused=*/true);

  auto grad_in = grad_in_v[0];
  if (!grad_in.defined()) {
    // Lean callers expect a concrete tensor result even when the input is unused.
    grad_in = torch::zeros_like(input_);
  }
  return fromTorchTensor(grad_in);
}

lean_object* lean_torch_backward(lean_obj_arg /* shape */, b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  auto out_ = borrowTensor(output);
  auto grad_output_ = borrowTensor(grad_output);
  out_.backward(grad_output_);

  lean_inc(output);
  return output;
}

int lean_torch_allclose(lean_obj_arg /* shape */, b_lean_obj_arg a, b_lean_obj_arg b, double rtol, double atol) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  return torch::allclose(a_, b_, rtol, atol);
}

lean_object* lean_torch_grad_of(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto out_ = borrowTensor(x);
    auto grad_out_ = out_.grad();
    return fromTorchTensor(grad_out_);
}

// Zero gradients
lean_object* lean_torch_zero_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    if (x_.grad().defined()) {
        x_.grad().zero_();
    }
    lean_inc(x);
    return x;
}

// Set requires_grad
lean_object* lean_torch_set_requires_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, uint8_t requires_grad) {
    auto x_ = borrowTensor(x);
    auto result_ = x_.requires_grad_(requires_grad);
    return fromTorchTensor(result_);
}

// Detach tensor from computation graph
lean_object* lean_torch_detach(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result_ = x_.detach();
    return fromTorchTensor(result_);
}

// Clone tensor with gradient
lean_object* lean_torch_clone(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result_ = x_.clone();
    return fromTorchTensor(result_);
}

// Retain grad (for non-leaf tensors)
lean_object* lean_torch_retain_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    x_.retain_grad();
    lean_inc(x);
    return x;
}

// Check if tensor is leaf (for autograd)
uint8_t lean_torch_is_leaf(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result = x_.is_leaf();
    return result;
}

// Get gradient function (for debugging)
uint8_t lean_torch_has_grad_fn(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = borrowTensor(x);
    auto result = x_.grad_fn() != nullptr;
    return result;
}

// Accumulate gradients
lean_object* lean_torch_accumulate_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = borrowTensor(x);
    auto grad_ = borrowTensor(grad);
    
    if (x_.grad().defined()) {
        x_.mutable_grad() += grad_;
    } else {
        x_.mutable_grad() = grad_.clone();
    }
    
    lean_inc(x);
    return x;
}

// Manual gradient setting
lean_object* lean_torch_set_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = borrowTensor(x);
    auto grad_ = borrowTensor(grad);
    
    x_.mutable_grad() = grad_.clone();
    
    lean_inc(x);
    return x;
}

// Enable/disable gradient computation globally
lean_object* lean_torch_set_grad_enabled(uint8_t enabled) {
    torch::autograd::GradMode::set_enabled(enabled);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_torch_is_grad_enabled() {
    return lean_io_result_mk_ok(lean_box(torch::autograd::GradMode::is_enabled()));
}

// No-grad context functions
lean_object* lean_torch_with_no_grad(lean_obj_arg /* shape */, lean_obj_arg func) {
    torch::NoGradGuard no_grad_guard;
    // Note: This would need special handling in Lean to call the function
    // For now, this is a placeholder for the infrastructure
    return func;
}

// Higher-order gradients
lean_object* lean_torch_grad_grad(lean_obj_arg /* sx */, lean_obj_arg /* sy */, lean_obj_arg /* sz */,
                                  b_lean_obj_arg z, b_lean_obj_arg y, b_lean_obj_arg x,
                                  b_lean_obj_arg grad_z, b_lean_obj_arg grad_y) {
    auto z_ = borrowTensor(z);
    auto y_ = borrowTensor(y);
    auto x_ = borrowTensor(x);
    auto grad_z_ = borrowTensor(grad_z);
    auto grad_y_ = borrowTensor(grad_y);

    // First compute dz/dy
    torch::autograd::variable_list out_v1({z_});
    torch::autograd::variable_list in_v1({y_});
    torch::autograd::variable_list grad_out_v1({grad_z_});
    auto grad_y_computed = torch::autograd::grad(out_v1, in_v1, grad_out_v1, true, true);

    // Then compute d(dz/dy)/dx
    torch::autograd::variable_list out_v2({grad_y_computed[0]});
    torch::autograd::variable_list in_v2({x_});
    torch::autograd::variable_list grad_out_v2({grad_y_});
    auto grad_x_second = torch::autograd::grad(out_v2, in_v2, grad_out_v2);
    // Intermediate tensors (grad_y_computed, etc.) auto-released on scope exit
    return fromTorchTensor(grad_x_second[0]);
}



lean_object* lean_torch_tensor_softmax(lean_obj_arg /**/, b_lean_obj_arg a, int32_t dim) {
  auto a_ = borrowTensor(a);
  auto c_ = torch::softmax(a_, dim);
  return fromTorchTensor(c_);
}

lean_object* lean_torch_tensor_cross_entropy(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto c_ = torch::nn::functional::cross_entropy(a_, b_);
  return fromTorchTensor(c_);
}

// Cross entropy with no reduction (per-element loss)
lean_object* lean_torch_cross_entropy_none(lean_obj_arg /*n*/, lean_obj_arg /*c*/, b_lean_obj_arg logits, b_lean_obj_arg targets) {
  auto logits_ = borrowTensor(logits);
  auto targets_ = borrowTensor(targets);
  auto options = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
  auto result_ = torch::nn::functional::cross_entropy(logits_, targets_, options);
  return fromTorchTensor(result_);
}


lean_object *lean_torch_to_string(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = borrowTensor(t);
  std::ostringstream stream;
  stream << tensor;

  return lean_mk_string(stream.str().c_str());
}

lean_object* lean_torch_tensor_print(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = borrowTensor(t);
  std::cout << tensor << std::endl;
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_torch_linear(
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
  lean_obj_arg /*b*/,
  b_lean_obj_arg x,
  b_lean_obj_arg M
) {
  auto M_ = borrowTensor(M);
  auto x_ = borrowTensor(x);
  auto y_ = torch::linear(x_, M_);
  return fromTorchTensor(y_);
}

lean_object* lean_torch_affine(
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
  lean_obj_arg /*b*/,
  b_lean_obj_arg x,
  b_lean_obj_arg M,
  b_lean_obj_arg b
) {
  auto M_ = borrowTensor(M);
  auto b_ = borrowTensor(b);
  auto x_ = borrowTensor(x);
  auto y_ = torch::linear(x_, M_, b_);
  return fromTorchTensor(y_);
}

// Linear projection for 3D input: [batch, seq, in] @ [out, in]^T -> [batch, seq, out]
lean_object* lean_torch_linear3d(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*in_dim*/,
  lean_obj_arg /*out_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg weight
) {
  auto x_ = borrowTensor(x);
  auto weight_ = borrowTensor(weight);
  // torch::linear does x @ weight.T, which is what we want
  auto y_ = torch::linear(x_, weight_);
  return fromTorchTensor(y_);
}

// Affine for 3D input: linear + bias with broadcasting
lean_object* lean_torch_affine3d(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*in_dim*/,
  lean_obj_arg /*out_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias
) {
  auto x_ = borrowTensor(x);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto y_ = torch::linear(x_, weight_, bias_);
  return fromTorchTensor(y_);
}

lean_object* lean_torch_conv2d(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);

  auto options = torch::nn::functional::Conv2dFuncOptions()
    .stride(stride_)
    .padding(padding_)
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv2d(input_, weight_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv2d_bias(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  lean_obj_arg /*bias_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);

  // Use at::conv2d which takes bias as a parameter
  auto output_ = at::conv2d(input_, weight_, bias_, stride_, padding_, dilation_);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_conv1d(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  int stride,
  int padding,
  int dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto options = torch::nn::functional::Conv1dFuncOptions()
    .stride(stride)
    .padding(padding) 
    .dilation(dilation);
  auto output_ = torch::nn::functional::conv1d(input_, weight_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_adaptive_avg_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg output_size
) {
  auto input_ = borrowTensor(input);
  auto output_size_ = getShape(output_size);
  auto output_ = torch::nn::functional::adaptive_avg_pool2d(input_, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(output_size_));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_avg_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  auto input_ = borrowTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  
  auto options = torch::nn::functional::AvgPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::avg_pool2d(input_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_max_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  auto input_ = borrowTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  
  auto options = torch::nn::functional::MaxPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::max_pool2d(input_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_dropout(lean_obj_arg /*s*/, b_lean_obj_arg input, double p, int training) {
  auto input_ = borrowTensor(input);
  auto output_ = torch::nn::functional::dropout(input_, torch::nn::functional::DropoutFuncOptions().p(p).training(training));
  return lean_io_result_mk_ok(fromTorchTensor(output_));
}

lean_object* lean_torch_mse_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  auto reduction_str = lean_string_cstr(reduction);
  torch::nn::functional::MSELossFuncOptions options;
  if (std::string(reduction_str) == "mean") {
    options.reduction(torch::kMean);
  } else if (std::string(reduction_str) == "sum") {
    options.reduction(torch::kSum);
  } else {
    options.reduction(torch::kNone);
  }
  auto output_ = torch::nn::functional::mse_loss(input_, target_, options);
  return fromTorchTensor(output_);
}

lean_object* lean_torch_log_softmax(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = borrowTensor(input);
  auto output_ = torch::nn::functional::log_softmax(input_, torch::nn::functional::LogSoftmaxFuncOptions(dim));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_leaky_relu(lean_obj_arg /*s*/, b_lean_obj_arg input, double negative_slope) {
  auto input_ = borrowTensor(input);
  auto output_ = torch::nn::functional::leaky_relu(input_, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
  return fromTorchTensor(output_);
}

lean_object* lean_torch_batch_norm(
  lean_obj_arg /*b*/,
  lean_obj_arg /*c*/,
  lean_obj_arg /*h*/,
  lean_obj_arg /*w*/,
  b_lean_obj_arg input,
  lean_obj_arg weight,
  lean_obj_arg bias,
  lean_obj_arg running_mean,
  lean_obj_arg running_var,
  int training,
  double momentum,
  double eps
) {
  auto input_ = borrowTensor(input);
  torch::Tensor running_mean_ = torch::Tensor();
  torch::Tensor running_var_ = torch::Tensor();
  
  // Handle optional running statistics
  if (!lean_is_scalar(running_mean)) {
    lean_object* rm = lean_ctor_get(running_mean, 0);
    running_mean_ = borrowTensor(rm);
  }
  if (!lean_is_scalar(running_var)) {
    lean_object* rv = lean_ctor_get(running_var, 0);
    running_var_ = borrowTensor(rv);
  }

  // Dec arguments as we own them
  lean_dec(weight);
  lean_dec(bias);
  lean_dec(running_mean);
  lean_dec(running_var);
  
  torch::nn::functional::BatchNormFuncOptions options;
  options.training(training).momentum(momentum).eps(eps);
  
  auto output_ = torch::nn::functional::batch_norm(
    input_, 
    running_mean_,
    running_var_,
    options
  );
  
  return fromTorchTensor(output_);
}

// Layer normalization
lean_object* lean_torch_layer_norm(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  lean_obj_arg normalized_shape,
  lean_obj_arg weight,
  lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto shape_array = getShape(normalized_shape); lean_dec(normalized_shape);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  lean_dec(weight);
  lean_dec(bias);

  auto output_ = torch::layer_norm(
    input_,
    c10::IntArrayRef(shape_array.data(), shape_array.size()),
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );


  return fromTorchTensor(output_);
}

// Group normalization
lean_object* lean_torch_group_norm(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  uint64_t num_groups,
  lean_obj_arg weight,
  lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  lean_dec(weight);
  lean_dec(bias);

  auto output_ = torch::group_norm(
    input_,
    num_groups,
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );


  return fromTorchTensor(output_);
}

// Instance normalization
lean_object* lean_torch_instance_norm(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  lean_obj_arg running_mean,
  lean_obj_arg running_var,
  lean_obj_arg weight,
  lean_obj_arg bias,
  uint8_t use_input_stats,
  double momentum,
  double eps
) {
  auto input_ = borrowTensor(input);

  // Initialize to empty tensors for safe conditional release
  torch::Tensor running_mean_ = torch::Tensor();
  torch::Tensor running_var_ = torch::Tensor();
  torch::Tensor weight_ = torch::Tensor();
  torch::Tensor bias_ = torch::Tensor();

  if (!lean_is_scalar(running_mean)) {
    lean_object* rm = lean_ctor_get(running_mean, 0);
    running_mean_ = borrowTensor(rm);
  }
  if (!lean_is_scalar(running_var)) {
    lean_object* rv = lean_ctor_get(running_var, 0);
    running_var_ = borrowTensor(rv);
  }
  if (!lean_is_scalar(weight)) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  if (!lean_is_scalar(bias)) {
    lean_object* b = lean_ctor_get(bias, 0);
    bias_ = borrowTensor(b);
  }
  lean_dec(running_mean);
  lean_dec(running_var);
  lean_dec(weight);
  lean_dec(bias);

  auto output_ = torch::instance_norm(
    input_,
    weight_.defined() ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    bias_.defined() ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    running_mean_.defined() ? c10::optional<torch::Tensor>(running_mean_) : c10::nullopt,
    running_var_.defined() ? c10::optional<torch::Tensor>(running_var_) : c10::nullopt,
    use_input_stats,
    momentum,
    eps,
    true  // cudnn_enabled
  );


  return fromTorchTensor(output_);
}

// Binary cross entropy loss
lean_object* lean_torch_binary_cross_entropy(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg weight, lean_obj_arg reduction) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  torch::Tensor weight_;
  bool has_weight = !lean_is_scalar(weight);

  if (has_weight) {
    lean_object* w = lean_ctor_get(weight, 0);
    weight_ = borrowTensor(w);
  }
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  lean_dec(weight);
  lean_dec(reduction);
  
  auto output_ = torch::binary_cross_entropy(
    input_, 
    target_,
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    reduction_mode
  );
  
  
  return fromTorchTensor(output_);
}

// L1 loss
lean_object* lean_torch_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  lean_dec(reduction);
  
  auto output_ = torch::l1_loss(
    input_, 
    target_,
    reduction_mode
  );
  
  
  return fromTorchTensor(output_);
}

// Smooth L1 loss 
lean_object* lean_torch_smooth_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction, double beta) {
  auto input_ = borrowTensor(input);
  auto target_ = borrowTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  lean_dec(reduction);
  
  auto output_ = torch::smooth_l1_loss(
    input_, 
    target_,
    reduction_mode,
    beta
  );
  
  
  return fromTorchTensor(output_);
}

// Conv3D
lean_object* lean_torch_conv3d(
  lean_obj_arg /*input_shape*/,
  lean_obj_arg /*weight_shape*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg stride,
  lean_obj_arg padding,
  lean_obj_arg dilation
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride); lean_dec(stride);
  auto padding_ = getShape(padding); lean_dec(padding);
  auto dilation_ = getShape(dilation); lean_dec(dilation);
  
  auto options = torch::nn::functional::Conv3dFuncOptions()
    .stride(stride_)
    .padding(padding_) 
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv3d(input_, weight_, options);
  
  
  return fromTorchTensor(output_);
}

// Embedding for 2D input: [batch, seq] -> [batch, seq, embed]
lean_object* lean_torch_embedding(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*vocab*/,
  lean_obj_arg /*embed*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg padding_idx,
  lean_obj_arg max_norm,
  double norm_type,
  uint8_t scale_grad_by_freq,
  uint8_t sparse
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    lean_object* idx = lean_ctor_get(padding_idx, 0);
    padding_idx_val = static_cast<int64_t>(lean_int64_of_int(idx));
  }

  lean_dec(padding_idx);
  lean_dec(max_norm); // Unused but must dec

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  return fromTorchTensor(output_);
}

// Embedding for 1D input: [seq] -> [seq, embed]
lean_object* lean_torch_embedding_1d(
  lean_obj_arg /*seq*/,
  lean_obj_arg /*vocab*/,
  lean_obj_arg /*embed*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  lean_obj_arg padding_idx,
  lean_obj_arg max_norm,
  double norm_type,
  uint8_t scale_grad_by_freq,
  uint8_t sparse
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    lean_object* idx = lean_ctor_get(padding_idx, 0);
    padding_idx_val = static_cast<int64_t>(lean_int64_of_int(idx));
  }

  lean_dec(padding_idx);
  lean_dec(max_norm);

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  return fromTorchTensor(output_);
}

// Matrix multiplication functions (critical for transformers)
lean_object* lean_torch_matmul(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg other) {
  auto input_ = borrowTensor(input);
  auto other_ = borrowTensor(other);
  auto result_ = torch::matmul(input_, other_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_bmm(
    uint64_t /*b*/,
    uint64_t /*m*/,
    uint64_t /*n*/,
    uint64_t /*k*/,
    b_lean_obj_arg input,
    b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::bmm(input_, mat2_);
  return fromTorchTensor(result_);
}

// 4D batched matmul for multi-head attention: [b,h,m,k] @ [b,h,k,n] -> [b,h,m,n]
lean_object* lean_torch_bmm4d(
    lean_obj_arg /*b*/,
    lean_obj_arg /*h*/,
    lean_obj_arg /*m*/,
    lean_obj_arg /*k*/,
    lean_obj_arg /*n*/,
    b_lean_obj_arg input,
    b_lean_obj_arg mat2
) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  // torch::matmul handles 4D broadcasting correctly
  auto result_ = torch::matmul(input_, mat2_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mm(uint64_t m, uint64_t n, uint64_t k, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::mm(input_, mat2_);
  return fromTorchTensor(result_);
}

// Tensor operations (essential for attention mechanisms)
lean_object* lean_torch_transpose(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t dim0, uint64_t dim1) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, static_cast<int64_t>(dim0), static_cast<int64_t>(dim1));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_cat(lean_obj_arg /*s*/, b_lean_obj_arg tensors, lean_obj_arg dim_obj) {
  int64_t dim = static_cast<int64_t>(lean_int64_of_int(dim_obj));
  lean_dec(dim_obj);
  if (!lean_is_array(tensors)) {
    lean_internal_panic("lean_torch_cat: expected array");
  }
  std::vector<torch::Tensor> tensor_list;
  size_t arr_size = lean_array_size(tensors);
  for (size_t i = 0; i < arr_size; i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  // tensors is borrowed (@& in Lean), do not lean_dec here
  auto result_ = torch::cat(tensor_list, dim);
  // tensor_list tensors are auto-released when vector goes out of scope
  return fromTorchTensor(result_);
}

// Direct 2-tensor concatenation without intermediate reshapes
lean_object* lean_torch_cat2(
    lean_obj_arg /*s1*/,
    lean_obj_arg /*s2*/,
    b_lean_obj_arg t1,
    b_lean_obj_arg t2,
    lean_obj_arg dim_obj
) {
  int64_t dim = static_cast<int64_t>(lean_int64_of_int(dim_obj));
  lean_dec(dim_obj);
  auto t1_ = borrowTensor(t1);
  auto t2_ = borrowTensor(t2);
  auto result_ = torch::cat({t1_, t2_}, dim);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_stack(lean_obj_arg /*s*/, lean_obj_arg tensors, int dim) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  lean_dec(tensors); // Decrement array
  auto result_ = torch::stack(tensor_list, dim);
  // tensor_list tensors are auto-released when vector goes out of scope
  return fromTorchTensor(result_);
}

// Attention mechanism functions
lean_object* lean_torch_scaled_dot_product_attention(
  lean_obj_arg /*s*/, 
  b_lean_obj_arg query, 
  b_lean_obj_arg key, 
  b_lean_obj_arg value,
  lean_obj_arg attn_mask,
  double dropout_p,
  uint8_t is_causal
) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);
  
  torch::Tensor attn_mask_;
  bool has_mask = !lean_is_scalar(attn_mask);
  if (has_mask) {
    lean_object* m = lean_ctor_get(attn_mask, 0);
    attn_mask_ = borrowTensor(m);
  }
  lean_dec(attn_mask);
  
  auto result_ = torch::scaled_dot_product_attention(
    query_, 
    key_, 
    value_,
    has_mask ? c10::optional<torch::Tensor>(attn_mask_) : c10::nullopt,
    dropout_p,
    is_causal
  );
  
  
  return fromTorchTensor(result_);
}

// Softmax with dimension parameter (needed for attention)
lean_object* lean_torch_softmax_dim(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::softmax(input_, dim);
  return fromTorchTensor(result_);
}

// Extended activation functions with parameters
lean_object* lean_torch_hardtanh_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double min_val, double max_val) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::hardtanh(input_, min_val, max_val);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_celu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double alpha) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::celu(input_, alpha);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rrelu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double lower, double upper, uint8_t training) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::rrelu(input_, lower, upper, training);
  return fromTorchTensor(result_);
}

// Additional tensor operations
lean_object* lean_torch_split(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int split_size, int dim) {
  auto tensor_ = borrowTensor(tensor);
  auto result_tensors = torch::split(tensor_, split_size, dim);
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_chunk(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int chunks, int dim) {
  auto tensor_ = borrowTensor(tensor);
  auto result_tensors = torch::chunk(tensor_, chunks, dim);
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_unsqueeze(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg dim) {
  auto input_ = borrowTensor(input);
  auto dim_ = lean_uint64_of_nat(dim);
  auto result_ = torch::unsqueeze(input_, static_cast<int64_t>(dim_));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_squeeze(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg dim) {
  auto input_ = borrowTensor(input);
  auto dim_ = lean_uint64_of_nat(dim);
  auto result_ = torch::squeeze(input_, static_cast<int64_t>(dim_));
  return fromTorchTensor(result_);
}

// Mathematical operations
lean_object* lean_torch_sqrt(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::sqrt(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rsqrt(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::rsqrt(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg other) {
  auto input_ = borrowTensor(input);
  auto other_ = borrowTensor(other);
  auto result_ = torch::div(input_, other_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow(lean_obj_arg /*s*/, b_lean_obj_arg input, double exponent) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::pow(input_, exponent);
  return fromTorchTensor(result_);
}

// Reduction operations
lean_object* lean_torch_sum(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  torch::Tensor result_;
  
  if (lean_is_scalar(dim)) {
    // none: sum all elements
    result_ = torch::sum(input_);
  } else {
    // some: dim is a constructor wrapping Array UInt64
    lean_object* dims_obj = lean_ctor_get(dim, 0);
    auto dims = getShape(dims_obj);
    lean_dec(dim);
    result_ = torch::sum(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mean(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  torch::Tensor result_;

  if (lean_is_scalar(dim)) {
    // none: mean of all elements
    result_ = torch::mean(input_);
  } else {
    // some: dim is a constructor wrapping Array UInt64
    lean_object* dims_obj = lean_ctor_get(dim, 0);
    auto dims = getShape(dims_obj);
    lean_dec(dim);
    result_ = torch::mean(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }

  return fromTorchTensor(result_);
}

lean_object* lean_torch_abs(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::abs(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_max_all(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::max(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_min_all(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::min(input_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_max(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  auto result_tuple = torch::max(input_, dim, keepdim);
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

lean_object* lean_torch_min(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  auto result_tuple = torch::min(input_, dim, keepdim);
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Masking and conditional operations
lean_object* lean_torch_masked_fill(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mask, double value) {
  auto input_ = borrowTensor(input);
  auto mask_ = borrowTensor(mask);
  auto result_ = input_.masked_fill(mask_, value);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_where(lean_obj_arg /*s*/, b_lean_obj_arg condition, b_lean_obj_arg x, b_lean_obj_arg y) {
  auto condition_ = borrowTensor(condition);
  auto x_ = borrowTensor(x);
  auto y_ = borrowTensor(y);
  auto result_ = torch::where(condition_, x_, y_);
  return fromTorchTensor(result_);
}

// Broadcasting and expanding
lean_object* lean_torch_expand(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg size) {
  auto input_ = borrowTensor(input);
  auto size_ = getShape(size); lean_dec(size);
  auto result_ = input_.expand(c10::IntArrayRef(size_.data(), size_.size()));
  return fromTorchTensor(result_);
}

lean_object* lean_torch_repeat(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg repeats) {
  auto input_ = borrowTensor(input);
  auto repeats_ = getShape(repeats); lean_dec(repeats);
  auto result_ = input_.repeat(c10::IntArrayRef(repeats_.data(), repeats_.size()));
  return fromTorchTensor(result_);
}

// Scalar operations (tensor-scalar arithmetic)
lean_object* lean_torch_mul_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ * scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ / scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_add_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ + scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_sub_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ - scalar;
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow_tensor(lean_obj_arg /*s*/, b_lean_obj_arg input, double exponent) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::pow(input_, exponent);
  return fromTorchTensor(result_);
}

// Lower triangular (for causal masking)
lean_object* lean_torch_tril(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t diagonal) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::tril(input_, diagonal);
  return fromTorchTensor(result_);
}

// Top-k values (for sampling)
lean_object* lean_torch_topk_values(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t k, int64_t dim) {
  auto input_ = borrowTensor(input);
  auto result_tuple = torch::topk(input_, k, dim);
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Multinomial sampling
lean_object* lean_torch_multinomial(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t num_samples, uint8_t replacement, lean_object* w) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::multinomial(input_, num_samples, replacement);
  return lean_io_result_mk_ok(fromTorchTensor(result_));
}

// Gradient clipping (in-place, returns norm)
lean_object* lean_torch_clip_grad_norm_(lean_obj_arg /*s*/, b_lean_obj_arg param, double max_norm, lean_object* w) {
  auto param_ = borrowTensor(param);
  double total_norm = 0.0;
  if (param_.grad().defined()) {
    auto grad = param_.grad();
    auto norm = grad.norm();
    total_norm = norm.item<double>();
    if (total_norm > max_norm) {
      grad.mul_(max_norm / total_norm);
    }
  }
  return lean_io_result_mk_ok(lean_box_float(total_norm));
}

// Item extraction (get scalar value from 0-d tensor)
double lean_torch_item(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  double result = input_.item<double>();
  return result;
}

// Item extraction as int64
int64_t lean_torch_item_int(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  int64_t result = input_.item<int64_t>();
  return result;
}

// Argmax
// Int64 is a single-field structure wrapping UInt64, so it's passed unboxed as uint64_t
lean_object* lean_torch_argmax(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::argmax(input_, static_cast<int64_t>(dim));
  return fromTorchTensor(result_);
}

// Create tensor from Array Int64
// Array Int64 is a regular array (not scalar array) where each element is a boxed Int64
// Int64 is stored as a ctor with the uint64 value at scalar offset 0
lean_object* lean_torch_from_int64_array(b_lean_obj_arg arr) {
  size_t len = lean_array_size(arr);

  // Allocate buffer and unbox each element
  std::vector<int64_t> data(len);
  for (size_t i = 0; i < len; i++) {
    lean_object* elem = lean_array_get_core(arr, i);
    // Int64 is stored as a ctor with uint64 at offset 0 (same as UInt64)
    data[i] = static_cast<int64_t>(lean_ctor_get_uint64(elem, 0));
  }

  // Create tensor by copying data
  auto tensor = torch::from_blob(data.data(), {static_cast<int64_t>(len)}, torch::kInt64).clone();
  return fromTorchTensor(tensor);
}

// Backward pass
lean_object* lean_torch_backward_unit(lean_obj_arg /*s*/, b_lean_obj_arg output, b_lean_obj_arg grad_output, lean_object* w) {
  auto output_ = borrowTensor(output);
  auto grad_output_ = borrowTensor(grad_output);
  output_.backward(grad_output_);
  return lean_io_result_mk_ok(lean_box(0));
}

// Get number of uint16 tokens in a binary file
lean_object* lean_torch_bin_file_token_count(b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to open file: " + std::string(path)).c_str())));
  }

  std::streamsize size = file.tellg();
  uint64_t num_tokens = size / sizeof(uint16_t);

  return lean_io_result_mk_ok(lean_box_uint64(num_tokens));
}

// Load binary file of uint16 tokens into a 1D tensor with known size
lean_object* lean_torch_load_u16_bin(uint64_t n, b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);
  uint64_t expected_tokens = n;

  // Open file
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to open file: " + std::string(path)).c_str())));
  }

  // Read as uint16
  std::vector<uint16_t> buffer(expected_tokens);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), expected_tokens * sizeof(uint16_t))) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Failed to read file")));
  }

  // Convert to int64 tensor (PyTorch uses int64 for indices)
  std::vector<int64_t> data(expected_tokens);
  for (size_t i = 0; i < expected_tokens; i++) {
    data[i] = static_cast<int64_t>(buffer[i]);
  }

  auto tensor = torch::from_blob(data.data(), {static_cast<int64_t>(expected_tokens)},
                                  torch::kInt64).clone();

  return lean_io_result_mk_ok(fromTorchTensor(tensor));
}

// Index select: gather elements along a dimension
lean_object* lean_torch_index_select(
  lean_obj_arg /*s*/,
  lean_obj_arg /*k*/,
  b_lean_obj_arg input,
  int64_t dim,
  b_lean_obj_arg indices
) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  // Ensure indices are int64 (index_select requires int32 or int64)
  if (indices_.dtype() != torch::kInt64 && indices_.dtype() != torch::kInt32) {
    indices_ = indices_.to(torch::kInt64);
  }
  auto result_ = torch::index_select(input_, dim, indices_);
  return fromTorchTensor(result_);
}

// Slice a 1D tensor: data[start:end]
lean_object* lean_torch_slice_1d(
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  int64_t start,
  int64_t end
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(0, start, end);
  return fromTorchTensor(result_);
}

// Slice a tensor along a specified dimension: data.slice(dim, start, start+len)
// Fully dependently typed - output shape is computed from input shape and slice params
lean_object* lean_torch_slice_along_dim(
  lean_obj_arg /*s*/,
  b_lean_obj_arg input,
  uint64_t dim,
  uint64_t start,
  uint64_t len
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(static_cast<int64_t>(dim),
                               static_cast<int64_t>(start),
                               static_cast<int64_t>(start + len));
  return fromTorchTensor(result_);
}

// Slice a 2D tensor along dimension 0: data[start:start+len, :]
lean_object* lean_torch_slice_2d(
  uint64_t /*n*/,
  uint64_t /*d*/,
  b_lean_obj_arg input,
  uint64_t start,
  uint64_t len
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.slice(0,
                               static_cast<int64_t>(start),
                               static_cast<int64_t>(start + len));
  return fromTorchTensor(result_);
}

// Stack multiple 1D tensors into 2D
lean_object* lean_torch_stack_1d(
  lean_obj_arg tensors,
  int64_t dim
) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  auto result_ = torch::stack(tensor_list, dim);
  // tensor_list tensors are auto-released when vector goes out of scope
  return fromTorchTensor(result_);
}

// Convert tensor to Long (int64) dtype
lean_object* lean_torch_to_long(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.to(torch::kLong);
  return fromTorchTensor(result_);
}

// Backward pass that only requires the loss (assumes gradient of 1.0)
lean_object* lean_torch_backward_loss(lean_obj_arg /*s*/, b_lean_obj_arg loss, lean_object* w) {
  auto loss_ = borrowTensor(loss);
  loss_.backward();
  return lean_io_result_mk_ok(lean_box(0));
}

// Shape-aware matmul for 3D @ 2D: [batch, seq, k] @ [k, n] -> [batch, seq, n]
lean_object* lean_torch_matmul3d_2d(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*k*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg a,
  b_lean_obj_arg b
) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::matmul(a_, b_);
  return fromTorchTensor(result_);
}

// Transpose 2D tensor: [m, n] -> [n, m]
lean_object* lean_torch_transpose_2d(
  uint64_t /*m*/,
  uint64_t /*n*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, 0, 1);
  return fromTorchTensor(result_);
}

// Transpose 3D tensor dims 1 and 2: [a, b, c] -> [a, c, b]
lean_object* lean_torch_transpose3d_12(
  lean_obj_arg /*a*/,
  lean_obj_arg /*b*/,
  lean_obj_arg /*c*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, 1, 2);
  return fromTorchTensor(result_);
}

// Transpose for attention: [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
lean_object* lean_torch_transpose_for_attention(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n_head*/,
  lean_obj_arg /*head_dim*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  // [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  return fromTorchTensor(result_);
}

// Transpose from attention: [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
lean_object* lean_torch_transpose_from_attention(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*n_head*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*head_dim*/,
  b_lean_obj_arg input
) {
  auto input_ = borrowTensor(input);
  // [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  return fromTorchTensor(result_);
}

// Layer norm for 3D tensors with separate weight/bias shapes
lean_object* lean_torch_layer_norm_3d(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  // Get the last dimension size from the weight tensor
  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto output_ = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );


  return fromTorchTensor(output_);
}

// ============================================================================
// Fused layer_norm + activation operations
// These avoid intermediate tensor allocation between layer_norm and activation
// ============================================================================

// Fused layer_norm + GELU for transformer attention/MLP
lean_object* lean_torch_layer_norm_gelu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::gelu(normed));
}

// Fused layer_norm + ReLU
lean_object* lean_torch_layer_norm_relu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::relu(normed));
}

// Fused layer_norm + SiLU (for SwiGLU MLP)
lean_object* lean_torch_layer_norm_silu(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight,
  b_lean_obj_arg bias,
  double eps
) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto bias_ = borrowTensor(bias);

  int64_t n = weight_.size(0);
  std::vector<int64_t> normalized_shape = {n};

  auto normed = torch::layer_norm(
    input_,
    c10::IntArrayRef(normalized_shape.data(), normalized_shape.size()),
    weight_,
    bias_,
    eps
  );

  return fromTorchTensor(torch::silu(normed));
}

// Cross entropy for 2D logits: [N, C] + targets [N] -> scalar
lean_object* lean_torch_cross_entropy_2d(
  lean_obj_arg /*n*/,
  lean_obj_arg /*c*/,
  b_lean_obj_arg logits,
  b_lean_obj_arg targets
) {
  auto logits_ = borrowTensor(logits);
  auto targets_ = borrowTensor(targets);
  auto result_ = torch::nn::functional::cross_entropy(logits_, targets_);
  return fromTorchTensor(result_);
}

// Scaled dot-product attention for 4D tensors
lean_object* lean_torch_sdpa_4d(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*n_head*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal
) {
  auto query_ = borrowTensor(query);
  auto key_ = borrowTensor(key);
  auto value_ = borrowTensor(value);

  auto result_ = torch::scaled_dot_product_attention(
    query_,
    key_,
    value_,
    c10::nullopt,  // attn_mask
    dropout_p,
    is_causal
  );


  return fromTorchTensor(result_);
}

// Save tensor to a binary file
lean_object* lean_torch_save_tensor(lean_obj_arg /*s*/, b_lean_obj_arg tensor, b_lean_obj_arg path_obj, lean_object* w) {
  auto tensor_ = borrowTensor(tensor);
  const char* path = lean_string_cstr(path_obj);

  try {
    // Use torch::save with a vector containing our tensor
    std::vector<torch::Tensor> tensors = {tensor_};
    torch::save(tensors, path);
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const c10::Error& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to save tensor: " + std::string(e.what())).c_str())));
  }
}

// Load tensor from a binary file with expected shape
lean_object* lean_torch_load_tensor(lean_obj_arg shape, b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);

  try {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, path);

    if (tensors.empty()) {
      lean_dec(shape);
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("No tensors found in file")));
    }

    // Return the first tensor (reshape to expected shape if needed)
    auto tensor = tensors[0];
    auto expected_shape = getShape(shape); lean_dec(shape);

    // Verify shape matches (or reshape if possible)
    if (tensor.numel() != std::accumulate(expected_shape.begin(), expected_shape.end(),
                                           1LL, std::multiplies<int64_t>())) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("Tensor size mismatch with expected shape")));
    }

    auto result = tensor.reshape(expected_shape);
    return lean_io_result_mk_ok(fromTorchTensor(result));
  } catch (const c10::Error& e) {
    lean_dec(shape);
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to load tensor: " + std::string(e.what())).c_str())));
  }
}

// Check if a file exists
lean_object* lean_torch_file_exists(b_lean_obj_arg path_obj, lean_object* w) {
  const char* path = lean_string_cstr(path_obj);
  std::ifstream file(path);
  return lean_io_result_mk_ok(lean_box(file.good() ? 1 : 0));
}

// Save image tensor as PPM file
// Expects tensor of shape [batch, 3, height, width] with values in [-1, 1]
lean_object* lean_torch_save_ppm(
  lean_obj_arg shape,
  b_lean_obj_arg tensor_obj,
  b_lean_obj_arg path_obj,
  lean_object* w
) {
  const char* path = lean_string_cstr(path_obj);
  lean_dec(shape);

  try {
    auto tensor = borrowTensor(tensor_obj);

    // Get dimensions
    int64_t batch = tensor.size(0);
    int64_t channels = tensor.size(1);
    int64_t height = tensor.size(2);
    int64_t width = tensor.size(3);

    if (channels != 3) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string("PPM requires 3 channels (RGB)")));
    }

    // Take first image, convert from [-1, 1] to [0, 255]
    auto img = tensor[0];  // [3, height, width]
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch::kUInt8);

    // Transpose to [height, width, 3]
    img = img.permute({1, 2, 0}).contiguous();

    // Write PPM file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open file for writing: " + std::string(path)).c_str())));
    }

    // PPM header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Write pixel data
    auto data_ptr = img.data_ptr<uint8_t>();
    file.write(reinterpret_cast<const char*>(data_ptr), height * width * 3);

    file.close();
    return lean_io_result_mk_ok(lean_box(0));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("Failed to save PPM: " + std::string(e.what())).c_str())));
  }
}

// ===========================================================================
// SafeTensors loading
// ===========================================================================

// Helper: parse a simple JSON-like header (basic implementation)
// SafeTensors format: 8-byte header size (LE) + JSON header + tensor data
struct SafeTensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t data_offset;
  int64_t data_size;
};

static torch::ScalarType parseDtype(const std::string& dtype) {
  if (dtype == "F32" || dtype == "float32") return torch::kFloat32;
  if (dtype == "F16" || dtype == "float16") return torch::kFloat16;
  if (dtype == "BF16" || dtype == "bfloat16") return torch::kBFloat16;
  if (dtype == "I64" || dtype == "int64") return torch::kInt64;
  if (dtype == "I32" || dtype == "int32") return torch::kInt32;
  if (dtype == "I16" || dtype == "int16") return torch::kInt16;
  if (dtype == "I8" || dtype == "int8") return torch::kInt8;
  if (dtype == "U8" || dtype == "uint8") return torch::kUInt8;
  if (dtype == "BOOL" || dtype == "bool") return torch::kBool;
  // FP8 types (PyTorch 2.1+)
  if (dtype == "F8_E4M3" || dtype == "float8_e4m3fn") return torch::kFloat8_e4m3fn;
  if (dtype == "F8_E5M2" || dtype == "float8_e5m2") return torch::kFloat8_e5m2;
  return torch::kFloat32; // Default
}

static size_t dtypeSize(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32: return 4;
    case torch::kFloat64: return 8;
    case torch::kFloat16: return 2;
    case torch::kBFloat16: return 2;
    case torch::kFloat8_e4m3fn: return 1;
    case torch::kFloat8_e5m2: return 1;
    case torch::kInt64: return 8;
    case torch::kInt32: return 4;
    case torch::kInt16: return 2;
    case torch::kInt8: return 1;
    case torch::kUInt8: return 1;
    case torch::kBool: return 1;
    default: return 4;
  }
}

// Load tensor from SafeTensors file by name
lean_object* lean_torch_safetensors_load(
  b_lean_obj_arg path_obj,
  b_lean_obj_arg name_obj,
  lean_obj_arg shape,
  lean_object* w
) {
  const char* path = lean_string_cstr(path_obj);
  const char* tensor_name = lean_string_cstr(name_obj);
  auto expected_shape = getShape(shape);
  lean_dec(shape);

  try {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Failed to open safetensors file: " + std::string(path)).c_str())));
    }

    // Read header size (8 bytes, little-endian)
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Check for reasonable header size (avoid huge allocations)
    if (header_size > 100000000) { // 100MB max header
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Unreasonable header size: " + std::to_string(header_size)).c_str())));
    }

    // Read JSON header
    std::string header(header_size, '\0');
    file.read(&header[0], header_size);

    // Simple JSON parsing to find tensor info
    // Look for: "tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}
    std::string search_key = "\"" + std::string(tensor_name) + "\"";
    size_t pos = header.find(search_key);
    if (pos == std::string::npos) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Tensor not found in safetensors: " + std::string(tensor_name)).c_str())));
    }

    // Find dtype
    size_t dtype_pos = header.find("\"dtype\"", pos);
    size_t dtype_start = header.find("\"", dtype_pos + 7) + 1;
    size_t dtype_end = header.find("\"", dtype_start);
    std::string dtype_str = header.substr(dtype_start, dtype_end - dtype_start);
    torch::ScalarType dtype = parseDtype(dtype_str);

    // Find shape
    std::vector<int64_t> shape_vec;
    size_t shape_pos = header.find("\"shape\"", pos);
    size_t shape_start = header.find("[", shape_pos);
    size_t shape_end = header.find("]", shape_start);
    std::string shape_str = header.substr(shape_start + 1, shape_end - shape_start - 1);

    // Parse shape array
    std::istringstream shape_stream(shape_str);
    std::string dim;
    while (std::getline(shape_stream, dim, ',')) {
      // Trim whitespace
      dim.erase(0, dim.find_first_not_of(" \t"));
      dim.erase(dim.find_last_not_of(" \t") + 1);
      if (!dim.empty()) {
        shape_vec.push_back(std::stoll(dim));
      }
    }

    // Find data offsets
    size_t offsets_pos = header.find("\"data_offsets\"", pos);
    size_t offsets_start = header.find("[", offsets_pos);
    size_t offsets_end = header.find("]", offsets_start);
    std::string offsets_str = header.substr(offsets_start + 1, offsets_end - offsets_start - 1);

    int64_t data_start = 0, data_end = 0;
    size_t comma = offsets_str.find(",");
    if (comma != std::string::npos) {
      data_start = std::stoll(offsets_str.substr(0, comma));
      data_end = std::stoll(offsets_str.substr(comma + 1));
    }

    int64_t data_size = data_end - data_start;

    // Seek to data position (after header)
    file.seekg(8 + header_size + data_start);

    // Read tensor data
    std::vector<char> buffer(data_size);
    file.read(buffer.data(), data_size);

    // Create tensor from buffer
    torch::Tensor tensor;
    try {
      tensor = torch::from_blob(
        buffer.data(),
        shape_vec,
        torch::TensorOptions().dtype(dtype)
      );
      tensor = tensor.clone(); // Clone to own the memory
    } catch (const std::exception& e) {
      return lean_io_result_mk_error(lean_mk_io_user_error(
        lean_mk_string(("Tensor creation error: " + std::string(e.what())).c_str())));
    }

    // Reshape to expected shape if needed
    if (!expected_shape.empty()) {
      tensor = tensor.reshape(expected_shape);
    }

    return lean_io_result_mk_ok(fromTorchTensor(tensor));
  } catch (const std::exception& e) {
    return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string(("SafeTensors load error: " + std::string(e.what())).c_str())));
  }
}

// Load tensor from sharded SafeTensors files
lean_object* lean_torch_safetensors_load_sharded(
  b_lean_obj_arg dir_obj,
  b_lean_obj_arg name_obj,
  lean_obj_arg shape,
  lean_object* w
) {
  const char* dir = lean_string_cstr(dir_obj);
  const char* tensor_name = lean_string_cstr(name_obj);

  // Try common shard patterns
  std::vector<std::string> patterns = {
    std::string(dir) + "/model.safetensors",
    std::string(dir) + "/model-00001-of-00002.safetensors",
    std::string(dir) + "/model-00002-of-00002.safetensors",
    std::string(dir) + "/model-00001-of-00003.safetensors",
    std::string(dir) + "/model-00002-of-00003.safetensors",
    std::string(dir) + "/model-00003-of-00003.safetensors",
  };

  for (const auto& path : patterns) {
    std::ifstream test(path);
    if (test.good()) {
      test.close();
      // Try loading from this shard
      lean_inc(shape);
      lean_object* name_copy = lean_mk_string(tensor_name);
      lean_object* path_str = lean_mk_string(path.c_str());
      lean_object* result = lean_torch_safetensors_load(path_str, name_copy, shape, w);

      // Check if successful
      if (!lean_io_result_is_error(result)) {
        return result;
      }
      lean_dec(result);
    }
  }

  lean_dec(shape);
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string(("Tensor not found in any shard: " + std::string(tensor_name)).c_str())));
}

// Find positions of BOS tokens in a 1D tensor
// Returns a 1D int64 tensor containing indices where tokens == bos_token
lean_object* lean_torch_find_bos_positions(
    lean_obj_arg /*n*/,
    b_lean_obj_arg tokens,
    int64_t bos_token,
    lean_object* w
) {
  auto tokens_ = borrowTensor(tokens);

  // Create mask where tokens == bos_token
  auto mask = tokens_ == bos_token;

  // Get indices where mask is true
  auto positions = torch::nonzero(mask);

  // nonzero returns [N, 1] for 1D input, squeeze to get [N]
  if (positions.dim() > 1) {
    positions = positions.squeeze(1);
  }

  // Ensure int64 dtype
  positions = positions.to(torch::kInt64);

  return lean_io_result_mk_ok(fromTorchTensor(positions));
}

// Convert a 1D int64 tensor to a Lean Array UInt64
lean_object* lean_torch_tensor_to_uint64_array(
    lean_obj_arg /*n*/,
    b_lean_obj_arg tensor,
    lean_object* w
) {
  auto t = borrowTensor(tensor);

  // Ensure tensor is on CPU and contiguous
  t = t.to(torch::kCPU).contiguous().to(torch::kInt64);

  // Get number of elements
  int64_t numel = t.numel();

  // Create Lean array
  lean_object* arr = lean_mk_empty_array();

  // Get data pointer
  auto* ptr = t.data_ptr<int64_t>();

  // Push each element
  for (int64_t i = 0; i < numel; i++) {
    arr = lean_array_push(arr, lean_box_uint64(static_cast<uint64_t>(ptr[i])));
  }

  return lean_io_result_mk_ok(arr);
}

// Convert a dynamically-shaped int64 tensor to a Lean Array UInt64
// (Same implementation, but without the n parameter for shape-erased tensors)
lean_object* lean_torch_tensor_to_uint64_array_dynamic(
    b_lean_obj_arg tensor,
    lean_object* w
) {
  auto t = borrowTensor(tensor);

  // Ensure tensor is on CPU and contiguous
  t = t.to(torch::kCPU).contiguous().to(torch::kInt64);

  // Get number of elements
  int64_t numel = t.numel();

  // Create Lean array
  lean_object* arr = lean_mk_empty_array();

  // Get data pointer
  auto* ptr = t.data_ptr<int64_t>();

  // Push each element
  for (int64_t i = 0; i < numel; i++) {
    arr = lean_array_push(arr, lean_box_uint64(static_cast<uint64_t>(ptr[i])));
  }

  return lean_io_result_mk_ok(arr);
}

// ============================================================================
// NanoProof FFI bindings
// ============================================================================

// RMSNorm: x / sqrt(mean(x^2) + eps) - no learnable parameters
lean_object* lean_torch_rms_norm(lean_obj_arg /*s*/, b_lean_obj_arg input, double eps) {
  auto input_ = borrowTensor(input);
  // Compute RMSNorm: x * rsqrt(mean(x^2) + eps)
  auto variance = input_.pow(2).mean(-1, /*keepdim=*/true);
  auto result_ = input_ * torch::rsqrt(variance + eps);
  return fromTorchTensor(result_);
}

// RMSNorm with learnable weight: (x / sqrt(mean(x^2) + eps)) * weight
// weight broadcasts over the last dimension
lean_object* lean_torch_rms_norm_weighted(lean_obj_arg /*s*/, lean_obj_arg /*w*/,
    b_lean_obj_arg input, b_lean_obj_arg weight, double eps) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  // Compute RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
  auto variance = input_.pow(2).mean(-1, /*keepdim=*/true);
  auto result_ = input_ * torch::rsqrt(variance + eps) * weight_;
  return fromTorchTensor(result_);
}

// ReLU squared activation: relu(x)^2
lean_object* lean_torch_relu_squared(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::relu(input_).square();
  return fromTorchTensor(result_);
}

// Logit softcap: cap * tanh(x / cap)
lean_object* lean_torch_softcap(lean_obj_arg /*s*/, b_lean_obj_arg input, double cap) {
  auto input_ = borrowTensor(input);
  auto result_ = cap * torch::tanh(input_ / cap);
  return fromTorchTensor(result_);
}

// Shared implementation for computing rotary embedding frequencies
static lean_object* compute_rotary_freqs_impl(uint64_t seq_len, uint64_t head_dim, double base) {
  auto device = torch::kCPU;

  // inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
  auto channel_range = torch::arange(0, static_cast<int64_t>(head_dim), 2,
    torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto inv_freq = 1.0 / torch::pow(base, channel_range / static_cast<double>(head_dim));

  // t = arange(seq_len)
  auto t = torch::arange(static_cast<int64_t>(seq_len),
    torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // freqs = outer(t, inv_freq) -> [seq_len, head_dim/2]
  auto freqs = torch::outer(t, inv_freq);
  auto cos = freqs.cos();
  auto sin = freqs.sin();

  // Return as Lean pair (Product type, ctor 0 with 2 fields)
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(cos));
  lean_ctor_set(result, 1, fromTorchTensor(sin));
  return result;
}

// Precompute rotary embedding frequencies (IO version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs(
  uint64_t seq_len,
  uint64_t head_dim,
  double base,
  lean_object* w
) {
  return lean_io_result_mk_ok(compute_rotary_freqs_impl(seq_len, head_dim, base));
}

// Precompute rotary embedding frequencies (pure version)
// Returns (cos, sin) tensors of shape [seq_len, head_dim/2]
lean_object* lean_torch_compute_rotary_freqs_pure(
  uint64_t seq_len,
  uint64_t head_dim,
  double base
) {
  return compute_rotary_freqs_impl(seq_len, head_dim, base);
}

// Apply rotary embeddings to Q or K
// x: [batch, seq, n_head, head_dim]
// cos, sin: [seq, head_dim/2] (will be broadcast)
lean_object* lean_torch_apply_rotary_emb(
  lean_obj_arg /*batch*/,
  lean_obj_arg /*seq*/,
  lean_obj_arg /*n_head*/,
  lean_obj_arg /*head_dim*/,
  b_lean_obj_arg x,
  b_lean_obj_arg cos,
  b_lean_obj_arg sin
) {
  auto x_ = borrowTensor(x);
  auto cos_ = borrowTensor(cos);
  auto sin_ = borrowTensor(sin);

  int64_t d = x_.size(3) / 2;
  auto x1 = x_.slice(3, 0, d);
  auto x2 = x_.slice(3, d, 2*d);

  // Broadcast cos/sin: [seq, d] -> [1, seq, 1, d]
  auto cos_b = cos_.unsqueeze(0).unsqueeze(2);
  auto sin_b = sin_.unsqueeze(0).unsqueeze(2);

  // Apply rotation (matches flux2 / standard RoPE):
  // [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
  auto y1 = x1 * cos_b - x2 * sin_b;
  auto y2 = x1 * sin_b + x2 * cos_b;
  auto result_ = torch::cat({y1, y2}, 3);

  // Ensure output dtype matches input
  result_ = result_.to(x_.dtype());

  return fromTorchTensor(result_);
}

// Create sinusoidal timestep embeddings
// t: [batch] timestep values
// dim: output dimension
// Returns: [batch, dim] positional embedding
lean_object* lean_torch_timestep_embedding(
  uint64_t /*batch*/,
  b_lean_obj_arg t_obj,
  uint64_t dim,
  double max_period,
  double time_factor
) {
  auto t = borrowTensor(t_obj);
  auto device = t.device();
  auto dtype = t.dtype();

  // Scale timesteps
  t = t * time_factor;

  // half = dim // 2
  int64_t half = dim / 2;

  // freqs = exp(-log(max_period) * arange(half) / half)
  auto indices = torch::arange(0, half, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto freqs = torch::exp(-std::log(max_period) * indices / static_cast<double>(half));

  // args = t[:, None] * freqs[None, :]
  auto args = t.unsqueeze(1).to(torch::kFloat32) * freqs.unsqueeze(0);

  // embedding = cat([cos(args), sin(args)], dim=-1)
  auto cos_emb = torch::cos(args);
  auto sin_emb = torch::sin(args);
  auto embedding = torch::cat({cos_emb, sin_emb}, -1);

  // Handle odd dimensions
  if (dim % 2 == 1) {
    auto zeros = torch::zeros({t.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    embedding = torch::cat({embedding, zeros}, -1);
  }

  // Cast back to input dtype if needed
  if (torch::is_floating_point(t)) {
    embedding = embedding.to(dtype);
  }

  return fromTorchTensor(embedding);
}

// Scaled dot-product attention with Group-Query Attention (GQA) support
// Q: [batch, n_head, seq, head_dim]
// K, V: [batch, n_kv_head, seq, head_dim]
lean_object* lean_torch_sdpa_gqa(
  uint64_t /*batch*/,
  uint64_t n_head,
  uint64_t n_kv_head,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa
) {
  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);

  // Handle GQA by expanding KV heads to match Q heads
  if (enable_gqa) {
    if (n_head != n_kv_head && n_kv_head > 0) {
      // Repeat KV heads: [batch, n_kv_head, seq, head_dim] -> [batch, n_head, seq, head_dim]
      auto repeat_factor = n_head / n_kv_head;
      // Use repeat_interleave along dim 1 (head dimension)
      k = k.repeat_interleave(repeat_factor, 1);
      v = v.repeat_interleave(repeat_factor, 1);
    }
  }

  auto result_ = torch::scaled_dot_product_attention(
    q, k, v,
    c10::nullopt,  // attn_mask
    dropout_p,
    is_causal
  );

  return fromTorchTensor(result_);
}

// Scaled dot-product attention with GQA and sliding window support
// Each position can only attend to the previous window_size positions
lean_object* lean_torch_sdpa_gqa_window(
  uint64_t /*batch*/,
  uint64_t n_head,
  uint64_t n_kv_head,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa,
  uint64_t window_size
) {
  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);

  // Handle GQA by expanding KV heads to match Q heads
  if (enable_gqa) {
    if (n_head != n_kv_head && n_kv_head > 0) {
      auto repeat_factor = n_head / n_kv_head;
      k = k.repeat_interleave(repeat_factor, 1);
      v = v.repeat_interleave(repeat_factor, 1);
    }
  }

  // Get sequence length from query shape: [batch, n_head, seq, head_dim]
  auto seq_len = q.size(2);

  // Create sliding window causal mask
  // mask[i,j] = True if position j is NOT attended to by position i
  // For sliding window: j is attended if (j <= i) AND (i - j < window_size)
  auto options = torch::TensorOptions().dtype(torch::kBool).device(q.device());

  // Create row indices [seq_len, 1] and col indices [1, seq_len]
  auto row_idx = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(1);
  auto col_idx = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(0);

  // Causal mask: mask where col > row (future positions)
  auto causal_mask = col_idx > row_idx;

  // Sliding window mask: mask where (row - col) >= window_size (too far in the past)
  auto window_mask = (row_idx - col_idx) >= static_cast<int64_t>(window_size);

  // Combined mask: either future or too far in the past
  auto combined_mask = causal_mask | window_mask;

  // Convert boolean mask to attention mask format expected by SDPA
  // SDPA expects: -inf for masked positions, 0 for unmasked
  auto attn_mask = torch::where(
    combined_mask,
    torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(), q.options()),
    torch::zeros({seq_len, seq_len}, q.options())
  );

  // Expand mask to [1, 1, seq, seq] for broadcasting over batch and heads
  attn_mask = attn_mask.unsqueeze(0).unsqueeze(0);

  auto result_ = torch::scaled_dot_product_attention(
    q, k, v,
    attn_mask,
    dropout_p,
    false  // is_causal=false since we're using explicit mask
  );

  return fromTorchTensor(result_);
}

// Scaled dot-product attention with GQA and explicit attention mask
// Q: [batch, n_head, seq, head_dim]
// K, V: [batch, n_kv_head, seq, head_dim]
// attn_mask: [batch, seq] - padding mask (1 for valid, 0 for padding)
lean_object* lean_torch_sdpa_gqa_mask(
  uint64_t /*batch*/,
  uint64_t n_head,
  uint64_t n_kv_head,
  uint64_t /*seq*/,
  uint64_t /*head_dim*/,
  b_lean_obj_arg query,
  b_lean_obj_arg key,
  b_lean_obj_arg value,
  b_lean_obj_arg mask,
  double dropout_p,
  uint8_t is_causal,
  uint8_t enable_gqa
) {
  auto q = borrowTensor(query);
  auto k = borrowTensor(key);
  auto v = borrowTensor(value);
  auto padding_mask = borrowTensor(mask);  // [batch, seq]

  // Handle GQA by expanding KV heads to match Q heads
  if (enable_gqa) {
    if (n_head != n_kv_head && n_kv_head > 0) {
      auto repeat_factor = n_head / n_kv_head;
      k = k.repeat_interleave(repeat_factor, 1);
      v = v.repeat_interleave(repeat_factor, 1);
    }
  }

  // Get sequence length from query shape: [batch, n_head, seq, head_dim]
  auto seq_len = q.size(2);

  // Convert padding mask [batch, seq] to attention mask format
  // SDPA expects: -inf for masked positions, 0 for unmasked
  // padding_mask: 1 for valid, 0 for padding

  // Expand padding mask to [batch, 1, 1, seq] for key/value masking
  auto key_mask = padding_mask.unsqueeze(1).unsqueeze(2);  // [batch, 1, 1, seq]

  // Create causal mask if needed: [1, 1, seq, seq]
  torch::Tensor attn_mask;
  if (is_causal) {
    auto row_idx = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(1);
    auto col_idx = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(q.device())).unsqueeze(0);
    auto causal_mask = col_idx > row_idx;  // True for positions to mask (future)

    // Combine with padding mask: mask if either causal or padding
    // key_mask: [batch, 1, 1, seq], causal_mask: [seq, seq]
    auto combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | (key_mask == 0);

    attn_mask = torch::where(
      combined_mask,
      torch::full({1, 1, seq_len, seq_len}, -std::numeric_limits<float>::infinity(), q.options()),
      torch::zeros({1, 1, seq_len, seq_len}, q.options())
    );
  } else {
    // Just padding mask
    attn_mask = torch::where(
      key_mask == 0,
      torch::full({1, 1, 1, seq_len}, -std::numeric_limits<float>::infinity(), q.options()),
      torch::zeros({1, 1, 1, seq_len}, q.options())
    );
  }

  auto result_ = torch::scaled_dot_product_attention(
    q, k, v,
    attn_mask,
    dropout_p,
    false  // is_causal=false since we're using explicit mask
  );

  return fromTorchTensor(result_);
}

// ============================================================================
// Comparison and conditional operations for diffusion
// ============================================================================

// Less than comparison: returns a boolean tensor
lean_object* lean_torch_lt(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::lt(a_, b_);
  return fromTorchTensor(result_);
}

// Less than scalar comparison
lean_object* lean_torch_lt_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ < scalar;
  return fromTorchTensor(result_);
}

// Greater than comparison
lean_object* lean_torch_gt(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::gt(a_, b_);
  return fromTorchTensor(result_);
}

// Greater than or equal comparison
lean_object* lean_torch_ge(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::ge(a_, b_);
  return fromTorchTensor(result_);
}

// Equality comparison: returns a boolean tensor
lean_object* lean_torch_eq(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::eq(a_, b_);
  return fromTorchTensor(result_);
}

// Equality with scalar comparison
lean_object* lean_torch_eq_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t scalar) {
  auto input_ = borrowTensor(input);
  auto result_ = input_ == scalar;
  return fromTorchTensor(result_);
}

// Full tensor with given shape and scalar value (int64 version for token ids)
lean_object* lean_torch_full_int(lean_obj_arg s, int64_t value) {
  auto shape = getShape(s);
  auto result_ = torch::full(shape, value, torch::kInt64);
  return fromTorchTensor(result_);
}

// Maximum along dimension with indices
lean_object* lean_torch_max_dim(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::max(input_, static_cast<int64_t>(dim));
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// Maximum along dimension for 3D tensors with proper output shape
// d0, d1, d2 are implicit type parameters
lean_object* lean_torch_max_dim_3d(uint64_t /*d0*/, uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::max(input_, static_cast<int64_t>(dim));
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// Boolean any (returns true if any element is true)
uint8_t lean_torch_any(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  return input_.any().item<bool>();
}

// Logical NOT for boolean tensors
lean_object* lean_torch_logical_not(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::logical_not(input_);
  return fromTorchTensor(result_);
}

// Logical AND for boolean tensors
lean_object* lean_torch_logical_and(lean_obj_arg /*s*/, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::logical_and(a_, b_);
  return fromTorchTensor(result_);
}

// Convert to float dtype
lean_object* lean_torch_to_float(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.to(torch::kFloat32);
  return fromTorchTensor(result_);
}

// Index select for 1D indexing into first dimension
lean_object* lean_torch_index_select_1d(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg indices) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto result_ = input_.index_select(0, indices_);
  return fromTorchTensor(result_);
}

// Clamp values to a range
lean_object* lean_torch_clamp(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t min_val, int64_t max_val) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::clamp(input_, min_val, max_val);
  return fromTorchTensor(result_);
}

// Top-k values and indices
lean_object* lean_torch_topk(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::topk(input_, k, dim);
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// Top-k for 2D tensors with proper output shape
// d1, d2 are the implicit shape parameters
lean_object* lean_torch_topk_2d(uint64_t /*d1*/, uint64_t /*d2*/,
    b_lean_obj_arg input, uint64_t k, uint64_t dim) {
  auto input_ = borrowTensor(input);
  auto [values, indices] = torch::topk(input_, k, dim);
  // Return as Lean pair
  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(values));
  lean_ctor_set(result, 1, fromTorchTensor(indices));
  return result;
}

// ============================================================================
// New operations: Device availability, gather/scatter, einsum, interpolate
// ============================================================================

// Check if CUDA is available
lean_object* lean_torch_cuda_is_available(lean_object* /*w*/) {
  return lean_io_result_mk_ok(lean_box(torch::cuda::is_available()));
}

// Check if MPS is available
lean_object* lean_torch_mps_is_available(lean_object* /*w*/) {
#ifdef __APPLE__
  bool available = at::hasMPS();
  return lean_io_result_mk_ok(lean_box(available));
#else
  return lean_io_result_mk_ok(lean_box(false));
#endif
}

// Transposed 2D convolution
lean_object* lean_torch_conv_transpose2d(
    lean_obj_arg /*input_shape*/, lean_obj_arg /*weight_shape*/,
    b_lean_obj_arg input, b_lean_obj_arg weight,
    lean_obj_arg stride, lean_obj_arg padding,
    lean_obj_arg output_padding, lean_obj_arg dilation) {
  auto input_ = borrowTensor(input);
  auto weight_ = borrowTensor(weight);
  auto stride_ = getShape(stride); lean_dec(stride);
  auto padding_ = getShape(padding); lean_dec(padding);
  auto output_padding_ = getShape(output_padding); lean_dec(output_padding);
  auto dilation_ = getShape(dilation); lean_dec(dilation);

  auto result_ = torch::nn::functional::conv_transpose2d(input_, weight_,
    torch::nn::functional::ConvTranspose2dFuncOptions()
      .stride(stride_)
      .padding(padding_)
      .output_padding(output_padding_)
      .dilation(dilation_));
  return fromTorchTensor(result_);
}

// 2D spatial dropout
lean_object* lean_torch_dropout2d(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/, lean_obj_arg /*h*/, lean_obj_arg /*w*/,
    b_lean_obj_arg input, double p, uint8_t training, lean_object* /*w*/) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::nn::functional::dropout2d(input_,
    torch::nn::functional::Dropout2dFuncOptions()
      .p(p)
      .training(training != 0));
  return lean_io_result_mk_ok(fromTorchTensor(result_));
}

// 3D spatial dropout
lean_object* lean_torch_dropout3d(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/, lean_obj_arg /*d*/, lean_obj_arg /*h*/, lean_obj_arg /*w_dim*/,
    b_lean_obj_arg input, double p, uint8_t training, lean_object* /*w*/) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::nn::functional::dropout3d(input_,
    torch::nn::functional::Dropout3dFuncOptions()
      .p(p)
      .training(training != 0));
  return lean_io_result_mk_ok(fromTorchTensor(result_));
}

// NLL loss
lean_object* lean_torch_nll_loss(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/,
    b_lean_obj_arg log_probs, b_lean_obj_arg targets,
    lean_obj_arg reduction) {
  auto log_probs_ = borrowTensor(log_probs);
  auto targets_ = borrowTensor(targets);
  auto reduction_str = std::string(lean_string_cstr(reduction));
  lean_dec(reduction);

  torch::nn::functional::NLLLossFuncOptions options;
  if (reduction_str == "mean") options.reduction(torch::kMean);
  else if (reduction_str == "sum") options.reduction(torch::kSum);
  else options.reduction(torch::kNone);

  auto result_ = torch::nn::functional::nll_loss(log_probs_, targets_, options);
  return fromTorchTensor(result_);
}

// NLL loss with no reduction
lean_object* lean_torch_nll_loss_none(
    lean_obj_arg /*n*/, lean_obj_arg /*c*/,
    b_lean_obj_arg log_probs, b_lean_obj_arg targets) {
  auto log_probs_ = borrowTensor(log_probs);
  auto targets_ = borrowTensor(targets);
  auto result_ = torch::nn::functional::nll_loss(log_probs_, targets_,
    torch::nn::functional::NLLLossFuncOptions().reduction(torch::kNone));
  return fromTorchTensor(result_);
}

// Gather along dimension
lean_object* lean_torch_gather(
    lean_obj_arg /*input_shape*/, lean_obj_arg /*index_shape*/,
    b_lean_obj_arg input, int64_t dim, b_lean_obj_arg indices) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto result_ = torch::gather(input_, dim, indices_);
  return fromTorchTensor(result_);
}

// Scatter along dimension
lean_object* lean_torch_scatter(
    lean_obj_arg /*s*/,
    b_lean_obj_arg input, int64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_(dim, indices_, src_);
  return fromTorchTensor(result_);
}

// Scatter with add reduction
lean_object* lean_torch_scatter_add(
    lean_obj_arg /*s*/,
    b_lean_obj_arg input, int64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_add_(dim, indices_, src_);
  return fromTorchTensor(result_);
}

// Scatter with different shapes (for top-k style operations)
// input: [batch, seq], indices: [batch, k], src: [batch, k] -> [batch, seq]
// batch, seq, k are the implicit shape parameters
lean_object* lean_torch_scatter_2d(uint64_t /*batch*/, uint64_t /*seq*/, uint64_t /*k*/,
    b_lean_obj_arg input, int64_t dim, b_lean_obj_arg indices, b_lean_obj_arg src) {
  auto input_ = borrowTensor(input);
  auto indices_ = borrowTensor(indices);
  auto src_ = borrowTensor(src);
  auto result_ = input_.clone().scatter_(dim, indices_, src_);
  return fromTorchTensor(result_);
}

// Einsum with array of tensors
lean_object* lean_torch_einsum(lean_obj_arg equation, b_lean_obj_arg tensors) {
  auto eq_str = std::string(lean_string_cstr(equation));
  lean_dec(equation);

  std::vector<torch::Tensor> tensor_vec;
  size_t n = lean_array_size(tensors);
  for (size_t i = 0; i < n; i++) {
    tensor_vec.push_back(borrowTensor(lean_array_get_core(tensors, i)));
  }

  auto result_ = torch::einsum(eq_str, tensor_vec);
  return fromTorchTensor(result_);
}

// Einsum with 2 tensors (common case)
lean_object* lean_torch_einsum2(
    lean_obj_arg /*s1*/, lean_obj_arg /*s2*/,
    lean_obj_arg equation, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto eq_str = std::string(lean_string_cstr(equation));
  lean_dec(equation);
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto result_ = torch::einsum(eq_str, {a_, b_});
  return fromTorchTensor(result_);
}

// Interpolate to target size
lean_object* lean_torch_interpolate(
    lean_obj_arg /*s*/, b_lean_obj_arg input,
    lean_obj_arg size, lean_obj_arg mode, uint8_t align_corners) {
  auto input_ = borrowTensor(input);
  auto size_ = getShape(size); lean_dec(size);
  auto mode_str = std::string(lean_string_cstr(mode));
  lean_dec(mode);

  torch::nn::functional::InterpolateFuncOptions options;
  options.size(size_);

  if (mode_str == "nearest") options.mode(torch::kNearest);
  else if (mode_str == "linear") options.mode(torch::kLinear);
  else if (mode_str == "bilinear") options.mode(torch::kBilinear);
  else if (mode_str == "bicubic") options.mode(torch::kBicubic);
  else if (mode_str == "trilinear") options.mode(torch::kTrilinear);
  else if (mode_str == "area") options.mode(torch::kArea);
  else options.mode(torch::kNearest);

  if (mode_str != "nearest" && mode_str != "area") {
    options.align_corners(align_corners != 0);
  }

  auto result_ = torch::nn::functional::interpolate(input_, options);
  return fromTorchTensor(result_);
}

// Interpolate with scale factor
lean_object* lean_torch_interpolate_scale(
    lean_obj_arg /*s*/, b_lean_obj_arg input,
    lean_obj_arg scale_factor, lean_obj_arg mode, uint8_t align_corners) {
  auto input_ = borrowTensor(input);
  auto mode_str = std::string(lean_string_cstr(mode));
  lean_dec(mode);

  // Extract scale factors
  std::vector<double> scales;
  size_t n = lean_array_size(scale_factor);
  for (size_t i = 0; i < n; i++) {
    scales.push_back(lean_unbox_float(lean_array_get_core(scale_factor, i)));
  }
  lean_dec(scale_factor);

  torch::nn::functional::InterpolateFuncOptions options;
  options.scale_factor(scales);

  if (mode_str == "nearest") options.mode(torch::kNearest);
  else if (mode_str == "linear") options.mode(torch::kLinear);
  else if (mode_str == "bilinear") options.mode(torch::kBilinear);
  else if (mode_str == "bicubic") options.mode(torch::kBicubic);
  else if (mode_str == "trilinear") options.mode(torch::kTrilinear);
  else if (mode_str == "area") options.mode(torch::kArea);
  else options.mode(torch::kNearest);

  if (mode_str != "nearest" && mode_str != "area") {
    options.align_corners(align_corners != 0);
  }

  auto result_ = torch::nn::functional::interpolate(input_, options);
  return fromTorchTensor(result_);
}

// Clip gradient values element-wise
lean_object* lean_torch_clip_grad_value_(lean_obj_arg /*s*/, b_lean_obj_arg param, double clip_value, lean_object* /*w*/) {
  auto param_ = borrowTensor(param);
  if (param_.grad().defined()) {
    param_.grad().clamp_(-clip_value, clip_value);
  }
  return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Sampling utilities for text generation
// ============================================================================

// Top-k filtering: set all logits outside top-k to -infinity
// logits: [..., vocab_size], k: number of top logits to keep
lean_object* lean_torch_topk_filter(
    lean_obj_arg /*s*/,
    b_lean_obj_arg logits,
    uint64_t k
) {
  auto logits_ = borrowTensor(logits);

  if (k == 0) {
    return fromTorchTensor(logits_.clone());
  }

  // Get top-k values along last dimension
  auto [topk_values, topk_indices] = logits_.topk(k, -1, true, true);

  // Get the minimum value in top-k (threshold)
  auto threshold = std::get<0>(topk_values.min(-1, true));

  // Mask: set logits below threshold to -inf
  auto mask = logits_ < threshold;
  auto result_ = logits_.masked_fill(mask, -std::numeric_limits<float>::infinity());

  return fromTorchTensor(result_);
}

// Top-p (nucleus) filtering: set logits outside cumulative probability p to -infinity
// logits: [..., vocab_size], p: cumulative probability threshold
lean_object* lean_torch_topp_filter(
    lean_obj_arg /*s*/,
    b_lean_obj_arg logits,
    double p
) {
  auto logits_ = borrowTensor(logits);

  if (p >= 1.0) {
    return fromTorchTensor(logits_.clone());
  }

  // Sort logits in descending order
  auto [sorted_logits, sorted_indices] = logits_.sort(-1, true);

  // Compute cumulative softmax probabilities
  auto softmax_sorted = torch::softmax(sorted_logits, -1);
  auto cumulative_probs = softmax_sorted.cumsum(-1);

  // Create mask for tokens to remove (cumulative prob > p, shifted by 1)
  // We shift by 1 so we always keep at least one token
  auto shifted_cumprobs = torch::zeros_like(cumulative_probs);
  if (cumulative_probs.size(-1) > 1) {
    shifted_cumprobs.slice(-1, 1) = cumulative_probs.slice(-1, 0, -1);
  }
  auto sorted_indices_to_remove = shifted_cumprobs > p;

  // Scatter the remove mask back to original positions
  auto indices_to_remove = torch::zeros_like(sorted_indices_to_remove);
  indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove);

  // Set removed positions to -inf
  auto result_ = logits_.masked_fill(indices_to_remove, -std::numeric_limits<float>::infinity());

  return fromTorchTensor(result_);
}

// Squeeze tensor along a specific dimension (with negative index support)
lean_object* lean_torch_squeeze_dim(
    lean_obj_arg /*s*/,
    b_lean_obj_arg input,
    int64_t dim
) {
  auto input_ = borrowTensor(input);
  auto result_ = input_.squeeze(dim);
  return fromTorchTensor(result_);
}

// ============================================================================
// Linear Algebra operations for manifold optimization
// ============================================================================

// QR Decomposition: A = Q @ R
// Returns (Q, R) where Q is orthogonal and R is upper triangular
// For an m×n matrix, returns Q as m×m (complete mode) and R as m×n
lean_object* lean_torch_qr(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Use full QR to get m×m Q matrix
  auto [Q, R] = torch::linalg_qr(A, "complete");

  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(Q));
  lean_ctor_set(result, 1, fromTorchTensor(R));
  return result;
}

// Reduced QR: returns Q as m×min(m,n) and R as min(m,n)×n
// More efficient when m > n
lean_object* lean_torch_qr_reduced(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Default "reduced" mode
  auto result_tuple = torch::linalg_qr(A);
  auto Q = std::get<0>(result_tuple);
  auto R = std::get<1>(result_tuple);

  // Debug: verify Q is orthogonal
  auto QtQ = torch::mm(Q.t(), Q);
  auto I = torch::eye(Q.size(1), Q.options());
  auto diff = (QtQ - I).abs().max().item<float>();
  if (diff > 1e-4) {
    std::cerr << "WARNING: QR produced non-orthogonal Q! Max diff from I: " << diff << std::endl;
    std::cerr << "A shape: " << A.sizes() << ", Q shape: " << Q.sizes() << std::endl;
  }

  lean_object* result = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(result, 0, fromTorchTensor(Q));
  lean_ctor_set(result, 1, fromTorchTensor(R));
  return result;
}

// Matrix exponential for square matrices: exp(A)
// Uses Padé approximation internally
lean_object* lean_torch_matrix_exp(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::linalg_matrix_exp(A);
  return fromTorchTensor(result);
}

// Matrix logarithm for square matrices: log(A)
// Implemented via eigendecomposition: log(A) = V @ diag(log(L)) @ V^{-1}
lean_object* lean_torch_matrix_log(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // Compute eigendecomposition: A = V @ diag(L) @ V^{-1}
  auto eig_result = torch::linalg_eig(A);
  auto L = std::get<0>(eig_result);  // eigenvalues (complex)
  auto V = std::get<1>(eig_result);  // eigenvectors (complex)
  // Take log of eigenvalues
  auto log_L = torch::log(L);
  // Reconstruct: log(A) = V @ diag(log_L) @ V^{-1}
  auto V_inv = torch::linalg_inv(V);
  auto result = torch::matmul(torch::matmul(V, torch::diag_embed(log_L)), V_inv);
  // Return real part (assuming input is real and result should be real)
  return fromTorchTensor(torch::real(result));
}

// Matrix inverse for square matrices: inv(A)
lean_object* lean_torch_inv(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::linalg_inv(A);
  return fromTorchTensor(result);
}

// SVD decomposition: A = U @ diag(S) @ V^T
// Returns (U, S, Vh) where Vh = V^T
lean_object* lean_torch_svd(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  // full_matrices=false gives reduced SVD
  auto result = torch::linalg_svd(A, false);
  auto U = std::get<0>(result);
  auto S = std::get<1>(result);
  auto Vh = std::get<2>(result);

  // Return as triple (U, S, Vh)
  lean_object* tuple = lean_alloc_ctor(0, 3, 0);
  lean_ctor_set(tuple, 0, fromTorchTensor(U));
  lean_ctor_set(tuple, 1, fromTorchTensor(S));
  lean_ctor_set(tuple, 2, fromTorchTensor(Vh));
  return tuple;
}

// SVD values only (singular values): returns just S from A = U @ diag(S) @ V^T
lean_object* lean_torch_svdvals(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto S = torch::linalg_svdvals(A);
  return fromTorchTensor(S);
}

// Extract diagonal of a matrix
lean_object* lean_torch_diag(
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto result = torch::diag(A);
  return fromTorchTensor(result);
}

// Create diagonal matrix from vector
lean_object* lean_torch_diagflat(
    uint64_t /*n*/,
    b_lean_obj_arg v_obj
) {
  auto v = borrowTensor(v_obj);
  auto result = torch::diag(v);
  return fromTorchTensor(result);
}

// ============================================================================
// Modular Norm operations
// ============================================================================

// Spectral norm: largest singular value σ_max(A)
double lean_torch_spectral_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto sv = torch::linalg_svdvals(A);
  return sv[0].item<double>();
}

// Nuclear norm: sum of singular values Σσᵢ(A)
double lean_torch_nuclear_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto sv = torch::linalg_svdvals(A);
  return sv.sum().item<double>();
}

// Row-wise L2 norms: ||a_i||₂ for each row
lean_object* lean_torch_row_norms(
    uint64_t /*n*/,
    uint64_t /*d*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto norms = A.norm(2, /*dim=*/1);
  return fromTorchTensor(norms);
}

// Max row norm: max_i ||a_i||₂
double lean_torch_max_row_norm(
    uint64_t /*n*/,
    uint64_t /*d*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  auto norms = A.norm(2, /*dim=*/1);
  return norms.max().item<double>();
}

// L2 norm of a 1D tensor
double lean_torch_l2_norm(
    uint64_t /*n*/,
    b_lean_obj_arg v_obj
) {
  auto v = borrowTensor(v_obj);
  return v.norm(2).item<double>();
}

// Frobenius norm of a matrix
double lean_torch_frobenius_norm(
    uint64_t /*m*/,
    uint64_t /*n*/,
    b_lean_obj_arg A_obj
) {
  auto A = borrowTensor(A_obj);
  return A.norm().item<double>();
}

}
