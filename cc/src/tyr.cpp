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
static inline torch::Tensor borrowTensor(b_lean_obj_arg o) {
    auto impl = static_cast<torch::TensorImpl*>(lean_get_external_data(o));
    // unsafe_reclaim_from_nonowning increments the refcount
    return torch::Tensor(c10::intrusive_ptr<torch::TensorImpl>::unsafe_reclaim_from_nonowning(impl));
}

// Transfer ownership of a new tensor to Lean.
// The tensor's refcount is transferred to Lean's finalizer.
static inline lean_object *giveTensor(torch::Tensor t) {
  g_live_lean_tensors++; // Increment when a new tensor is given to Lean
  return lean_alloc_external(getTorchTensorImplClass(), t.unsafeReleaseTensorImpl());
}

// Alias for backward compatibility
static inline lean_object *fromTorchTensor(torch::Tensor t) {
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
    auto cuda_idx = lean_unbox_uint64(lean_ctor_get(device, 0));
    return torch::Device(torch::kCUDA, cuda_idx);
  } else {  // CPU
    return torch::Device(torch::kCPU);
  }
}

extern "C" {

lean_object* lean_torch_get_live_tensors(lean_object* /* w */) {
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(g_live_lean_tensors.load())));
}


// --
// tensor creation api
lean_object* lean_torch_randn(lean_obj_arg s, int requires_grad) {
  auto t = torch::randn(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}
lean_object* lean_torch_rand(lean_obj_arg s, int requires_grad) {
  auto t = torch::rand(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_randint(int64_t low, int64_t high, lean_obj_arg s, int /*requires_grad*/) {
  // randint always returns Long (int64) dtype - integral types don't support requires_grad
  auto t = torch::randint(low, high, getShape(s), torch::kLong);
  lean_dec(s);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_full(lean_obj_arg s, double value, int requires_grad) {
  auto t = torch::full(getShape(s), value, torch::TensorOptions().requires_grad(requires_grad));
  lean_dec(s);
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros(lean_obj_arg s, int requires_grad) {
  auto t = torch::zeros(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  lean_dec(s);
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones(lean_obj_arg s, int requires_grad) {
  auto t = torch::ones(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  lean_dec(s);
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
UNOP_FUN(tanh)
#undef UNOP_FUN

lean_object* lean_torch_tensor_grad(lean_obj_arg /* s */, lean_obj_arg /* s' */, b_lean_obj_arg output, b_lean_obj_arg input, b_lean_obj_arg grad_output) {
  auto output_ = borrowTensor(output);
  auto input_ = borrowTensor(input);
  auto grad_output_ = borrowTensor(grad_output);

  torch::autograd::variable_list out_v({output_});
  torch::autograd::variable_list in_v({input_});
  torch::autograd::variable_list grad_out_v({grad_output_});    
  auto grad_in_v = torch::autograd::grad(out_v, in_v, grad_out_v);

  
  return fromTorchTensor(grad_in_v[0]);
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
  auto res_ = torch::allclose(a_, b_, rtol, atol);

  return res_;
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
void lean_torch_set_grad_enabled(uint8_t enabled) {
    torch::autograd::GradMode::set_enabled(enabled);
}

uint8_t lean_torch_is_grad_enabled() {
    return torch::autograd::GradMode::is_enabled();
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



lean_object* lean_torch_tensor_softmax(lean_obj_arg /**/, b_lean_obj_arg a) {
  auto a_ = borrowTensor(a);
  auto c_ = torch::softmax(a_, 0);
  return fromTorchTensor(c_);
}

lean_object* lean_torch_tensor_cross_entropy(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = borrowTensor(a);
  auto b_ = borrowTensor(b);
  auto c_ = torch::nn::functional::cross_entropy(a_, b_);
  return fromTorchTensor(c_);
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
    running_mean_ = borrowTensor(running_mean);
  }
  if (!lean_is_scalar(running_var)) {
    running_var_ = borrowTensor(running_var);
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
    weight_ = borrowTensor(weight);
  }
  if (!lean_is_scalar(bias)) {
    bias_ = borrowTensor(bias);
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
    weight_ = borrowTensor(weight);
  }
  if (!lean_is_scalar(bias)) {
    bias_ = borrowTensor(bias);
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
    running_mean_ = borrowTensor(running_mean);
  }
  if (!lean_is_scalar(running_var)) {
    running_var_ = borrowTensor(running_var);
  }
  if (!lean_is_scalar(weight)) {
    weight_ = borrowTensor(weight);
  }
  if (!lean_is_scalar(bias)) {
    bias_ = borrowTensor(bias);
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
    weight_ = borrowTensor(weight);
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
    padding_idx_val = lean_scalar_to_int64(padding_idx);
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
    padding_idx_val = lean_scalar_to_int64(padding_idx);
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

lean_object* lean_torch_bmm(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::bmm(input_, mat2_);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mm(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = borrowTensor(input);
  auto mat2_ = borrowTensor(mat2);
  auto result_ = torch::mm(input_, mat2_);
  return fromTorchTensor(result_);
}

// Tensor operations (essential for attention mechanisms)
lean_object* lean_torch_transpose(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim0, int dim1) {
  auto input_ = borrowTensor(input);
  auto result_ = torch::transpose(input_, dim0, dim1);
  return fromTorchTensor(result_);
}

lean_object* lean_torch_cat(lean_obj_arg /*s*/, lean_obj_arg tensors, int dim) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(borrowTensor(tensor_obj));
  }
  lean_dec(tensors); // Decrement array
  auto result_ = torch::cat(tensor_list, dim);
  // tensor_list tensors are auto-released when vector goes out of scope
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
    attn_mask_ = borrowTensor(attn_mask);
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
    // Sum all elements
    result_ = torch::sum(input_);
  } else {
    // Sum along specific dimensions
    auto dims = getShape(dim); lean_dec(dim);
    result_ = torch::sum(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mean(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  auto input_ = borrowTensor(input);
  torch::Tensor result_;
  
  if (lean_is_scalar(dim)) {
    // Mean of all elements
    result_ = torch::mean(input_);
  } else {
    // Mean along specific dimensions
    auto dims = getShape(dim); lean_dec(dim);
    result_ = torch::mean(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
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
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
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

}
