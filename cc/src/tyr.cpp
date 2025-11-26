#include <iostream>
#include <fstream>
#include <lean/lean.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

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

static inline torch::Tensor toTorchTensor(b_lean_obj_arg o) {
    // lean_assert(lean_external_class(o) == getTorchTensorImplClass());
    auto impl = c10::intrusive_ptr<torch::TensorImpl>::reclaim(static_cast<torch::TensorImpl*>(lean_get_external_data(o)));
    return torch::Tensor(impl);
}

static inline lean_object *fromTorchTensor(torch::Tensor t) {
  return lean_alloc_external(getTorchTensorImplClass(), t.unsafeReleaseTensorImpl());
}


extern "C" {
  
std::vector<int64_t> getShape(lean_obj_arg s) {
  std::vector<int64_t> shape;  
  for (size_t i = 0; i<lean_array_size(s); i++) {
    shape.push_back(lean_unbox_uint64(lean_array_get_core(s, i)));
  }
  return shape;
}

torch::Device getDevice(lean_obj_arg device) {
  auto tag = lean_obj_tag(device);
  if (tag == 0) {  // CUDA
    auto cuda_idx = lean_unbox_uint64(lean_ctor_get(device, 0));
    return torch::Device(torch::kCUDA, cuda_idx);
  } else {  // CPU
    return torch::Device(torch::kCPU);
  }
}


lean_object* backward(lean_obj_arg /* shape */, b_lean_obj_arg output, b_lean_obj_arg grad_output) {
    auto output_ = toTorchTensor(output);
    auto grad_output_ = toTorchTensor(grad_output);
    output_.backward(grad_output_);
    grad_output_.unsafeReleaseTensorImpl();
    return fromTorchTensor(output_);
}

// --
// tensor creation api
lean_object* lean_torch_randn(lean_obj_arg s, int requires_grad) {
  auto t = torch::randn(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  return lean_io_result_mk_ok(fromTorchTensor(t));
}
lean_object* lean_torch_rand(lean_obj_arg s, int requires_grad) {
  auto t = torch::rand(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_randint(int64_t low, int64_t high, lean_obj_arg s, int /*requires_grad*/) {
  // randint always returns Long (int64) dtype - integral types don't support requires_grad
  auto t = torch::randint(low, high, getShape(s), torch::kLong);
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_full(lean_obj_arg s, double value, int requires_grad) {
  auto t = torch::full(getShape(s), value, torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_zeros(lean_obj_arg s, int requires_grad) {
  auto t = torch::zeros(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}

lean_object* lean_torch_ones(lean_obj_arg s, int requires_grad) {
  auto t = torch::ones(getShape(s), torch::TensorOptions().requires_grad(requires_grad));
  return fromTorchTensor(t);
}


lean_object* lean_torch_arange(int start, int stop, int step) {
  auto t = torch::arange(start, stop, step);
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
    auto x_ = toTorchTensor(x);
    auto res_ = x_.requires_grad();
    x_.unsafeReleaseTensorImpl();
    return res_;
}

//

lean_object* lean_torch_reshape(lean_obj_arg /*s*/, b_lean_obj_arg self, b_lean_obj_arg shape) {
  auto shape_ = getShape(shape);
  auto self_ = toTorchTensor(self);
  auto res = torch::reshape(self_, shape_);
  self_.unsafeReleaseTensorImpl();
  return fromTorchTensor(res);
}

lean_object* lean_torch_permute(lean_obj_arg /*s*/, b_lean_obj_arg self, b_lean_obj_arg permutation) {
  auto permutation_ = getShape(permutation);
  auto self_ = toTorchTensor(self);
  auto res = torch::permute(self_, permutation_);
  self_.unsafeReleaseTensorImpl();
  return fromTorchTensor(res);
}

lean_object* lean_torch_get(lean_obj_arg /*s*/, b_lean_obj_arg self, int idx) {
  auto self_ = toTorchTensor(self);
  auto res = self_.index({idx});
  self_.unsafeReleaseTensorImpl();
  return fromTorchTensor(res);
}

lean_object* lean_torch_to(lean_obj_arg /*s*/, b_lean_obj_arg self, lean_obj_arg device) {
  auto self_ = toTorchTensor(self);
  auto device_ = getDevice(device);
  auto res = self_.to(device_);
  self_.unsafeReleaseTensorImpl();
  return fromTorchTensor(res);
}

lean_object* lean_torch_slice(lean_obj_arg /*s*/, 
  b_lean_obj_arg self, 
  b_lean_obj_arg dim,
  int start,
  int stop,
  int step,
  lean_obj_arg /* */,
  lean_obj_arg /* */, 
  lean_obj_arg /* */) {

  auto dim_ = lean_uint64_of_nat(dim);
  auto self_ = toTorchTensor(self);
  auto res = self_.slice(dim_, start, stop, step);
  self_.unsafeReleaseTensorImpl();
  return fromTorchTensor(res);
}

#define BINOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) { \
  auto a_ = toTorchTensor(a); \
  auto b_ = toTorchTensor(b); \
  auto c_ = torch::F(a_, b_); \
  a_.unsafeReleaseTensorImpl(); \
  b_.unsafeReleaseTensorImpl(); \
  return fromTorchTensor(c_); \
}

BINOP_FUN(add)
BINOP_FUN(sub)
BINOP_FUN(mul)
#undef BINOP_FUN

#define UNOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a) { \
  auto a_ = toTorchTensor(a); \
  auto c_ = torch::F(a_); \
  a_.unsafeReleaseTensorImpl(); \
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
  auto output_ = toTorchTensor(output);
  auto input_ = toTorchTensor(input);
  auto grad_output_ = toTorchTensor(grad_output);

  torch::autograd::variable_list out_v({output_});
  torch::autograd::variable_list in_v({input_});
  torch::autograd::variable_list grad_out_v({grad_output_});    
  auto grad_in_v = torch::autograd::grad(out_v, in_v, grad_out_v);

  output_.unsafeReleaseTensorImpl();
  input_.unsafeReleaseTensorImpl();
  grad_output_.unsafeReleaseTensorImpl();
  
  return fromTorchTensor(grad_in_v[0]);
}

lean_object* lean_torch_backward(lean_obj_arg /* shape */, b_lean_obj_arg output, b_lean_obj_arg grad_output) {
  auto out_ = toTorchTensor(output);
  auto grad_output_ = toTorchTensor(grad_output);
  out_.backward(grad_output_);

  grad_output_.unsafeReleaseTensorImpl();
  return output;
}

int lean_torch_allclose(lean_obj_arg /* shape */, b_lean_obj_arg a, b_lean_obj_arg b, double rtol, double atol) {
  auto a_ = toTorchTensor(a);
  auto b_ = toTorchTensor(b);
  auto res_ = torch::allclose(a_, b_, rtol, atol);
  a_.unsafeReleaseTensorImpl();
  b_.unsafeReleaseTensorImpl();

  return res_;
}

lean_object* lean_torch_grad_of(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto out_ = toTorchTensor(x);
    auto grad_out_ = out_.grad();
    out_.unsafeReleaseTensorImpl();
    return fromTorchTensor(grad_out_);
}

// Zero gradients
lean_object* lean_torch_zero_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    if (x_.grad().defined()) {
        x_.grad().zero_();
    }
    x_.unsafeReleaseTensorImpl();
    return x;
}

// Set requires_grad
lean_object* lean_torch_set_requires_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, uint8_t requires_grad) {
    auto x_ = toTorchTensor(x);
    auto result_ = x_.requires_grad_(requires_grad);
    x_.unsafeReleaseTensorImpl();
    return fromTorchTensor(result_);
}

// Detach tensor from computation graph
lean_object* lean_torch_detach(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    auto result_ = x_.detach();
    x_.unsafeReleaseTensorImpl();
    return fromTorchTensor(result_);
}

// Clone tensor with gradient
lean_object* lean_torch_clone(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    auto result_ = x_.clone();
    x_.unsafeReleaseTensorImpl();
    return fromTorchTensor(result_);
}

// Retain grad (for non-leaf tensors)
lean_object* lean_torch_retain_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    x_.retain_grad();
    x_.unsafeReleaseTensorImpl();
    return x;
}

// Check if tensor is leaf (for autograd)
uint8_t lean_torch_is_leaf(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    auto result = x_.is_leaf();
    x_.unsafeReleaseTensorImpl();
    return result;
}

// Get gradient function (for debugging)
uint8_t lean_torch_has_grad_fn(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    auto result = x_.grad_fn() != nullptr;
    x_.unsafeReleaseTensorImpl();
    return result;
}

// Accumulate gradients
lean_object* lean_torch_accumulate_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = toTorchTensor(x);
    auto grad_ = toTorchTensor(grad);
    
    if (x_.grad().defined()) {
        x_.mutable_grad() += grad_;
    } else {
        x_.mutable_grad() = grad_.clone();
    }
    
    x_.unsafeReleaseTensorImpl();
    grad_.unsafeReleaseTensorImpl();
    return x;
}

// Manual gradient setting
lean_object* lean_torch_set_grad(lean_obj_arg /* shape */, b_lean_obj_arg x, b_lean_obj_arg grad) {
    auto x_ = toTorchTensor(x);
    auto grad_ = toTorchTensor(grad);
    
    x_.mutable_grad() = grad_.clone();
    
    x_.unsafeReleaseTensorImpl();
    grad_.unsafeReleaseTensorImpl();
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
    auto z_ = toTorchTensor(z);
    auto y_ = toTorchTensor(y);
    auto x_ = toTorchTensor(x);
    auto grad_z_ = toTorchTensor(grad_z);
    auto grad_y_ = toTorchTensor(grad_y);
    
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
    
    z_.unsafeReleaseTensorImpl();
    y_.unsafeReleaseTensorImpl();
    x_.unsafeReleaseTensorImpl();
    grad_z_.unsafeReleaseTensorImpl();
    grad_y_.unsafeReleaseTensorImpl();
    
    return fromTorchTensor(grad_x_second[0]);
}



lean_object* lean_torch_tensor_softmax(lean_obj_arg /**/, b_lean_obj_arg a) {
  auto a_ = toTorchTensor(a);
  auto c_ = torch::softmax(a_, 0);
  a_.unsafeReleaseTensorImpl();
  return fromTorchTensor(c_);
}

lean_object* lean_torch_tensor_cross_entropy(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) {
  auto a_ = toTorchTensor(a);
  auto b_ = toTorchTensor(b);
  auto c_ = torch::nn::functional::cross_entropy(a_, b_);
  a_.unsafeReleaseTensorImpl();
  b_.unsafeReleaseTensorImpl();
  return fromTorchTensor(c_);
}


lean_object *lean_torch_to_string(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = toTorchTensor(t);
  std::ostringstream stream;
  stream << tensor;

  tensor.unsafeReleaseTensorImpl();
  return lean_mk_string(stream.str().c_str());
}

lean_object* lean_torch_tensor_print(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = toTorchTensor(t);
  std::cout << tensor << std::endl;
  tensor.unsafeReleaseTensorImpl();
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_torch_linear(
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
  lean_obj_arg /*b*/,
  b_lean_obj_arg x,
  b_lean_obj_arg M
) {
  auto M_ = toTorchTensor(M);
  auto x_ = toTorchTensor(x);
  auto y_ = torch::linear(x_, M_);
  M_.unsafeReleaseTensorImpl();
  x_.unsafeReleaseTensorImpl();
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
  auto M_ = toTorchTensor(M);
  auto b_ = toTorchTensor(b);
  auto x_ = toTorchTensor(x);
  auto y_ = torch::linear(x_, M_, b_);
  M_.unsafeReleaseTensorImpl();
  x_.unsafeReleaseTensorImpl();
  b_.unsafeReleaseTensorImpl();
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
  auto x_ = toTorchTensor(x);
  auto weight_ = toTorchTensor(weight);
  // torch::linear does x @ weight.T, which is what we want
  auto y_ = torch::linear(x_, weight_);
  x_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
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
  auto x_ = toTorchTensor(x);
  auto weight_ = toTorchTensor(weight);
  auto bias_ = toTorchTensor(bias);
  auto y_ = torch::linear(x_, weight_, bias_);
  x_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
  bias_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);
  
  auto options = torch::nn::functional::Conv2dFuncOptions()
    .stride(stride_)
    .padding(padding_) 
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv2d(input_, weight_, options);
  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);
  auto options = torch::nn::functional::Conv1dFuncOptions()
    .stride(stride)
    .padding(padding) 
    .dilation(dilation);
  auto output_ = torch::nn::functional::conv1d(input_, weight_, options);
  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_adaptive_avg_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg output_size
) {
  auto input_ = toTorchTensor(input);
  auto output_size_ = getShape(output_size);
  auto output_ = torch::nn::functional::adaptive_avg_pool2d(input_, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(output_size_));
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_avg_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  auto input_ = toTorchTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  
  auto options = torch::nn::functional::AvgPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::avg_pool2d(input_, options);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_max_pool2d(
  lean_obj_arg /*input_shape*/,
  b_lean_obj_arg input,
  lean_obj_arg kernel_size,
  lean_obj_arg stride,
  lean_obj_arg padding
) {
  auto input_ = toTorchTensor(input);
  auto kernel_size_ = getShape(kernel_size);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  
  auto options = torch::nn::functional::MaxPool2dFuncOptions(kernel_size_);
  if (stride_.size() > 0) {
    options.stride(stride_);
  }
  options.padding(padding_);
  
  auto output_ = torch::nn::functional::max_pool2d(input_, options);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_dropout(lean_obj_arg /*s*/, b_lean_obj_arg input, double p, int training) {
  auto input_ = toTorchTensor(input);
  auto output_ = torch::nn::functional::dropout(input_, torch::nn::functional::DropoutFuncOptions().p(p).training(training));
  input_.unsafeReleaseTensorImpl();
  return lean_io_result_mk_ok(fromTorchTensor(output_));
}

lean_object* lean_torch_mse_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = toTorchTensor(input);
  auto target_ = toTorchTensor(target);
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
  input_.unsafeReleaseTensorImpl();
  target_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_log_softmax(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = toTorchTensor(input);
  auto output_ = torch::nn::functional::log_softmax(input_, torch::nn::functional::LogSoftmaxFuncOptions(dim));
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

lean_object* lean_torch_leaky_relu(lean_obj_arg /*s*/, b_lean_obj_arg input, double negative_slope) {
  auto input_ = toTorchTensor(input);
  auto output_ = torch::nn::functional::leaky_relu(input_, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
  input_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  torch::Tensor running_mean_ = torch::Tensor();
  torch::Tensor running_var_ = torch::Tensor();
  
  // Handle optional running statistics
  if (!lean_is_scalar(running_mean)) {
    running_mean_ = toTorchTensor(running_mean);
  }
  if (!lean_is_scalar(running_var)) {
    running_var_ = toTorchTensor(running_var);
  }
  
  torch::nn::functional::BatchNormFuncOptions options;
  options.training(training).momentum(momentum).eps(eps);
  
  auto output_ = torch::nn::functional::batch_norm(
    input_, 
    running_mean_,
    running_var_,
    options
  );
  
  input_.unsafeReleaseTensorImpl();
  if (running_mean_.defined()) running_mean_.unsafeReleaseTensorImpl();
  if (running_var_.defined()) running_var_.unsafeReleaseTensorImpl();
  
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
  auto input_ = toTorchTensor(input);
  auto shape_array = getShape(normalized_shape);
  
  torch::Tensor weight_, bias_;
  bool has_weight = !lean_is_scalar(weight);
  bool has_bias = !lean_is_scalar(bias);
  
  if (has_weight) {
    weight_ = toTorchTensor(weight);
  }
  if (has_bias) {
    bias_ = toTorchTensor(bias);
  }
  
  auto output_ = torch::layer_norm(
    input_, 
    c10::IntArrayRef(shape_array.data(), shape_array.size()),
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    has_bias ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );
  
  input_.unsafeReleaseTensorImpl();
  if (has_weight) weight_.unsafeReleaseTensorImpl();
  if (has_bias) bias_.unsafeReleaseTensorImpl();
  
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
  auto input_ = toTorchTensor(input);
  
  torch::Tensor weight_, bias_;
  bool has_weight = !lean_is_scalar(weight);
  bool has_bias = !lean_is_scalar(bias);
  
  if (has_weight) {
    weight_ = toTorchTensor(weight);
  }
  if (has_bias) {
    bias_ = toTorchTensor(bias);
  }
  
  auto output_ = torch::group_norm(
    input_, 
    num_groups,
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    has_bias ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    eps
  );
  
  input_.unsafeReleaseTensorImpl();
  if (has_weight) weight_.unsafeReleaseTensorImpl();
  if (has_bias) bias_.unsafeReleaseTensorImpl();
  
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
  auto input_ = toTorchTensor(input);
  
  torch::Tensor running_mean_, running_var_, weight_, bias_;
  bool has_running_mean = !lean_is_scalar(running_mean);
  bool has_running_var = !lean_is_scalar(running_var);
  bool has_weight = !lean_is_scalar(weight);
  bool has_bias = !lean_is_scalar(bias);
  
  if (has_running_mean) {
    running_mean_ = toTorchTensor(running_mean);
  }
  if (has_running_var) {
    running_var_ = toTorchTensor(running_var);
  }
  if (has_weight) {
    weight_ = toTorchTensor(weight);
  }
  if (has_bias) {
    bias_ = toTorchTensor(bias);
  }
  
  auto output_ = torch::instance_norm(
    input_, 
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    has_bias ? c10::optional<torch::Tensor>(bias_) : c10::nullopt,
    has_running_mean ? c10::optional<torch::Tensor>(running_mean_) : c10::nullopt,
    has_running_var ? c10::optional<torch::Tensor>(running_var_) : c10::nullopt,
    use_input_stats,
    momentum,
    eps,
    true  // cudnn_enabled
  );
  
  input_.unsafeReleaseTensorImpl();
  if (has_running_mean) running_mean_.unsafeReleaseTensorImpl();
  if (has_running_var) running_var_.unsafeReleaseTensorImpl();
  if (has_weight) weight_.unsafeReleaseTensorImpl();
  if (has_bias) bias_.unsafeReleaseTensorImpl();
  
  return fromTorchTensor(output_);
}

// Binary cross entropy loss
lean_object* lean_torch_binary_cross_entropy(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg weight, lean_obj_arg reduction) {
  auto input_ = toTorchTensor(input);
  auto target_ = toTorchTensor(target);
  
  torch::Tensor weight_;
  bool has_weight = !lean_is_scalar(weight);
  
  if (has_weight) {
    weight_ = toTorchTensor(weight);
  }
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  auto output_ = torch::binary_cross_entropy(
    input_, 
    target_,
    has_weight ? c10::optional<torch::Tensor>(weight_) : c10::nullopt,
    reduction_mode
  );
  
  input_.unsafeReleaseTensorImpl();
  target_.unsafeReleaseTensorImpl();
  if (has_weight) weight_.unsafeReleaseTensorImpl();
  
  return fromTorchTensor(output_);
}

// L1 loss
lean_object* lean_torch_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction) {
  auto input_ = toTorchTensor(input);
  auto target_ = toTorchTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  auto output_ = torch::l1_loss(
    input_, 
    target_,
    reduction_mode
  );
  
  input_.unsafeReleaseTensorImpl();
  target_.unsafeReleaseTensorImpl();
  
  return fromTorchTensor(output_);
}

// Smooth L1 loss 
lean_object* lean_torch_smooth_l1_loss(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg target, lean_obj_arg reduction, double beta) {
  auto input_ = toTorchTensor(input);
  auto target_ = toTorchTensor(target);
  
  auto reduction_str = lean_string_cstr(reduction);
  int64_t reduction_mode = 1; // mean
  if (strcmp(reduction_str, "none") == 0) {
    reduction_mode = 0;
  } else if (strcmp(reduction_str, "sum") == 0) {
    reduction_mode = 2;
  }
  
  auto output_ = torch::smooth_l1_loss(
    input_, 
    target_,
    reduction_mode,
    beta
  );
  
  input_.unsafeReleaseTensorImpl();
  target_.unsafeReleaseTensorImpl();
  
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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);
  auto stride_ = getShape(stride);
  auto padding_ = getShape(padding);
  auto dilation_ = getShape(dilation);
  
  auto options = torch::nn::functional::Conv3dFuncOptions()
    .stride(stride_)
    .padding(padding_) 
    .dilation(dilation_);
  auto output_ = torch::nn::functional::conv3d(input_, weight_, options);
  
  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
  
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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    padding_idx_val = lean_scalar_to_int64(padding_idx);
  }

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();

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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);

  int64_t padding_idx_val = -1;
  if (!lean_is_scalar(padding_idx)) {
    padding_idx_val = lean_scalar_to_int64(padding_idx);
  }

  auto output_ = torch::embedding(
    weight_,
    input_,
    padding_idx_val,
    scale_grad_by_freq,
    sparse
  );

  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();

  return fromTorchTensor(output_);
}

// Matrix multiplication functions (critical for transformers)
lean_object* lean_torch_matmul(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg other) {
  auto input_ = toTorchTensor(input);
  auto other_ = toTorchTensor(other);
  auto result_ = torch::matmul(input_, other_);
  input_.unsafeReleaseTensorImpl();
  other_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_bmm(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = toTorchTensor(input);
  auto mat2_ = toTorchTensor(mat2);
  auto result_ = torch::bmm(input_, mat2_);
  input_.unsafeReleaseTensorImpl();
  mat2_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mm(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mat2) {
  auto input_ = toTorchTensor(input);
  auto mat2_ = toTorchTensor(mat2);
  auto result_ = torch::mm(input_, mat2_);
  input_.unsafeReleaseTensorImpl();
  mat2_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Tensor operations (essential for attention mechanisms)
lean_object* lean_torch_transpose(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim0, int dim1) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::transpose(input_, dim0, dim1);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_cat(lean_obj_arg /*s*/, lean_obj_arg tensors, int dim) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(toTorchTensor(tensor_obj));
  }
  auto result_ = torch::cat(tensor_list, dim);
  
  // Release all input tensors
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    auto temp = toTorchTensor(tensor_obj);
    temp.unsafeReleaseTensorImpl();
  }
  
  return fromTorchTensor(result_);
}

lean_object* lean_torch_stack(lean_obj_arg /*s*/, lean_obj_arg tensors, int dim) {
  std::vector<torch::Tensor> tensor_list;
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    tensor_list.push_back(toTorchTensor(tensor_obj));
  }
  auto result_ = torch::stack(tensor_list, dim);
  
  // Release all input tensors
  for (size_t i = 0; i < lean_array_size(tensors); i++) {
    auto tensor_obj = lean_array_get_core(tensors, i);
    auto temp = toTorchTensor(tensor_obj);
    temp.unsafeReleaseTensorImpl();
  }
  
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
  auto query_ = toTorchTensor(query);
  auto key_ = toTorchTensor(key);
  auto value_ = toTorchTensor(value);
  
  torch::Tensor attn_mask_;
  bool has_mask = !lean_is_scalar(attn_mask);
  if (has_mask) {
    attn_mask_ = toTorchTensor(attn_mask);
  }
  
  auto result_ = torch::scaled_dot_product_attention(
    query_, 
    key_, 
    value_,
    has_mask ? c10::optional<torch::Tensor>(attn_mask_) : c10::nullopt,
    dropout_p,
    is_causal
  );
  
  query_.unsafeReleaseTensorImpl();
  key_.unsafeReleaseTensorImpl();
  value_.unsafeReleaseTensorImpl();
  if (has_mask) attn_mask_.unsafeReleaseTensorImpl();
  
  return fromTorchTensor(result_);
}

// Softmax with dimension parameter (needed for attention)
lean_object* lean_torch_softmax_dim(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::softmax(input_, dim);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Extended activation functions with parameters
lean_object* lean_torch_hardtanh_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double min_val, double max_val) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::hardtanh(input_, min_val, max_val);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_celu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double alpha) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::celu(input_, alpha);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rrelu_params(lean_obj_arg /*s*/, b_lean_obj_arg input, double lower, double upper, uint8_t training) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::rrelu(input_, lower, upper, training);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Additional tensor operations
lean_object* lean_torch_split(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int split_size, int dim) {
  auto tensor_ = toTorchTensor(tensor);
  auto result_tensors = torch::split(tensor_, split_size, dim);
  tensor_.unsafeReleaseTensorImpl();
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_chunk(lean_obj_arg /*s*/, b_lean_obj_arg tensor, int chunks, int dim) {
  auto tensor_ = toTorchTensor(tensor);
  auto result_tensors = torch::chunk(tensor_, chunks, dim);
  tensor_.unsafeReleaseTensorImpl();
  
  // Create Lean array of tensors
  lean_object* result_array = lean_alloc_array(result_tensors.size(), result_tensors.size());
  for (size_t i = 0; i < result_tensors.size(); i++) {
    lean_array_set_core(result_array, i, fromTorchTensor(result_tensors[i]));
  }
  return result_array;
}

lean_object* lean_torch_unsqueeze(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::unsqueeze(input_, dim);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_squeeze(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::squeeze(input_, dim);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Mathematical operations
lean_object* lean_torch_sqrt(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::sqrt(input_);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_rsqrt(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::rsqrt(input_);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg other) {
  auto input_ = toTorchTensor(input);
  auto other_ = toTorchTensor(other);
  auto result_ = torch::div(input_, other_);
  input_.unsafeReleaseTensorImpl();
  other_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow(lean_obj_arg /*s*/, b_lean_obj_arg input, double exponent) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::pow(input_, exponent);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Reduction operations
lean_object* lean_torch_sum(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  auto input_ = toTorchTensor(input);
  torch::Tensor result_;
  
  if (lean_is_scalar(dim)) {
    // Sum all elements
    result_ = torch::sum(input_);
  } else {
    // Sum along specific dimensions
    auto dims = getShape(dim);
    result_ = torch::sum(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_mean(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg dim, uint8_t keepdim) {
  auto input_ = toTorchTensor(input);
  torch::Tensor result_;
  
  if (lean_is_scalar(dim)) {
    // Mean of all elements
    result_ = torch::mean(input_);
  } else {
    // Mean along specific dimensions
    auto dims = getShape(dim);
    result_ = torch::mean(input_, c10::IntArrayRef(dims.data(), dims.size()), keepdim);
  }
  
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_max(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = toTorchTensor(input);
  auto result_tuple = torch::max(input_, dim, keepdim);
  input_.unsafeReleaseTensorImpl();
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

lean_object* lean_torch_min(lean_obj_arg /*s*/, b_lean_obj_arg input, int dim, uint8_t keepdim) {
  auto input_ = toTorchTensor(input);
  auto result_tuple = torch::min(input_, dim, keepdim);
  input_.unsafeReleaseTensorImpl();
  
  // Return just the values (first element of tuple)
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Masking and conditional operations
lean_object* lean_torch_masked_fill(lean_obj_arg /*s*/, b_lean_obj_arg input, b_lean_obj_arg mask, double value) {
  auto input_ = toTorchTensor(input);
  auto mask_ = toTorchTensor(mask);
  auto result_ = input_.masked_fill(mask_, value);
  input_.unsafeReleaseTensorImpl();
  mask_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_where(lean_obj_arg /*s*/, b_lean_obj_arg condition, b_lean_obj_arg x, b_lean_obj_arg y) {
  auto condition_ = toTorchTensor(condition);
  auto x_ = toTorchTensor(x);
  auto y_ = toTorchTensor(y);
  auto result_ = torch::where(condition_, x_, y_);
  condition_.unsafeReleaseTensorImpl();
  x_.unsafeReleaseTensorImpl();
  y_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Broadcasting and expanding
lean_object* lean_torch_expand(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg size) {
  auto input_ = toTorchTensor(input);
  auto size_ = getShape(size);
  auto result_ = input_.expand(c10::IntArrayRef(size_.data(), size_.size()));
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_repeat(lean_obj_arg /*s*/, b_lean_obj_arg input, lean_obj_arg repeats) {
  auto input_ = toTorchTensor(input);
  auto repeats_ = getShape(repeats);
  auto result_ = input_.repeat(c10::IntArrayRef(repeats_.data(), repeats_.size()));
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Scalar operations (tensor-scalar arithmetic)
lean_object* lean_torch_mul_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_ * scalar;
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_div_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_ / scalar;
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_add_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_ + scalar;
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_sub_scalar(lean_obj_arg /*s*/, b_lean_obj_arg input, double scalar) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_ - scalar;
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

lean_object* lean_torch_pow_tensor(lean_obj_arg /*s*/, b_lean_obj_arg input, double exponent) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::pow(input_, exponent);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Lower triangular (for causal masking)
lean_object* lean_torch_tril(lean_obj_arg /*s*/, b_lean_obj_arg input, int64_t diagonal) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::tril(input_, diagonal);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Top-k values (for sampling)
lean_object* lean_torch_topk_values(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t k, int64_t dim) {
  auto input_ = toTorchTensor(input);
  auto result_tuple = torch::topk(input_, k, dim);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(std::get<0>(result_tuple));
}

// Multinomial sampling
lean_object* lean_torch_multinomial(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t num_samples, uint8_t replacement, lean_object* w) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::multinomial(input_, num_samples, replacement);
  input_.unsafeReleaseTensorImpl();
  return lean_io_result_mk_ok(fromTorchTensor(result_));
}

// Gradient clipping (in-place, returns norm)
lean_object* lean_torch_clip_grad_norm_(lean_obj_arg /*s*/, b_lean_obj_arg param, double max_norm, lean_object* w) {
  auto param_ = toTorchTensor(param);
  double total_norm = 0.0;
  if (param_.grad().defined()) {
    auto grad = param_.grad();
    auto norm = grad.norm();
    total_norm = norm.item<double>();
    if (total_norm > max_norm) {
      grad.mul_(max_norm / total_norm);
    }
  }
  param_.unsafeReleaseTensorImpl();
  return lean_io_result_mk_ok(lean_box_float(total_norm));
}

// Item extraction (get scalar value from 0-d tensor)
double lean_torch_item(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = toTorchTensor(input);
  double result = input_.item<double>();
  input_.unsafeReleaseTensorImpl();
  return result;
}

// Item extraction as int64
int64_t lean_torch_item_int(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = toTorchTensor(input);
  int64_t result = input_.item<int64_t>();
  input_.unsafeReleaseTensorImpl();
  return result;
}

// Argmax
// Int64 is a single-field structure wrapping UInt64, so it's passed unboxed as uint64_t
lean_object* lean_torch_argmax(lean_obj_arg /*s*/, b_lean_obj_arg input, uint64_t dim) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::argmax(input_, static_cast<int64_t>(dim));
  input_.unsafeReleaseTensorImpl();
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
  auto output_ = toTorchTensor(output);
  auto grad_output_ = toTorchTensor(grad_output);
  output_.backward(grad_output_);
  output_.unsafeReleaseTensorImpl();
  grad_output_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  auto indices_ = toTorchTensor(indices);
  auto result_ = torch::index_select(input_, dim, indices_);
  input_.unsafeReleaseTensorImpl();
  indices_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Slice a 1D tensor: data[start:end]
lean_object* lean_torch_slice_1d(
  lean_obj_arg /*n*/,
  b_lean_obj_arg input,
  int64_t start,
  int64_t end
) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_.slice(0, start, end);
  input_.unsafeReleaseTensorImpl();
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
    tensor_list.push_back(toTorchTensor(tensor_obj));
  }
  auto result_ = torch::stack(tensor_list, dim);

  for (auto& t : tensor_list) {
    t.unsafeReleaseTensorImpl();
  }

  return fromTorchTensor(result_);
}

// Convert tensor to Long (int64) dtype
lean_object* lean_torch_to_long(lean_obj_arg /*s*/, b_lean_obj_arg input) {
  auto input_ = toTorchTensor(input);
  auto result_ = input_.to(torch::kLong);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Backward pass that only requires the loss (assumes gradient of 1.0)
lean_object* lean_torch_backward_loss(lean_obj_arg /*s*/, b_lean_obj_arg loss, lean_object* w) {
  auto loss_ = toTorchTensor(loss);
  loss_.backward();
  loss_.unsafeReleaseTensorImpl();
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
  auto a_ = toTorchTensor(a);
  auto b_ = toTorchTensor(b);
  auto result_ = torch::matmul(a_, b_);
  a_.unsafeReleaseTensorImpl();
  b_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Transpose 2D tensor: [m, n] -> [n, m]
lean_object* lean_torch_transpose_2d(
  lean_obj_arg /*m*/,
  lean_obj_arg /*n*/,
  b_lean_obj_arg input
) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::transpose(input_, 0, 1);
  input_.unsafeReleaseTensorImpl();
  return fromTorchTensor(result_);
}

// Transpose 3D tensor dims 1 and 2: [a, b, c] -> [a, c, b]
lean_object* lean_torch_transpose3d_12(
  lean_obj_arg /*a*/,
  lean_obj_arg /*b*/,
  lean_obj_arg /*c*/,
  b_lean_obj_arg input
) {
  auto input_ = toTorchTensor(input);
  auto result_ = torch::transpose(input_, 1, 2);
  input_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  // [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  input_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  // [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim]
  auto result_ = input_.permute({0, 2, 1, 3}).contiguous();
  input_.unsafeReleaseTensorImpl();
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
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);
  auto bias_ = toTorchTensor(bias);

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

  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
  bias_.unsafeReleaseTensorImpl();

  return fromTorchTensor(output_);
}

// Cross entropy for 2D logits: [N, C] + targets [N] -> scalar
lean_object* lean_torch_cross_entropy_2d(
  lean_obj_arg /*n*/,
  lean_obj_arg /*c*/,
  b_lean_obj_arg logits,
  b_lean_obj_arg targets
) {
  auto logits_ = toTorchTensor(logits);
  auto targets_ = toTorchTensor(targets);
  auto result_ = torch::nn::functional::cross_entropy(logits_, targets_);
  logits_.unsafeReleaseTensorImpl();
  targets_.unsafeReleaseTensorImpl();
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
  auto query_ = toTorchTensor(query);
  auto key_ = toTorchTensor(key);
  auto value_ = toTorchTensor(value);

  auto result_ = torch::scaled_dot_product_attention(
    query_,
    key_,
    value_,
    c10::nullopt,  // attn_mask
    dropout_p,
    is_causal
  );

  query_.unsafeReleaseTensorImpl();
  key_.unsafeReleaseTensorImpl();
  value_.unsafeReleaseTensorImpl();

  return fromTorchTensor(result_);
}

}