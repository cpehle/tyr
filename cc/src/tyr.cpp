#include <iostream>
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


bool lean_torch_requires_grad(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto x_ = toTorchTensor(x);
    auto res_ = x_.requires_grad();
    x_.unsafeReleaseTensorImpl();
    return res_;
}

//

lean_object* lean_torch_get(lean_obj_arg /*s*/, b_lean_obj_arg self, int idx) {
  auto self_ = toTorchTensor(self);
  auto res = self_.index({idx});
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

lean_object* lean_torch_grad_of(lean_obj_arg /* shape */, b_lean_obj_arg x) {
    auto out_ = toTorchTensor(x);
    auto grad_out_ = out_.grad();
    out_.unsafeReleaseTensorImpl();
    return fromTorchTensor(grad_out_);
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


extern "C" lean_object *lean_torch_to_string(lean_object /* s */, b_lean_obj_arg t) {
  auto tensor = toTorchTensor(t);
  std::ostringstream stream;
  stream << tensor;

  tensor.unsafeReleaseTensorImpl();
  return lean_mk_string(stream.str().c_str());
}

extern "C" lean_object* lean_torch_tensor_print(lean_object /* s */, b_lean_obj_arg t) {
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

lean_object* lean_torch_conv2d(
  lean_obj_arg /*b*/,
  lean_obj_arg /*ic*/,
  lean_obj_arg /*ih*/,
  lean_obj_arg /*iw*/,
  lean_obj_arg /*oc */,
  lean_obj_arg /*kh */,
  lean_obj_arg /*kw*/,
  b_lean_obj_arg input,
  b_lean_obj_arg weight
) {
  auto input_ = toTorchTensor(input);
  auto weight_ = toTorchTensor(weight);
  auto output_ = torch::conv2d(input_, weight_);
  input_.unsafeReleaseTensorImpl();
  weight_.unsafeReleaseTensorImpl();
  return fromTorchTensor(output_);
}

}