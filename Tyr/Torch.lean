import Tyr.Basic

namespace torch

-- | Tensor Creation API
-- arange: Returns a tensor with a sequence of integers,
@[extern "lean_torch_arange"] constant arange (start : UInt64) (stop : UInt64) (step : UInt64 := 1) : T #[(stop - start)/step]
-- empty: Returns a tensor with uninitialized values,
-- eye: Returns an identity matrix,
-- full: Returns a tensor filled with a single value,
-- linspace: Returns a tensor with values linearly spaced in some interval,
-- logspace: Returns a tensor with values logarithmically spaced in some interval,
-- ones: Returns a tensor filled with all ones,
@[extern "lean_torch_ones"] constant ones (s : Shape) (requires_grad : Bool := false) : T s
-- rand: Returns a tensor filled with values drawn from a uniform distribution on [0, 1).
@[extern "lean_torch_rand"] constant rand (s : Shape) (requires_grad : Bool := false) : IO (T s)
-- randint: Returns a tensor with integers randomly drawn from an interval,
-- zeros: Returns a tensor filled with all zeros
@[extern "lean_torch_zeros"] constant zeros (s : Shape) (requires_grad : Bool := false) : T s
-- randn: Returns a tensor filled with values drawn from a unit normal distribution,
@[extern "lean_torch_randn"] constant randn (s : Shape) (requires_grad : Bool := false) : IO (T s)


@[extern "lean_torch_requires_grad"] constant T.requires_grad {s : Shape} (t : @& T s) : Bool

instance (shape : Shape) : Inhabited (T shape) := ⟨zeros shape⟩

@[extern "lean_torch_tensor_add"] constant add {s : Shape} (t t' : T s) : T s
@[extern "lean_torch_tensor_sub"] constant sub {s : Shape} (t t' : T s) : T s
@[extern "lean_torch_tensor_mul"] constant mul {s : Shape} (t t' : T s) : T s

instance {shape : Shape} : Add (T shape) where
  add := add
instance {shape : Shape} : Sub (T shape) where
  sub := sub
instance {shape : Shape} : Mul (T shape) where
  mul := mul

@[extern "lean_torch_get"] constant T.getOp {s : Shape} (self : @& T s) (idx : Int) : T (s[1:].toArray)
@[extern "lean_torch_to"] constant T.to {s : Shape} (self : @& T s) (device : Device) : T s


structure Conv2dOption where
  in_channels : UInt64
  out_channels : UInt64
  kernel_size : UInt64 × UInt64
  stride : UInt64 × UInt64 := (1,1)
  padding : UInt64 × UInt64 := (0,0)
  dilation : UInt64 × UInt64 := (1,1)
  groups : UInt64 := 1
  bias : Bool := true
  deriving Repr

def convOption : Conv2dOption := {
  in_channels := 16, 
  out_channels := 33,
  kernel_size := (3,3)
}

def conv2dshape (copt:Conv2dOption) (s:Shape) : Shape :=
  let (n, cIn, hin, win) := (s[0], s[1], s[2], s[3]);
  let (px,py) := copt.padding;
  let (sx,sy) := copt.stride;
  let (dx,dy) := copt.dilation;
  let (kx,ky) := copt.kernel_size;
  let hout := (hin + 2 * px - dx * (kx-1) - 1)/sx + 1;
  let wout := (win + 2 * py - dy * (ky-1) - 1)/sy + 1;
  #[n, cIn, hout, wout]

@[extern "lean_torch_linear"] constant linear {m n b : UInt64} (x : T #[b, m]) (M : T #[n,m]) : T #[b, n]
@[extern "lean_torch_affine"] constant affine {m n b : UInt64} (x : T #[b, m]) (M : T #[n,m]) (bias : T #[n]) : T #[b, n]

def slicedShape (s : Array UInt64) (dim : Nat := 0)  (start : UInt64 := 0) (stop : UInt64  := s[dim]) (step : UInt64 := 1) : Array UInt64 :=
  let s' := s[:dim].toArray;
  let s'' := s[dim+1:].toArray;
  let d := (stop - start) / step;
  s' ++ #[d] ++ s''

@[extern "lean_torch_slice "] constant T.slice {s : Shape} (self : @& T s) (dim : @& Nat := 0)  (start : UInt64 := 0) (stop : UInt64  := s[dim] ) (step : UInt64 := 1) 
  (startPositive : start >= 0 := by simp)
  (dimInBounds : dim < s.size := by simp)
  (stopInBounds : stop <= s[dim] := by simp) : T (slicedShape s dim start stop step)


namespace autograd

@[extern "lean_torch_tensor_grad"] constant grad {sx sy : Shape} (y : (T sy)) (x : (T sx)) (dy : (T sy)) : (T sx)

def pullback {sx sy : Shape} (f : T sx → T sy) (x : T sx) (dy : T sy) : T sx :=
    grad (f x) x dy

@[extern "lean_torch_grad_of"] constant grad_of {s : Shape} (x : T s) : T s
end autograd



class differentiable (α : Type _) :=
    (cotangent_space : Type _)
    (grad : α → cotangent_space)

instance {s : Shape} : differentiable (torch.T s) := ⟨torch.T s, fun (t : (torch.T s)) => torch.autograd.grad_of t⟩

structure Linear {n m : UInt64} :=
    (w : T #[n, m])

namespace Linear
instance  {n m : UInt64} : differentiable (@Linear n m) := ⟨@Linear n m, fun (m : Linear) => ⟨differentiable.grad m.w⟩⟩
end Linear

structure Affine {n m : UInt64} :=
    (w : T #[m, n])
    (b : T #[n])

namespace Affine
instance {n m : UInt64}: differentiable (@Affine n m) := ⟨@Affine n m, fun (a : Affine) => ⟨differentiable.grad a.w, differentiable.grad a.b⟩⟩
end Affine


structure LSTM {n m : UInt64} :=
    (w_i : @Affine n m)  
    (r_i : @Affine n n)
    (w_f : @Affine n m)
    (r_f : @Affine n n)
    (w_o : @Affine n m)
    (r_o : @Affine n n)
    (w_c : @Affine n m)
    (r_c : @Affine n n)

namespace LSTM
instance {n m : UInt64}: differentiable (@LSTM n m) := ⟨@LSTM n m, fun (a : LSTM) => ⟨
  differentiable.grad a.w_i, 
  differentiable.grad a.r_i, 
  differentiable.grad a.w_f, 
  differentiable.grad a.r_f, 
  differentiable.grad a.w_o, 
  differentiable.grad a.r_o, 
  differentiable.grad a.w_c, 
  differentiable.grad a.r_c⟩
⟩
end LSTM


-- def rjvp {a b : Type} [differentiable a] [differentiable b] (f : a → b) (x : a) : b × (differentiable.cotangent_space b → differentiable.cotangent_space a) :=
--     let y := f x;
--     let fT := λ (dy : differentiable.cotangent_space b) => torch.backward y dy
--     (y, fT)


namespace nn





-- torch::nn::functional::_interp_output_size
-- torch::nn::functional::_narrow_with_range
-- torch::nn::functional::_pad_circular
-- torch::nn::functional::_smooth_l1_loss
-- torch::nn::functional::_unpool_output_size
-- torch::nn::functional::adaptive_avg_pool1d
-- torch::nn::functional::adaptive_avg_pool2d
-- torch::nn::functional::adaptive_avg_pool3d
-- torch::nn::functional::adaptive_max_pool1d
-- torch::nn::functional::adaptive_max_pool2d
-- torch::nn::functional::adaptive_max_pool2d_with_indices
-- torch::nn::functional::adaptive_max_pool3d
-- torch::nn::functional::adaptive_max_pool3d_with_indices
-- torch::nn::functional::affine_grid
-- torch::nn::functional::alpha_dropout
-- torch::nn::functional::avg_pool1d
-- torch::nn::functional::avg_pool2d
-- torch::nn::functional::avg_pool3d
-- torch::nn::functional::batch_norm
-- torch::nn::functional::bilinear
-- torch::nn::functional::binary_cross_entropy
-- torch::nn::functional::binary_cross_entropy_with_logits
-- torch::nn::functional::celu
-- torch::nn::functional::conv1d
-- torch::nn::functional::conv2d
@[extern "lean_torch_conv2d"] constant conv2d {b ic ih iw oc kh kw : UInt64} (input : T #[b,ic,ih,iw]) (weight : T #[oc, ic, kh, kw]) : T #[b, oc, (ih - (kh - 1) -1) + 1, (iw - (kw - 1) - 1) + 1]
-- torch::nn::functional::conv3d
-- torch::nn::functional::conv_transpose1d
-- torch::nn::functional::conv_transpose2d
-- torch::nn::functional::conv_transpose3d
-- torch::nn::functional::cosine_embedding_loss
-- torch::nn::functional::cosine_similarity
-- torch::nn::functional::cross_entropy
@[extern "lean_torch_tensor_cross_entropy"] constant cross_entropy {s : Shape} (t t' : T s) : T s
-- torch::nn::functional::ctc_loss
-- torch::nn::functional::dropout
-- torch::nn::functional::dropout2d
-- torch::nn::functional::dropout3d
-- torch::nn::functional::elu
@[extern "lean_torch_tensor_elu"] constant elu {s : Shape} (t : T s) : T s
-- torch::nn::functional::embedding
-- torch::nn::functional::embedding_bag
-- torch::nn::functional::feature_alpha_dropout
-- torch::nn::functional::fold
-- torch::nn::functional::fractional_max_pool2d
-- torch::nn::functional::fractional_max_pool2d_with_indices
-- torch::nn::functional::fractional_max_pool3d
-- torch::nn::functional::fractional_max_pool3d_with_indices
-- torch::nn::functional::gelu
@[extern "lean_torch_tensor_gelu"] constant gelu {s : Shape} (t : T s) : T s
-- torch::nn::functional::glu
-- torch::nn::functional::grid_sample
-- torch::nn::functional::group_norm
-- torch::nn::functional::gumbel_softmax
-- torch::nn::functional::hardshrink
-- torch::nn::functional::hardtanh
-- torch::nn::functional::hinge_embedding_loss
-- torch::nn::functional::huber_loss
-- torch::nn::functional::instance_norm
-- torch::nn::functional::interpolate
-- torch::nn::functional::kl_div
-- torch::nn::functional::l1_loss
-- torch::nn::functional::layer_norm
-- torch::nn::functional::leaky_relu
-- torch::nn::functional::linear
-- torch::nn::functional::local_response_norm
-- torch::nn::functional::log_softmax
-- torch::nn::functional::logsigmoid
-- torch::nn::functional::lp_pool1d
-- torch::nn::functional::lp_pool2d
-- torch::nn::functional::margin_ranking_loss
-- torch::nn::functional::max_pool1d
-- torch::nn::functional::max_pool1d_with_indices
-- torch::nn::functional::max_pool2d
-- torch::nn::functional::max_pool2d_with_indices
-- torch::nn::functional::max_pool3d
-- torch::nn::functional::max_pool3d_with_indices
-- torch::nn::functional::max_unpool1d
-- torch::nn::functional::max_unpool2d
-- torch::nn::functional::max_unpool3d
-- torch::nn::functional::mish
-- torch::nn::functional::mse_loss
-- torch::nn::functional::multi_head_attention_forward
-- torch::nn::functional::multi_margin_loss
-- torch::nn::functional::multilabel_margin_loss
-- torch::nn::functional::multilabel_soft_margin_loss
-- torch::nn::functional::nll_loss
-- torch::nn::functional::normalize
-- torch::nn::functional::one_hot
-- torch::nn::functional::pad
-- torch::nn::functional::pairwise_distance
-- torch::nn::functional::pdist
-- torch::nn::functional::pixel_shuffle
-- torch::nn::functional::pixel_unshuffle
-- torch::nn::functional::poisson_nll_loss
-- torch::nn::functional::prelu
-- torch::nn::functional::relu
@[extern "lean_torch_tensor_relu"] constant relu {s : Shape} (t : T s) : T s
-- torch::nn::functional::relu6
-- torch::nn::functional::rrelu
-- torch::nn::functional::selu
-- torch::nn::functional::silu
-- torch::nn::functional::smooth_l1_loss
-- torch::nn::functional::soft_margin_loss
-- torch::nn::functional::softmax
@[extern "lean_torch_tensor_softmax"] constant softmax {s : Shape} (t : T s) : T s
-- torch::nn::functional::softmin
-- torch::nn::functional::softplus
-- torch::nn::functional::softshrink
-- torch::nn::functional::softsign
-- torch::nn::functional::tanhshrink
-- torch::nn::functional::threshold
-- torch::nn::functional::triplet_margin_loss
-- torch::nn::functional::triplet_margin_with_distance_loss
-- torch::nn::functional::unfold


end nn
end torch

