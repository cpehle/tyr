namespace torch
def Shape := Array UInt64

inductive DType where
| UInt8
| Int8
| Int16
| Int32
| Int64
| Float16
| Float32
| Float64

inductive Device where
| CUDA : Nat → Device
| CPU

constant TSpec : PointedType
def T (s : Shape) : Type :=  TSpec.type
instance (s : Shape) : Inhabited (T s) := {
  default := TSpec.val
}

@[extern "lean_torch_to_string"] constant T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] constant T.print {s : Shape} (t : @& T s) : IO Unit

-- arange: Returns a tensor with a sequence of integers,
-- empty: Returns a tensor with uninitialized values,
-- eye: Returns an identity matrix,
-- full: Returns a tensor filled with a single value,
-- linspace: Returns a tensor with values linearly spaced in some interval,
-- logspace: Returns a tensor with values logarithmically spaced in some interval,
-- ones: Returns a tensor filled with all ones,
@[extern "lean_torch_ones"] constant ones (s : Shape) : T s
-- rand: Returns a tensor filled with values drawn from a uniform distribution on [0, 1).
@[extern "lean_torch_rand"] constant rand (s : Shape) : IO (T s)
-- randint: Returns a tensor with integers randomly drawn from an interval,
-- zeros: Returns a tensor filled with all zeros
@[extern "lean_torch_zeros"] constant zeros (s : Shape) : T s
-- randn: Returns a tensor filled with values drawn from a unit normal distribution,
@[extern "lean_torch_randn"] constant randn (s : Shape) : IO (T s)

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


-- def main : IO Unit := do
--  let x <- torch.randn #[5,5]
--   let y <- torch.rand #[5,5]
-- let M := torch.zeros #[5,5]
-- let b := torch.ones #[5]
-- let z := (x + M)
-- let a := (x - x)
--  M[1].print
-- (torch.ones #[5,5]).print
-- z.print
-- IO.println "------------------------"
-- x.print
-- (torch.linear x M).print
-- (torch.affine x M b).print
-- ((torch.zeros #[4,5,10]).slice (dim := 1) (start := 2)).print
-- IO.println "------------------------"
-- (torch.nn.softmax (torch.ones #[2,3])).print
-- pure ()
