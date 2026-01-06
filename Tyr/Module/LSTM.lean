import Tyr.Basic
import Tyr.Torch
import Tyr.Module.Affine

namespace torch

structure LSTM {n m : UInt64} where
  w_i : @Affine n m
  r_i : @Affine n n
  w_f : @Affine n m
  r_f : @Affine n n
  w_o : @Affine n m
  r_o : @Affine n n
  w_c : @Affine n m
  r_c : @Affine n n

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

def LSTM.step {b n m : UInt64} (tfm : @torch.LSTM n m) (x : torch.T #[b, m]) (h : torch.T #[b, m]) (c : torch.T #[b, m]) : torch.T #[b, n] × torch.T #[b, m] :=
  let ix_t := tfm.w_i.step x;
  let ih_t := tfm.r_i.step h;
  let fx_t := tfm.w_f.step x;
  let fh_t := tfm.r_f.step h;
  let ox_t := tfm.w_o.step x;
  let oh_t := tfm.r_o.step h;
  let c'x_t := tfm.w_c.step x;
  let c'c_t := tfm.r_c.step c;
  let i_t := torch.nn.sigmoid (ix_t + ih_t);
  let f_t := torch.nn.sigmoid (fx_t + fh_t);
  let _o_t := torch.nn.sigmoid (ox_t + fh_t);
  let c'_t := torch.nn.tanh (c'x_t + c'c_t);
  let c_t := f_t * c'_t + i_t * c'_t;
  let h_t := oh_t * torch.nn.tanh c_t;
  (h_t, c_t)
