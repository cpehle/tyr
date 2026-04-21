/- Runtime smoke test for the high-level `tyr::flash_attn` bridge. -/
import Tyr.Torch
import Tyr.GPU.Ops.FlashAttn

namespace Examples.GPU

open torch
open Tyr.GPU.Ops.FlashAttn

private def routeName : DispatchRoute → String
  | .tkKernel => "tkKernel"
  | .portable => "portable"

private def reportCase
    (name : String)
    (route : DispatchRoute)
    (routeOk outOk dqOk dkOk dvOk : Bool)
    (outMae dqMae dkMae dvMae : Float)
    (outMax dqMax dkMax dvMax : Float) : IO Unit :=
  IO.println s!"{name} route={routeName route} route_ok={routeOk} out_ok={outOk} dq_ok={dqOk} dk_ok={dkOk} dv_ok={dvOk} out_mae={outMae} dq_mae={dqMae} dk_mae={dkMae} dv_mae={dvMae} out_max={outMax} dq_max={dqMax} dk_max={dkMax} dv_max={dvMax}"

private def nativeDenseCase : IO Bool := do
  let device := Device.CUDA 0
  let q := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 128, 64] false device)) true
  let k := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 128, 64] false device)) true
  let v := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 128, 64] false device)) true
  let dO := torch.toBFloat16' (← torch.randn #[1, 1, 128, 64] false device)

  let (route, out) := flashAttnWithRoute q k v
  torch.autograd.backward out dO
  let _ ← torch.cuda_synchronize
  let dq := torch.toFloat' (torch.autograd.grad_of q)
  let dk := torch.toFloat' (torch.autograd.grad_of k)
  let dv := torch.toFloat' (torch.autograd.grad_of v)

  let qRef := torch.autograd.set_requires_grad (torch.autograd.detach q) true
  let kRef := torch.autograd.set_requires_grad (torch.autograd.detach k) true
  let vRef := torch.autograd.set_requires_grad (torch.autograd.detach v) true
  let outRef := torch.nn.scaled_dot_product_attention qRef kRef vRef 0.0 false
  torch.autograd.backward outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := torch.toFloat' (torch.autograd.grad_of qRef)
  let dkRef := torch.toFloat' (torch.autograd.grad_of kRef)
  let dvRef := torch.toFloat' (torch.autograd.grad_of vRef)

  let routeOk := route == .tkKernel
  let outOk := torch.allclose outRef out 3e-2 3e-2
  let dqOk := torch.allclose dqRef dq 3e-2 3e-2
  let dkOk := torch.allclose dkRef dk 3e-2 3e-2
  let dvOk := torch.allclose dvRef dv 3e-2 3e-2
  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - outRef)))
  let dqMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dq - dqRef)))
  let dkMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dk - dkRef)))
  let dvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dv - dvRef)))
  let outMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - outRef)))
  let dqMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dq - dqRef)))
  let dkMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dk - dkRef)))
  let dvMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dv - dvRef)))
  reportCase "flash_attn_native_dense" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

private def portableCausalCase : IO Bool := do
  let device := Device.CUDA 0
  let q := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 96, 64] false device)) true
  let k := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 96, 64] false device)) true
  let v := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 1, 96, 64] false device)) true
  let dO := torch.toBFloat16' (← torch.randn #[1, 1, 96, 64] false device)

  let (route, out) := flashAttnWithRoute q k v none 0.0 true none false
  torch.autograd.backward out dO
  let _ ← torch.cuda_synchronize
  let dq := torch.toFloat' (torch.autograd.grad_of q)
  let dk := torch.toFloat' (torch.autograd.grad_of k)
  let dv := torch.toFloat' (torch.autograd.grad_of v)

  let qRef := torch.autograd.set_requires_grad (torch.autograd.detach q) true
  let kRef := torch.autograd.set_requires_grad (torch.autograd.detach k) true
  let vRef := torch.autograd.set_requires_grad (torch.autograd.detach v) true
  let outRef := torch.nn.scaled_dot_product_attention qRef kRef vRef 0.0 true
  torch.autograd.backward outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := torch.toFloat' (torch.autograd.grad_of qRef)
  let dkRef := torch.toFloat' (torch.autograd.grad_of kRef)
  let dvRef := torch.toFloat' (torch.autograd.grad_of vRef)

  let routeOk := route == .portable
  let outOk := torch.allclose outRef out 1e-5 1e-5
  let dqOk := torch.allclose dqRef dq 1e-5 1e-5
  let dkOk := torch.allclose dkRef dk 1e-5 1e-5
  let dvOk := torch.allclose dvRef dv 1e-5 1e-5
  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - outRef)))
  let dqMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dq - dqRef)))
  let dkMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dk - dkRef)))
  let dvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dv - dvRef)))
  let outMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - outRef)))
  let dqMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dq - dqRef)))
  let dkMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dk - dkRef)))
  let dvMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dv - dvRef)))
  reportCase "flash_attn_portable_causal" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

private def portableGqaCase : IO Bool := do
  let device := Device.CUDA 0
  let q := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 4, 96, 64] false device)) true
  let k := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 2, 96, 64] false device)) true
  let v := torch.autograd.set_requires_grad (torch.toBFloat16' (← torch.randn #[1, 2, 96, 64] false device)) true
  let dO := torch.toBFloat16' (← torch.randn #[1, 4, 96, 64] false device)

  let (route, out) := flashAttnWithRoute q k v none 0.0 false none true
  torch.autograd.backward out dO
  let _ ← torch.cuda_synchronize
  let dq := torch.toFloat' (torch.autograd.grad_of q)
  let dk := torch.toFloat' (torch.autograd.grad_of k)
  let dv := torch.toFloat' (torch.autograd.grad_of v)

  let qRef := torch.autograd.set_requires_grad (torch.autograd.detach q) true
  let kRef := torch.autograd.set_requires_grad (torch.autograd.detach k) true
  let vRef := torch.autograd.set_requires_grad (torch.autograd.detach v) true
  let outRef := torch.nn.scaledDotProductAttentionGQAQKV qRef kRef vRef 0.0 false true
  torch.autograd.backward outRef dO
  let _ ← torch.cuda_synchronize
  let dqRef := torch.toFloat' (torch.autograd.grad_of qRef)
  let dkRef := torch.toFloat' (torch.autograd.grad_of kRef)
  let dvRef := torch.toFloat' (torch.autograd.grad_of vRef)

  let routeOk := route == .portable
  let outOk := torch.allclose outRef out 1e-5 1e-5
  let dqOk := torch.allclose dqRef dq 1e-5 1e-5
  let dkOk := torch.allclose dkRef dk 1e-5 1e-5
  let dvOk := torch.allclose dvRef dv 1e-5 1e-5
  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - outRef)))
  let dqMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dq - dqRef)))
  let dkMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dk - dkRef)))
  let dvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dv - dvRef)))
  let outMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - outRef)))
  let dqMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dq - dqRef)))
  let dkMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dk - dkRef)))
  let dvMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dv - dvRef)))
  reportCase "flash_attn_portable_gqa" route routeOk outOk dqOk dkOk dvOk outMae dqMae dkMae dvMae outMax dqMax dkMax dvMax
  pure (routeOk && outOk && dqOk && dkOk && dvOk)

def main (_args : List String) : IO UInt32 := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return 1

  torch.manualSeed 20260421
  let okNative ← nativeDenseCase
  let okPortableCausal ← portableCausalCase
  let okPortableGqa ← portableGqaCase
  pure <| if okNative && okPortableCausal && okPortableGqa then 0 else 1

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
