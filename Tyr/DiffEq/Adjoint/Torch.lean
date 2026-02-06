import Tyr.DiffEq.Adjoint.Core
import Tyr.Module.Derive

namespace torch
namespace DiffEq

/-! ## Torch Adjoint Backend -/

open _root_.torch

private def numel (s : Shape) : Float :=
  s.foldl (fun acc d => acc * d.toFloat) 1.0

private def sumAllCompat {s : Shape} (t : T s) : T #[] :=
  let mean := nn.meanAll t
  let scale := numel (T.shape t)
  mul_scalar mean scale

private def dotStruct [TensorStruct Y] (x y : Y) : T #[] :=
  let prod := TensorStruct.zipWith (fun a b => mul a b) x y
  TensorStruct.fold (fun t acc => add acc (sumAllCompat t)) (zeros #[]) prod

private def gradStruct [TensorStruct α] (loss : T #[]) (x : α) : α :=
  let one := ones #[]
  TensorStruct.map (fun t => autograd.grad loss t one) x

private def vjp
    [TensorStruct Y] [TensorStruct Args]
    (vf : Time → Y → Args → Y)
    (t : Time) (y : Y) (args : Args) (adjY : Y) : (Y × Y × Args) :=
  let yReq := TensorStruct.makeLeafParams y
  let argsReq := TensorStruct.makeLeafParams args
  let fReq := vf t yReq argsReq
  let adjDet := TensorStruct.detach adjY
  let loss := dotStruct fReq adjDet
  let gradY := gradStruct loss yReq
  let gradArgs := gradStruct loss argsReq
  let fDet := TensorStruct.detach fReq
  (fDet, gradY, gradArgs)

instance [TensorStruct Y] [TensorStruct Args] : AdjointBackend Y Args where
  vjp := vjp

private def vjpFn
    [TensorStruct Y] [TensorStruct Args]
    (f : Y → Args → Y)
    (y : Y) (args : Args) (adjY : Y) : (Y × Args) :=
  let yReq := TensorStruct.makeLeafParams y
  let argsReq := TensorStruct.makeLeafParams args
  let out := f yReq argsReq
  let adjDet := TensorStruct.detach adjY
  let loss := dotStruct out adjDet
  let gradY := gradStruct loss yReq
  let gradArgs := gradStruct loss argsReq
  (gradY, gradArgs)

instance [TensorStruct Y] [TensorStruct Args] : AdjointFnBackend Y Args where
  vjpFn := vjpFn

end DiffEq
end torch
