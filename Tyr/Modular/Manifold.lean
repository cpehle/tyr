import Tyr.Modular.Norm
import Tyr.Modular.Budget
import Tyr.Manifolds.Optimizer
import Tyr.Optim.DualOptimizer

namespace Tyr.Modular

open torch
open Tyr.AD

/-- Matrix-manifold carrier interface for manifold-native linear layers. -/
class MatrixManifoldCarrier (M : UInt64 → UInt64 → Type) where
  /-- Project an ambient matrix to the manifold. -/
  project : (m n : UInt64) → T #[m, n] → M m n
  /-- Convert manifold value to its ambient matrix representation. -/
  toMatrix : {m n : UInt64} → M m n → T #[m, n]
  /-- Take one geometry-aware step and retract on-manifold. -/
  dualMapStep : {m n : UInt64} → M m n → T #[m, n] → Float → M m n

instance : MatrixManifoldCarrier Stiefel where
  project := Stiefel.project
  toMatrix := fun x => x.matrix
  dualMapStep := by
    intro m n x g lr
    let tg := StiefelTangent.project x g
    exact DualMapGeometry.dualMapStep x tg lr

instance : MatrixManifoldCarrier Grassmann where
  project := Grassmann.project
  toMatrix := fun x => x.matrix
  dualMapStep := by
    intro m n x g lr
    let tg := GrassmannTangent.project x g
    exact DualMapGeometry.dualMapStep x tg lr

/--
Linear layer whose weight is stored on a matrix manifold.

The weight is natively represented by `M out_dim in_dim`, so retractions are
part of the layer-level update path rather than an external afterthought.
-/
structure ManifoldLinear (M : UInt64 → UInt64 → Type) [MatrixManifoldCarrier M]
    (in_dim out_dim : UInt64) where
  weight : M out_dim in_dim
  bias : Option (T #[out_dim]) := none

namespace ManifoldLinear

private def makeLeafWeight [MatrixManifoldCarrier M] {in_dim out_dim : UInt64}
    (w : M out_dim in_dim) : M out_dim in_dim :=
  MatrixManifoldCarrier.project out_dim in_dim
    (autograd.set_requires_grad (autograd.detach (MatrixManifoldCarrier.toMatrix w)) true)

/-- Initialize manifold-linear weights by projecting a random matrix. -/
def init [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    (withBias : Bool := true) : IO (ManifoldLinear M in_dim out_dim) := do
  let w0 ← randn #[out_dim, in_dim] false
  let w := MatrixManifoldCarrier.project out_dim in_dim w0
  let weight := makeLeafWeight w
  let bias ← if withBias then
      let b := autograd.set_requires_grad (zeros #[out_dim]) true
      pure (some b)
    else
      pure none
  pure { weight, bias }

/-- Forward pass for `[batch, in_dim] -> [batch, out_dim]`. -/
def forward2d [MatrixManifoldCarrier M] {in_dim out_dim batch : UInt64}
    (lin : ManifoldLinear M in_dim out_dim)
    (x : T #[batch, in_dim]) : T #[batch, out_dim] :=
  let y := linear x (MatrixManifoldCarrier.toMatrix lin.weight)
  match lin.bias with
  | some b => add y (nn.expand b #[batch, out_dim])
  | none => y

/-- Apply a manifold-native step to weights and an Euclidean step to bias. -/
def applyDualMapUpdate [MatrixManifoldCarrier M] {in_dim out_dim : UInt64}
    (lin : ManifoldLinear M in_dim out_dim)
    (weightGrad : T #[out_dim, in_dim])
    (biasGrad? : Option (T #[out_dim]))
    (lr : Float) : ManifoldLinear M in_dim out_dim :=
  let weight' := MatrixManifoldCarrier.dualMapStep lin.weight weightGrad lr
  let weight' := makeLeafWeight weight'
  let bias' :=
    match lin.bias, biasGrad? with
    | some b, some g =>
      some (autograd.set_requires_grad (autograd.detach (b - mul_scalar g lr)) true)
    | some b, none => some b
    | none, _ => none
  { lin with weight := weight', bias := bias' }

end ManifoldLinear

instance [MatrixManifoldCarrier M] (in_dim out_dim : UInt64) :
    TensorStruct (ManifoldLinear M in_dim out_dim) where
  map f lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let weight := MatrixManifoldCarrier.project out_dim in_dim (f w)
    let bias := lin.bias.map f
    { weight, bias }
  mapM f lin := do
    let w' ← f (MatrixManifoldCarrier.toMatrix lin.weight)
    let weight := MatrixManifoldCarrier.project out_dim in_dim w'
    let bias ← match lin.bias with
      | some b => some <$> f b
      | none => pure none
    pure { weight, bias }
  zipWith f a b :=
    let w := f (MatrixManifoldCarrier.toMatrix a.weight) (MatrixManifoldCarrier.toMatrix b.weight)
    let weight := MatrixManifoldCarrier.project out_dim in_dim w
    let bias := match a.bias, b.bias with
      | some x, some y => some (f x y)
      | _, _ => none
    { weight, bias }
  fold f init lin :=
    let acc := f (MatrixManifoldCarrier.toMatrix lin.weight) init
    match lin.bias with
    | some b => f b acc
    | none => acc

instance [MatrixManifoldCarrier M] (in_dim out_dim : UInt64) :
    NormedModule (ManifoldLinear M in_dim out_dim) where
  norm lin :=
    let weightNorm := linalg.spectralNorm (MatrixManifoldCarrier.toMatrix lin.weight)
    match lin.bias with
    | some b => floatMax weightNorm (linalg.l2Norm b)
    | none => weightNorm
  dualNorm lin :=
    let weightDual := linalg.nuclearNorm (MatrixManifoldCarrier.toMatrix lin.weight)
    match lin.bias with
    | some b => weightDual + linalg.l2Norm b
    | none => weightDual
  nu lin := linalg.spectralNorm (MatrixManifoldCarrier.toMatrix lin.weight)
  mu _ := 1.0
  normalize lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let σ := linalg.spectralNorm w
    let w' := if σ == 0.0 then w else mul_scalar w (1.0 / σ)
    let weight := MatrixManifoldCarrier.project out_dim in_dim w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := linalg.l2Norm b
        if bNorm == 0.0 then some b else some (mul_scalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }
  normalizeDual lin :=
    let w := MatrixManifoldCarrier.toMatrix lin.weight
    let fnorm := linalg.frobeniusNorm w
    let w' := if fnorm == 0.0 then w else mul_scalar w (1.0 / fnorm)
    let weight := MatrixManifoldCarrier.project out_dim in_dim w'
    let bias :=
      match lin.bias with
      | some b =>
        let bNorm := linalg.l2Norm b
        if bNorm == 0.0 then some b else some (mul_scalar b (1.0 / bNorm))
      | none => none
    { lin with weight, bias }

/-- Convenience alias: manifold-linear with Stiefel-constrained weights. -/
abbrev StiefelLinear (in_dim out_dim : UInt64) := ManifoldLinear Stiefel in_dim out_dim

/-- Convenience alias: manifold-linear with Grassmann-constrained weights. -/
abbrev GrassmannLinear (in_dim out_dim : UInt64) := ManifoldLinear Grassmann in_dim out_dim

private def meanOrOne (xs : Array Float) : Float :=
  if xs.isEmpty then 1.0 else xs.foldl (fun acc x => acc + x) 0.0 / xs.size.toFloat

/--
Compile a matrix-group budget multiplier from modular sensitivities of
manifold-native modules and inject it into `DualOptimizer.Config`.
-/
def applyMatrixBudgetFromModules [NormedModule M]
    (optCfg : torch.Optim.DualOptimizer.Config)
    (budgetCfg : BudgetConfig := {})
    (modules : Array M)
    : torch.Optim.DualOptimizer.Config :=
  let scales := sequentialDownstreamScales budgetCfg modules
  let matrixMul := meanOrOne scales
  { optCfg with
    useModularBudget := true
    budget := { optCfg.budget with matrix := matrixMul } }

end Tyr.Modular
