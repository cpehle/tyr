import Tyr.Basic
import Tyr.Torch 

/-!
# `Tyr.Module.Affine`

Neural-module building block for Affine with type-safe tensor interfaces.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch

structure Affine {n m : UInt64} :=
    (w : T #[m, n])
    (b : T #[n])

instance {n m : UInt64}: differentiable (@Affine n m) := ⟨@Affine n m, fun (a : Affine) => ⟨differentiable.grad a.w, differentiable.grad a.b⟩⟩

def Affine.step {b n m : UInt64} (a : @torch.Affine n m) (input : torch.T #[b, m]) :  torch.T #[b, n] :=
    @torch.affine b n m a.w a.b input

