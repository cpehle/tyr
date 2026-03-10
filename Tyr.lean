-- Core infrastructure
import Tyr.Basic
import Tyr.Torch
import Tyr.SafeTensors
import Tyr.TensorStruct
import Tyr.Mctx

-- AutoGrad (JAX-style AD on IR)
import Tyr.AD

-- Neural network modules
import Tyr.Module

-- Optimizers
import Tyr.Optim
import Tyr.Optim.ManifoldMuon
import Tyr.Optim.RiemannianSGD
import Tyr.Optim.RiemannianTreeSGD
import Tyr.Optim.Scheduler
import Tyr.Optim.Schedule
import Tyr.Optim.DualOptimizer
import Tyr.Optim.DistAdam

-- Modular and manifold optimization primitives
import Tyr.Modular
import Tyr.Manifolds

-- Distributed training
import Tyr.Distributed
import Tyr.Sharding

-- Data loading and tokenization
import Tyr.Data
import Tyr.DataLoader
import Tyr.Tokenizer

-- Checkpointing
import Tyr.Checkpoint

-- Pipeline orchestration
import Tyr.Pipeline

-- Training loops
import Tyr.Train.RunLedger
import Tyr.Train.ChatSFT
import Tyr.RL.GRPO

-- Metrics
import Tyr.Metrics

-- GPU kernel abstraction
import Tyr.GPU
import Tyr.GPU.AD
import Tyr.GPU.AutoGrad

/-!
# Tyr

`Tyr` is a dependently typed deep learning library for Lean 4.
Its central goal is to make tensor programs safer by tracking key invariants
(especially shapes) in types while still exposing practical training and inference tooling.

## Major Components

This root module re-exports the core, general-purpose surface area:

- `Tyr.Basic`: foundational tensor aliases and shared utilities.
- `Tyr.Torch`: low-level tensor operations and libtorch bindings.
- `Tyr.SafeTensors`: typed SafeTensors schema and loading support.
- `Tyr.TensorStruct`: generic traversal/mapping over tensor-containing structures.
- `Tyr.Mctx`: Monte Carlo tree search infrastructure.
- `Tyr.AutoGrad`: JAX-style automatic differentiation over Tyr IR.
- `Tyr.Module`: neural network module abstractions and layers.
- `Tyr.Optim` (+ scheduler/dist variants): optimizers and learning-rate scheduling.
- `Tyr.Distributed` and `Tyr.Sharding`: distributed training and partitioning utilities.
- `Tyr.Data`, `Tyr.DataLoader`, `Tyr.Tokenizer`: data ingestion, batching, and tokenization.
- `Tyr.Checkpoint`: model/optimizer checkpoint save and restore.
- `Tyr.Pipeline`: training/inference orchestration helpers.
- `Tyr.Metrics`: metric definitions and reporting helpers.
- `Tyr.GPU` (+ `Tyr.GPU.AD`, `Tyr.GPU.AutoGrad`): GPU kernel abstractions and AD support.

## Example

```lean
import Tyr

open torch

def toyForward : T #[2, 4] :=
  let x : T #[2, 8] := zeros #[2, 8]
  let w : T #[4, 8] := zeros #[4, 8]
  linear x w
```

`import Tyr` gives a batteries-included entrypoint; shapes remain encoded in types.

## Scope

This file is intentionally a stable import surface for common Tyr workflows.
Task- and model-specific code lives in dedicated modules (for example `Examples` and `Tyr.Model`).
-/
