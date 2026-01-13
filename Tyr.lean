/-
  Tyr.lean

  Root module for Tyr - a deep learning library for Lean 4.

  This module exports the core general-purpose infrastructure:
  - Tensor operations (Torch bindings)
  - Neural network modules
  - Optimizers (AdamW, Muon, DistAdam)
  - Distributed training (DDP, reduce-scatter, sharding)
  - Data loading
  - Tokenization
  - Checkpointing
  - Metrics

  Model-specific implementations are in Examples/:
  - Examples/GPT/ - Simple GPT training
  - Examples/NanoChat/ - NanoChat (modded GPT with tools)
  - Examples/NanoProof/ - Theorem-proving transformer
  - Examples/Diffusion/ - Diffusion models
-/

-- Core infrastructure
import Tyr.Basic
import Tyr.Torch
import Tyr.TensorStruct

-- Neural network modules
import Tyr.Module

-- Optimizers
import Tyr.Optim
import Tyr.Optim.Scheduler
import Tyr.Optim.Schedule
import Tyr.Optim.DualOptimizer
import Tyr.Optim.DistAdam

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

-- Metrics
import Tyr.Metrics
