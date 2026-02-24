-- Core infrastructure
import Tyr.Basic
import Tyr.Torch
import Tyr.TensorStruct

-- AutoGrad (JAX-style AD on IR)
import Tyr.AutoGrad

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

-- GPU kernel abstraction (ThunderKittens)
import Tyr.GPU
import Tyr.GPU.AD
import Tyr.GPU.AutoGrad

/-!
# `Tyr`

Root module that re-exports Tyr core tensor, module, optimizer, distributed, data, tokenizer, checkpoint, pipeline, and GPU infrastructure.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

