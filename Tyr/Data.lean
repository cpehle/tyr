/-
  Tyr/Data.lean

  Data loading and pipeline infrastructure.

  Exports:
  - Task: Task-based data loading for instruction tuning
  - Tasks: Concrete task implementations (identity, math, spelling, etc.)
  - Pipeline: Multi-stage training configuration
  - Pretraining: Streaming data loader for pretraining

  Note: LLM-specific tasks (ARC, MMLU, GSM8K, HumanEval, etc.) are in Tyr.Tasks.LLM
-/

import Tyr.Data.Task
import Tyr.Data.TaskClass
import Tyr.Data.Tasks
import Tyr.Data.Pipeline
import Tyr.Data.Pretraining

/-!
# `Tyr.Data`

Entry point for Tyr data components, including tasks, pipelines, pretraining streams, and related utilities.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

