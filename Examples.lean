/-
  Examples.lean

  Root module for Tyr examples.

  This module imports all example implementations:
  - GPT: Simple GPT-2 style training (Shakespeare, etc.)
  - NanoChat: Advanced chat model with tools (modded-nanogpt port)
  - NanoProof: Theorem-proving transformer
  - Diffusion: Diffusion models
-/

-- Simple GPT
import Examples.GPT.GPT
import Examples.GPT.Train
import Examples.GPT.GPTDataLoader
import Examples.GPT.Pretraining

-- NanoChat (modded GPT + tools + tasks)
import Examples.NanoChat.NanoChat


-- Diffusion
import Examples.Diffusion.Diffusion
import Examples.Diffusion.DiffusionSchedule
import Examples.Diffusion.DiffusionTrain
import Examples.Diffusion.DiffusionCheckpoint

-- Executable entry points excluded (they define global `main`):
-- import Examples.TrainGPT
-- import Examples.TrainDiffusion
-- import Examples.NanoChat.TrainNanoChat
