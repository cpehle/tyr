/-
  Examples/NanoProof/NanoProof.lean

  Root module for NanoProof - a theorem-proving transformer.

  NanoProof is designed for automated theorem proving with:
  - Rotary embeddings (no positional)
  - RMSNorm without learnable parameters
  - QK normalization
  - ReLU² activation
  - Group-Query Attention (GQA)
  - Dual heads: policy (for action selection) + value (for MCTS)
-/

-- Model
import Examples.NanoProof.Model

-- Training (requires MCTS/Prover - currently broken)
-- import Examples.NanoProof.RLTrain

/-!
# `Examples.NanoProof.NanoProof`

NanoProof example component for Nano Proof, used in theorem-oriented model experiments.

## Overview
- Example module intended for runnable workflows and reference usage patterns.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

