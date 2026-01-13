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
