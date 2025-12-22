/-
  Checkpoint System for GPT Training

  Provides save/load functionality for model parameters and optimizer state.
  Enables training resumption and model export.
-/
import Tyr.Train
import Tyr.TensorStruct

namespace torch.checkpoint

open torch
open torch.gpt
open torch.train

/-- Checkpoint metadata -/
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  deriving Repr, Inhabited

/-- Save a single BlockParams to a directory -/
def saveBlockParams {n_embd : UInt64} (params : BlockParams n_embd) (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/block_{blockIdx}_"
  data.saveTensor params.ln1_weight (pfx ++ "ln1_weight.pt")
  data.saveTensor params.ln1_bias (pfx ++ "ln1_bias.pt")
  data.saveTensor params.q_proj (pfx ++ "q_proj.pt")
  data.saveTensor params.k_proj (pfx ++ "k_proj.pt")
  data.saveTensor params.v_proj (pfx ++ "v_proj.pt")
  data.saveTensor params.c_proj (pfx ++ "c_proj.pt")
  data.saveTensor params.c_proj_bias (pfx ++ "c_proj_bias.pt")
  data.saveTensor params.ln2_weight (pfx ++ "ln2_weight.pt")
  data.saveTensor params.ln2_bias (pfx ++ "ln2_bias.pt")
  data.saveTensor params.mlp_fc (pfx ++ "mlp_fc.pt")
  data.saveTensor params.mlp_fc_bias (pfx ++ "mlp_fc_bias.pt")
  data.saveTensor params.mlp_proj (pfx ++ "mlp_proj.pt")
  data.saveTensor params.mlp_proj_bias (pfx ++ "mlp_proj_bias.pt")

/-- Load a single BlockParams from a directory (without making leaf params) -/
private def loadBlockParamsRaw (n_embd : UInt64) (dir : String) (blockIdx : Nat) : IO (BlockParams n_embd) := do
  let pfx := s!"{dir}/block_{blockIdx}_"
  return {
    ln1_weight := ← data.loadTensor #[n_embd] (pfx ++ "ln1_weight.pt")
    ln1_bias := ← data.loadTensor #[n_embd] (pfx ++ "ln1_bias.pt")
    q_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "q_proj.pt")
    k_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "k_proj.pt")
    v_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "v_proj.pt")
    c_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "c_proj.pt")
    c_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "c_proj_bias.pt")
    ln2_weight := ← data.loadTensor #[n_embd] (pfx ++ "ln2_weight.pt")
    ln2_bias := ← data.loadTensor #[n_embd] (pfx ++ "ln2_bias.pt")
    mlp_fc := ← data.loadTensor #[4 * n_embd, n_embd] (pfx ++ "mlp_fc.pt")
    mlp_fc_bias := ← data.loadTensor #[4 * n_embd] (pfx ++ "mlp_fc_bias.pt")
    mlp_proj := ← data.loadTensor #[n_embd, 4 * n_embd] (pfx ++ "mlp_proj.pt")
    mlp_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "mlp_proj_bias.pt")
  }

/-- Save GPT model parameters to a directory -/
def saveGPTParams {cfg : Config} (params : GPTParams cfg) (dir : String) : IO Unit := do
  -- Save embeddings
  data.saveTensor params.wte (dir ++ "/wte.pt")
  data.saveTensor params.wpe (dir ++ "/wpe.pt")
  -- Save blocks using zipWithIndex pattern
  let mut idx := 0
  for block in params.blocks do
    saveBlockParams block dir idx
    idx := idx + 1
  -- Save final layer norm
  data.saveTensor params.ln_f_weight (dir ++ "/ln_f_weight.pt")
  data.saveTensor params.ln_f_bias (dir ++ "/ln_f_bias.pt")
  IO.println s!"Model saved to {dir}"

/-- Load GPT model parameters from a directory -/
def loadGPTParams (cfg : Config) (dir : String) : IO (GPTParams cfg) := do
  -- Load embeddings
  let wte ← data.loadTensor #[cfg.vocab_size, cfg.n_embd] (dir ++ "/wte.pt")
  let wpe ← data.loadTensor #[cfg.block_size, cfg.n_embd] (dir ++ "/wpe.pt")
  -- Load blocks
  let mut blocks : Array (BlockParams cfg.n_embd) := #[]
  for i in [:cfg.n_layer.toNat] do
    let block ← loadBlockParamsRaw cfg.n_embd dir i
    blocks := blocks.push block
  -- Load final layer norm
  let ln_f_weight ← data.loadTensor #[cfg.n_embd] (dir ++ "/ln_f_weight.pt")
  let ln_f_bias ← data.loadTensor #[cfg.n_embd] (dir ++ "/ln_f_bias.pt")
  IO.println s!"Model loaded from {dir}"
  -- Apply makeLeafParams to top-level tensors (blocks already processed)
  let model : GPTParams cfg := {
    wte := wte
    wpe := wpe
    blocks := blocks
    ln_f_weight := ln_f_weight
    ln_f_bias := ln_f_bias
  }
  return TensorStruct.makeLeafParams model

/-- Save checkpoint metadata to a file -/
def saveCheckpointMeta (m : CheckpointMeta) (path : String) : IO Unit := do
  let content := s!"iteration={m.iteration}\nbestValLoss={m.bestValLoss}\ntrainLoss={m.trainLoss}"
  IO.FS.writeFile path content

/-- Parse checkpoint metadata from a file -/
def loadCheckpointMeta (path : String) : IO CheckpointMeta := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n"
  let mut iteration : Nat := 0
  let mut bestValLoss : Float := 1e10
  let mut trainLoss : Float := 0.0
  for line in lines do
    if line.startsWith "iteration=" then
      iteration := (line.drop 10).toNat!
    else if line.startsWith "bestValLoss=" then
      let valStr := line.drop 12
      bestValLoss := valStr.toNat!.toFloat
    else if line.startsWith "trainLoss=" then
      let valStr := line.drop 10
      trainLoss := valStr.toNat!.toFloat
  return { iteration, bestValLoss, trainLoss }

/-- Save full checkpoint (model + metadata) -/
def saveCheckpoint {cfg : Config}
    (params : GPTParams cfg)
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String) : IO Unit := do
  saveGPTParams params dir
  saveCheckpointMeta { iteration, bestValLoss, trainLoss } (dir ++ "/meta.txt")
  IO.println s!"Checkpoint saved at iteration {iteration}"

/-- Load checkpoint (model + metadata) -/
def loadCheckpoint (cfg : Config) (dir : String) : IO (GPTParams cfg × CheckpointMeta) := do
  let params ← loadGPTParams cfg dir
  let m ← loadCheckpointMeta (dir ++ "/meta.txt")
  IO.println s!"Checkpoint loaded from iteration {m.iteration}"
  return (params, m)

/-- Check if a checkpoint exists -/
def checkpointExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/meta.txt")

end torch.checkpoint
