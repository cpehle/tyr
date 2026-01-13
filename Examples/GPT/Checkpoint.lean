/-
  Checkpoint System for GPT Training

  Provides save/load functionality for model parameters and optimizer state.
  Enables training resumption and model export.
-/
import Examples.GPT.Train
import Tyr.TensorStruct

namespace torch.checkpoint

open torch
open torch.gpt
open torch.train
open torch.Optim

/-- Checkpoint metadata -/
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0  -- Adam step count for bias correction
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

/-! ## Optimizer State Checkpointing

Save and load AdamW optimizer state (mu, nu, count) to enable training resumption
with proper momentum preservation.
-/

/-- Helper: Save ScaleByAdamState for a BlockParams -/
private def saveBlockAdamState {n_embd : UInt64}
    (mu nu : BlockParams n_embd) (dir : String) (blockIdx : Nat) : IO Unit := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_"
  -- Save mu (first moment)
  data.saveTensor mu.ln1_weight (pfx ++ "mu_ln1_weight.pt")
  data.saveTensor mu.ln1_bias (pfx ++ "mu_ln1_bias.pt")
  data.saveTensor mu.q_proj (pfx ++ "mu_q_proj.pt")
  data.saveTensor mu.k_proj (pfx ++ "mu_k_proj.pt")
  data.saveTensor mu.v_proj (pfx ++ "mu_v_proj.pt")
  data.saveTensor mu.c_proj (pfx ++ "mu_c_proj.pt")
  data.saveTensor mu.c_proj_bias (pfx ++ "mu_c_proj_bias.pt")
  data.saveTensor mu.ln2_weight (pfx ++ "mu_ln2_weight.pt")
  data.saveTensor mu.ln2_bias (pfx ++ "mu_ln2_bias.pt")
  data.saveTensor mu.mlp_fc (pfx ++ "mu_mlp_fc.pt")
  data.saveTensor mu.mlp_fc_bias (pfx ++ "mu_mlp_fc_bias.pt")
  data.saveTensor mu.mlp_proj (pfx ++ "mu_mlp_proj.pt")
  data.saveTensor mu.mlp_proj_bias (pfx ++ "mu_mlp_proj_bias.pt")
  -- Save nu (second moment)
  data.saveTensor nu.ln1_weight (pfx ++ "nu_ln1_weight.pt")
  data.saveTensor nu.ln1_bias (pfx ++ "nu_ln1_bias.pt")
  data.saveTensor nu.q_proj (pfx ++ "nu_q_proj.pt")
  data.saveTensor nu.k_proj (pfx ++ "nu_k_proj.pt")
  data.saveTensor nu.v_proj (pfx ++ "nu_v_proj.pt")
  data.saveTensor nu.c_proj (pfx ++ "nu_c_proj.pt")
  data.saveTensor nu.c_proj_bias (pfx ++ "nu_c_proj_bias.pt")
  data.saveTensor nu.ln2_weight (pfx ++ "nu_ln2_weight.pt")
  data.saveTensor nu.ln2_bias (pfx ++ "nu_ln2_bias.pt")
  data.saveTensor nu.mlp_fc (pfx ++ "nu_mlp_fc.pt")
  data.saveTensor nu.mlp_fc_bias (pfx ++ "nu_mlp_fc_bias.pt")
  data.saveTensor nu.mlp_proj (pfx ++ "nu_mlp_proj.pt")
  data.saveTensor nu.mlp_proj_bias (pfx ++ "nu_mlp_proj_bias.pt")

/-- Helper: Load BlockParams for optimizer state -/
private def loadBlockAdamStateRaw (n_embd : UInt64) (dir : String) (blockIdx : Nat)
    : IO (BlockParams n_embd × BlockParams n_embd) := do
  let pfx := s!"{dir}/optim_block_{blockIdx}_"
  let mu : BlockParams n_embd := {
    ln1_weight := ← data.loadTensor #[n_embd] (pfx ++ "mu_ln1_weight.pt")
    ln1_bias := ← data.loadTensor #[n_embd] (pfx ++ "mu_ln1_bias.pt")
    q_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "mu_q_proj.pt")
    k_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "mu_k_proj.pt")
    v_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "mu_v_proj.pt")
    c_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "mu_c_proj.pt")
    c_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "mu_c_proj_bias.pt")
    ln2_weight := ← data.loadTensor #[n_embd] (pfx ++ "mu_ln2_weight.pt")
    ln2_bias := ← data.loadTensor #[n_embd] (pfx ++ "mu_ln2_bias.pt")
    mlp_fc := ← data.loadTensor #[4 * n_embd, n_embd] (pfx ++ "mu_mlp_fc.pt")
    mlp_fc_bias := ← data.loadTensor #[4 * n_embd] (pfx ++ "mu_mlp_fc_bias.pt")
    mlp_proj := ← data.loadTensor #[n_embd, 4 * n_embd] (pfx ++ "mu_mlp_proj.pt")
    mlp_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "mu_mlp_proj_bias.pt")
  }
  let nu : BlockParams n_embd := {
    ln1_weight := ← data.loadTensor #[n_embd] (pfx ++ "nu_ln1_weight.pt")
    ln1_bias := ← data.loadTensor #[n_embd] (pfx ++ "nu_ln1_bias.pt")
    q_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "nu_q_proj.pt")
    k_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "nu_k_proj.pt")
    v_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "nu_v_proj.pt")
    c_proj := ← data.loadTensor #[n_embd, n_embd] (pfx ++ "nu_c_proj.pt")
    c_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "nu_c_proj_bias.pt")
    ln2_weight := ← data.loadTensor #[n_embd] (pfx ++ "nu_ln2_weight.pt")
    ln2_bias := ← data.loadTensor #[n_embd] (pfx ++ "nu_ln2_bias.pt")
    mlp_fc := ← data.loadTensor #[4 * n_embd, n_embd] (pfx ++ "nu_mlp_fc.pt")
    mlp_fc_bias := ← data.loadTensor #[4 * n_embd] (pfx ++ "nu_mlp_fc_bias.pt")
    mlp_proj := ← data.loadTensor #[n_embd, 4 * n_embd] (pfx ++ "nu_mlp_proj.pt")
    mlp_proj_bias := ← data.loadTensor #[n_embd] (pfx ++ "nu_mlp_proj_bias.pt")
  }
  return (mu, nu)

/-- Save AdamW optimizer state for GPT model -/
def saveAdamWState {cfg : Config}
    (state : AdamWState (GPTParams cfg))
    (dir : String) : IO Unit := do
  let adamState := state.fst  -- ScaleByAdamState is first in chain
  let pfx := dir ++ "/optim_"
  -- Save count
  IO.FS.writeFile (pfx ++ "count.txt") (toString adamState.count)
  -- Save embedding mu/nu
  data.saveTensor adamState.mu.wte (pfx ++ "mu_wte.pt")
  data.saveTensor adamState.mu.wpe (pfx ++ "mu_wpe.pt")
  data.saveTensor adamState.mu.ln_f_weight (pfx ++ "mu_ln_f_weight.pt")
  data.saveTensor adamState.mu.ln_f_bias (pfx ++ "mu_ln_f_bias.pt")
  data.saveTensor adamState.nu.wte (pfx ++ "nu_wte.pt")
  data.saveTensor adamState.nu.wpe (pfx ++ "nu_wpe.pt")
  data.saveTensor adamState.nu.ln_f_weight (pfx ++ "nu_ln_f_weight.pt")
  data.saveTensor adamState.nu.ln_f_bias (pfx ++ "nu_ln_f_bias.pt")
  -- Save block states
  let mut idx := 0
  for (muBlock, nuBlock) in adamState.mu.blocks.zip adamState.nu.blocks do
    saveBlockAdamState muBlock nuBlock dir idx
    idx := idx + 1
  IO.println s!"Optimizer state saved to {dir}"

/-- Load AdamW optimizer state for GPT model -/
def loadAdamWState (cfg : Config) (dir : String) : IO (AdamWState (GPTParams cfg)) := do
  let pfx := dir ++ "/optim_"
  -- Load count
  let countStr ← IO.FS.readFile (pfx ++ "count.txt")
  let count := countStr.trimAscii.toString.toNat!
  -- Load embedding mu/nu
  let mu_wte ← data.loadTensor #[cfg.vocab_size, cfg.n_embd] (pfx ++ "mu_wte.pt")
  let mu_wpe ← data.loadTensor #[cfg.block_size, cfg.n_embd] (pfx ++ "mu_wpe.pt")
  let mu_ln_f_weight ← data.loadTensor #[cfg.n_embd] (pfx ++ "mu_ln_f_weight.pt")
  let mu_ln_f_bias ← data.loadTensor #[cfg.n_embd] (pfx ++ "mu_ln_f_bias.pt")
  let nu_wte ← data.loadTensor #[cfg.vocab_size, cfg.n_embd] (pfx ++ "nu_wte.pt")
  let nu_wpe ← data.loadTensor #[cfg.block_size, cfg.n_embd] (pfx ++ "nu_wpe.pt")
  let nu_ln_f_weight ← data.loadTensor #[cfg.n_embd] (pfx ++ "nu_ln_f_weight.pt")
  let nu_ln_f_bias ← data.loadTensor #[cfg.n_embd] (pfx ++ "nu_ln_f_bias.pt")
  -- Load block states
  let mut muBlocks : Array (BlockParams cfg.n_embd) := #[]
  let mut nuBlocks : Array (BlockParams cfg.n_embd) := #[]
  for i in [:cfg.n_layer.toNat] do
    let (muBlock, nuBlock) ← loadBlockAdamStateRaw cfg.n_embd dir i
    muBlocks := muBlocks.push muBlock
    nuBlocks := nuBlocks.push nuBlock
  let mu : GPTParams cfg := {
    wte := mu_wte, wpe := mu_wpe, blocks := muBlocks
    ln_f_weight := mu_ln_f_weight, ln_f_bias := mu_ln_f_bias
  }
  let nu : GPTParams cfg := {
    wte := nu_wte, wpe := nu_wpe, blocks := nuBlocks
    ln_f_weight := nu_ln_f_weight, ln_f_bias := nu_ln_f_bias
  }
  IO.println s!"Optimizer state loaded from {dir} (count={count})"
  return { fst := { count, mu, nu }, snd := { fst := {}, snd := {} } }

/-- Save full checkpoint with optimizer state -/
def saveFullCheckpoint {cfg : Config}
    (params : GPTParams cfg)
    (optState : AdamWState (GPTParams cfg))
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String) : IO Unit := do
  saveGPTParams params dir
  saveAdamWState optState dir
  let ckptMeta : CheckpointMeta := {
    iteration, bestValLoss, trainLoss
    optimCount := optState.fst.count
  }
  saveCheckpointMeta ckptMeta (dir ++ "/meta.txt")
  IO.println s!"Full checkpoint saved at iteration {iteration}"

/-- Load full checkpoint with optimizer state -/
def loadFullCheckpoint (cfg : Config) (dir : String)
    : IO (GPTParams cfg × AdamWState (GPTParams cfg) × CheckpointMeta) := do
  let params ← loadGPTParams cfg dir
  let optState ← loadAdamWState cfg dir
  let ckptMeta ← loadCheckpointMeta (dir ++ "/meta.txt")
  IO.println s!"Full checkpoint loaded from iteration {ckptMeta.iteration}"
  return (params, optState, ckptMeta)

/-- Check if optimizer state exists in checkpoint -/
def optimStateExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/optim_count.txt")

end torch.checkpoint
