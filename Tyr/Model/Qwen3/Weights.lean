/-
  Tyr/Model/Qwen3/Weights.lean

  Pretrained weight loading for standalone Qwen3 causal-LM.
-/
import Tyr.Torch
import Tyr.Model.Qwen.Weights
import Tyr.Model.Qwen3.Model

namespace torch.qwen3

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (toFloat' t) false

private def tryLoadTensorSharded (modelDir : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensorSharded modelDir name s
    pure (some t)
  catch _ =>
    pure none

private def tryLoadTensor (path : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensor path name s
    pure (some t)
  catch _ =>
    pure none

namespace Qwen3ForCausalLM

/-- Load Qwen3 causal-LM from a sharded HF SafeTensors directory.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen3_4B)
    : IO (Qwen3ForCausalLM cfg) := do
  IO.println s!"Loading Qwen3ForCausalLM from {modelDir}..."
  let model ← qwen.loadQwen3ModelSharded modelDir cfg false

  let lmHeadOpt ← tryLoadTensorSharded modelDir "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; using tied embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  IO.println "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

/-- Load Qwen3 causal-LM from a single SafeTensors file.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def load (path : String) (cfg : Config := Config.qwen3_4B)
    : IO (Qwen3ForCausalLM cfg) := do
  IO.println s!"Loading Qwen3ForCausalLM from {path}..."
  let model ← qwen.loadQwen3Model path cfg

  let lmHeadOpt ← tryLoadTensor path "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; using tied embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  IO.println "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

end Qwen3ForCausalLM

end torch.qwen3
