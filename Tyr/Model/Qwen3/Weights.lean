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

private def isMissingTensorError (msg : String) (name : String) : Bool :=
  (msg.contains name) &&
  ((msg.contains "not found") || (msg.contains "missing") || (msg.contains "No tensor"))

private def tryLoadOptionalTensorSharded (modelDir : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensorSharded modelDir name s
    pure (some t)
  catch e =>
    let msg := toString e
    if isMissingTensorError msg name then
      pure none
    else
      throw e

private def tryLoadOptionalTensor (path : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensor path name s
    pure (some t)
  catch e =>
    let msg := toString e
    if isMissingTensorError msg name then
      pure none
    else
      throw e

namespace Qwen3ForCausalLM

/-- Load Qwen3 causal-LM from a sharded HF SafeTensors directory.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen3_4B)
    : IO (Qwen3ForCausalLM cfg) := do
  IO.println s!"Loading Qwen3ForCausalLM from {modelDir}..."
  let model ← qwen.loadQwen3ModelSharded modelDir cfg false

  let lmHeadOpt ← tryLoadOptionalTensorSharded modelDir "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; using tied embeddings."
      pure (model.embed_tokens, true)

  IO.println "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

/-- Load Qwen3 causal-LM from a single SafeTensors file.
    Falls back to tied embeddings if `lm_head.weight` is absent. -/
def load (path : String) (cfg : Config := Config.qwen3_4B)
    : IO (Qwen3ForCausalLM cfg) := do
  IO.println s!"Loading Qwen3ForCausalLM from {path}..."
  let model ← qwen.loadQwen3Model path cfg

  let lmHeadOpt ← tryLoadOptionalTensor path "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; using tied embeddings."
      pure (model.embed_tokens, true)

  IO.println "Loaded Qwen3ForCausalLM weights."
  pure { model, lmHead, tieWordEmbeddings }

end Qwen3ForCausalLM

end torch.qwen3
