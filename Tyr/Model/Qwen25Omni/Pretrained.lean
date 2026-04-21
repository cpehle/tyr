/-
  Tyr/Model/Qwen25Omni/Pretrained.lean

  Resolve/load Qwen2.5-Omni thinker text checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen25Omni.ConfigIO
import Tyr.Model.Qwen25Omni.Weights

namespace torch.qwen25omni.hub

open torch.Hub

/-- Qwen2.5-Omni model ids (HF collection `Qwen/qwen25-omni`). -/
def qwen25OmniCollectionRepoIds : Array String := #[
  "Qwen/Qwen2.5-Omni-3B",
  "Qwen/Qwen2.5-Omni-7B",
  "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
  "Qwen/Qwen2.5-Omni-7B-AWQ"
]

def isQwen25OmniCollectionRepoId (repoId : String) : Bool :=
  qwen25OmniCollectionRepoIds.contains repoId

/-- Qwen2.5-Omni tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
  "merges.txt"
]

/-- Re-export generic hub utilities under the qwen25omni namespace for backward compatibility. -/
export Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot?)

end torch.qwen25omni.hub

namespace torch.qwen25omni

namespace Qwen25OmniForCausalLM

/-- Load Qwen2.5-Omni thinker text checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen25omni_3B)
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
    : IO (Sigma (fun cfg => Qwen25OmniForCausalLM cfg)) := do
  let modelDir ← Hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  } hub.tokenizerFiles
  let cfg ← Config.loadFromPretrainedDir modelDir defaults
  let isSharded ← Hub.detectWeightLayout modelDir
  if isSharded then
    let m ← Qwen25OmniForCausalLM.loadSharded modelDir cfg
    pure ⟨cfg, m⟩
  else
    let m ← Qwen25OmniForCausalLM.load s!"{modelDir}/model.safetensors" cfg
    pure ⟨cfg, m⟩

end Qwen25OmniForCausalLM

end torch.qwen25omni
