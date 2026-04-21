/-
  Tyr/Model/Qwen35/Pretrained.lean

  Resolve/load Qwen3.5 checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.VLWeights

namespace torch.qwen35.hub

open torch.Hub

/-- Public Qwen3.5/Qwen3.6 repo ids explicitly covered by Tyr as of 2026-04-16. -/
def qwen35CollectionRepoIds : Array String := #[
  "Qwen/Qwen3.5-0.8B",
  "Qwen/Qwen3.5-0.8B-Base",
  "Qwen/Qwen3.5-397B-A17B",
  "Qwen/Qwen3.5-397B-A17B-FP8",
  "Qwen/Qwen3.5-122B-A14B",
  "Qwen/Qwen3.5-122B-A14B-FP8",
  "Qwen/Qwen3.5-35B-A3B",
  "Qwen/Qwen3.5-35B-A3B-FP8",
  "Qwen/Qwen3.5-35B-A3B-Base",
  "Qwen/Qwen3.5-27B",
  "Qwen/Qwen3.5-27B-FP8",
  "Qwen/Qwen3.6-35B-A3B"
]

def isQwen35CollectionRepoId (repoId : String) : Bool :=
  qwen35CollectionRepoIds.contains repoId

/-- Qwen tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
  "merges.txt"
]

export torch.Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot? shardFilesFromIndexFile)

end torch.qwen35.hub

namespace torch.qwen35

namespace Qwen35ForCausalLM

/-- Load a text Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen35_9B)
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Qwen35ForCausalLM cfg)) := do
  let modelDir ← Hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  } hub.tokenizerFiles
  let cfg ← Config.loadFromPretrainedDir modelDir defaults
  let isSharded ← Hub.detectWeightLayout modelDir
  if isSharded then
    let m ← Qwen35ForCausalLM.loadSharded modelDir cfg device
    pure ⟨cfg, m⟩
  else
    let m ← Qwen35ForCausalLM.load s!"{modelDir}/model.safetensors" cfg device
    pure ⟨cfg, m⟩

end Qwen35ForCausalLM

namespace Qwen35ForConditionalGeneration

/-- Load a multimodal Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : VLConfig := {})
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
    (device : Device := Device.CPU)
    : IO (Sigma (fun cfg => Qwen35ForConditionalGeneration cfg)) := do
  let modelDir ← Hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  } hub.tokenizerFiles
  let cfg ← VLConfig.loadFromPretrainedDir modelDir defaults
  let isSharded ← Hub.detectWeightLayout modelDir
  if isSharded then
    let m ← Qwen35ForConditionalGeneration.loadSharded modelDir cfg device
    pure ⟨cfg, m⟩
  else
    let m ← Qwen35ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg device
    pure ⟨cfg, m⟩

end Qwen35ForConditionalGeneration

end torch.qwen35
