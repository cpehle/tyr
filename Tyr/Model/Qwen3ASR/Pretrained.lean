/-
  Tyr/Model/Qwen3ASR/Pretrained.lean

  Resolve/load Qwen3-ASR checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen3ASR.ConfigIO
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Model.Qwen3ASR.Weights

namespace torch.qwen3asr.hub

open torch.Hub

structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  includeTokenizer : Bool := true
  includePreprocessor : Bool := true
  deriving Repr, Inhabited

/-- Official Qwen3-ASR model ids in HF collection `Qwen/Qwen3-ASR` as of 2026-02-28. -/
def qwen3ASRCollectionRepoIds : Array String := #[
  "Qwen/Qwen3-ASR-0.6B",
  "Qwen/Qwen3-ASR-1.7B"
]

def isQwen3ASRCollectionRepoId (repoId : String) : Bool :=
  qwen3ASRCollectionRepoIds.contains repoId

private def hasCompleteWeightFiles (dir : String) : IO Bool := do
  let sharded := s!"{dir}/model.safetensors.index.json"
  let single := s!"{dir}/model.safetensors"
  if ← fileExists sharded then
    try
      let shardFiles ← shardFilesFromIndexFile sharded
      if shardFiles.isEmpty then
        pure false
      else
        let mut allPresent := true
        for shard in shardFiles do
          if !(← fileExists s!"{dir}/{shard}") then
            allPresent := false
        pure allPresent
    catch _ =>
      pure false
  else
    fileExists single

private def hasTokenizerFiles (dir : String) : IO Bool := do
  let tokJson := s!"{dir}/tokenizer.json"
  let tokCfg := s!"{dir}/tokenizer_config.json"
  let vocab := s!"{dir}/vocab.json"
  let merges := s!"{dir}/merges.txt"
  pure <| (← fileExists tokJson) || ((← fileExists tokCfg) && (← fileExists vocab) && (← fileExists merges))

private def hasPreprocessorFiles (dir : String) : IO Bool :=
  fileExists s!"{dir}/preprocessor_config.json"

/-- Try resolving a repo id against existing HuggingFace cache snapshots. -/
def findCachedSnapshot? (repoId : String) (revision : String := "main") : IO (Option String) := do
  let hubDir ← expandHome (← defaultHFHubDir)
  let repoDir : System.FilePath := ⟨s!"{hubDir}/{hfRepoDirName repoId}"⟩
  if !(← repoDir.pathExists) then
    return none

  let snapshotsDir := repoDir / "snapshots"
  if !(← snapshotsDir.pathExists) then
    return none

  let mut candidates : Array System.FilePath := #[]
  let directRevPath := snapshotsDir / revision
  if ← directRevPath.pathExists then
    candidates := candidates.push directRevPath

  if let some resolvedRev ← maybeReadRef repoDir revision then
    let resolvedPath := snapshotsDir / resolvedRev
    if (← resolvedPath.pathExists) && !candidates.contains resolvedPath then
      candidates := candidates.push resolvedPath

  let entries ← snapshotsDir.readDir
  for entry in entries do
    if (← entry.path.isDir) && !candidates.contains entry.path then
      candidates := candidates.push entry.path

  for c in candidates do
    let cStr := c.toString
    if (← fileExists s!"{cStr}/config.json") && (← hasCompleteWeightFiles cStr) then
      return some cStr
  pure none

/-- Resolve a source into a local model directory.
    - If `source` is an existing local directory, return it.
    - Otherwise treat it as HF `repo_id` and resolve/download locally. -/
def resolvePretrainedDir (source : String) (opts : DownloadOptions := {}) : IO String := do
  let sourceExpanded ← expandHome source
  if ← dirExists sourceExpanded then
    return sourceExpanded

  if let some snap ← findCachedSnapshot? source opts.revision then
    let hasTok ←
      if opts.includeTokenizer then
        hasTokenizerFiles snap
      else
        pure true
    let hasPre ←
      if opts.includePreprocessor then
        hasPreprocessorFiles snap
      else
        pure true
    if hasTok && hasPre then
      ensureModelWeights source opts.revision snap
      return snap

  let modelDir ← modelDirForRepo opts.cacheDir source opts.revision
  IO.FS.createDirAll ⟨modelDir⟩
  ensureRemoteFile source opts.revision "config.json" s!"{modelDir}/config.json"
  ensureModelWeights source opts.revision modelDir
  if opts.includeTokenizer then
    let _ ← tryRemoteFile source opts.revision "tokenizer.json" s!"{modelDir}/tokenizer.json"
    let _ ← tryRemoteFile source opts.revision "tokenizer_config.json" s!"{modelDir}/tokenizer_config.json"
    let _ ← tryRemoteFile source opts.revision "vocab.json" s!"{modelDir}/vocab.json"
    let _ ← tryRemoteFile source opts.revision "merges.txt" s!"{modelDir}/merges.txt"
    pure ()
  if opts.includePreprocessor then
    let _ ← tryRemoteFile source opts.revision "preprocessor_config.json" s!"{modelDir}/preprocessor_config.json"
    pure ()
  pure modelDir

export torch.Hub (detectWeightLayout)

end torch.qwen3asr.hub

namespace torch.qwen3asr

namespace Qwen3ASRForConditionalGeneration

/-- Load Qwen3-ASR from a local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Qwen3ASRConfig := {})
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    : IO (Sigma (fun cfg => Qwen3ASRForConditionalGeneration cfg)) := do
  let modelDir ← hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := false
    includePreprocessor := false
  }
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir defaults
  let m ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  pure ⟨cfg, m⟩

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
