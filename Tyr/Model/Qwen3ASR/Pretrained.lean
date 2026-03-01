/-  Tyr/Model/Qwen3ASR/Pretrained.lean

  Resolve/load Qwen3-ASR checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Model.Qwen3ASR.ConfigIO
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Model.Qwen3ASR.Weights
import Lean.Data.Json

namespace torch.qwen3asr.hub

open Lean

structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
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

private def expandHome (path : String) : IO String := do
  if path.startsWith "~" then
    match (← IO.getEnv "HOME") with
    | some home => pure (path.replace "~" home)
    | none => pure path
  else
    pure path

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getString? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok j => pure j
  | .error err => throw <| IO.userError s!"Failed to parse JSON at {path}: {err}"

private def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

/-- Parse a sharded `model.safetensors.index.json` and return unique shard file names. -/
def shardFilesFromIndexFile (indexPath : String) : IO (Array String) := do
  let root ← parseJsonFile indexPath
  let mut out : Array String := #[]
  match getObjVal? root "weight_map" with
  | some (.obj kvs) =>
    for (_, shardJson) in kvs do
      match getString? shardJson with
      | some shard => out := pushUnique out shard
      | none => pure ()
  | _ =>
    throw <| IO.userError s!"Missing or invalid weight_map in {indexPath}"
  pure out

private def ensureParentDir (path : String) : IO Unit := do
  let p : System.FilePath := ⟨path⟩
  match p.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()

private def fileExists (path : String) : IO Bool := do
  let p : System.FilePath := ⟨path⟩
  p.pathExists

private def dirExists (path : String) : IO Bool := do
  let p : System.FilePath := ⟨path⟩
  if !(← p.pathExists) then
    pure false
  else
    p.isDir

private def downloadFile (url : String) (dest : String) : IO Bool := do
  if ← fileExists dest then
    return true

  ensureParentDir dest
  let tmp := s!"{dest}.tmp"
  let token? ← IO.getEnv "HF_TOKEN"
  let args :=
    match token? with
    | some tok =>
      #[
        "-fL", "--retry", "3", "--retry-delay", "1",
        "-H", s!"Authorization: Bearer {tok}",
        "-o", tmp, url
      ]
    | none =>
      #[
        "-fL", "--retry", "3", "--retry-delay", "1",
        "-o", tmp, url
      ]

  let out ← IO.Process.output { cmd := "curl", args := args }
  if out.exitCode == 0 then
    IO.FS.rename ⟨tmp⟩ ⟨dest⟩
    pure true
  else
    if ← fileExists tmp then
      IO.FS.removeFile ⟨tmp⟩
    pure false

private def ensureRemoteFile (repoId revision relPath dest : String) : IO Unit := do
  if ← fileExists dest then
    pure ()
  else
    let url := s!"https://huggingface.co/{repoId}/resolve/{revision}/{relPath}"
    let ok ← downloadFile url dest
    if !ok then
      throw <| IO.userError s!"Failed to download {repoId}:{revision}:{relPath}"

private def tryRemoteFile (repoId revision relPath dest : String) : IO Bool := do
  if ← fileExists dest then
    pure true
  else
    let url := s!"https://huggingface.co/{repoId}/resolve/{revision}/{relPath}"
    downloadFile url dest

private def sanitizeRepoId (repoId : String) : String :=
  repoId.replace "/" "__"

private def modelDirForRepo (cacheDir repoId revision : String) : IO String := do
  let cacheDir ← expandHome cacheDir
  pure s!"{cacheDir}/{sanitizeRepoId repoId}/{revision}"

private def maybeReadRef (repoDir : System.FilePath) (revision : String) : IO (Option String) := do
  let refPath := repoDir / "refs" / revision
  if ← refPath.pathExists then
    let ref := (← IO.FS.readFile refPath).trimAscii.toString
    if ref.isEmpty then pure none else pure (some ref)
  else
    pure none

private def defaultHFHubDir : IO String := do
  match (← IO.getEnv "HF_HOME") with
  | some home => pure s!"{home}/hub"
  | none => pure "~/.cache/huggingface/hub"

private def hfRepoDirName (repoId : String) : String :=
  s!"models--{repoId.replace "/" "--"}"

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

private def hasPreprocessorFiles (dir : String) : IO Bool := do
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

private def ensureTokenizerFiles (repoId revision modelDir : String) : IO Unit := do
  let _ ← tryRemoteFile repoId revision "tokenizer.json" s!"{modelDir}/tokenizer.json"
  let _ ← tryRemoteFile repoId revision "tokenizer_config.json" s!"{modelDir}/tokenizer_config.json"
  let _ ← tryRemoteFile repoId revision "vocab.json" s!"{modelDir}/vocab.json"
  let _ ← tryRemoteFile repoId revision "merges.txt" s!"{modelDir}/merges.txt"
  pure ()

private def ensurePreprocessorFiles (repoId revision modelDir : String) : IO Unit := do
  let _ ← tryRemoteFile repoId revision "preprocessor_config.json" s!"{modelDir}/preprocessor_config.json"
  pure ()

private def ensureModelWeights (repoId revision modelDir : String) : IO Unit := do
  let indexPath := s!"{modelDir}/model.safetensors.index.json"
  let singlePath := s!"{modelDir}/model.safetensors"

  if ← fileExists indexPath then
    let shardFiles ← shardFilesFromIndexFile indexPath
    for shard in shardFiles do
      ensureRemoteFile repoId revision shard s!"{modelDir}/{shard}"
  else if ← fileExists singlePath then
    pure ()
  else
    let gotIndex ← tryRemoteFile repoId revision "model.safetensors.index.json" indexPath
    if gotIndex then
      let shardFiles ← shardFilesFromIndexFile indexPath
      for shard in shardFiles do
        ensureRemoteFile repoId revision shard s!"{modelDir}/{shard}"
    else
      ensureRemoteFile repoId revision "model.safetensors" singlePath

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
    ensureTokenizerFiles source opts.revision modelDir
  if opts.includePreprocessor then
    ensurePreprocessorFiles source opts.revision modelDir
  pure modelDir

/-- Detect whether local model files are sharded or single-file. -/
def detectWeightLayout (modelDir : String) : IO Bool := do
  let sharded := s!"{modelDir}/model.safetensors.index.json"
  if ← fileExists sharded then
    pure true
  else if ← fileExists s!"{modelDir}/model.safetensors" then
    pure false
  else
    throw <| IO.userError s!"No model.safetensors(.index.json) found in {modelDir}"

end torch.qwen3asr.hub

namespace torch.qwen3asr

namespace Qwen3ASRForConditionalGeneration

/-- Load Qwen3-ASR from a local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Qwen3ASRConfig := {})
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
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
