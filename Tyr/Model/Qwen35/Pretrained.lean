/-  
  Tyr/Model/Qwen35/Pretrained.lean

  Resolve/load Qwen3.5 checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.VLWeights
import Lean.Data.Json

namespace torch.qwen35.hub

open Lean

structure DownloadOptions where
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  includeTokenizer : Bool := true
  deriving Repr, Inhabited

/-- Qwen3.5 model ids in HF collection `Qwen/qwen35` as of 2026-02-28. -/
def qwen35CollectionRepoIds : Array String := #[
  "Qwen/Qwen3.5-397B-A17B",
  "Qwen/Qwen3.5-397B-A17B-FP8",
  "Qwen/Qwen3.5-122B-A14B",
  "Qwen/Qwen3.5-122B-A14B-FP8",
  "Qwen/Qwen3.5-35B-A3B",
  "Qwen/Qwen3.5-35B-A3B-FP8",
  "Qwen/Qwen3.5-35B-A3B-Base",
  "Qwen/Qwen3.5-27B",
  "Qwen/Qwen3.5-27B-FP8"
]

def isQwen35CollectionRepoId (repoId : String) : Bool :=
  qwen35CollectionRepoIds.contains repoId

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
  | some home =>
    pure s!"{home}/hub"
  | none =>
    pure "~/.cache/huggingface/hub"

private def hfRepoDirName (repoId : String) : String :=
  s!"models--{repoId.replace "/" "--"}"

private def hasQwenWeightFiles (dir : String) : IO Bool := do
  let sharded := s!"{dir}/model.safetensors.index.json"
  let single := s!"{dir}/model.safetensors"
  pure ((← fileExists sharded) || (← fileExists single))

/-- Try resolving a repo id against existing HuggingFace cache snapshots. -/
def findCachedSnapshot? (repoId : String) (revision : String := "main") : IO (Option String) := do
  let hubDir ← expandHome (← defaultHFHubDir)
  let repoDir : System.FilePath := ⟨s!"{hubDir}/{hfRepoDirName repoId}"⟩
  if !(← repoDir.pathExists) then
    return none

  let snapshotsDir := repoDir / "snapshots"
  if !(← snapshotsDir.pathExists) then
    return none

  -- Prefer requested revision:
  let mut candidates : Array System.FilePath := #[]
  let directRevPath := snapshotsDir / revision
  if ← directRevPath.pathExists then
    candidates := candidates.push directRevPath

  -- Then hash from refs/<revision>, if present:
  if let some resolvedRev ← maybeReadRef repoDir revision then
    let resolvedPath := snapshotsDir / resolvedRev
    if (← resolvedPath.pathExists) && !candidates.contains resolvedPath then
      candidates := candidates.push resolvedPath

  -- Fallback: any snapshot dir:
  let entries ← snapshotsDir.readDir
  for entry in entries do
    if (← entry.path.isDir) && !candidates.contains entry.path then
      candidates := candidates.push entry.path

  for c in candidates do
    let cStr := c.toString
    if (← fileExists s!"{cStr}/config.json") && (← hasQwenWeightFiles cStr) then
      return some cStr
  pure none

private def ensureTokenizerFiles (repoId revision modelDir : String) : IO Unit := do
  let _ ← tryRemoteFile repoId revision "tokenizer.json" s!"{modelDir}/tokenizer.json"
  let _ ← tryRemoteFile repoId revision "tokenizer_config.json" s!"{modelDir}/tokenizer_config.json"
  let _ ← tryRemoteFile repoId revision "vocab.json" s!"{modelDir}/vocab.json"
  let _ ← tryRemoteFile repoId revision "merges.txt" s!"{modelDir}/merges.txt"
  pure ()

private def ensureModelWeights (repoId revision modelDir : String) : IO Unit := do
  let indexPath := s!"{modelDir}/model.safetensors.index.json"
  let singlePath := s!"{modelDir}/model.safetensors"

  if (← fileExists indexPath) || (← fileExists singlePath) then
    pure ()
  else
    -- Prefer sharded if an index exists upstream; fallback to single-file.
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

  -- Repo id path: first try existing HF cache snapshots.
  if let some snap ← findCachedSnapshot? source opts.revision then
    return snap

  -- Then use Tyr-managed cache and download required files.
  let modelDir ← modelDirForRepo opts.cacheDir source opts.revision
  IO.FS.createDirAll ⟨modelDir⟩
  ensureRemoteFile source opts.revision "config.json" s!"{modelDir}/config.json"
  ensureModelWeights source opts.revision modelDir
  if opts.includeTokenizer then
    ensureTokenizerFiles source opts.revision modelDir
  pure modelDir

/-- Detect whether local model files are sharded or single-file. -/
def detectWeightLayout (modelDir : String) : IO Bool := do
  -- true = sharded, false = single-file
  let sharded := s!"{modelDir}/model.safetensors.index.json"
  if ← fileExists sharded then
    pure true
  else if ← fileExists s!"{modelDir}/model.safetensors" then
    pure false
  else
    throw <| IO.userError s!"No model.safetensors(.index.json) found in {modelDir}"

end torch.qwen35.hub

namespace torch.qwen35

namespace Qwen35ForCausalLM

/-- Load a text Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen35_9B)
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
    : IO (Sigma (fun cfg => Qwen35ForCausalLM cfg)) := do
  let modelDir ← hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  }
  let cfg ← Config.loadFromPretrainedDir modelDir defaults
  let isSharded ← hub.detectWeightLayout modelDir
  if isSharded then
    let m ← Qwen35ForCausalLM.loadSharded modelDir cfg
    pure ⟨cfg, m⟩
  else
    let m ← Qwen35ForCausalLM.load s!"{modelDir}/model.safetensors" cfg
    pure ⟨cfg, m⟩

end Qwen35ForCausalLM

namespace Qwen35ForConditionalGeneration

/-- Load a multimodal Qwen3.5 checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : VLConfig := {})
    (revision : String := "main")
    (cacheDir : String := "~/.cache/huggingface/tyr-models")
    : IO (Sigma (fun cfg => Qwen35ForConditionalGeneration cfg)) := do
  let modelDir ← hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
  }
  let cfg ← VLConfig.loadFromPretrainedDir modelDir defaults
  let isSharded ← hub.detectWeightLayout modelDir
  if isSharded then
    let m ← Qwen35ForConditionalGeneration.loadSharded modelDir cfg
    pure ⟨cfg, m⟩
  else
    let m ← Qwen35ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg
    pure ⟨cfg, m⟩

end Qwen35ForConditionalGeneration

end torch.qwen35
