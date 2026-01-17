/-
  Tyr/Data/HuggingFace.lean

  HuggingFace dataset loading via parquet files using FFI.

  Uses the Apache Arrow parquet reader via C++ FFI bindings,
  avoiding the need for Python dependencies.
-/
import Tyr.Data.Download
import Tyr.Data.Pretraining
import Lean.Data.Json

namespace torch.Data.HuggingFace

open torch.Data.Download
open torch.Data.Pretraining (readParquetAsJson readRowGroupAsJson getParquetMetadata)

/-- HuggingFace dataset configuration -/
structure HFDatasetConfig where
  repoId : String
  subset : String := "default"
  split : String := "train"
  cacheDir : String := "~/.cache/huggingface"
  deriving Repr, Inhabited

/-- Read parquet file and return JSON objects per row using FFI.
    This uses the Apache Arrow library via C++ bindings. -/
def readParquetToJson (filePath : String) : IO (Array Lean.Json) := do
  -- Use FFI to read parquet as JSON strings
  let jsonStrings ← readParquetAsJson filePath

  -- Parse each JSON string
  let mut jsons : Array Lean.Json := #[]
  for jsonStr in jsonStrings do
    match Lean.Json.parse jsonStr with
    | .ok json => jsons := jsons.push json
    | .error e =>
      -- Log parse error but continue
      IO.eprintln s!"Warning: Failed to parse JSON: {e}"
  return jsons

/-- Read parquet file row group by row group for large files.
    More memory efficient than reading the entire file at once. -/
def readParquetToJsonStreaming (filePath : String) : IO (Array Lean.Json) := do
  let metadata ← getParquetMetadata filePath
  let mut allJsons : Array Lean.Json := #[]

  for rgIdx in [:metadata.numRowGroups] do
    let jsonStrings ← readRowGroupAsJson filePath rgIdx.toUInt64
    for jsonStr in jsonStrings do
      match Lean.Json.parse jsonStr with
      | .ok json => allJsons := allJsons.push json
      | .error _ => pure ()

  return allJsons

/-- Load ARC dataset from HuggingFace -/
def loadARC (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "allenai/ai2_arc" subset split cacheDir
  readParquetToJson localPath

/-- Load GSM8K dataset from HuggingFace -/
def loadGSM8K (subset : String := "main") (split : String := "train")
    (cacheDir : String := "~/.cache/huggingface") : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "openai/gsm8k" subset split cacheDir
  readParquetToJson localPath

/-- Load MMLU dataset from HuggingFace -/
def loadMMLU (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO (Array Lean.Json) := do
  let localPath ← ensureHFParquet "cais/mmlu" subset split cacheDir
  readParquetToJson localPath

/-- Load JSONL file from URL -/
def loadJsonlFromUrl (url : String) (cacheDir : String) (filename : String)
    : IO (Array Lean.Json) := do
  let expandedCacheDir ← expandHome cacheDir
  let localPath := s!"{expandedCacheDir}/{filename}"
  let _ ← downloadWithRetry url localPath 3
  let content ← IO.FS.readFile ⟨localPath⟩
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut jsons : Array Lean.Json := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json => jsons := jsons.push json
    | .error _ => pure ()
  return jsons

/-- Load JSONL file from local path -/
def loadJsonlFromFile (path : String) : IO (Array Lean.Json) := do
  let expandedPath ← expandHome path
  let content ← IO.FS.readFile ⟨expandedPath⟩
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut jsons : Array Lean.Json := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json => jsons := jsons.push json
    | .error _ => pure ()
  return jsons

/-- Get string field from JSON object -/
def getJsonString (json : Lean.Json) (field : String) : Option String :=
  (json.getObjValAs? String field).toOption

def getJsonNat (json : Lean.Json) (field : String) : Option Nat :=
  (json.getObjValAs? Nat field).toOption

def getJsonBool (json : Lean.Json) (field : String) : Option Bool :=
  (json.getObjValAs? Bool field).toOption

def getJsonStringArray (json : Lean.Json) (field : String) : Option (Array String) :=
  (json.getObjValAs? (Array String) field).toOption

end torch.Data.HuggingFace
