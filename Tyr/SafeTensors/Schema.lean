/-
  Tyr/SafeTensors/Schema.lean

  SafeTensors schema introspection for Lean:
  - parse tensor headers from `.safetensors` files
  - introspect single-file or sharded-directory layouts
-/
import Tyr.Torch
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace torch.safetensors

open Lean

/-- One tensor entry discovered in a SafeTensors source. -/
structure TensorSchema where
  name : String
  dtype : String
  shape : Shape
  /-- For directory sources this is the shard filename; for single-file sources it is empty. -/
  sourceFile : String := ""
  deriving Inhabited, Repr, BEq

/-- Full schema discovered from a SafeTensors source path. -/
structure Schema where
  source : String
  sourceIsDirectory : Bool
  tensors : Array TensorSchema
  deriving Inhabited, Repr

/-- Look up a tensor schema by exact tensor name. -/
def Schema.find? (schema : Schema) (tensorName : String) : Option TensorSchema :=
  schema.tensors.findSome? fun t => if t.name == tensorName then some t else none

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr xs => some xs
  | _ => none

private def getStr? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def getNat? (j : Json) : Option Nat :=
  match (FromJson.fromJson? j : Except String Nat) with
  | .ok n => some n
  | .error _ => none

private def readU64LE? (bytes : ByteArray) (offset : Nat) : Option UInt64 :=
  if offset + 8 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    let b2 := bytes[offset + 2]!
    let b3 := bytes[offset + 3]!
    let b4 := bytes[offset + 4]!
    let b5 := bytes[offset + 5]!
    let b6 := bytes[offset + 6]!
    let b7 := bytes[offset + 7]!
    some (
      b0.toUInt64 |||
      (b1.toUInt64 <<< 8) |||
      (b2.toUInt64 <<< 16) |||
      (b3.toUInt64 <<< 24) |||
      (b4.toUInt64 <<< 32) |||
      (b5.toUInt64 <<< 40) |||
      (b6.toUInt64 <<< 48) |||
      (b7.toUInt64 <<< 56)
    )

private def parseTensorEntry (tensorName sourceFile : String) (entryJson : Json)
    : Except String TensorSchema := do
  let dtypeJson ←
    match getObjVal? entryJson "dtype" with
    | some j => pure j
    | none => .error s!"SafeTensors entry '{tensorName}' is missing 'dtype'"
  let dtype ←
    match getStr? dtypeJson with
    | some s => pure s
    | none => .error s!"SafeTensors entry '{tensorName}' has non-string 'dtype'"

  let shapeJson ←
    match getObjVal? entryJson "shape" with
    | some j => pure j
    | none => .error s!"SafeTensors entry '{tensorName}' is missing 'shape'"
  let dimsJson ←
    match getArr? shapeJson with
    | some xs => pure xs
    | none => .error s!"SafeTensors entry '{tensorName}' has non-array 'shape'"

  let mut shape : Shape := #[]
  for dimJson in dimsJson do
    let n ←
      match getNat? dimJson with
      | some x => pure x
      | none => .error s!"SafeTensors entry '{tensorName}' has non-natural shape dimension"
    shape := shape.push n.toUInt64

  pure {
    name := tensorName
    dtype := dtype
    shape := shape
    sourceFile := sourceFile
  }

private def parseHeaderEntries (sourceFile : String) (headerJson : Json)
    : Except String (Array TensorSchema) := do
  let pairs ←
    match headerJson with
    | .obj kvs => pure kvs.toList
    | _ => .error "SafeTensors header is not a JSON object"

  let mut entries : Array TensorSchema := #[]
  for (tensorName, entryJson) in pairs do
    if tensorName != "__metadata__" then
      let parsed ← parseTensorEntry tensorName sourceFile entryJson
      entries := entries.push parsed
  pure entries

private def parseSafeTensorFile (path sourceFile : String) : IO (Array TensorSchema) := do
  let bytes ← IO.FS.readBinFile path
  let headerSize ←
    match readU64LE? bytes 0 with
    | some n => pure n
    | none => throw <| IO.userError s!"Invalid SafeTensors file '{path}': missing 8-byte header size"
  let headerSizeNat := headerSize.toNat
  if 8 + headerSizeNat > bytes.size then
    throw <| IO.userError
      s!"Invalid SafeTensors file '{path}': header exceeds file size ({headerSizeNat} bytes)"

  let headerBytes := bytes.extract 8 (8 + headerSizeNat)
  let headerStr ←
    match String.fromUTF8? headerBytes with
    | some s => pure s
    | none => throw <| IO.userError s!"Invalid SafeTensors file '{path}': header is not UTF-8"
  let headerJson ←
    match Json.parse headerStr with
    | .ok j => pure j
    | .error err =>
      throw <| IO.userError s!"Invalid SafeTensors file '{path}': failed to parse header JSON: {err}"

  match parseHeaderEntries sourceFile headerJson with
  | .ok entries => pure entries
  | .error err => throw <| IO.userError s!"Invalid SafeTensors file '{path}': {err}"

private def listShardFiles (dir : System.FilePath) : IO (Array String) := do
  let entries ← dir.readDir
  let mut shardFiles : Array String := #[]
  for entry in entries do
    if !(← entry.path.isDir) && entry.path.extension == some "safetensors" then
      shardFiles := shardFiles.push entry.fileName
  if shardFiles.isEmpty then
    throw <| IO.userError s!"No '.safetensors' files found in directory '{dir}'"
  pure <| shardFiles.qsort (· < ·)

private def ensureUniqueNames (tensors : Array TensorSchema) : IO Unit := do
  let mut seen : Std.HashMap String String := {}
  for t in tensors do
    match seen.get? t.name with
    | some otherSource =>
      throw <| IO.userError
        s!"Duplicate tensor name '{t.name}' in sources '{otherSource}' and '{t.sourceFile}'"
    | none =>
      seen := seen.insert t.name t.sourceFile

private def sortedByName (tensors : Array TensorSchema) : Array TensorSchema :=
  tensors.qsort (fun a b => a.name < b.name)

/-- Introspect tensor schema from either:
    - a single `.safetensors` file, or
    - a directory containing shard `.safetensors` files. -/
def introspect (source : String) : IO Schema := do
  let sourcePath : System.FilePath := ⟨source⟩
  if !(← sourcePath.pathExists) then
    throw <| IO.userError s!"SafeTensors source does not exist: {source}"

  if ← sourcePath.isDir then
    let shardFiles ← listShardFiles sourcePath
    let mut tensors : Array TensorSchema := #[]
    for shardFile in shardFiles do
      let fullPath := (sourcePath / shardFile).toString
      let shardEntries ← parseSafeTensorFile fullPath shardFile
      tensors := tensors ++ shardEntries
    ensureUniqueNames tensors
    pure {
      source := source
      sourceIsDirectory := true
      tensors := sortedByName tensors
    }
  else
    if sourcePath.extension != some "safetensors" then
      throw <| IO.userError
        s!"SafeTensors file must use '.safetensors' extension: {source}"
    let tensors ← parseSafeTensorFile source ""
    pure {
      source := source
      sourceIsDirectory := false
      tensors := sortedByName tensors
    }

end torch.safetensors
