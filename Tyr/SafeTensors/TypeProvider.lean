/-
  Tyr/SafeTensors/TypeProvider.lean

  Command-level SafeTensors type provider:

    safetensors_type_provider "path/to/model.safetensors" as MyWeights
    safetensors_type_provider "path/to/sharded_dir" as MyWeights

  The command introspects tensor schema at elaboration time and emits:
  - typed tensor specs
  - typed per-tensor load helpers
  - hierarchical typed checkpoint record (`Weights`) with `loadAll`
  - generated schema table for introspection from Lean
-/
import Lean
import Tyr.SafeTensors.Schema

open Lean Elab Command

namespace torch.safetensors

syntax (name := safetensorsTypeProviderCmd)
  "safetensors_type_provider " str " as " ident : command

private def leanKeywords : List String :=
  [ "abbrev", "axiom", "class", "def", "deriving", "do", "else", "end", "example", "forall"
  , "fun", "if", "import", "in", "inductive", "instance", "let", "macro", "match", "mutual"
  , "namespace", "noncomputable", "opaque", "open", "private", "protected", "section", "set_option"
  , "structure", "syntax", "term", "then", "theorem", "universe", "unsafe", "where", "with"
  ]

private def replaceInvalidChars (raw : String) : String :=
  String.ofList <| raw.toList.map fun c => if c.isAlphanum || c == '_' then c else '_'

private def ensureValidStart (raw : String) : String :=
  match raw.toList with
  | [] => "tensor"
  | c :: _ => if c.isAlpha || c == '_' then raw else s!"t_{raw}"

private def sanitizeIdent (raw fallback : String) : String :=
  let cleaned := replaceInvalidChars raw
  let cleaned := if cleaned.isEmpty then fallback else cleaned
  let cleaned := ensureValidStart cleaned
  if leanKeywords.contains cleaned then cleaned ++ "_" else cleaned

private partial def freshNameAux (used : List String) (base : String) (idx : Nat) : String :=
  let candidate := if idx == 0 then base else s!"{base}_{idx}"
  if used.contains candidate then
    freshNameAux used base (idx + 1)
  else
    candidate

private def freshName (used : List String) (base : String) : String :=
  freshNameAux used base 0

private def tensorDeclBaseName (raw : String) : String :=
  sanitizeIdent (replaceInvalidChars raw |>.map Char.toLower) "tensor"

private def splitWords (raw : String) : List String :=
  let spaced := String.ofList <| raw.toList.map fun c => if c.isAlphanum then c else ' '
  (spaced.splitOn " ").filter (fun part => !part.isEmpty)

private def capitalizeWord (word : String) : String :=
  match word.toList with
  | [] => ""
  | c :: cs => String.ofList (c.toUpper :: cs.map Char.toLower)

private def typeNameBase (path : List String) : String :=
  if path.isEmpty then
    "Weights"
  else
    let suffix :=
      path.foldl (init := "") fun acc part =>
        let merged := (splitWords part).foldl (fun a w => a ++ capitalizeWord w) ""
        acc ++ capitalizeWord (if merged.isEmpty then part else merged)
    "Weights" ++ suffix

private def escapeLeanStringChars : List Char → List Char
  | [] => []
  | c :: cs =>
      let escaped :=
        match c with
        | '\"' => ['\\', '\"']
        | '\\' => ['\\', '\\']
        | '\n' => ['\\', 'n']
        | '\r' => ['\\', 'r']
        | '\t' => ['\\', 't']
        | other => [other]
      escaped ++ escapeLeanStringChars cs

private def escapeLeanString (raw : String) : String :=
  String.ofList <| escapeLeanStringChars raw.toList

private def leanStringLit (raw : String) : String :=
  "\"" ++ escapeLeanString raw ++ "\""

private def renderShapeLit (shape : Shape) : String :=
  let dims := shape.toList.map (fun d => toString d)
  "#[" ++ String.intercalate ", " dims ++ "]"

private def parseGeneratedCommand (source : String) : CommandElabM Syntax := do
  let env ← getEnv
  match Parser.runParserCategory env `command source "<safetensors_type_provider>" with
  | .ok stx => pure stx
  | .error err =>
    throwError "safetensors_type_provider generated invalid Lean command:\n{err}\n\n{source}"

private def elabGeneratedCommand (source : String) : CommandElabM Unit := do
  elabCommand (← parseGeneratedCommand source)

private def indentBlock (n : Nat) (source : String) : String :=
  let pad := String.ofList (List.replicate n ' ')
  String.intercalate "\n" <| (source.splitOn "\n").map (fun line => pad ++ line)

private def renderTensorDecls (tensor : TensorSchema) (declName : String) : Array String :=
  let shapeLit := renderShapeLit tensor.shape
  let shapeDecl :=
    s!"/-- Static shape for tensor `{tensor.name}`. -/\n" ++
    s!"abbrev {declName}Shape : _root_.torch.Shape := {shapeLit}\n"
  let specDecl :=
    s!"/-- Tensor schema for `{tensor.name}`. -/\n" ++
    s!"def {declName}Spec : _root_.torch.safetensors.TensorSchema :=\n" ++
    "  {\n" ++
    s!"    name := {leanStringLit tensor.name},\n" ++
    s!"    dtype := {leanStringLit tensor.dtype},\n" ++
    s!"    shape := {declName}Shape,\n" ++
    s!"    sourceFile := {leanStringLit tensor.sourceFile}\n" ++
    "  }\n"
  let loadDecl :=
    s!"/-- Load `{tensor.name}` with a type-safe shape. -/\n" ++
    s!"def load_{declName} (source : String := defaultSource) : IO (_root_.torch.T {declName}Shape) := do\n" ++
    "  let filePath : String :=\n" ++
    "    if sourceIsDirectory then\n" ++
    "      if " ++ declName ++ "Spec.sourceFile.isEmpty then source else source ++ \"/\" ++ " ++ declName ++ "Spec.sourceFile\n" ++
    "    else\n" ++
    "      source\n" ++
    s!"  _root_.torch.safetensors.loadTensor filePath {declName}Spec.name {declName}Shape\n"
  #[shapeDecl, specDecl, loadDecl]

private structure LeafInfo where
  declName : String
  tensorName : String
  dtype : String
  shape : Shape

private structure PathNode where
  leaf? : Option LeafInfo := none
  named : Array (String × PathNode) := #[]
  indexed : Array (Nat × PathNode) := #[]

private def emptyPathNode : PathNode := {}

private def upsertNamed (children : Array (String × PathNode)) (key : String) (child : PathNode)
    : Array (String × PathNode) :=
  let updated :=
    match children.findIdx? (fun pair => pair.1 == key) with
    | some i => children.set! i (key, child)
    | none => children.push (key, child)
  updated.qsort (fun a b => a.1 < b.1)

private def upsertIndexed (children : Array (Nat × PathNode)) (key : Nat) (child : PathNode)
    : Array (Nat × PathNode) :=
  let updated :=
    match children.findIdx? (fun pair => pair.1 == key) with
    | some i => children.set! i (key, child)
    | none => children.push (key, child)
  updated.qsort (fun a b => a.1 < b.1)

private partial def insertPath (node : PathNode) (segments : List String) (leaf : LeafInfo)
    : Except String PathNode := do
  match segments with
  | [] =>
      match node.leaf? with
      | some _ => .error s!"duplicate tensor path '{leaf.tensorName}'"
      | none => pure { node with leaf? := some leaf }
  | seg :: rest =>
      match seg.toNat? with
      | some idx =>
          let child := (node.indexed.findSome? fun (k, c) => if k == idx then some c else none).getD emptyPathNode
          let child' ← insertPath child rest leaf
          pure { node with indexed := upsertIndexed node.indexed idx child' }
      | none =>
          let child := (node.named.findSome? fun (k, c) => if k == seg then some c else none).getD emptyPathNode
          let child' ← insertPath child rest leaf
          pure { node with named := upsertNamed node.named seg child' }

private def namedChildren (node : PathNode) : Array (String × PathNode) := node.named
private def indexedChildren (node : PathNode) : Array (Nat × PathNode) := node.indexed

private def shapeFingerprint (shape : Shape) : String :=
  "[" ++ String.intercalate "," (shape.toList.map toString) ++ "]"

private partial def nodeFingerprint (node : PathNode) : String :=
  let leafPart :=
    match node.leaf? with
    | none => "N"
    | some leaf => s!"L({leaf.dtype}:{shapeFingerprint leaf.shape})"
  let namedPart :=
    String.intercalate "," <| (namedChildren node).toList.map fun (k, child) =>
      s!"{k}:{nodeFingerprint child}"
  let indexedPart :=
    String.intercalate "," <| (indexedChildren node).toList.map fun (i, child) =>
      s!"{i}:{nodeFingerprint child}"
  leafPart ++ "{" ++ namedPart ++ "}[" ++ indexedPart ++ "]"

private def pathLabel (path : List String) : String :=
  if path.isEmpty then "<root>" else String.intercalate "." path

private def validateIndexedGroup (path : List String) (children : Array (Nat × PathNode))
    : CommandElabM Unit := do
  let mut expected : Nat := 0
  for (idx, _) in children do
    if idx != expected then
      throwError
        "safetensors_type_provider: non-contiguous numeric path segment under '{pathLabel path}'. expected index {expected}, found {idx}."
    expected := expected + 1

  let mut firstFp? : Option String := none
  for (_, child) in children do
    let fp := nodeFingerprint child
    match firstFp? with
    | none => firstFp? := some fp
    | some firstFp =>
        if fp != firstFp then
          throwError
            "safetensors_type_provider: non-uniform indexed subtree under '{pathLabel path}'. All numeric siblings must share the same schema to form an indexed collection."

private inductive FieldSource where
  | value
  | named (segment : String)
  | indexed

private inductive TypeRepr where
  | leaf (declName : String)
  | array (elem : TypeRepr) (count : Nat)
  | struct (typeName : String) (fields : Array (String × FieldSource × TypeRepr))

private partial def TypeRepr.typeExpr : TypeRepr → String
  | .leaf declName => s!"_root_.torch.T {declName}Shape"
  | .array elem _ => s!"Array {elem.typeExpr}"
  | .struct typeName _ => typeName

private partial def buildTypeRepr
    (node : PathNode)
    (path : List String)
    (usedTypeNames : List String)
    : CommandElabM (TypeRepr × Array String × List String) := do
  let hasLeaf := node.leaf?.isSome
  let named := namedChildren node
  let indexed := indexedChildren node

  if hasLeaf && named.isEmpty && indexed.isEmpty then
    let some leaf := node.leaf?
      | throwError "safetensors_type_provider internal error: missing leaf at '{pathLabel path}'"
    return (.leaf leaf.declName, #[], usedTypeNames)

  if !hasLeaf && named.isEmpty && !indexed.isEmpty then
    validateIndexedGroup path indexed
    let some (_, firstChild) := indexed[0]?
      | throwError "safetensors_type_provider internal error: missing first indexed child"
    let (elemRepr, elemDecls, used') ← buildTypeRepr firstChild (path ++ ["item"]) usedTypeNames
    return (.array elemRepr indexed.size, elemDecls, used')

  let mut decls : Array String := #[]
  let mut used := usedTypeNames
  let mut fields : Array (String × FieldSource × TypeRepr) := #[]
  let mut usedFieldNames : List String := []

  if let some leaf := node.leaf? then
    let fieldName := freshName usedFieldNames "value"
    usedFieldNames := fieldName :: usedFieldNames
    fields := fields.push (fieldName, .value, .leaf leaf.declName)

  for (seg, child) in named do
    let baseField := sanitizeIdent (replaceInvalidChars seg |>.map Char.toLower) "field"
    let fieldName := freshName usedFieldNames baseField
    usedFieldNames := fieldName :: usedFieldNames
    let (childRepr, childDecls, used') ← buildTypeRepr child (path ++ [fieldName]) used
    used := used'
    decls := decls ++ childDecls
    fields := fields.push (fieldName, .named seg, childRepr)

  if !indexed.isEmpty then
    validateIndexedGroup (path ++ ["<index>"]) indexed
    let fieldName := freshName usedFieldNames "items"
    usedFieldNames := fieldName :: usedFieldNames
    let some (_, firstChild) := indexed[0]?
      | throwError "safetensors_type_provider internal error: missing first indexed child for struct field"
    let (elemRepr, elemDecls, used') ← buildTypeRepr firstChild (path ++ [fieldName]) used
    used := used'
    decls := decls ++ elemDecls
    fields := fields.push (fieldName, .indexed, .array elemRepr indexed.size)

  let typeName := freshName used (typeNameBase path)
  used := typeName :: used
  let fieldLines :=
    fields.map fun (fieldName, _, childRepr) => s!"  {fieldName} : {childRepr.typeExpr}"
  let fieldsBlock := if fieldLines.isEmpty then "" else String.intercalate "\n" fieldLines.toList ++ "\n"
  let structDecl :=
    s!"/-- Generated hierarchical checkpoint node `{typeName}`. -/\n" ++
    s!"structure {typeName} where\n" ++
    fieldsBlock ++
    "deriving Repr, Inhabited\n"
  decls := decls.push structDecl
  return (.struct typeName fields, decls, used)

private partial def buildLoadExpr
    (node : PathNode)
    (repr : TypeRepr)
    (sourceExpr : String)
    (hint : String)
    : CommandElabM String := do
  match repr with
  | .leaf _ =>
      let some leaf := node.leaf?
        | throwError "safetensors_type_provider internal error: missing leaf while building load expression"
      pure s!"load_{leaf.declName} {sourceExpr}"
  | .array elem expectedCount =>
      let indexed := indexedChildren node
      if indexed.size != expectedCount then
        throwError
          "safetensors_type_provider internal error: indexed collection size mismatch (expected {expectedCount}, got {indexed.size})."
      let mut lines : Array String := #["do"]
      let mut valueNames : Array String := #[]
      let mut i : Nat := 0
      for (idx, child) in indexed do
        let valueName := sanitizeIdent s!"{hint}_{idx}" s!"item_{i}"
        let childExpr ← buildLoadExpr child elem sourceExpr valueName
        lines := lines.push s!"  let {valueName} ←"
        lines := lines.push (indentBlock 4 childExpr)
        valueNames := valueNames.push valueName
        i := i + 1
      let arrayLiteral := "#[" ++ String.intercalate ", " valueNames.toList ++ "]"
      lines := lines.push s!"  pure {arrayLiteral}"
      pure <| String.intercalate "\n" lines.toList
  | .struct typeName fields =>
      let mut lines : Array String := #["do"]
      let mut assignments : Array String := #[]
      let mut i : Nat := 0
      for (fieldName, source, childRepr) in fields do
        let childNode ←
          match source with
          | .value => pure node
          | .named segment =>
              match node.named.findSome? (fun (k, c) => if k == segment then some c else none) with
              | some child => pure child
              | none =>
                  throwError
                    "safetensors_type_provider internal error: missing named child segment '{segment}' while building hierarchical checkpoint type."
          | .indexed => pure node
        let valueName := sanitizeIdent s!"{hint}_{fieldName}" s!"field_{i}"
        let childExpr ← buildLoadExpr childNode childRepr sourceExpr valueName
        lines := lines.push s!"  let {valueName} ←"
        lines := lines.push (indentBlock 4 childExpr)
        assignments := assignments.push s!"{fieldName} := {valueName}"
        i := i + 1
      let recordLiteral := "{" ++ String.intercalate ", " assignments.toList ++ "}"
      lines := lines.push s!"  pure ({recordLiteral} : {typeName})"
      pure <| String.intercalate "\n" lines.toList

@[command_elab safetensorsTypeProviderCmd]
def elabSafeTensorsTypeProvider : CommandElab
  | `(safetensors_type_provider $sourcePath:str as $ns:ident) => do
      let source := sourcePath.getString
      let discovered ←
        try
          liftIO <| introspect source
        catch _ =>
          throwError "safetensors_type_provider failed while introspecting source '{source}'."

      let namespaceName := ns.getId.toString
      elabGeneratedCommand s!"namespace {namespaceName}"
      try
        elabGeneratedCommand <|
          s!"/-- Provider default source path (file or directory). -/\n" ++
          s!"def defaultSource : String := {leanStringLit discovered.source}\n"
        elabGeneratedCommand <|
          s!"/-- True when `defaultSource` is a directory containing shard files. -/\n" ++
          s!"def sourceIsDirectory : Bool := {if discovered.sourceIsDirectory then "true" else "false"}\n"

        let mut usedDeclNames : List String := []
        let mut specNames : Array String := #[]
        let mut tensorFieldPairs : Array (String × String) := #[]
        let mut root : PathNode := emptyPathNode

        for tensor in discovered.tensors do
          let baseName := tensorDeclBaseName tensor.name
          let declName := freshName usedDeclNames baseName
          usedDeclNames := declName :: usedDeclNames
          specNames := specNames.push s!"{declName}Spec"
          tensorFieldPairs := tensorFieldPairs.push (declName, tensor.name)
          for decl in renderTensorDecls tensor declName do
            elabGeneratedCommand decl

          let segments := (tensor.name.splitOn ".").filter (fun s => !s.isEmpty)
          if segments.isEmpty then
            throwError "safetensors_type_provider found invalid empty tensor path."
          let leaf : LeafInfo := {
            declName := declName
            tensorName := tensor.name
            dtype := tensor.dtype
            shape := tensor.shape
          }
          match insertPath root segments leaf with
          | .ok root' => root := root'
          | .error err =>
              throwError "safetensors_type_provider failed while building hierarchy: {err}"

        if tensorFieldPairs.isEmpty then
          elabGeneratedCommand <|
            "/-- Typed checkpoint record generated by `safetensors_type_provider`. -/\n" ++
            "abbrev Weights := Unit\n"
          elabGeneratedCommand <|
            "/-- Load all tensors into a typed checkpoint record. -/\n" ++
            "def loadAll (source : String := defaultSource) : IO Weights :=\n" ++
            "  pure ()\n"
        else
          let (weightsRepr, weightsTypeDecls, _) ← buildTypeRepr root [] []
          for decl in weightsTypeDecls do
            elabGeneratedCommand decl
          if weightsRepr.typeExpr != "Weights" then
            elabGeneratedCommand s!"abbrev Weights := {weightsRepr.typeExpr}\n"
          let loadExpr ← buildLoadExpr root weightsRepr "source" "weights"
          let loadAllDecl :=
            "/-- Load all tensors into a typed checkpoint record. -/\n" ++
            "def loadAll (source : String := defaultSource) : IO Weights :=\n" ++
            indentBlock 2 loadExpr ++ "\n"
          elabGeneratedCommand loadAllDecl

        let fieldPairsLit := tensorFieldPairs.map fun (fieldName, tensorName) =>
          s!"({leanStringLit fieldName}, {leanStringLit tensorName})"
        elabGeneratedCommand <|
          "/-- Mapping from generated flat tensor identifiers to original tensor names. -/\n" ++
          "def fieldToTensorName : List (String × String) :=\n" ++
          "  [" ++ String.intercalate ", " fieldPairsLit.toList ++ "]\n"

        let schemaArray :=
          if specNames.isEmpty then
            "#[]"
          else
            "#[" ++ String.intercalate ", " specNames.toList ++ "]"
        elabGeneratedCommand <|
          s!"/-- All tensors discovered by the provider. -/\n" ++
          s!"def schema : Array _root_.torch.safetensors.TensorSchema := {schemaArray}\n"
        elabGeneratedCommand <|
          s!"/-- Number of discovered tensors. -/\n" ++
          s!"def tensorCount : Nat := {discovered.tensors.size}\n"
        elabGeneratedCommand <|
          "/-- Check whether a tensor exists in the generated schema. -/\n" ++
          "def hasTensor (tensorName : String) : Bool :=\n" ++
          "  schema.any (fun t => t.name == tensorName)\n"
        elabGeneratedCommand <|
          "/-- Look up tensor schema by tensor name. -/\n" ++
          "def find? (tensorName : String) : Option _root_.torch.safetensors.TensorSchema :=\n" ++
          "  schema.findSome? (fun t => if t.name == tensorName then some t else none)\n"
      finally
        elabGeneratedCommand s!"end {namespaceName}"
  | _ => throwUnsupportedSyntax

end torch.safetensors
