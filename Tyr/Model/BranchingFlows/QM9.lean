import Lean.Data.Json
import Tyr.Model.BranchingFlows.Molecule

/-!
  Preprocessed QM9 molecule records for BranchingFlows.

  The paper-specific chemistry preprocessing should stay outside Lean for now:
  RDKit/OpenBabel can canonicalize atom order and emit a compact JSON/JSONL
  artifact.  This module defines that artifact boundary and turns each molecule
  into the native `BranchingState MoleculeAtom` representation used by the
  bridge, sampler, and training packer.
-/

namespace torch.branching

open Lean

namespace Vec3

def isFinite (v : Vec3) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y && Float.isFinite v.z

end Vec3

structure QM9AtomRecord where
  coord : Vec3
  label : Nat
  deriving Repr, BEq, Inhabited

structure QM9MoleculeRecord where
  atoms : Array QM9AtomRecord
  name? : Option String := none
  smiles? : Option String := none
  deriving Repr, Inhabited

/-- Validation and state-bookkeeping knobs for terminal QM9 records. -/
structure QM9StateConfig where
  group : Int := 0
  firstId : Int := 1
  vocabSize? : Option Nat := none
  maskToken? : Option Nat := none
  allowMaskLabel : Bool := false
  deriving Repr, Inhabited

namespace QM9AtomRecord

def toMoleculeAtom (atom : QM9AtomRecord) : MoleculeAtom :=
  { coord := atom.coord, label := atom.label }

def validate (cfg : QM9StateConfig) (atom : QM9AtomRecord) : Except String Unit := do
  if !atom.coord.isFinite then
    throw s!"non-finite coordinate {reprStr atom.coord}"
  match cfg.vocabSize? with
  | some vocabSize =>
      if atom.label >= vocabSize then
        throw s!"label {atom.label} is outside vocabulary size {vocabSize}"
  | none => pure ()
  match cfg.maskToken? with
  | some maskToken =>
      if !cfg.allowMaskLabel && atom.label == maskToken then
        throw s!"terminal molecule label uses reserved mask token {maskToken}"
  | none => pure ()

private def getFloatField (j : Json) (field : String) : Except String Float :=
  match j.getObjValAs? Float field with
  | .ok x => pure x
  | .error e => throw s!"missing/invalid Float field '{field}': {e}"

private def getNatField (j : Json) (field : String) : Except String Nat :=
  match j.getObjValAs? Nat field with
  | .ok n => pure n
  | .error _ =>
      match j.getObjValAs? Int field with
      | .ok i =>
          if i < 0 then
            throw s!"field '{field}' must be non-negative, got {i}"
          else
            pure i.toNat
      | .error e => throw s!"missing/invalid Nat field '{field}': {e}"

private def getFirstNatField (j : Json) (fields : Array String) : Except String Nat := do
  let mut lastError := ""
  for field in fields do
    match getNatField j field with
    | .ok n => return n
    | .error e => lastError := e
  throw s!"expected one of Nat fields {fields}; last error: {lastError}"

private def coordFromJson (j : Json) : Except String Vec3 := do
  match (j.getObjVal? "coord").toOption with
  | some coordJson =>
      match (fromJson? coordJson : Except String (Array Float)) with
      | .ok coord =>
          if coord.size != 3 then
            throw s!"coord must have length 3, got {coord.size}"
          let v : Vec3 := { x := coord[0]!, y := coord[1]!, z := coord[2]! }
          if !v.isFinite then
            throw s!"coord contains non-finite value {reprStr v}"
          pure v
      | .error e => throw s!"invalid coord array: {e}"
  | none =>
      let v : Vec3 := {
        x := ← getFloatField j "x"
        y := ← getFloatField j "y"
        z := ← getFloatField j "z"
      }
      if !v.isFinite then
        throw s!"coord contains non-finite value {reprStr v}"
      pure v

/--
Parse one preprocessed atom.

Accepted atom forms:

```
{"label": 6, "x": 0.0, "y": 0.1, "z": -0.2}
{"label": 6, "coord": [0.0, 0.1, -0.2]}
```

`atom_label` is also accepted as a label-field alias for preprocessing scripts.
-/
def fromJson (j : Json) : Except String QM9AtomRecord := do
  let coord ← coordFromJson j
  let label ← getFirstNatField j #["label", "atom_label"]
  pure { coord, label }

end QM9AtomRecord

namespace QM9MoleculeRecord

def validate (cfg : QM9StateConfig) (mol : QM9MoleculeRecord) : Except String Unit := do
  if mol.atoms.isEmpty then
    throw "molecule must contain at least one atom"
  for i in [:mol.atoms.size] do
    match QM9AtomRecord.validate cfg mol.atoms[i]! with
    | .ok _ => pure ()
    | .error e => throw s!"atoms[{i}]: {e}"

def toBranchingState
    (mol : QM9MoleculeRecord)
    (cfg : QM9StateConfig := {}) :
    Except String (BranchingState MoleculeAtom) := do
  mol.validate cfg
  let state := mol.atoms.map (fun atom => atom.toMoleculeAtom)
  let n := state.size
  pure {
    state
    groupings := Array.replicate n cfg.group
    del := Array.replicate n false
    ids := (Array.range n).map (fun i => cfg.firstId + Int.ofNat i)
    branchmask := Array.replicate n true
    flowmask := Array.replicate n true
    padmask := Array.replicate n true
  }

private def optionalString (j : Json) (field : String) : Option String :=
  (j.getObjValAs? String field).toOption

private def parseAtoms (atomsJson : Array Json) : Except String (Array QM9AtomRecord) := do
  let mut atoms : Array QM9AtomRecord := #[]
  for i in [:atomsJson.size] do
    match QM9AtomRecord.fromJson atomsJson[i]! with
    | .ok atom => atoms := atoms.push atom
    | .error e => throw s!"atoms[{i}]: {e}"
  pure atoms

/--
Parse one preprocessed molecule object.

The required field is `atoms`; optional metadata fields are `name`, `smiles`,
and `canonical_smiles`.
-/
def fromJson (j : Json) : Except String QM9MoleculeRecord := do
  let atomsJson ←
    match j.getObjValAs? (Array Json) "atoms" with
    | .ok atomsJson => pure atomsJson
    | .error e => throw s!"missing/invalid atoms array: {e}"
  let atoms ← parseAtoms atomsJson
  let smiles? := (optionalString j "smiles").orElse fun _ => optionalString j "canonical_smiles"
  pure {
    atoms
    name? := optionalString j "name"
    smiles?
  }

end QM9MoleculeRecord

private def parseMoleculeArray (molecules : Array Json) :
    Except String (Array QM9MoleculeRecord) := do
  let mut out : Array QM9MoleculeRecord := #[]
  for i in [:molecules.size] do
    match QM9MoleculeRecord.fromJson molecules[i]! with
    | .ok mol => out := out.push mol
    | .error e => throw s!"molecules[{i}]: {e}"
  pure out

/-- Parse exactly one preprocessed molecule JSON object. -/
def parseQM9MoleculeJson (raw : String) : Except String QM9MoleculeRecord := do
  match Json.parse raw with
  | .ok json => QM9MoleculeRecord.fromJson json
  | .error e => throw s!"JSON parse failed: {e}"

/--
Parse a JSON dataset.  The top level may be:

- one molecule object,
- an array of molecule objects, or
- an object with a `molecules` array.
-/
def parseQM9MoleculeDatasetJson (raw : String) : Except String (Array QM9MoleculeRecord) := do
  let json ←
    match Json.parse raw with
    | .ok json => pure json
    | .error e => throw s!"JSON parse failed: {e}"
  match json with
  | .arr molecules => parseMoleculeArray molecules
  | .obj _ =>
      match (json.getObjValAs? (Array Json) "molecules").toOption with
      | some molecules => parseMoleculeArray molecules
      | none => pure #[← QM9MoleculeRecord.fromJson json]
  | _ => throw "expected molecule object, molecule array, or object with molecules array"

/-- Parse newline-delimited preprocessed molecule objects. -/
def parseQM9MoleculeJsonl (raw : String) : Except String (Array QM9MoleculeRecord) := do
  let mut out : Array QM9MoleculeRecord := #[]
  let mut lineNo := 0
  for rawLine in raw.splitOn "\n" do
    lineNo := lineNo + 1
    let line := rawLine.trimAscii.toString
    if line.isEmpty then
      continue
    match parseQM9MoleculeJson line with
    | .ok mol => out := out.push mol
    | .error e => throw s!"line {lineNo}: {e}"
  pure out

def qm9RecordsToBranchingStates
    (records : Array QM9MoleculeRecord)
    (cfg : QM9StateConfig := {}) :
    Except String (Array (BranchingState MoleculeAtom)) := do
  let mut out : Array (BranchingState MoleculeAtom) := #[]
  for i in [:records.size] do
    match records[i]!.toBranchingState cfg with
    | .ok state => out := out.push state
    | .error e => throw s!"molecules[{i}]: {e}"
  pure out

/-- Length-one masked source state used by the QM9 generation setup. -/
def qm9InitialMaskedState
    (cfg : MoleculeBridgeConfig)
    (group : Int := 0)
    (coord : Vec3 := Vec3.zero) :
    BranchingState MoleculeAtom :=
  BranchingState.mkDefault #[cfg.maskedAtom coord] #[group]

/--
Default XYZ symbol mapping for preprocessed records whose labels are atomic
numbers.  Token-id vocabularies can pass their own `labelToSymbol` function to
`moleculeStateToXYZ`.
-/
def qm9AtomicNumberSymbol (label : Nat) : String :=
  match label with
  | 1 => "H"
  | 5 => "B"
  | 6 => "C"
  | 7 => "N"
  | 8 => "O"
  | 9 => "F"
  | _ => "X"

def labelSymbolFromArray (symbols : Array String) (fallback : String := "X") (label : Nat) : String :=
  symbols.getD label fallback

def moleculeAtomXYZLine
    (atom : MoleculeAtom)
    (labelToSymbol : Nat → String := qm9AtomicNumberSymbol) : String :=
  let c := atom.coord
  s!"{labelToSymbol atom.label} {c.x} {c.y} {c.z}"

def moleculeStateToXYZ
    (state : BranchingState MoleculeAtom)
    (comment : String := "generated by Tyr BranchingFlows")
    (labelToSymbol : Nat → String := qm9AtomicNumberSymbol) : String :=
  let lines := state.state.map (fun atom => moleculeAtomXYZLine atom labelToSymbol)
  String.intercalate "\n" ((toString state.state.size) :: comment :: lines.toList) ++ "\n"

def writeMoleculeXYZ
    (path : System.FilePath)
    (state : BranchingState MoleculeAtom)
    (comment : String := "generated by Tyr BranchingFlows")
    (labelToSymbol : Nat → String := qm9AtomicNumberSymbol) : IO Unit :=
  IO.FS.writeFile path (moleculeStateToXYZ state comment labelToSymbol)

def loadQM9MoleculeJson (path : System.FilePath) : IO QM9MoleculeRecord := do
  let raw ← IO.FS.readFile path
  match parseQM9MoleculeJson raw with
  | .ok mol => pure mol
  | .error e => throw (IO.userError s!"{path}: {e}")

def loadQM9MoleculeDatasetJson (path : System.FilePath) : IO (Array QM9MoleculeRecord) := do
  let raw ← IO.FS.readFile path
  match parseQM9MoleculeDatasetJson raw with
  | .ok mols => pure mols
  | .error e => throw (IO.userError s!"{path}: {e}")

def loadQM9MoleculeJsonl (path : System.FilePath) : IO (Array QM9MoleculeRecord) := do
  let raw ← IO.FS.readFile path
  match parseQM9MoleculeJsonl raw with
  | .ok mols => pure mols
  | .error e => throw (IO.userError s!"{path}: {e}")

end torch.branching
