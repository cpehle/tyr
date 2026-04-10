import Tyr.GPU.Codegen.TileIR.Render

/-!
# Tyr.GPU.Codegen.TileIR.Toolchain

Driver utilities for compiling rendered TileIR through NVIDIA's public tools:

- `cuda-tile-opt`
- `cuda-tile-translate`
- `tileiras`
-/

namespace Tyr.GPU.Codegen.TileIR

open System

inductive OutputKind where
  | mlir
  | bytecode
  | cubin
  deriving Repr, Inhabited, BEq, DecidableEq

instance : ToString OutputKind where
  toString
    | .mlir => "mlir"
    | .bytecode => "bytecode"
    | .cubin => "cubin"

structure Toolchain where
  cudaTileOpt? : Option FilePath := none
  cudaTileTranslate? : Option FilePath := none
  tileiras? : Option FilePath := none
  searchDirs : Array FilePath := #[]
  deriving Repr, Inhabited

structure CompileOptions where
  gpuName : String := "sm_100"
  optLevel : Nat := 3
  lineInfo : Bool := true
  normalizeMlir : Bool := true
  outputKind : OutputKind := .cubin
  deriving Repr, Inhabited

structure ExternalCommand where
  executable : FilePath
  args : Array String
  deriving Repr, Inhabited

structure CompilationStep where
  stage : String
  command? : Option ExternalCommand := none
  output : FilePath
  deriving Repr, Inhabited

structure CompilationManifest where
  moduleName : String
  outputKind : OutputKind
  gpuName : String
  optLevel : Nat
  lineInfo : Bool
  normalizeMlir : Bool
  inputMlir : FilePath
  optimizedMlir : Option FilePath := none
  bytecode : Option FilePath := none
  cubin : Option FilePath := none
  steps : Array CompilationStep := #[]
  deriving Repr, Inhabited

structure LaunchGrid where
  x : Nat := 1
  y : Nat := 1
  z : Nat := 1
  deriving Repr, Inhabited, BEq, DecidableEq

structure LauncherSpec where
  moduleName : String
  entryName : String
  cubin : FilePath
  grid : LaunchGrid := {}
  block : LaunchGrid := {}
  sharedMemoryBytes : Nat := 0
  args : Array String := #[]
  deriving Repr, Inhabited

structure ArtifactPaths where
  inputMlir : FilePath
  optimizedMlir : FilePath
  bytecode : FilePath
  cubin : FilePath
  manifest : FilePath
  deriving Repr, Inhabited

namespace ArtifactPaths

def outputPath (paths : ArtifactPaths) (kind : OutputKind) : FilePath :=
  match kind with
  | .mlir => paths.inputMlir
  | .bytecode => paths.bytecode
  | .cubin => paths.cubin

end ArtifactPaths

inductive ToolError where
  | missingTool (tool : String) (searched : Array FilePath)
  | processFailed (command : String) (exitCode : UInt32) (stderr : String)
  | invalidConfig (message : String)
  deriving Repr, Inhabited

namespace ToolError

def message : ToolError → String
  | .missingTool tool searched =>
      let dirs :=
        if searched.isEmpty then
          "<none>"
        else
          String.intercalate ", " <| searched.toList.map FilePath.toString
      s!"Missing required tool '{tool}'. Searched: {dirs}"
  | .processFailed command exitCode stderr =>
      s!"Command failed (exit={exitCode}): {command}\n{stderr}"
  | .invalidConfig message =>
      message

end ToolError

private def searchSeparator : String :=
  if System.Platform.isWindows then ";" else ":"

private def splitSearchPath (pathValue : String) : Array FilePath :=
  (pathValue.splitOn searchSeparator).foldl
    (fun acc part =>
      let trimmed := part.trimAscii.toString
      if trimmed.isEmpty then acc else acc.push ⟨trimmed⟩)
    #[]

private def appendEnvBinDir (dirs : Array FilePath) (envName : String) : IO (Array FilePath) := do
  match (← IO.getEnv envName) with
  | some value =>
      let trimmed := value.trimAscii.toString
      if trimmed.isEmpty then
        pure dirs
      else
        pure (dirs.push (⟨trimmed⟩ / "bin"))
  | none =>
      pure dirs

private def findInDirs? (name : String) (dirs : Array FilePath) : IO (Option FilePath) := do
  for dir in dirs do
    let candidate := dir / name
    if ← candidate.pathExists then
      return some candidate
  return none

private def resolveTool? (envName : String) (binaryName : String) (dirs : Array FilePath) : IO (Option FilePath) := do
  match (← IO.getEnv envName) with
  | some explicit =>
      let trimmed := explicit.trimAscii.toString
      if trimmed.isEmpty then
        findInDirs? binaryName dirs
      else
        let path : FilePath := ⟨trimmed⟩
        if ← path.pathExists then
          pure (some path)
        else
          pure none
  | none =>
      findInDirs? binaryName dirs

/-- Detect NVIDIA TileIR tools from explicit env vars, `PATH`, and CUDA bin dirs. -/
def detectToolchain : IO Toolchain := do
  let pathDirs := splitSearchPath ((← IO.getEnv "PATH").getD "")
  let dirs ← appendEnvBinDir pathDirs "CUDA_HOME"
  let dirs ← appendEnvBinDir dirs "CUDA_PATH"
  let cudaTileOpt? ← resolveTool? "CUDA_TILE_OPT" "cuda-tile-opt" dirs
  let cudaTileTranslate? ← resolveTool? "CUDA_TILE_TRANSLATE" "cuda-tile-translate" dirs
  let tileiras? ← resolveTool? "TILEIRAS" "tileiras" dirs
  pure {
    cudaTileOpt?
    cudaTileTranslate?
    tileiras?
    searchDirs := dirs
  }

private def requireTool (name : String) (value : Option FilePath) (searched : Array FilePath)
    : Except ToolError FilePath :=
  match value with
  | some path => .ok path
  | none => .error <| .missingTool name searched

private def sanitizeStem (name : String) : String :=
  String.map (fun c => if c.isAlphanum then c else '_') name

/-- Deterministic artifact layout for a rendered TileIR module. -/
def artifactPaths (outDir : FilePath) (mod : Module) : ArtifactPaths :=
  let stem := sanitizeStem mod.name
  {
    inputMlir := outDir / s!"{stem}.mlir"
    optimizedMlir := outDir / s!"{stem}.opt.mlir"
    bytecode := outDir / s!"{stem}.tilebc"
    cubin := outDir / s!"{stem}.cubin"
    manifest := outDir / s!"{stem}.manifest.txt"
  }

def renderShellCommand (command : ExternalCommand) : String :=
  let parts := command.executable.toString :: command.args.toList
  String.intercalate " " parts

def buildOptCommand (toolchain : Toolchain) (input output : FilePath)
    : Except ToolError ExternalCommand := do
  let exe ← requireTool "cuda-tile-opt" toolchain.cudaTileOpt? toolchain.searchDirs
  pure {
    executable := exe
    args := #["-no-implicit-module", input.toString, "-o", output.toString]
  }

def buildTranslateCommand (toolchain : Toolchain) (input output : FilePath)
    : Except ToolError ExternalCommand := do
  let exe ← requireTool "cuda-tile-translate" toolchain.cudaTileTranslate? toolchain.searchDirs
  pure {
    executable := exe
    args := #["-mlir-to-cudatilebc", "-no-implicit-module", input.toString, "-o", output.toString]
  }

def buildTileirasCommand (toolchain : Toolchain) (input output : FilePath) (opts : CompileOptions)
    : Except ToolError ExternalCommand := do
  let exe ← requireTool "tileiras" toolchain.tileiras? toolchain.searchDirs
  let optLevel :=
    if opts.optLevel > 3 then 3 else opts.optLevel
  let args :=
    #[
      input.toString,
      "-o", output.toString,
      "--gpu-name", opts.gpuName,
      s!"-O{optLevel}"
    ] ++
    (if opts.lineInfo then #["--lineinfo"] else #[])
  pure { executable := exe, args := args }

private def runCommand (command : ExternalCommand) : IO (Except ToolError Unit) := do
  let result ← IO.Process.output {
    cmd := command.executable.toString
    args := command.args
  }
  if result.exitCode == 0 then
    pure (.ok ())
  else
    pure <| .error <|
      .processFailed (renderShellCommand command) result.exitCode result.stderr.trimAscii.toString

private def renderOptionalPath (path? : Option FilePath) : String :=
  match path? with
  | some path => path.toString
  | none => "<none>"

private def renderCompilationStep (step : CompilationStep) : String :=
  let command :=
    match step.command? with
    | some command => renderShellCommand command
    | none => "<none>"
  s!"- {step.stage}\n  output: {step.output}\n  command: {command}"

def renderCompilationManifest (manifest : CompilationManifest) : String :=
  let header :=
    s!"module: {manifest.moduleName}\n" ++
      s!"output-kind: {manifest.outputKind}\n" ++
      s!"gpu-name: {manifest.gpuName}\n" ++
      s!"opt-level: {manifest.optLevel}\n" ++
      s!"line-info: {manifest.lineInfo}\n" ++
      s!"normalize-mlir: {manifest.normalizeMlir}\n" ++
      s!"input-mlir: {manifest.inputMlir}\n" ++
      s!"optimized-mlir: {renderOptionalPath manifest.optimizedMlir}\n" ++
      s!"bytecode: {renderOptionalPath manifest.bytecode}\n" ++
      s!"cubin: {renderOptionalPath manifest.cubin}"
  let steps :=
    if manifest.steps.isEmpty then
      "\nsteps: []"
    else
      "\nsteps:\n" ++ String.intercalate "\n" (manifest.steps.toList.map renderCompilationStep)
  header ++ steps ++ "\n"

def renderLauncherSpec (spec : LauncherSpec) : String :=
  String.intercalate "\n" [
    s!"module: {spec.moduleName}",
    s!"entry: {spec.entryName}",
    s!"cubin: {spec.cubin}",
    s!"grid: ({spec.grid.x}, {spec.grid.y}, {spec.grid.z})",
    s!"block: ({spec.block.x}, {spec.block.y}, {spec.block.z})",
    s!"shared-memory-bytes: {spec.sharedMemoryBytes}",
    s!"args: [{String.intercalate ", " spec.args.toList}]"
  ] ++ "\n"

private partial def compilationSteps
    (paths : ArtifactPaths)
    (opts : CompileOptions)
    (toolchain : Toolchain)
    : Except ToolError (Array CompilationStep) := do
  match opts.outputKind with
  | .mlir =>
      pure #[
        {
          stage := "render"
          output := paths.inputMlir
        }
      ]
  | .bytecode =>
      let renderStep : CompilationStep :=
        {
          stage := "render"
          output := paths.inputMlir
        }
      if opts.normalizeMlir then
        let .ok optCmd := buildOptCommand toolchain paths.inputMlir paths.optimizedMlir
          | throw <| .invalidConfig "Unable to construct cuda-tile-opt command."
        let .ok translateCmd := buildTranslateCommand toolchain paths.optimizedMlir paths.bytecode
          | throw <| .invalidConfig "Unable to construct cuda-tile-translate command."
        pure #[
          renderStep,
          {
            stage := "normalize"
            command? := some optCmd
            output := paths.optimizedMlir
          },
          {
            stage := "bytecode"
            command? := some translateCmd
            output := paths.bytecode
          }
        ]
      else
        let .ok translateCmd := buildTranslateCommand toolchain paths.inputMlir paths.bytecode
          | throw <| .invalidConfig "Unable to construct cuda-tile-translate command."
        pure #[
          renderStep,
          {
            stage := "bytecode"
            command? := some translateCmd
            output := paths.bytecode
          }
        ]
  | .cubin =>
      let bytecodeOpts := { opts with outputKind := .bytecode }
      let .ok bytecodeSteps := compilationSteps paths bytecodeOpts toolchain
        | throw <| .invalidConfig "Unable to construct bytecode compilation plan."
      let .ok tileirasCmd := buildTileirasCommand toolchain paths.bytecode paths.cubin opts
        | throw <| .invalidConfig "Unable to construct tileiras command."
      pure <| bytecodeSteps.push {
        stage := "cubin"
        command? := some tileirasCmd
        output := paths.cubin
      }

private def buildCompilationManifest
    (mod : Module)
    (paths : ArtifactPaths)
    (opts : CompileOptions)
    (toolchain : Toolchain)
    : Except ToolError CompilationManifest := do
  let steps ← compilationSteps paths opts toolchain
  pure {
    moduleName := mod.name
    outputKind := opts.outputKind
    gpuName := opts.gpuName
    optLevel := opts.optLevel
    lineInfo := opts.lineInfo
    normalizeMlir := opts.normalizeMlir
    inputMlir := paths.inputMlir
    optimizedMlir := some paths.optimizedMlir
    bytecode := if opts.outputKind == .mlir then none else some paths.bytecode
    cubin := if opts.outputKind == .cubin then some paths.cubin else none
    steps := steps
  }

private def compileBytecodeArtifacts
    (toolchain : Toolchain)
    (paths : ArtifactPaths)
    (opts : CompileOptions)
    : IO (Except ToolError Unit) := do
  if opts.normalizeMlir then
    match buildOptCommand toolchain paths.inputMlir paths.optimizedMlir with
    | .error err => pure (.error err)
    | .ok command =>
        match ← runCommand command with
        | .error err => pure (.error err)
        | .ok _ =>
            match buildTranslateCommand toolchain paths.optimizedMlir paths.bytecode with
            | .error err => pure (.error err)
            | .ok translateCmd =>
                match ← runCommand translateCmd with
                | .error err => pure (.error err)
                | .ok _ => pure (.ok ())
  else
    match buildTranslateCommand toolchain paths.inputMlir paths.bytecode with
    | .error err => pure (.error err)
    | .ok command =>
        match ← runCommand command with
        | .error err => pure (.error err)
        | .ok _ => pure (.ok ())

/-- Render a module and optionally compile it as far as MLIR, TileIR bytecode, or cubin output. -/
partial def compileModuleAt
    (mod : Module)
    (outDir : FilePath)
    (opts : CompileOptions := {})
    (toolchain? : Option Toolchain := none)
    : IO (Except ToolError ArtifactPaths) := do
  let paths := artifactPaths outDir mod
  IO.FS.createDirAll outDir
  IO.FS.writeFile paths.inputMlir (renderModule mod)
  match opts.outputKind with
  | .mlir =>
      let manifest : CompilationManifest := {
        moduleName := mod.name
        outputKind := .mlir
        gpuName := opts.gpuName
        optLevel := opts.optLevel
        lineInfo := opts.lineInfo
        normalizeMlir := opts.normalizeMlir
        inputMlir := paths.inputMlir
        optimizedMlir := some paths.optimizedMlir
        bytecode := none
        cubin := none
        steps := #[
          {
            stage := "render"
            output := paths.inputMlir
          }
        ]
      }
      IO.FS.writeFile paths.manifest (renderCompilationManifest manifest)
      pure (.ok paths)
  | .bytecode =>
      let toolchain ← match toolchain? with
        | some tc => pure tc
        | none => detectToolchain
      match buildCompilationManifest mod paths opts toolchain with
      | .error err =>
          pure (.error err)
      | .ok manifest =>
          IO.FS.writeFile paths.manifest (renderCompilationManifest manifest)
          match ← compileBytecodeArtifacts toolchain paths opts with
          | .error err => pure (.error err)
          | .ok _ => pure (.ok paths)
  | .cubin =>
      let toolchain ← match toolchain? with
        | some tc => pure tc
        | none => detectToolchain
      match buildCompilationManifest mod paths opts toolchain with
      | .error err =>
          pure (.error err)
      | .ok manifest =>
          IO.FS.writeFile paths.manifest (renderCompilationManifest manifest)
          let bytecodeOpts := { opts with outputKind := .bytecode }
          match ← compileBytecodeArtifacts toolchain paths bytecodeOpts with
          | .error err => pure (.error err)
          | .ok _ =>
              match buildTileirasCommand toolchain paths.bytecode paths.cubin opts with
              | .error err => pure (.error err)
              | .ok command =>
                  match ← runCommand command with
                  | .error err => pure (.error err)
                  | .ok _ => pure (.ok paths)

instance : ToString ToolError where
  toString := ToolError.message

end Tyr.GPU.Codegen.TileIR
