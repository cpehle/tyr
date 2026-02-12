import Lean
import Tyr.GPU.Codegen.FFI

open Lean

namespace Tyr.GPU.Codegen

structure CliConfig where
  outDir : System.FilePath := ⟨"cc/src/generated"⟩
  clean : Bool := true
  modules : Array Name := #[]
  deriving Inhabited

def usage : String :=
  String.intercalate "\n" [
    "Usage: lake exe GenerateGpuKernels [OPTIONS] <Module> [<Module> ...]",
    "",
    "Positional arguments:",
    "  <Module>                 Lean module to import (e.g. Tyr.GPU.Kernels.TestNewStyle)",
    "",
    "Options:",
    "  --out-dir <path>         Output directory for generated .cu files (default: cc/src/generated)",
    "  --no-clean               Keep existing generated .cu files in output directory",
    "  --help                   Show this help"
  ]

def parseModuleName (s : String) : Except String Name := do
  let parts := (s.splitOn ".").filter (!·.isEmpty)
  if parts.isEmpty then
    throw s!"Invalid module name '{s}'"
  return parts.foldl (init := Name.anonymous) fun n p => Name.str n p

partial def parseArgs (cfg : CliConfig) : List String → Except String CliConfig
  | [] =>
    if cfg.modules.isEmpty then
      throw "At least one module is required."
    else
      return cfg
  | "--out-dir" :: dir :: rest =>
    parseArgs { cfg with outDir := ⟨dir⟩ } rest
  | "--out-dir" :: [] =>
    throw "--out-dir expects a path argument."
  | "--no-clean" :: rest =>
    parseArgs { cfg with clean := false } rest
  | "--help" :: _ =>
    return cfg
  | arg :: rest =>
    if arg.startsWith "-" then
      throw s!"Unknown option '{arg}'"
    else
      match parseModuleName arg with
      | .error err => throw err
      | .ok moduleName =>
        parseArgs { cfg with modules := cfg.modules.push moduleName } rest

unsafe def main (args : List String) : IO UInt32 := do
  if args.contains "--help" then
    IO.println usage
    return (0 : UInt32)

  match parseArgs {} args with
  | .error err =>
    IO.eprintln err
    IO.eprintln ""
    IO.eprintln usage
    return (1 : UInt32)
  | .ok cfg =>
    try
      Lean.initSearchPath (← Lean.findSysroot)
      -- Match LeanTest-style driver behavior: extension initializers must run
      -- so imported modules can register entries during import.
      Lean.enableInitializersExecution
      let imports :=
        #[({ module := `Tyr.GPU.Codegen.Attribute } : Import)] ++
        cfg.modules.map (fun m => ({ module := m } : Import))
      let env ← Lean.importModules imports {} (loadExts := true)
      let regs ← collectRegisteredKernelsFromEnvModules env cfg.modules
      let written ← writeKernelCudaUnitsByModuleFrom regs cfg.outDir (clean := cfg.clean)
      IO.println s!"Generated {written.size} CUDA translation unit(s) in {cfg.outDir}"
      for path in written do
        IO.println s!"  {path}"
      return (0 : UInt32)
    catch e =>
      IO.eprintln s!"Kernel generation failed: {e.toString}"
      return (1 : UInt32)

end Tyr.GPU.Codegen

unsafe def main : List String → IO UInt32 :=
  Tyr.GPU.Codegen.main
