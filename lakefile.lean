import Lake
open Lake DSL
open System (FilePath)

package tyr where
  srcDir := "."
  buildDir := ".lake/build"
  moreServerArgs := #["-Dpp.unicode.fun=true"]

require LeanTest from "../LeanTest"

/-! ## Platform Detection

Lake doesn't expose direct platform detection, so we use conditional compilation
based on the target triple. For now, we detect macOS vs Linux at build time.
-/

/-- Check if we're on macOS by checking for typical macOS paths -/
def isMacOS : IO Bool := do
  -- Check if Homebrew libomp exists (macOS indicator)
  let ompPath : FilePath := "/opt/homebrew/opt/libomp/lib"
  return ← ompPath.pathExists

/-- OpenMP library path - macOS uses Homebrew, Linux uses system path -/
def getOmpLibPath : IO FilePath := do
  if ← isMacOS then
    return "/opt/homebrew/opt/libomp/lib"
  else
    return "/usr/lib"

/-! ## C++ Library Build -/

/-- External library target for the C++ bindings.
    This wraps the Makefile build for now - a future enhancement could
    use Lake's native C++ compilation. -/
extern_lib libtyr pkg := do
  let tyrCLib := pkg.dir / "cc" / "build" / "libTyrC.a"

  -- Watch Makefile and key source files
  -- Note: For full dependency tracking, we'd need to list all .cpp/.h files,
  -- but Lake's API makes this complex. The Makefile handles internal deps.
  let srcJob ← inputTextFile <| pkg.dir / "cc" / "Makefile"

  buildFileAfterDep tyrCLib srcJob fun _srcFile => do
    let sysroot ← getLeanSysroot
    proc {
      cmd := "make"
      args := #["-C", (pkg.dir / "cc").toString]
      env := #[("LEAN_HOME", sysroot.toString)]
    }

/-! ## Common Link Arguments

These are platform-aware at runtime. On macOS, we use @executable_path for rpath.
On Linux, we use $ORIGIN. The actual platform is determined by the linker.
-/

/-- Common linker arguments for all executables.
    Note: rpath uses macOS syntax - Linux builds may need adjustment. -/
def commonLinkArgs : Array String := #[
  -- Tyr C++ library
  "-Lcc/build", "-lTyrC",
  -- LibTorch
  "-Lexternal/libtorch/lib",
  "-ltorch", "-ltorch_cpu", "-lc10",
  -- OpenMP (try both common locations)
  "-L/opt/homebrew/opt/libomp/lib",  -- macOS Homebrew
  "-L/usr/lib",                       -- Linux fallback
  "-lomp",
  -- Runtime library paths (macOS style - Linux uses different syntax)
  "-Wl,-rpath,@executable_path/../../external/libtorch/lib",
  "-Wl,-rpath,/opt/homebrew/opt/libomp/lib"
]

/-- Link arguments for Linux builds (use when building on Linux) -/
def linuxLinkArgs : Array String := #[
  "-Lcc/build", "-lTyrC",
  "-Lexternal/libtorch/lib",
  "-ltorch", "-ltorch_cpu", "-lc10",
  "-L/usr/lib", "-lomp",
  "-Wl,-rpath,$ORIGIN/../../external/libtorch/lib"
]

/-! ## Lean Library -/

/-- Main Lean library containing all Tyr modules -/
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  -- Disable precompilation to avoid needing dylib at compile time
  precompileModules := false

/-- Test library containing all tests -/
lean_lib Tests where
  roots := #[`Tests.Test, `Tests.TestDiffusion, `Tests.TestDataLoader, `Tests.TestModdedGPT]
  precompileModules := false

/-! ## Executables -/

/-- Main test runner using LeanTest -/
@[test_driver]
lean_exe test_runner where
  root := `Tests.RunTests
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Memory test executable -/
lean_exe memtest where
  root := `MemoryTest
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPT training executable -/
lean_exe TrainGPT where
  root := `Examples.TrainGPT
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Diffusion training executable -/
lean_exe TrainDiffusion where
  root := `Examples.TrainDiffusion
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Diffusion tests executable -/
lean_exe TestDiffusion where
  root := `Tests.TestDiffusion
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- DataLoader test executable -/
lean_exe TestDataLoader where
  root := `Tests.TestDataLoader
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-! ## Scripts -/

/-- Script to run the test executable with proper environment.
    Usage: lake script run -/
script run do
  let rootPath := (← getWorkspace).root.dir
  let exe := rootPath / ".lake" / "build" / "bin" / "test_runner"

  -- Build library paths
  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let leanLibPath ← getLeanSysroot
  let leanLib := leanLibPath / "lib" / "lean"

  let onMac ← isMacOS
  let ompPath ← getOmpLibPath
  let libPath := s!"{libtorchPath}:{ompPath}:{leanLib}"
  let libEnvVar := if onMac then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"

  IO.println s!"Running: {exe}"
  IO.println s!"{libEnvVar}={libPath}"
  IO.println ""
  IO.println s!"To run manually:"
  IO.println s!"export {libEnvVar}=\"{libPath}\""
  IO.println s!"\"{exe}\""

  return 0

/-- Script to run TrainGPT with proper environment.
    Usage: lake script train -/
script train do
  let rootPath := (← getWorkspace).root.dir
  let exe := rootPath / ".lake" / "build" / "bin" / "TrainGPT"

  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let leanLibPath ← getLeanSysroot
  let leanLib := leanLibPath / "lib" / "lean"

  let onMac ← isMacOS
  let ompPath ← getOmpLibPath
  let libPath := s!"{libtorchPath}:{ompPath}:{leanLib}"
  let libEnvVar := if onMac then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"

  IO.println s!"To run training:"
  IO.println s!"export {libEnvVar}=\"{libPath}\""
  IO.println s!"\"{exe}\""

  return 0
