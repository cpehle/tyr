import Lake
open Lake DSL
open System (FilePath)

def packageLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet",
      "-Wl,-rpath,@loader_path/../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/usr/lib", "-lgomp", "-lstdc++",
      "-larrow", "-lparquet",
      "-Wl,-rpath,$ORIGIN/../../external/libtorch/lib"
    ]

def commonLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      "-Lcc/build", "-lTyrC",
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet",
      "-Wl,-rpath,@executable_path/../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[
      "-Lcc/build", "-lTyrC",
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/usr/lib", "-lgomp", "-lstdc++",
      "-larrow", "-lparquet",
      "-Wl,-rpath,$ORIGIN/../../external/libtorch/lib"
    ]

package tyr where
  srcDir := "."
  buildDir := ".lake/build"
  moreServerArgs := #["-Dpp.unicode.fun=true"]
  -- Link arguments for extern_lib shared library
  moreLinkArgs := packageLinkArgs

require LeanTest from git "https://github.com/cpehle/lean_test.git" @ "b42cd3d78716e5a2de5b640ac82d7fe3f05f2a4c"

/-! ## Platform Detection

Use `System.Platform` for compile-time platform-specific link arguments and
runtime environment setup in scripts.
-/

/-- Check if we're on macOS. -/
def isMacOS : Bool :=
  System.Platform.isOSX

/-- OpenMP library path - macOS uses Homebrew, Linux uses system path -/
def getOmpLibPath : IO FilePath := do
  if isMacOS then
    let armPath : FilePath := "/opt/homebrew/opt/libomp/lib"
    if ← armPath.pathExists then
      return armPath
    let intelPath : FilePath := "/usr/local/opt/libomp/lib"
    if ← intelPath.pathExists then
      return intelPath
    return armPath
  else
    match (← IO.getEnv "EBROOTGCCCORE") with
    | some root =>
      let p : FilePath := root / "lib64"
      if (← p.pathExists) then
        return p
      else
        return "/usr/lib"
    | none =>
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

/-! ## Lean Library -/

/-- Main Lean library containing all Tyr modules -/
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  precompileModules := true

/-- Test library containing all tests -/
lean_lib Tests where
  roots := #[`Tests]
  precompileModules := false

/-- Experimental tests that track in-progress modules. -/
lean_lib TestsExperimental where
  roots := #[`TestsExperimental]
  precompileModules := false

/-- Examples library -/
lean_lib Examples where
  roots := #[`Examples]
  precompileModules := false

/-! ## Executables -/

/-- Main test runner using LeanTest -/
@[test_driver]
lean_exe test_runner where
  root := `Tests.RunTests
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Generate CUDA translation units from registered @[gpu_kernel] declarations. -/
lean_exe GenerateGpuKernels where
  root := `Tyr.GPU.Codegen.GenerateMain
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Experimental test runner for unstable/in-progress modules. -/
lean_exe test_runner_experimental where
  root := `Tests.RunTestsExperimental
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

/-- NanoChat training executable (modded GPT + distributed) -/
lean_exe TrainNanoChat where
  root := `Examples.NanoChat.TrainNanoChat
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NanoChat multi-stage pipeline executable. -/
lean_exe NanoChatPipeline where
  root := `Examples.NanoChat.Pipeline
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- NanoChat checkpoint-backed chat/inference executable. -/
lean_exe NanoChatChat where
  root := `Examples.NanoChat.RunChat
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

/-- Flux image generation demo -/
lean_exe FluxDemo where
  root := `Examples.Flux.FluxDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Flux debug harness (saves intermediate tensors) -/
lean_exe FluxDebug where
  root := `Examples.Flux.FluxDebug
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end demo for a minimal ThunderKittens-style copy kernel. -/
lean_exe RunCopy where
  root := `Examples.GPU.RunCopy
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end rotary fixture validation using a ThunderKittens-style kernel. -/
lean_exe RunRotary where
  root := `Examples.GPU.RunRotary
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens layernorm fixture validation. -/
lean_exe RunLayerNorm where
  root := `Examples.GPU.RunLayerNorm
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens flash attention fixture validation. -/
lean_exe RunFlashAttn where
  root := `Examples.GPU.RunFlashAttn
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end ThunderKittens `mha_h100` forward/backward fixture validation. -/
lean_exe RunMhaH100 where
  root := `Examples.GPU.RunMhaH100
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end `mha_h100` training/benchmark demo (kernel + optional torch baseline). -/
lean_exe RunMhaH100Train where
  root := `Examples.GPU.RunMhaH100Train
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end multi-block `mha_h100` validation (`seq=768`, `d=64`). -/
lean_exe RunMhaH100Seq768 where
  root := `Examples.GPU.RunMhaH100Seq768
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-! ## Scripts -/

/-- Script to run the test executable with proper environment.
    Usage: lake run -/
script run (args) do
  let rootPath := (← getWorkspace).root.dir
  let exe := rootPath / ".lake" / "build" / "bin" / "test_runner"

  let tyrCLib := rootPath / "cc" / "build"
  let lakeLib := rootPath / ".lake" / "build" / "lib"
  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let leanLibPath ← getLeanSysroot
  let leanLib := leanLibPath / "lib" / "lean"
  let ompPath ← getOmpLibPath
  let libEnvVar := if isMacOS then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"
  -- Prepend our runtime deps but keep the user's existing env (e.g. module-provided libstdc++).
  -- Prefer the EasyBuild GCCcore runtime if present, since the system libstdc++ may be too old.
  let gccCoreLibPath? ← do
    match (← IO.getEnv "EBROOTGCCCORE") with
    | none => pure none
    | some root =>
      let p : FilePath := root / "lib64"
      if (← p.pathExists) then
        pure (some p)
      else
        pure none
  let arrowLibPath? ← do
    match (← IO.getEnv "EBROOTARROW") with
    | none => pure none
    | some root =>
      let p : FilePath := root / "lib"
      if (← p.pathExists) then
        pure (some p)
      else
        let p64 : FilePath := root / "lib64"
        if (← p64.pathExists) then
          pure (some p64)
        else
          pure none
  let inheritedLibPath := (← IO.getEnv libEnvVar)
  let baseLibPath := s!"{tyrCLib}:{lakeLib}:{libtorchPath}:{ompPath}:{leanLib}"
  let baseLibPath :=
    match arrowLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  let libPathPrefix :=
    match gccCoreLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  let libPath :=
    match inheritedLibPath with
    | some v => s!"{libPathPrefix}:{v}"
    | none => libPathPrefix

  let child ← IO.Process.spawn {
    cmd := exe.toString
    args := args.toArray
    env := #[(libEnvVar, some libPath)]
    stdin := .inherit
    stdout := .inherit
    stderr := .inherit
  }
  return ← child.wait

/-- Script to run TrainGPT with proper environment.
    Usage: lake run train -/
script train (args) do
  let rootPath := (← getWorkspace).root.dir
  let exe := rootPath / ".lake" / "build" / "bin" / "TrainGPT"

  let tyrCLib := rootPath / "cc" / "build"
  let lakeLib := rootPath / ".lake" / "build" / "lib"
  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let leanLibPath ← getLeanSysroot
  let leanLib := leanLibPath / "lib" / "lean"

  let ompPath ← getOmpLibPath
  let libEnvVar := if isMacOS then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH"
  -- Prepend our runtime deps but keep the user's existing env (e.g. module-provided libstdc++).
  -- Prefer the EasyBuild GCCcore runtime if present, since the system libstdc++ may be too old.
  let gccCoreLibPath? ← do
    match (← IO.getEnv "EBROOTGCCCORE") with
    | none => pure none
    | some root =>
      let p : FilePath := root / "lib64"
      if (← p.pathExists) then
        pure (some p)
      else
        pure none
  let arrowLibPath? ← do
    match (← IO.getEnv "EBROOTARROW") with
    | none => pure none
    | some root =>
      let p : FilePath := root / "lib"
      if (← p.pathExists) then
        pure (some p)
      else
        let p64 : FilePath := root / "lib64"
        if (← p64.pathExists) then
          pure (some p64)
        else
          pure none
  let inheritedLibPath := (← IO.getEnv libEnvVar)
  let baseLibPath := s!"{tyrCLib}:{lakeLib}:{libtorchPath}:{ompPath}:{leanLib}"
  let baseLibPath :=
    match arrowLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  let libPathPrefix :=
    match gccCoreLibPath? with
    | some p => s!"{baseLibPath}:{p}"
    | none => baseLibPath
  let libPath :=
    match inheritedLibPath with
    | some v => s!"{libPathPrefix}:{v}"
    | none => libPathPrefix

  let child ← IO.Process.spawn {
    cmd := exe.toString
    args := args.toArray
    env := #[(libEnvVar, some libPath)]
    stdin := .inherit
    stdout := .inherit
    stderr := .inherit
  }
  return ← child.wait
