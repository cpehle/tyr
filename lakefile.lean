import Lake
open Lake DSL
open System (FilePath)

def linuxSystemLinkDirs : Array String :=
  #[
    "-L/usr/lib/x86_64-linux-gnu",
    "-L/lib/x86_64-linux-gnu",
    "-L/usr/lib/gcc/x86_64-linux-gnu/13",
    "-L/usr/lib/gcc/x86_64-linux-gnu/14",
    "-L/usr/lib/aarch64-linux-gnu",
    "-L/lib/aarch64-linux-gnu",
    "-L/usr/lib/gcc/aarch64-linux-gnu/13",
    "-L/usr/lib/gcc/aarch64-linux-gnu/14",
    "-L/usr/lib"
  ]

/-- Return `none` for blank strings after trimming whitespace. -/
def nonEmptyTrimmed? (s : String) : Option String :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then none else some trimmed

/-- Resolve the macOS SDK root from env or `xcrun` without hard-coded Xcode/CLT paths. -/
def macOSSDKRoot? : Option String := run_io do
  let envSdk? ← do
    match (← IO.getEnv "TYR_MACOS_SDKROOT") with
    | some p => pure (some p)
    | none => IO.getEnv "SDKROOT"
  match envSdk?.bind nonEmptyTrimmed? with
  | some p => pure (some p)
  | none =>
    try
      let out ← IO.Process.output {
        cmd := "xcrun"
        args := #["--sdk", "macosx", "--show-sdk-path"]
      }
      if out.exitCode == 0 then
        pure (nonEmptyTrimmed? out.stdout)
      else
        pure none
    catch _ =>
      pure none

/-- Optional macOS SDK search flags when an SDK root can be discovered. -/
def macOSSDKLinkArgs : Array String :=
  match macOSSDKRoot? with
  | some sdk =>
    #[
      s!"-F{sdk}/System/Library/Frameworks",
      s!"-Wl,-syslibroot,{sdk}"
    ]
  | none => #[]

/-- Apple system frameworks used by the C++ bridge/runtime on macOS. -/
def macOSFrameworkArgs : Array String :=
  #[
    "-framework", "Foundation",
    "-framework", "CoreFoundation",
    "-framework", "CoreGraphics",
    "-framework", "ImageIO",
    "-framework", "AVFoundation",
    "-framework", "CoreMedia",
    "-framework", "CoreVideo",
    "-framework", "VideoToolbox",
    "-framework", "Accelerate",
    "-framework", "AudioToolbox"
  ]

def packageLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet", "-lsoxr"
    ] ++ macOSSDKLinkArgs ++ macOSFrameworkArgs ++ #[
      "-Wl,-rpath,@loader_path/../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10"
    ] ++ linuxSystemLinkDirs ++ #[
      "-l:libgomp.so.1", "-l:libstdc++.so.6",
      "-larrow", "-lparquet", "-lsoxr",
      "-Wl,-rpath,$ORIGIN/../../../external/libtorch/lib"
    ]

def commonLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      "-Lcc/build", "-lTyrC",
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet", "-lsoxr"
    ] ++ macOSSDKLinkArgs ++ macOSFrameworkArgs ++ #[
      "-Wl,-rpath,@executable_path/../../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[
      "-Lcc/build", "-lTyrC",
      "-Lexternal/libtorch/lib",
      "-ltorch", "-ltorch_cpu", "-lc10"
    ] ++ linuxSystemLinkDirs ++ #[
      "-l:libgomp.so.1", "-l:libstdc++.so.6",
      "-larrow", "-lparquet", "-lsoxr",
      "-Wl,-rpath,$ORIGIN/../../../external/libtorch/lib"
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

  -- Track Makefile plus C/CUDA sources/headers so Lake reruns `make` when FFI changes.
  let makefileJob ← inputTextFile <| pkg.dir / "cc" / "Makefile"
  let srcJob ← inputDir (pkg.dir / "cc" / "src") (text := true) fun p =>
    p.toString.endsWith ".cpp" || p.toString.endsWith ".mm" ||
      p.toString.endsWith ".cu" || p.toString.endsWith ".h"
  let depJob := makefileJob.mix srcJob

  buildFileAfterDep tyrCLib depJob fun _ => do
    let sysroot ← getLeanSysroot
    proc {
      cmd := "make"
      args := #["-C", (pkg.dir / "cc").toString, "lib"]
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

/-- Live microphone streaming Qwen3-ASR demo (macOS AudioToolbox input). -/
lean_exe Qwen3ASRLiveMic where
  root := `Examples.Qwen3ASR.LiveMic
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Separate streaming-native ASR session executable (parallel path). -/
lean_exe Qwen3ASRLiveMicTrueStream where
  root := `Examples.Qwen3ASR.LiveMicTrueStream
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Diffusion tests executable -/
lean_exe TestDiffusion where
  root := `Tests.TestDiffusion
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- DataLoader test executable -/
lean_exe TestDataLoader where
  root := `Tests.RunTestDataLoader
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Differential equation baseline test executable. -/
lean_exe TestDiffEq where
  root := `Tests.TestDiffEq
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjoint where
  root := `Tests.TestDiffEqAdjoint
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Core adjoint differential equation test executable. -/
lean_exe TestDiffEqAdjointCore where
  root := `Tests.TestDiffEqAdjointCore
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPU DSL regression test executable. -/
lean_exe TestGPUDSL where
  root := `Tests.TestGPUDSL
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- GPU kernel fixture test executable. -/
lean_exe TestGPUKernels where
  root := `Tests.TestGPUKernels
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Flux image generation demo -/
lean_exe FluxDemo where
  root := `Examples.Flux.FluxDemo
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- End-to-end Qwen3-TTS demo (Lean talker + Python speech-tokenizer decode). -/
lean_exe Qwen3TTSEndToEnd where
  root := `Examples.Qwen3TTS.EndToEnd
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Offline Qwen3-ASR transcription demo (fully Lean pipeline). -/
lean_exe Qwen3ASRTranscribe where
  root := `Examples.Qwen3ASR.Transcribe
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Qwen3.5 model loader/generation demo with HF repo-id resolution. -/
lean_exe Qwen35RunHF where
  root := `Examples.Qwen35.RunHF
  supportInterpreter := true
  moreLinkArgs := commonLinkArgs

/-- Qwen2.5-Omni thinker text loader/generation demo (3B/7B). -/
lean_exe Qwen25OmniRunHF where
  root := `Examples.Qwen25Omni.RunHF
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
