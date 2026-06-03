import Lake
open Lake DSL
open System

def capnpLeanDir : FilePath :=
  __dir__ / ".." / ".." / ".." / "capnproto-lean"

def capnpBuildDir : FilePath :=
  capnpLeanDir / "extern" / "capnproto" / "build" / "c++"

def capnpLibDir : FilePath :=
  capnpBuildDir / "src" / "capnp"

def kjLibDir : FilePath :=
  capnpBuildDir / "src" / "kj"

def capnpCompilerPath : FilePath :=
  capnpLibDir / "capnp"

def capnpPluginSourcePath : FilePath :=
  capnpLeanDir / "extern" / "capnproto" / "c++" / "src" / "capnp" / "compiler" / "capnpc-lean4.c++"

def tyrRootDir : FilePath :=
  __dir__ / ".." / ".."

def nonEmptyTrimmed? (s : String) : Option String :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then none else some trimmed

def normalizeMacOSSDKRoot (sdk : String) : IO String := do
  let sdkPath : FilePath := ⟨sdk⟩
  match sdkPath.parent with
  | some parent =>
      let stablePath := parent / "MacOSX.sdk"
      if ← stablePath.pathExists then
        pure stablePath.toString
      else
        pure sdk
  | none =>
      pure sdk

def macOSSDKRoot? : Option String := run_io do
  let envSdk? ← do
    match (← IO.getEnv "TYR_MACOS_SDKROOT") with
    | some p => pure (some p)
    | none => IO.getEnv "SDKROOT"
  match envSdk?.bind nonEmptyTrimmed? with
  | some p =>
      let normalized ← normalizeMacOSSDKRoot p
      if ← (⟨normalized⟩ : FilePath).pathExists then
        pure (some normalized)
      else
        pure none
  | none =>
      try
        let out ← IO.Process.output {
          cmd := "xcrun"
          args := #["--sdk", "macosx", "--show-sdk-path"]
        }
        if out.exitCode == 0 then
          match nonEmptyTrimmed? out.stdout with
          | some sdk =>
              pure (some (← normalizeMacOSSDKRoot sdk))
          | none =>
              pure none
        else
          pure none
      catch _ =>
        pure none

def macOSSDKLinkArgs : Array String :=
  match macOSSDKRoot? with
  | some sdk =>
      #[
        s!"-F{sdk}/System/Library/Frameworks",
        s!"-Wl,-syslibroot,{sdk}"
      ]
  | none => #[]

def macOSDeploymentTarget : String := run_io do
  let envTarget? ← do
    match (← IO.getEnv "TYR_MACOS_DEPLOYMENT_TARGET") with
    | some t => pure (some t)
    | none => IO.getEnv "MACOSX_DEPLOYMENT_TARGET"
  match envTarget?.bind nonEmptyTrimmed? with
  | some t => pure t
  | none => pure "14.0"

def macOSDeploymentLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[s!"-mmacosx-version-min={macOSDeploymentTarget}"]
  else
    #[]

def tyrLinuxSystemLinkDirs : Array String :=
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

def tyrMacOSFrameworkArgs : Array String :=
  #[
    "-framework", "Foundation",
    "-framework", "CoreFoundation",
    "-framework", "Metal",
    "-framework", "CoreGraphics",
    "-framework", "ImageIO",
    "-framework", "AVFoundation",
    "-framework", "CoreMedia",
    "-framework", "CoreVideo",
    "-framework", "VideoToolbox",
    "-framework", "Accelerate",
    "-framework", "AudioToolbox"
  ]

def tyrSoxrLinkArgs : Array String :=
  #[s!"-L{tyrRootDir / "cc" / "build" / "soxr" / "src"}", "-lsoxr"]

def tyrModelServerExeLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      s!"{tyrRootDir / "cc" / "build" / "libTyrC.a"}",
      s!"-L{tyrRootDir / "external" / "libtorch" / "lib"}",
      "-ltorch", "-ltorch_cpu", "-lc10",
      "-L/opt/homebrew/opt/libomp/lib", "-lomp",
      "-L/opt/homebrew/lib", "-larrow", "-lparquet",
      "-lobjc"
    ] ++ tyrSoxrLinkArgs ++ macOSSDKLinkArgs ++ macOSDeploymentLinkArgs ++ tyrMacOSFrameworkArgs ++ #[
      "-Wl,-rpath,@executable_path/../../../../../external/libtorch/lib",
      "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
      "-Wl,-rpath,/opt/homebrew/lib"
    ]
  else
    #[
      s!"{tyrRootDir / "cc" / "build" / "libTyrC.a"}",
      s!"-L{tyrRootDir / "external" / "libtorch" / "lib"}",
      "-ltorch", "-ltorch_cpu", "-lc10"
    ] ++ tyrLinuxSystemLinkDirs ++ tyrSoxrLinkArgs ++ #[
      "-l:libgomp.so.1", "-l:libstdc++.so.6",
      "-larrow", "-lparquet",
      "-Wl,-rpath,$ORIGIN/../../../../../external/libtorch/lib"
    ]

def capnpLinkArgs : Array String :=
  if System.Platform.isOSX then
    #[
      "-L", capnpLibDir.toString,
      "-L", kjLibDir.toString,
      "-L/opt/homebrew/lib",
      "-L/usr/local/lib",
      "-L/opt/homebrew/opt/openssl@3/lib",
      "-L/usr/local/opt/openssl@3/lib",
      "-L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib",
      "-lcapnp-rpc", "-lcapnp", "-lkj-http", "-lkj-gzip", "-lkj-tls", "-lkj-async", "-lkj",
      "-lssl", "-lcrypto", "-lz", "-lc++"
    ] ++ macOSSDKLinkArgs ++ macOSDeploymentLinkArgs
  else
    #[
      "-L", capnpLibDir.toString,
      "-L", kjLibDir.toString,
      "-lcapnp-rpc", "-lcapnp", "-lkj-http", "-lkj-gzip", "-lkj-tls", "-lkj-async", "-lkj",
      "-lssl", "-lcrypto", "-lstdc++", "-lz", "-pthread"
    ]

package TyrModelServer where
  extraDepTargets := #[`generateModelGatewayCapnp]
  moreLeanArgs := #["-DmaxHeartbeats=2000000"]
  moreLinkArgs := capnpLinkArgs

require tyr from "../.."
require capnproto_lean from "../../../capnproto-lean"

def capnpPluginBinaryPath (pkg : NPackage __name__) : FilePath :=
  pkg.buildDir / "tools" / "capnpc-lean4"

private def runChecked (cwd : FilePath) (cmd : String) (args : Array String) : IO Unit := do
  let output ← IO.Process.output {
    cmd := cmd
    args := args
    cwd := cwd
  }
  if output.exitCode != 0 then
    error s!"{cmd} {args} failed with exit code {output.exitCode}:\n{output.stderr}"

private def writeFileIfChanged (path : FilePath) (content : String) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let shouldWrite ←
    if ← path.pathExists then
      pure ((← IO.FS.readFile path) != content)
    else
      pure true
  if shouldWrite then
    IO.FS.writeFile path content

private def schemaPath (pkg : NPackage __name__) : FilePath :=
  pkg.dir / "capnp" / "model_gateway.capnp"

private def generatedDir (pkg : NPackage __name__) : FilePath :=
  pkg.buildDir / "generated" / "model_gateway_capnp"

private def generatedPath (pkg : NPackage __name__) : FilePath :=
  generatedDir pkg / "Capnp" / "Gen" / "model_gateway.lean"

private def outputModulePath (pkg : NPackage __name__) : FilePath :=
  pkg.dir / "TyrModelServer" / "Capnp" / "model_gateway.lean"

private def rewriteNamespace (content : String) : String :=
  content.replace "Capnp.Gen.model_gateway" "TyrModelServer.Capnp.model_gateway"

target generateModelGatewayCapnp (pkg : NPackage __name__) : Unit := do
  unless (← capnpCompilerPath.pathExists) do
    error s!"missing Cap'n Proto compiler at {capnpCompilerPath}; build ../capnproto-lean first"
  unless (← capnpPluginSourcePath.pathExists) do
    error s!"missing capnpc-lean4 source at {capnpPluginSourcePath}"

  let pluginBinary := capnpPluginBinaryPath pkg
  IO.FS.createDirAll (pkg.buildDir / "tools")
  runChecked pkg.dir "c++" #[
    "-std=c++23",
    capnpPluginSourcePath.toString,
    "-I", (capnpLeanDir / "extern" / "capnproto" / "c++" / "src").toString,
    "-L", capnpLibDir.toString,
    "-L", kjLibDir.toString,
    "-lcapnp",
    "-lkj",
    "-o", pluginBinary.toString
  ]

  IO.FS.createDirAll (generatedDir pkg)
  runChecked pkg.dir capnpCompilerPath.toString #[
    "compile",
    s!"-o{pluginBinary}:{generatedDir pkg}",
    s!"--src-prefix={pkg.dir / "capnp"}",
    (schemaPath pkg).toString
  ]

  unless (← (generatedPath pkg).pathExists) do
    error s!"expected generated schema at {generatedPath pkg}"
  let content ← IO.FS.readFile (generatedPath pkg)
  writeFileIfChanged (outputModulePath pkg) (rewriteNamespace content)
  pure .nil

@[default_target]
lean_lib TyrModelServer where
  roots := #[`TyrModelServer, `TyrModelServer.Capnp.model_gateway]
  globs := #[.submodules `TyrModelServer]

lean_exe tyr_model_server where
  root := `Main
  supportInterpreter := true
  moreLinkArgs := capnpLinkArgs ++ tyrModelServerExeLinkArgs

@[test_driver]
lean_exe smoke where
  root := `Smoke
  supportInterpreter := true
  moreLinkArgs := capnpLinkArgs
