import Lake
open Lake DSL
open System (FilePath)

package tyr where
  srcDir := "."
  buildDir := ".lake/build"
  moreServerArgs := #["-Dpp.unicode.fun=true"]

-- External library target for the C++ code
-- This builds the C++ bindings for PyTorch and XLA
extern_lib libtyr pkg := do
  -- The actual C++ library is built in cc/build/libTyrC.a
  let tyrCLib := pkg.dir / "cc" / "build" / "libTyrC.a"

  -- Build the C++ library using make
  let srcJob ← inputTextFile <| pkg.dir / "cc" / "Makefile"
  buildFileAfterDep tyrCLib srcJob fun _srcFile => do
    let sysroot ← getLeanSysroot
    proc {
      cmd := "make"
      args := #["-C", (pkg.dir / "cc").toString]
      env := #[("LEAN_HOME", sysroot.toString)]
    }

-- Main Lean library
@[default_target]
lean_lib Tyr where
  roots := #[`Tyr]
  -- Only include working modules (exclude broken AutoGrad and Module subdirectory for now)
  -- These are the three main modules that build successfully
  -- Disable precompilation to avoid needing to link the dylib at compile time
  precompileModules := false

-- Executable target for tests
lean_exe test where
  root := `Test
  supportInterpreter := true
  moreLinkArgs := #[
    -- Tyr C++ library
    "-Lcc/build",
    "-lTyrC",
    -- LibTorch
    "-Lexternal/libtorch/lib",
    "-ltorch", "-ltorch_cpu", "-lc10",
    -- XLA extension
    "-Lexternal/xla_extension/lib",
    "-lxla_extension",
    -- libomp
    "-L/opt/homebrew/opt/libomp/lib",
    "-lomp",
    -- Runtime library paths (rpath)
    "-Wl,-rpath,@executable_path/../../external/libtorch/lib",
    "-Wl,-rpath,@executable_path/../../external/xla_extension/lib",
    "-Wl,-rpath,/opt/homebrew/opt/libomp/lib"
  ]

lean_exe memtest where
  root := `MemoryTest
  supportInterpreter := true
  moreLinkArgs := #[
    "-Lcc/build",
    "-lTyrC",
    "-Lexternal/libtorch/lib",
    "-ltorch", "-ltorch_cpu", "-lc10",
    "-Lexternal/xla_extension/lib",
    "-lxla_extension",
    "-L/opt/homebrew/opt/libomp/lib",
    "-lomp",
    "-Wl,-rpath,@executable_path/../../external/libtorch/lib",
    "-Wl,-rpath,@executable_path/../../external/xla_extension/lib",
    "-Wl,-rpath,/opt/homebrew/opt/libomp/lib"
  ]

-- Script to run the test with proper environment
-- Usage: lake script run
script run do
  let rootPath := (← getWorkspace).root.dir
  let exe := rootPath / ".lake" / "build" / "bin" / "test"

  -- Set DYLD_LIBRARY_PATH for macOS
  let libtorchPath := rootPath / "external" / "libtorch" / "lib"
  let xlaPath := rootPath / "external" / "xla_extension" / "lib"
  let ompPath : FilePath := "/opt/homebrew/opt/libomp/lib"
  let leanLibPath ← getLeanSysroot
  let leanLib := leanLibPath / "lib" / "lean"
  let libPath := s!"{libtorchPath}:{xlaPath}:{ompPath}:{leanLib}"

  IO.println s!"Running: {exe}"
  IO.println s!"DYLD_LIBRARY_PATH={libPath}"

  IO.println s!"To run manually:"
  IO.println s!"export DYLD_LIBRARY_PATH=\"{libPath}\""
  IO.println s!"\"{exe}\""
  return 0
