/-
  Tyr/Eval/Execution.lean

  Sandboxed code execution for evaluating generated code.

  Based on nanochat's execution.py:
  - Sandboxed Python execution in separate process
  - Timeout protection (3s default)
  - Memory limits (256MB default)
  - Stdout/stderr capture
  - Dangerous functions disabled

  Used for:
  - HumanEval pass@k evaluation
  - Code generation benchmarks
  - Test suite execution
-/
import Tyr.Torch

namespace torch.Eval.Execution

/-! ## Configuration -/

/-- Configuration for sandboxed execution -/
structure ExecConfig where
  /-- Timeout in milliseconds -/
  timeout : UInt64 := 3000
  /-- Memory limit in bytes -/
  memoryLimit : UInt64 := 256 * 1024 * 1024
  /-- Working directory (temp dir if empty) -/
  workDir : String := ""
  /-- Environment variables to set -/
  envVars : Array (String × String) := #[]
  deriving Repr, Inhabited

/-- Default execution config (3s timeout, 256MB memory) -/
def ExecConfig.default : ExecConfig := {}

/-- Strict execution config (1s timeout, 128MB memory) -/
def ExecConfig.strict : ExecConfig := {
  timeout := 1000
  memoryLimit := 128 * 1024 * 1024
}

/-! ## Results -/

/-- Exit status codes -/
inductive ExitStatus where
  | success     : ExitStatus  -- Exit code 0
  | failure     : ExitStatus  -- Non-zero exit code
  | timeout     : ExitStatus  -- Process timed out
  | memoryLimit : ExitStatus  -- Memory limit exceeded
  | signaled    : ExitStatus  -- Killed by signal
  | error       : ExitStatus  -- Failed to execute
  deriving Repr, BEq, Inhabited

/-- Result of executing code -/
structure ExecResult where
  /-- Exit status -/
  status : ExitStatus
  /-- Exit code (if applicable) -/
  exitCode : Int32 := 0
  /-- Captured stdout -/
  stdout : String := ""
  /-- Captured stderr -/
  stderr : String := ""
  /-- Execution time in milliseconds -/
  execTimeMs : UInt64 := 0
  deriving Repr, Inhabited

def ExecResult.isSuccess (r : ExecResult) : Bool :=
  r.status == .success

def ExecResult.isTimeout (r : ExecResult) : Bool :=
  r.status == .timeout

/-! ## FFI Declarations -/

/-- Execute Python code in a sandboxed subprocess.
    The code is written to a temp file and executed with resource limits.
    Returns captured output and exit status. -/
@[extern "lean_exec_python_sandboxed"]
opaque execPythonSandboxed (code : @& String) (config : @& ExecConfig) : IO ExecResult

/-- Execute a shell command in a sandboxed subprocess.
    Similar to execPythonSandboxed but runs arbitrary shell command. -/
@[extern "lean_exec_shell_sandboxed"]
opaque execShellSandboxed (command : @& String) (config : @& ExecConfig) : IO ExecResult

/-- Check if sandboxed execution is available on this platform -/
@[extern "lean_exec_is_available"]
opaque execIsAvailable : IO Bool

/-! ## High-Level API -/

/-- Execute Python code with test assertions.
    Prepends common imports and appends assertion handling. -/
def execPythonWithTests (code : String) (testCode : String)
    (config : ExecConfig := {}) : IO ExecResult := do
  let fullCode := s!"{code}\n\n{testCode}"
  execPythonSandboxed fullCode config

/-- Execute code and check if all tests pass.
    Returns true if exit code is 0 and no assertion errors. -/
def checkTestsPass (code : String) (testCode : String)
    (config : ExecConfig := {}) : IO Bool := do
  let result ← execPythonWithTests code testCode config
  return result.isSuccess

/-! ## HumanEval Helpers -/

/-- Standard HumanEval test runner code.
    Wraps the generated function and test cases in a try-except block. -/
def humanEvalTestRunner (entryPoint : String) (testCode : String) : String :=
  "import sys\n" ++
  "import math\n" ++
  "from typing import List, Dict, Tuple, Optional, Any, Union\n\n" ++
  "# Run tests\n" ++
  "def check(candidate):\n" ++
  testCode ++ "\n\n" ++
  "if __name__ == '__main__':\n" ++
  "    try:\n" ++
  "        check(" ++ entryPoint ++ ")\n" ++
  "        print('All tests passed')\n" ++
  "        sys.exit(0)\n" ++
  "    except AssertionError as e:\n" ++
  "        print(f'Assertion failed: {e}', file=sys.stderr)\n" ++
  "        sys.exit(1)\n" ++
  "    except Exception as e:\n" ++
  "        print(f'Error: {e}', file=sys.stderr)\n" ++
  "        sys.exit(1)\n"

/-- Run HumanEval-style evaluation on generated code.
    Returns true if all tests pass. -/
def evalHumanEval (generatedCode : String) (entryPoint : String)
    (testCode : String) (config : ExecConfig := {}) : IO ExecResult := do
  let fullCode := generatedCode ++ "\n\n" ++ humanEvalTestRunner entryPoint testCode
  execPythonSandboxed fullCode config

/-! ## Batch Evaluation -/

/-- Result of evaluating multiple solutions -/
structure BatchEvalResult where
  /-- Number of solutions that passed -/
  numPassed : Nat
  /-- Total number of solutions -/
  total : Nat
  /-- Individual results -/
  results : Array ExecResult
  deriving Repr

/-- Compute pass@k metric.
    Given n samples and c correct, estimates probability that
    at least one of k random samples is correct. -/
def passAtK (n c k : Nat) : Float :=
  if n < k || c == 0 then 0.0
  else if c >= n then 1.0
  else
    -- 1 - C(n-c, k) / C(n, k)
    -- = 1 - product_{i=0}^{k-1} (n-c-i) / (n-i)
    let prob := (List.range k).foldl (fun acc i =>
      acc * (n - c - i).toFloat / (n - i).toFloat) 1.0
    1.0 - prob

/-- Evaluate multiple code solutions for the same problem.
    Returns batch result with pass counts. -/
def evalBatch (solutions : Array String) (entryPoint : String)
    (testCode : String) (config : ExecConfig := {}) : IO BatchEvalResult := do
  let mut results : Array ExecResult := #[]
  let mut numPassed := 0

  for solution in solutions do
    let result ← evalHumanEval solution entryPoint testCode config
    results := results.push result
    if result.isSuccess then
      numPassed := numPassed + 1

  return { numPassed, total := solutions.size, results }

/-- Compute pass@k from batch evaluation result -/
def BatchEvalResult.computePassAtK (result : BatchEvalResult) (k : Nat) : Float :=
  torch.Eval.Execution.passAtK result.total result.numPassed k

end torch.Eval.Execution
