/-
  Tyr/GPU/Codegen/Emit.lean

  C++ code generation from kernel AST.
  Produces ThunderKittens-compatible CUDA code.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST

namespace Tyr.GPU.Codegen

/-- Generate C++ for a single expression -/
partial def generateExpr (indent : String := "  ") : KExpr → String
  -- Declarations
  | .declRT name dtype rows cols layout =>
    s!"{indent}rt<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {name};\n"
  | .declST name dtype rows cols layout =>
    s!"{indent}st<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {name};\n"
  | .declRV name dtype len =>
    s!"{indent}rv<{dtype.toCpp}, {len}> {name};\n"
  | .declSV name dtype len =>
    s!"{indent}sv<{dtype.toCpp}, {len}> {name};\n"

  -- Memory
  | .load dst src => s!"{indent}load({dst}, {src});\n"
  | .store dst src => s!"{indent}store({dst}, {src});\n"
  | .loadAsync dst src => s!"{indent}load_async({dst}, {src});\n"

  -- MMA
  | .mma trans dst a b c => s!"{indent}mma_{trans.toSuffix}({dst}, {a}, {b}, {c});\n"
  | .mm trans dst a b => s!"{indent}mm_{trans.toSuffix}({dst}, {a}, {b});\n"
  | .mmaFence dst => s!"{indent}mma_fence({dst});\n"
  | .mmaCommitGroup => s!"{indent}mma_commit_group();\n"
  | .mmaAsyncWait n => s!"{indent}mma_async_wait<{n}>();\n"

  -- Element-wise
  | .unary op dst src => s!"{indent}{op.toCpp}({dst}, {src});\n"
  | .binary op dst a b => s!"{indent}{op.toCpp}({dst}, {a}, {b});\n"

  -- Broadcasting
  | .broadcast axis dst vec => s!"{indent}broadcast{axis.toSuffix}({dst}, {vec});\n"
  | .binaryBroadcast op axis dst tile vec =>
    s!"{indent}{op.toCpp}{axis.toSuffix}({dst}, {tile}, {vec});\n"

  -- Reductions
  | .reduce op axis dst src =>
    s!"{indent}{axis.toPrefix}{op.toCpp}({dst}, {src});\n"
  | .reduceAccum op axis dst src accum =>
    s!"{indent}{axis.toPrefix}{op.toCpp}({dst}, {src}, {accum});\n"

  -- Conversions
  | .swapLayout dst src => s!"{indent}swap_layout({dst}, {src});\n"
  | .transpose dst src => s!"{indent}transpose({dst}, {src});\n"
  | .convert dst src => s!"{indent}copy({dst}, {src});\n"

  -- Masking
  | .mask op dst src fillVal =>
    let fillStr := fillVal.map (fun v => s!", {v}") |>.getD ""
    match op with
    | .Tril d => s!"{indent}tril({dst}, {src}, {d}{fillStr});\n"
    | .Triu d => s!"{indent}triu({dst}, {src}, {d}{fillStr});\n"
    | .MakeCausal => s!"{indent}make_causal({dst}, {src}{fillStr});\n"
    | .RightFill c => s!"{indent}right_fill({dst}, {src}, {c}{fillStr});\n"
    | .LeftFill c => s!"{indent}left_fill({dst}, {src}, {c}{fillStr});\n"
    | .UpperFill r => s!"{indent}upper_fill({dst}, {src}, {r}{fillStr});\n"
    | .LowerFill r => s!"{indent}lower_fill({dst}, {src}, {r}{fillStr});\n"

  -- Synchronization
  | .sync barrierId => s!"{indent}sync({barrierId});\n"
  | .arrive barrierId => s!"{indent}arrive({barrierId});\n"

  -- Control flow
  | .seq a b => generateExpr indent a ++ generateExpr indent b
  | .forLoop var lo hi body =>
    s!"{indent}for (int {var} = {lo}; {var} < {hi}; {var}++) \{\n" ++
    generateExpr (indent ++ "  ") body ++
    s!"{indent}}\n"
  | .comment text => s!"{indent}// {text}\n"

/-- Generate kernel parameter list -/
def generateParams (params : List KernelParam) : String :=
  let paramStrs := params.map fun p =>
    if p.isPointer then s!"{p.cppType}* {p.name}" else s!"{p.cppType} {p.name}"
  String.intercalate ", " paramStrs

/-- Generate full kernel C++ code -/
def generateCpp (k : KernelDef) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let body := generateExpr "  " k.body
  let footer := "}\n#endif\n"
  header ++ archGuard ++ signature ++ body ++ footer

/-- Generate kernel with extern shared memory -/
def generateCppWithShared (k : KernelDef) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let sharedDecl := if k.sharedMemBytes > 0
    then s!"  extern __shared__ char smem[{k.sharedMemBytes}];\n"
    else ""
  let body := generateExpr "  " k.body
  let footer := "}\n#endif\n"
  header ++ archGuard ++ signature ++ sharedDecl ++ body ++ footer

/-- Write kernel to file -/
def writeKernel (k : KernelDef) (path : String) : IO Unit := do
  let code := generateCpp k
  IO.FS.writeFile path code

/-- Generate launch configuration -/
structure LaunchConfig where
  gridDim : Nat × Nat × Nat := (1, 1, 1)
  blockDim : Nat × Nat × Nat := (128, 1, 1)
  sharedMem : Nat := 0
  deriving Repr, Inhabited

/-- Generate CUDA launch code -/
def generateLaunch (k : KernelDef) (cfg : LaunchConfig) (args : List String) : String :=
  let (gx, gy, gz) := cfg.gridDim
  let (bx, byy, bz) := cfg.blockDim
  let argStr := String.intercalate ", " args
  s!"{k.name}<<<dim3({gx}, {gy}, {gz}), dim3({bx}, {byy}, {bz}), {cfg.sharedMem}>>>({argStr});\n"

end Tyr.GPU.Codegen
