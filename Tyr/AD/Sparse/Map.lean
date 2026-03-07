import Tyr.AD.Sparse.Dim

/-!
# Tyr.AD.Sparse.Map

Core sparse linear-map representation.
-/

namespace Tyr.AD.Sparse

structure SparseEntry where
  src : Nat
  dst : Nat
  weight : Float := 1.0
  deriving Repr, Inhabited, BEq

/-- Input-role marker for semantic Jacobian tags. -/
inductive JacArgRole where
  | x
  | a
  | b
  | lhs
  | rhs
  | src
  | accum
  | tile
  | vec
  deriving Repr, Inhabited, BEq, DecidableEq

namespace JacArgRole

def toString : JacArgRole → String
  | .x => "x"
  | .a => "a"
  | .b => "b"
  | .lhs => "lhs"
  | .rhs => "rhs"
  | .src => "src"
  | .accum => "accum"
  | .tile => "tile"
  | .vec => "vec"

instance : ToString JacArgRole := ⟨toString⟩

end JacArgRole

/-- Semantic mode marker for structured local-Jacobian tags. -/
inductive JacMode where
  | none
  | rhsValue
  | lhsValue
  | reciprocalRhs
  | negLhsOverRhsSq
  | mask
  | complementMask
  | expand
  | contract
  | carry
  | permute
  | layoutPermute
  | cast
  | projection
  | inject
  | kronOther
  | prefix
  | prefixProduct
  | vecBroadcast
  | tileBroadcast
  deriving Repr, Inhabited, BEq, DecidableEq

namespace JacMode

def toString : JacMode → String
  | .none => "none"
  | .rhsValue => "rhs"
  | .lhsValue => "lhs"
  | .reciprocalRhs => "1/rhs"
  | .negLhsOverRhsSq => "-lhs/rhs^2"
  | .mask => "mask"
  | .complementMask => "1-mask"
  | .expand => "expand"
  | .contract => "contract"
  | .carry => "carry"
  | .permute => "permute"
  | .layoutPermute => "layout-permute"
  | .cast => "cast"
  | .projection => "projection"
  | .inject => "inject"
  | .kronOther => "kron(other)"
  | .prefix => "prefix"
  | .prefixProduct => "prefix-product"
  | .vecBroadcast => "vec-broadcast"
  | .tileBroadcast => "tile"

instance : ToString JacMode := ⟨toString⟩

end JacMode

/-- Structured metadata for `dot_general` local Jacobian semantics. -/
structure DotGeneralSemantics where
  variant : Lean.Name
  arg : JacArgRole
  lhsContract : Array Nat := #[]
  rhsContract : Array Nat := #[]
  lhsBatch : Array Nat := #[]
  rhsBatch : Array Nat := #[]
  deriving Repr, Inhabited, BEq, DecidableEq

/-- Structured semantic Jacobian tag used by rule packs. -/
inductive SparseSemanticTag where
  | unary (op : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | binary (op : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | reduce (op : Lean.Name) (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | reduceAccum (op : Lean.Name) (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | broadcast (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | binaryBroadcast (op : Lean.Name) (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | transpose (arg : JacArgRole) (mode : JacMode)
  | swapLayout (arg : JacArgRole) (mode : JacMode)
  | convert (arg : JacArgRole) (mode : JacMode)
  | sliceRows (arg : JacArgRole) (mode : JacMode)
  | sliceCols (arg : JacArgRole) (mode : JacMode)
  | concatCols (arg : JacArgRole) (mode : JacMode)
  | outer (arg : JacArgRole) (mode : JacMode)
  | cumsum (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | cumprod (axis : Lean.Name) (arg : JacArgRole) (mode : JacMode)
  | dotGeneral (spec : DotGeneralSemantics)
  deriving Repr, Inhabited, BEq, DecidableEq

namespace SparseSemanticTag

private def renderNatArray (xs : Array Nat) : String :=
  "[" ++ String.intercalate ", " (xs.toList.map (fun x => toString x)) ++ "]"

private def renderMode (mode : JacMode) : String :=
  match mode with
  | .none => ""
  | _ => s!"[{mode}]"

def toString : SparseSemanticTag → String
  | .unary op arg mode =>
      s!"d{op}/d{arg}{renderMode mode}"
  | .binary op arg mode =>
      s!"d{op}/d{arg}{renderMode mode}"
  | .reduce op axis arg mode =>
      s!"dreduce.{op}.{axis}/d{arg}{renderMode mode}"
  | .reduceAccum op axis arg mode =>
      s!"dreduceAccum.{op}.{axis}/d{arg}{renderMode mode}"
  | .broadcast axis arg mode =>
      s!"dbroadcast.{axis}/d{arg}{renderMode mode}"
  | .binaryBroadcast op axis arg mode =>
      s!"dbinaryBroadcast.{op}.{axis}/d{arg}{renderMode mode}"
  | .transpose arg mode =>
      s!"dtranspose/d{arg}{renderMode mode}"
  | .swapLayout arg mode =>
      s!"dswapLayout/d{arg}{renderMode mode}"
  | .convert arg mode =>
      s!"dconvert/d{arg}{renderMode mode}"
  | .sliceRows arg mode =>
      s!"dsliceRows/d{arg}{renderMode mode}"
  | .sliceCols arg mode =>
      s!"dsliceCols/d{arg}{renderMode mode}"
  | .concatCols arg mode =>
      s!"dconcatCols/d{arg}{renderMode mode}"
  | .outer arg mode =>
      s!"douter/d{arg}{renderMode mode}"
  | .cumsum axis arg mode =>
      s!"dcumsum.{axis}/d{arg}{renderMode mode}"
  | .cumprod axis arg mode =>
      s!"dcumprod.{axis}/d{arg}{renderMode mode}"
  | .dotGeneral spec =>
      let lhsContract := renderNatArray spec.lhsContract
      let rhsContract := renderNatArray spec.rhsContract
      let lhsBatch := renderNatArray spec.lhsBatch
      let rhsBatch := renderNatArray spec.rhsBatch
      s!"ddotGeneral.{spec.variant}/d{spec.arg}[lhsContract={lhsContract};rhsContract={rhsContract};lhsBatch={lhsBatch};rhsBatch={rhsBatch}]"

instance : ToString SparseSemanticTag := ⟨toString⟩

end SparseSemanticTag

/-- Typed semantic tag carried by sparse maps for diagnostics/debugging. -/
inductive SparseMapTag where
  | placeholder
  | identityLike
  | zero
  | identity (n : Nat)
  | semantic (tag : SparseSemanticTag)
  | named (label : Lean.Name)
  | add (lhs rhs : SparseMapTag)
  | compose (outMap inMap : SparseMapTag)
  deriving Repr, Inhabited, BEq, DecidableEq

namespace SparseMapTag

private def atom (s : String) : Lean.Name :=
  Lean.Name.str Lean.Name.anonymous s

def namedStr (s : String) : SparseMapTag :=
  .named (atom s)

def toString : SparseMapTag → String
  | .placeholder => "placeholder"
  | .identityLike => "identity-like"
  | .zero => "zero"
  | .identity n => s!"I[{n}]"
  | .semantic tag => SparseSemanticTag.toString tag
  | .named label => label.toString
  | .add lhs rhs => s!"({toString lhs} + {toString rhs})"
  | .compose outMap inMap => s!"({toString outMap} ∘ {toString inMap})"

instance : ToString SparseMapTag := ⟨toString⟩

end SparseMapTag

/--
Sparse linear map in coordinate format.

`inDim?`/`outDim?` are optional during early lowering and become concrete once
shape metadata is available.
-/
structure SparseLinearMap where
  repr : SparseMapTag := .placeholder
  inDim? : Option DimSize := none
  outDim? : Option DimSize := none
  entries : Array SparseEntry := #[]
  deriving Repr, Inhabited, BEq

def SparseLinearMap.isIdentityLike (m : SparseLinearMap) : Bool :=
  m.entries.isEmpty && m.repr == .identityLike

def SparseLinearMap.shape? (m : SparseLinearMap) : Option LinearShape := do
  let i ← m.inDim?
  let o ← m.outDim?
  pure { inDim := i, outDim := o }

end Tyr.AD.Sparse
