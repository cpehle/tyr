import Lean.CoreM
import Tyr.AD.Elim
import Tyr.AD.JaxprLike.Pipeline
import Tyr.AD.JaxprLike.RulePackKStmt
import Tyr.GPU.Codegen.IR

/-!
  Examples/AlphaGradPort/Tasks.lean

  Task specifications for AlphaGrad-style elimination planning ports.
  `RoeFlux_1d` remains the first fully hand-shaped baseline.
  `Perceptron`/`Encoder`/`RobotArm_6DOF`/`BlackScholes_Jacobian` are now
  materialized from real `KStmt` lowering.
-/

namespace Examples.AlphaGradPort

open Lean
open Tyr.AD.Elim
open Tyr.AD.JaxprLike
open Tyr.GPU.Codegen

structure TaskSpec where
  name : String
  description : String
  numVertices : Nat
  edges : Array LocalJacEdge
  envCfg : AlphaGradMctxConfig := {}
  mctsCfg : AlphaGradMctsConfig := {}
  deriving Repr, Inhabited

inductive TaskName where
  | roeFlux1d
  | perceptron
  | encoder
  | robotArm6DOF
  | blackScholesJacobian
  | humanHeartDipole
  | propaneCombustion
  deriving Repr, BEq, Inhabited

instance : ToString TaskName where
  toString
    | .roeFlux1d => "RoeFlux_1d"
    | .perceptron => "Perceptron"
    | .encoder => "Encoder"
    | .robotArm6DOF => "RobotArm_6DOF"
    | .blackScholesJacobian => "BlackScholes_Jacobian"
    | .humanHeartDipole => "HumanHeartDipole"
    | .propaneCombustion => "PropaneCombustion"

def parseTaskName? (s : String) : Option TaskName :=
  match s.trimAscii.toString with
  | "RoeFlux_1d" => some .roeFlux1d
  | "Perceptron" => some .perceptron
  | "Encoder" => some .encoder
  | "RobotArm_6DOF" => some .robotArm6DOF
  | "BlackScholes_Jacobian" => some .blackScholesJacobian
  | "HumanHeartDipole" => some .humanHeartDipole
  | "PropaneCombustion" => some .propaneCombustion
  | _ => none

def taskSequence : Array TaskName := #[
  .roeFlux1d,
  .perceptron,
  .encoder,
  .robotArm6DOF,
  .blackScholesJacobian,
  .humanHeartDipole,
  .propaneCombustion
]

private def scalarMap (repr : String) : Tyr.AD.Sparse.SparseLinearMap :=
  {
    repr := Tyr.AD.Sparse.SparseMapTag.namedStr repr
    inDim? := some 1
    outDim? := some 1
    entries := #[({ src := 0, dst := 0, weight := 1.0 } : Tyr.AD.Sparse.SparseEntry)]
  }

private def mkEdge (src dst : Nat) (repr : String := "d?") : LocalJacEdge :=
  { src := src, dst := dst, map := scalarMap repr }

private def denseBipartite
    (srcStart srcCount dstStart dstCount : Nat)
    (tag : String) :
    Array LocalJacEdge := Id.run do
  let mut out : Array LocalJacEdge := #[]
  for s in [:srcCount] do
    let src := srcStart + s
    for d in [:dstCount] do
      let dst := dstStart + d
      out := out.push (mkEdge src dst s!"{tag}.{s}->{d}")
  return out

private def chainEdges (start count : Nat) (tag : String) : Array LocalJacEdge := Id.run do
  let mut out : Array LocalJacEdge := #[]
  if count < 2 then
    return out
  for i in [:count - 1] do
    out := out.push (mkEdge (start + i) (start + i + 1) s!"{tag}.{i}")
  return out

private def roeFlux1dEdges : Array LocalJacEdge := #[
  -- ul/ur pre-processing
  mkEdge 1 2 "ul0->du0", mkEdge 4 2 "ur0->du0",
  mkEdge 1 3 "ul0->ulr0", mkEdge 4 3 "ur0->ulr0",
  mkEdge 1 4 "ul0->sqrt_ul0", mkEdge 4 5 "ur0->sqrt_ur0",
  mkEdge 4 6 "ur1->vr", mkEdge 1 7 "ul1->vl",
  mkEdge 1 8 "ul0->pl", mkEdge 7 8 "vl->pl", mkEdge 3 8 "ulr0->pl",
  mkEdge 4 9 "ur0->pr", mkEdge 6 9 "vr->pr", mkEdge 3 9 "ulr0->pr",
  mkEdge 1 10 "ul2->hl", mkEdge 8 10 "pl->hl",
  mkEdge 4 11 "ur2->hr", mkEdge 9 11 "pr->hr",
  mkEdge 9 12 "pr->dp", mkEdge 8 12 "pl->dp",
  mkEdge 6 13 "vr->dv", mkEdge 7 13 "vl->dv",
  -- Roe averages
  mkEdge 4 14 "sqrt_ul0->w1", mkEdge 5 14 "sqrt_ur0->w1",
  mkEdge 4 15 "sqrt_ul0->u_num", mkEdge 7 15 "vl->u_num",
  mkEdge 5 15 "sqrt_ur0->u_num", mkEdge 6 15 "vr->u_num",
  mkEdge 15 16 "u_num->u", mkEdge 14 16 "w1->u",
  mkEdge 4 17 "sqrt_ul0->h_num", mkEdge 10 17 "hl->h_num",
  mkEdge 5 17 "sqrt_ur0->h_num", mkEdge 11 17 "hr->h_num",
  mkEdge 17 18 "h_num->h", mkEdge 14 18 "w1->h",
  -- spectral terms
  mkEdge 16 19 "u->q2",
  mkEdge 18 20 "h->a2", mkEdge 19 20 "q2->a2",
  mkEdge 20 21 "a2->a",
  mkEdge 16 22 "u->lp", mkEdge 21 22 "a->lp",
  mkEdge 16 23 "u->l", mkEdge 16 24 "u->ln",
  mkEdge 21 25 "a->n", mkEdge 3 25 "ulr0->n",
  -- coefficients and final flux assembly
  mkEdge 2 26 "du0->c0", mkEdge 12 26 "dp->c0", mkEdge 23 26 "l->c0",
  mkEdge 13 27 "dv->c1", mkEdge 12 27 "dp->c1", mkEdge 22 27 "lp->c1",
  mkEdge 13 28 "dv->c2", mkEdge 12 28 "dp->c2", mkEdge 24 28 "ln->c2",
  mkEdge 3 29 "ulr0->c", mkEdge 21 29 "a->c",
  mkEdge 26 30 "c0->dF0", mkEdge 29 30 "c->dF0", mkEdge 27 30 "c1->dF0", mkEdge 28 30 "c2->dF0",
  mkEdge 26 31 "c0->dF1", mkEdge 16 31 "u->dF1", mkEdge 29 31 "c->dF1", mkEdge 27 31 "c1->dF1", mkEdge 28 31 "c2->dF1", mkEdge 21 31 "a->dF1",
  mkEdge 26 32 "c0->dF2", mkEdge 19 32 "q2->dF2", mkEdge 29 32 "c->dF2", mkEdge 27 32 "c1->dF2", mkEdge 28 32 "c2->dF2", mkEdge 18 32 "h->dF2", mkEdge 16 32 "u->dF2", mkEdge 21 32 "a->dF2",
  mkEdge 30 33 "dF0->phi0",
  mkEdge 31 34 "dF1->phi1",
  mkEdge 32 35 "dF2->phi2"
]

private def robotArmEdges : Array LocalJacEdge :=
  denseBipartite 1 6 7 10 "r.trig" ++
  denseBipartite 7 10 17 8 "r.pose" ++
  chainEdges 17 8 "r.euler" ++
  denseBipartite 20 5 25 4 "r.output"

private def blackScholesEdges : Array LocalJacEdge :=
  denseBipartite 1 5 6 8 "b.pricing" ++
  chainEdges 6 8 "b.normal_cdf" ++
  denseBipartite 10 4 14 5 "b.greeks" ++
  chainEdges 14 5 "b.aggregate"

private def heartDipoleEdges : Array LocalJacEdge :=
  denseBipartite 1 8 9 10 "h.geom" ++
  denseBipartite 9 10 19 6 "h.residual" ++
  chainEdges 19 6 "h.norm" ++
  denseBipartite 22 3 25 2 "h.output"

private def propaneEdges : Array LocalJacEdge :=
  denseBipartite 1 11 12 10 "pc.chem1" ++
  denseBipartite 12 10 22 8 "pc.chem2" ++
  chainEdges 22 8 "pc.balance" ++
  denseBipartite 26 4 30 3 "pc.output"

private def runCoreMResult (x : CoreM α) : IO (Except String α) := do
  let env ← mkEmptyEnvironment
  let ctx : Core.Context := {
    fileName := "<AlphaGradPort.Tasks>"
    fileMap := default
  }
  let st : Core.State := { env := env }
  let eio := x.run ctx st
  let out ← EIO.toBaseIO eio
  match out with
  | .ok (value, _st') =>
    pure (.ok value)
  | .error err =>
    let msg ← err.toMessageData.toString
    pure (.error msg)

private def inferDenseVertexCount? (edges : Array LocalJacEdge) : Except String Nat := Id.run do
  if edges.isEmpty then
    return .error "KStmt lowering produced no local-Jac edges."

  let maxVertex :=
    edges.foldl (init := 0) fun acc e =>
      max acc (max e.src e.dst)

  if maxVertex = 0 then
    return .error "KStmt lowering produced only vertex 0; expected 1-based vertex IDs."

  let mut seen : Array Bool := Array.replicate (maxVertex + 1) false
  for e in edges do
    seen := seen.set! e.src true
    seen := seen.set! e.dst true

  let mut missing : Array Nat := #[]
  for v in [1:maxVertex + 1] do
    if !(seen.getD v false) then
      missing := missing.push v

  if !missing.isEmpty then
    return .error s!"KStmt lowering produced a non-dense vertex domain [1..{maxVertex}]; missing: {missing}"

  return .ok maxVertex

private def v (idx : Nat) : VarId := { idx := idx }

private def perceptronKStmts : Array KStmt := #[
  -- Inputs/params are v1..v8 (x,y,W1,b1,W2,b2,gamma,beta).
  KStmt.outer (v 9) (v 3) (v 1),
  KStmt.reduce .Sum .Col (v 10) (v 9),
  KStmt.binary .Add (v 11) (v 10) (v 4),
  KStmt.unary .Tanh (v 12) (v 11),

  -- Layer-norm style normalization path.
  KStmt.reduce .Sum .Row (v 13) (v 12),
  KStmt.binaryBroadcast .Sub .Row (v 14) (v 12) (v 13),
  KStmt.unary .Square (v 15) (v 14),
  KStmt.reduce .Sum .Row (v 16) (v 15),
  KStmt.unary .Sqrt (v 17) (v 16),
  KStmt.binaryBroadcast .Div .Row (v 18) (v 14) (v 17),
  KStmt.binaryBroadcast .Mul .Row (v 19) (v 18) (v 7),
  KStmt.binaryBroadcast .Add .Row (v 20) (v 19) (v 8),

  -- Output layer + softmax-cross-entropy style loss skeleton.
  KStmt.outer (v 21) (v 5) (v 20),
  KStmt.reduce .Sum .Col (v 22) (v 21),
  KStmt.binary .Add (v 23) (v 22) (v 6),
  KStmt.unary .Tanh (v 24) (v 23),
  KStmt.unary .Exp (v 25) (v 24),
  KStmt.reduce .Sum .Full (v 26) (v 25),
  KStmt.binaryBroadcast .Div .Row (v 27) (v 25) (v 26),
  KStmt.unary .Log (v 28) (v 27),
  KStmt.binary .Mul (v 29) (v 2) (v 28),
  KStmt.reduce .Sum .Full (v 30) (v 29),
  KStmt.unary .Neg (v 31) (v 30)
]

private def encoderKStmts : Array KStmt := #[
  -- Inputs/params are v1..v16.
  -- Block 1: q/k/v + attention-like aggregation + residual + norm + FFN.
  KStmt.outer (v 17) (v 3) (v 1),
  KStmt.reduce .Sum .Col (v 18) (v 17),
  KStmt.outer (v 19) (v 4) (v 1),
  KStmt.reduce .Sum .Col (v 20) (v 19),
  KStmt.outer (v 21) (v 5) (v 1),
  KStmt.reduce .Sum .Col (v 22) (v 21),

  KStmt.outer (v 23) (v 18) (v 20),
  KStmt.reduce .Sum .Col (v 24) (v 23),
  KStmt.outer (v 25) (v 24) (v 22),
  KStmt.reduce .Sum .Col (v 26) (v 25),
  KStmt.binary .Add (v 27) (v 1) (v 26),

  KStmt.reduce .Sum .Row (v 28) (v 27),
  KStmt.binaryBroadcast .Sub .Row (v 29) (v 27) (v 28),
  KStmt.unary .Square (v 30) (v 29),
  KStmt.reduce .Sum .Row (v 31) (v 30),
  KStmt.unary .Sqrt (v 32) (v 31),
  KStmt.binaryBroadcast .Div .Row (v 33) (v 29) (v 32),
  KStmt.binaryBroadcast .Mul .Row (v 34) (v 33) (v 8),
  KStmt.binaryBroadcast .Add .Row (v 35) (v 34) (v 9),

  KStmt.outer (v 36) (v 6) (v 35),
  KStmt.reduce .Sum .Col (v 37) (v 36),
  KStmt.binary .Add (v 38) (v 37) (v 7),
  KStmt.unary .Silu (v 39) (v 38),

  -- Block 2: same pattern on block-1 output.
  KStmt.outer (v 40) (v 10) (v 39),
  KStmt.reduce .Sum .Col (v 41) (v 40),
  KStmt.outer (v 42) (v 11) (v 39),
  KStmt.reduce .Sum .Col (v 43) (v 42),
  KStmt.outer (v 44) (v 12) (v 39),
  KStmt.reduce .Sum .Col (v 45) (v 44),

  KStmt.outer (v 46) (v 41) (v 43),
  KStmt.reduce .Sum .Col (v 47) (v 46),
  KStmt.outer (v 48) (v 47) (v 45),
  KStmt.reduce .Sum .Col (v 49) (v 48),
  KStmt.binary .Add (v 50) (v 39) (v 49),

  KStmt.reduce .Sum .Row (v 51) (v 50),
  KStmt.binaryBroadcast .Sub .Row (v 52) (v 50) (v 51),
  KStmt.unary .Square (v 53) (v 52),
  KStmt.reduce .Sum .Row (v 54) (v 53),
  KStmt.unary .Sqrt (v 55) (v 54),
  KStmt.binaryBroadcast .Div .Row (v 56) (v 52) (v 55),
  KStmt.binaryBroadcast .Mul .Row (v 57) (v 56) (v 15),
  KStmt.binaryBroadcast .Add .Row (v 58) (v 57) (v 16),

  KStmt.outer (v 59) (v 13) (v 58),
  KStmt.reduce .Sum .Col (v 60) (v 59),
  KStmt.binary .Add (v 61) (v 60) (v 14),
  KStmt.unary .Silu (v 62) (v 61),

  -- Classification loss proxy against y.
  KStmt.binary .Sub (v 63) (v 62) (v 2),
  KStmt.unary .Square (v 64) (v 63),
  KStmt.reduce .Sum .Full (v 65) (v 64)
]

private def robotArmKStmts : Array KStmt := #[
  -- Inputs are v1..v6 (t1..t6).
  KStmt.unary .Cos (v 7) (v 1),
  KStmt.unary .Cos (v 8) (v 2),
  KStmt.unary .Cos (v 9) (v 4),
  KStmt.unary .Cos (v 10) (v 5),
  KStmt.unary .Cos (v 11) (v 6),
  KStmt.unary .Sin (v 12) (v 1),
  KStmt.unary .Sin (v 13) (v 2),
  KStmt.unary .Sin (v 14) (v 4),
  KStmt.unary .Sin (v 15) (v 5),
  KStmt.unary .Sin (v 16) (v 6),

  KStmt.binary .Add (v 17) (v 2) (v 3),
  KStmt.unary .Cos (v 18) (v 17),
  KStmt.unary .Sin (v 19) (v 17),

  -- Axes and orientation proxies.
  KStmt.binary .Mul (v 20) (v 7) (v 18),
  KStmt.binary .Mul (v 21) (v 20) (v 9),
  KStmt.binary .Mul (v 22) (v 12) (v 14),
  KStmt.binary .Add (v 23) (v 21) (v 22),
  KStmt.binary .Mul (v 24) (v 15) (v 23),
  KStmt.binary .Mul (v 25) (v 7) (v 19),
  KStmt.binary .Mul (v 26) (v 25) (v 10),
  KStmt.binary .Add (v 27) (v 24) (v 26),

  KStmt.binary .Mul (v 28) (v 12) (v 18),
  KStmt.binary .Mul (v 29) (v 28) (v 9),
  KStmt.binary .Mul (v 30) (v 7) (v 14),
  KStmt.binary .Sub (v 31) (v 29) (v 30),
  KStmt.binary .Mul (v 32) (v 15) (v 31),
  KStmt.binary .Mul (v 33) (v 12) (v 19),
  KStmt.binary .Mul (v 34) (v 33) (v 10),
  KStmt.binary .Add (v 35) (v 32) (v 34),

  KStmt.binary .Mul (v 36) (v 19) (v 9),
  KStmt.binary .Mul (v 37) (v 36) (v 15),
  KStmt.binary .Mul (v 38) (v 18) (v 10),
  KStmt.binary .Sub (v 39) (v 37) (v 38),

  KStmt.binary .Mul (v 40) (v 18) (v 15),
  KStmt.binary .Mul (v 41) (v 36) (v 10),
  KStmt.binary .Add (v 42) (v 40) (v 41),
  KStmt.binary .Mul (v 43) (v 11) (v 42),
  KStmt.binary .Mul (v 44) (v 19) (v 14),
  KStmt.binary .Mul (v 45) (v 44) (v 16),
  KStmt.binary .Sub (v 46) (v 43) (v 45),

  KStmt.unary .Neg (v 47) (v 16),
  KStmt.binary .Mul (v 48) (v 47) (v 42),
  KStmt.binary .Mul (v 49) (v 44) (v 11),
  KStmt.binary .Sub (v 50) (v 48) (v 49),

  -- Position proxies.
  KStmt.binary .Add (v 51) (v 8) (v 18),
  KStmt.binary .Add (v 52) (v 51) (v 19),
  KStmt.binary .Mul (v 53) (v 7) (v 52),
  KStmt.binary .Add (v 54) (v 27) (v 53),
  KStmt.binary .Mul (v 55) (v 12) (v 52),
  KStmt.binary .Add (v 56) (v 35) (v 55),
  KStmt.binary .Add (v 57) (v 13) (v 19),
  KStmt.binary .Sub (v 58) (v 57) (v 18),
  KStmt.binary .Add (v 59) (v 58) (v 39),

  -- Final aggregated outputs analogous to (px, py, pz, z, y_, z'').
  KStmt.binary .Add (v 60) (v 54) (v 56),
  KStmt.binary .Add (v 61) (v 59) (v 46),
  KStmt.binary .Add (v 62) (v 50) (v 60)
]

private def blackScholesJacobianKStmts : Array KStmt := #[
  -- Inputs are v1..v5 (S, K, r, sigma, T).
  KStmt.binary .Mul (v 6) (v 3) (v 5),
  KStmt.unary .Exp (v 7) (v 6),
  KStmt.binary .Mul (v 8) (v 7) (v 1),
  KStmt.binary .Div (v 9) (v 8) (v 2),
  KStmt.unary .Log (v 10) (v 9),
  KStmt.unary .Square (v 11) (v 4),
  KStmt.binary .Mul (v 12) (v 11) (v 5),
  KStmt.binary .Add (v 13) (v 10) (v 12),
  KStmt.unary .Sqrt (v 14) (v 5),
  KStmt.binary .Mul (v 15) (v 4) (v 14),
  KStmt.binary .Div (v 16) (v 13) (v 15),
  KStmt.binary .Mul (v 17) (v 4) (v 14),
  KStmt.binary .Sub (v 18) (v 16) (v 17),

  -- Phi-like smooth proxies.
  KStmt.unary .Sigmoid (v 19) (v 16),
  KStmt.unary .Sigmoid (v 20) (v 18),
  KStmt.binary .Mul (v 21) (v 8) (v 19),
  KStmt.binary .Mul (v 22) (v 2) (v 20),
  KStmt.binary .Sub (v 23) (v 21) (v 22),
  KStmt.unary .Neg (v 24) (v 6),
  KStmt.unary .Exp (v 25) (v 24),
  KStmt.binary .Mul (v 26) (v 25) (v 23),

  -- Gradient and second-order-greek proxies.
  KStmt.binary .Add (v 27) (v 1) (v 2),
  KStmt.binary .Div (v 28) (v 26) (v 27),
  KStmt.binary .Div (v 29) (v 26) (v 2),
  KStmt.unary .Neg (v 30) (v 29),
  KStmt.binary .Mul (v 31) (v 26) (v 5),
  KStmt.binary .Mul (v 32) (v 26) (v 4),
  KStmt.binary .Mul (v 33) (v 26) (v 14),
  KStmt.binary .Add (v 34) (v 1) (v 5),
  KStmt.binary .Div (v 35) (v 28) (v 34),
  KStmt.binary .Mul (v 36) (v 32) (v 4),
  KStmt.unary .Neg (v 37) (v 33),
  KStmt.binary .Mul (v 38) (v 31) (v 3),

  -- Jacobian/Hessian-style structural path.
  KStmt.binary .Add (v 39) (v 28) (v 30),
  KStmt.binary .Add (v 40) (v 31) (v 32),
  KStmt.binary .Add (v 41) (v 33) (v 35),
  KStmt.binary .Add (v 42) (v 36) (v 37),
  KStmt.binary .Add (v 43) (v 38) (v 26),
  KStmt.outer (v 44) (v 39) (v 40),
  KStmt.outer (v 45) (v 41) (v 42),
  KStmt.concatCols (v 46) (v 44) (v 45),
  KStmt.reduce .Sum .Full (v 47) (v 46),
  KStmt.binary .Add (v 48) (v 47) (v 43),
  KStmt.cumsum .Row (v 49) (v 46),
  KStmt.cumprod .Row (v 50) (v 49),
  KStmt.reduce .Sum .Full (v 51) (v 50),
  KStmt.binary .Add (v 52) (v 48) (v 51)
]

private def materializeFromKStmts
    (name : String)
    (description : String)
    (stmts : Array KStmt)
    (envCfg : AlphaGradMctxConfig := {})
    (mctsCfg : AlphaGradMctsConfig := {}) :
    IO (Except String TaskSpec) := do
  let lowered ← runCoreMResult do
    registerKStmtAllSupportedHybridRules
    buildAndExtractFromKStmts {} stmts

  match lowered with
  | .error msg =>
    return .error s!"{name}: CoreM failure while lowering KStmt graph: {msg}"
  | .ok (.error err) =>
    return .error s!"{name}: {buildExtractErrorToString err}"
  | .ok (.ok (_jaxpr, edges)) =>
    match inferDenseVertexCount? edges with
    | .error msg =>
      return .error s!"{name}: {msg}"
    | .ok numVertices =>
      return .ok {
        name := name
        description := description
        numVertices := numVertices
        edges := edges
        envCfg := envCfg
        mctsCfg := mctsCfg
      }

private def taskSpecStatic : TaskName → TaskSpec
  | .roeFlux1d =>
    {
      name := "RoeFlux_1d"
      description := "Graphax RoeFlux_1d-inspired elimination graph; first end-to-end AlphaGrad port target."
      numVertices := 35
      edges := roeFlux1dEdges
      envCfg := { terminalBonus := 0.0, discount := 1.0 }
      mctsCfg := { numSimulations := 50, maxNumConsideredActions := 5, gumbelScale := 1.0 }
    }
  | .perceptron =>
    panic! "Perceptron task is now KStmt-lowered. Use materializeTask instead of taskSpec."
  | .encoder =>
    panic! "Encoder task is now KStmt-lowered. Use materializeTask instead of taskSpec."
  | .robotArm6DOF =>
    panic! "RobotArm_6DOF task is now KStmt-lowered. Use materializeTask instead of taskSpec."
  | .blackScholesJacobian =>
    panic! "BlackScholes_Jacobian task is now KStmt-lowered. Use materializeTask instead of taskSpec."
  | .humanHeartDipole =>
    {
      name := "HumanHeartDipole"
      description := "HumanHeartDipole-style surrogate graph used for staged AlphaGrad porting."
      numVertices := 26
      edges := heartDipoleEdges
      envCfg := { terminalBonus := 0.0, discount := 1.0 }
      mctsCfg := { numSimulations := 48, maxNumConsideredActions := 8, gumbelScale := 1.0 }
    }
  | .propaneCombustion =>
    {
      name := "PropaneCombustion"
      description := "PropaneCombustion-style surrogate graph used for staged AlphaGrad porting."
      numVertices := 32
      edges := propaneEdges
      envCfg := { terminalBonus := 0.0, discount := 1.0 }
      mctsCfg := { numSimulations := 56, maxNumConsideredActions := 9, gumbelScale := 1.0 }
    }

/--
Materialize task spec. Selected benchmark tasks are lowered from `KStmt`
at call-time; remaining tasks use static in-file specs.
-/
def materializeTask : TaskName → IO (Except String TaskSpec)
  | .perceptron =>
    materializeFromKStmts
      "Perceptron"
      "Perceptron graph lowered from Tyr KStmt IR (tanh + layer-norm + cross-entropy-style path)."
      perceptronKStmts
      { terminalBonus := 0.0, discount := 1.0 }
      { numSimulations := 36, maxNumConsideredActions := 7, gumbelScale := 1.0 }
  | .encoder =>
    materializeFromKStmts
      "Encoder"
      "Encoder graph lowered from Tyr KStmt IR (two attention-like blocks + normalization + loss proxy)."
      encoderKStmts
      { terminalBonus := 0.0, discount := 1.0 }
      { numSimulations := 40, maxNumConsideredActions := 10, gumbelScale := 1.0 }
  | .robotArm6DOF =>
    materializeFromKStmts
      "RobotArm_6DOF"
      "RobotArm_6DOF graph lowered from Tyr KStmt IR (6-DOF trig/pose dependency path)."
      robotArmKStmts
      { terminalBonus := 0.0, discount := 1.0 }
      { numSimulations := 48, maxNumConsideredActions := 8, gumbelScale := 1.0 }
  | .blackScholesJacobian =>
    materializeFromKStmts
      "BlackScholes_Jacobian"
      "BlackScholes_Jacobian graph lowered from Tyr KStmt IR (pricing + Jacobian/Hessian proxy path)."
      blackScholesJacobianKStmts
      { terminalBonus := 0.0, discount := 1.0 }
      { numSimulations := 40, maxNumConsideredActions := 6, gumbelScale := 1.0 }
  | task =>
    pure (.ok (taskSpecStatic task))

/--
Static task spec accessor.
For KStmt-lowered tasks (`Perceptron`, `Encoder`, `RobotArm_6DOF`,
`BlackScholes_Jacobian`) call `materializeTask`.
-/
def taskSpec (taskName : TaskName) : TaskSpec :=
  taskSpecStatic taskName

def taskNamesCsv : String :=
  String.intercalate ", " <| taskSequence.toList.map (fun t => toString t)

end Examples.AlphaGradPort
