import Lean
import Std.Data.HashMap
import Std.Data.HashSet
import Tyr.GPU.Codegen.TileIR.Expr

/-!
# Tyr.GPU.Codegen.TileIR.Passes

Optimization and normalization passes over the Lean-side TileIR AST.

The goal is to mirror the overall shape of cuTile's compiler pipeline while
staying appropriate for the current backend-first Lean representation:

- run pure AST-to-AST normalizations before rendering/compilation,
- keep the raw `Expr` layer as the ground truth IR,
- and reserve the more elaborate frontend translation for `@[tileir_kernel]`
  declarations later.
-/

namespace Tyr.GPU.Codegen.TileIR

open Std

abbrev NameSet := Std.HashSet String
abbrev NameSubst := Std.HashMap String String
abbrev ExprKeyMap := Std.HashMap String String

structure Pass where
  occurrence : Nat := 0
  name : Lean.Name
  run : Module → Module

structure PassSnapshot where
  name : Lean.Name
  occurrence : Nat := 0
  mod : Module

instance : Inhabited PassSnapshot where
  default := { name := .anonymous, mod := default }

structure PassManager where
  passes : Array Pass
  deriving Inhabited

structure BlockInfo where
  stmts : Array Stmt
  liveBefore : NameSet
  hasEffect : Bool
  deriving Inhabited

private def setInsertAll (set : NameSet) (names : Array String) : NameSet :=
  names.foldl (fun acc name => acc.insert name) set

private def setEraseAll (set : NameSet) (names : Array String) : NameSet :=
  names.foldl (fun acc name => acc.erase name) set

private def arrayAnyLive (set : NameSet) (names : Array String) : Bool :=
  names.any set.contains

private def eraseNamesFromSubst (subst : NameSubst) (names : Array String) : NameSubst :=
  names.foldl (fun acc name => acc.erase name) subst

private partial def resolveSubst (subst : NameSubst) (name : String) : String :=
  match subst.get? name with
  | some next =>
      if next == name then name else resolveSubst subst next
  | none => name

private def stmtDefsQuery? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .comment _ => some #[]
  | .const dst _ => some #[dst.name]
  | .getNumTileBlocks x y z => some #[x.name, y.name, z.name]
  | .getTileBlockId x y z => some #[x.name, y.name, z.name]
  | .iota dst => some #[dst.name]
  | .getGlobal dst _ => some #[dst.name]
  | _ => none

private def stmtDefsAlgebra? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .unary dst _ _ => some #[dst.name]
  | .binary dst _ _ _ => some #[dst.name]
  | .cmpf dst _ _ _ _ _ => some #[dst.name]
  | .cmpi dst _ _ _ _ _ => some #[dst.name]
  | .cat dst _ _ _ _ _ => some #[dst.name]
  | .mmaf dst _ _ _ _ _ _ => some #[dst.name]
  | .mmai dst _ _ _ _ _ _ _ _ => some #[dst.name]
  | .cast dst _ _ _ => some #[dst.name]
  | .broadcast dst _ _ => some #[dst.name]
  | .reshape dst _ _ => some #[dst.name]
  | .permute dst _ _ _ => some #[dst.name]
  | .extract dst _ _ _ => some #[dst.name]
  | .select dst _ _ _ _ _ => some #[dst.name]
  | .offset dst _ _ _ _ => some #[dst.name]
  | .makeTensorView dst _ _ _ => some #[dst.name]
  | .makePartitionView dst _ => some #[dst.name]
  | _ => none

private def stmtDefsMemory? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .loadPtrTko value token _ _ _ _ => some #[value.name, token.name]
  | .loadViewTko value token _ _ _ _ _ => some #[value.name, token.name]
  | .storePtrTko token _ _ _ _ _ _ => some #[token.name]
  | .storeViewTko token _ _ _ _ _ _ _ => some #[token.name]
  | .printTko token _ => some #[token.name]
  | .assertOp _ _ _ => some #[]
  | _ => none

private def stmtDefsStructured? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .ifOp results _ _ _ => some (results.map Binding.name)
  | .forOp results _ _ _ _ _ _ => some (results.map Binding.name)
  | .yieldOp _ => some #[]
  | .continueOp _ => some #[]
  | .breakOp _ => some #[]
  | _ => none

private def stmtDefs (stmt : Stmt) : Array String :=
  match stmtDefsQuery? stmt with
  | some defs => defs
  | none =>
      match stmtDefsAlgebra? stmt with
      | some defs => defs
      | none =>
          match stmtDefsMemory? stmt with
          | some defs => defs
          | none => (stmtDefsStructured? stmt).getD #[]

private def stmtLeafUsesQuery? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .comment _ => some #[]
  | .const _ _ => some #[]
  | .getNumTileBlocks .. => some #[]
  | .getTileBlockId .. => some #[]
  | .iota _ => some #[]
  | .printTko _ _ => some #[]
  | .getGlobal _ _ => some #[]
  | _ => none

private def stmtLeafUsesAlgebra? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .unary _ _ src => some #[src]
  | .binary _ _ lhs rhs => some #[lhs, rhs]
  | .cmpf _ _ _ lhs rhs _ => some #[lhs, rhs]
  | .cmpi _ _ lhs rhs _ _ => some #[lhs, rhs]
  | .cat _ lhs rhs _ _ _ => some #[lhs, rhs]
  | .mmaf _ a b c _ _ _ => some #[a, b, c]
  | .mmai _ a b c _ _ _ _ _ => some #[a, b, c]
  | .cast _ _ src _ => some #[src]
  | .broadcast _ src _ => some #[src]
  | .reshape _ src _ => some #[src]
  | .permute _ src _ _ => some #[src]
  | .extract _ src indices _ => some <| #[src] ++ indices
  | .select _ cond valIfTrue valIfFalse _ _ => some #[cond, valIfTrue, valIfFalse]
  | .offset _ ptr idx _ _ => some #[ptr, idx]
  | .makeTensorView _ base _ _ => some #[base]
  | .makePartitionView _ src => some #[src]
  | .assertOp cond _ _ => some #[cond]
  | _ => none

private def stmtLeafUsesMemory? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .loadPtrTko _ _ _ ptr inputToken _ => some #[ptr, inputToken]
  | .loadViewTko _ _ _ view indices inputToken _ => some <| #[view, inputToken] ++ indices
  | .storePtrTko _ _ ptr value inputToken _ _ => some #[ptr, value, inputToken]
  | .storeViewTko _ _ view indices value inputToken _ _ =>
      some <| #[view, value, inputToken] ++ indices
  | _ => none

private def stmtLeafUsesStructured? (stmt : Stmt) : Option (Array String) :=
  match stmt with
  | .ifOp _ cond _ _ => some #[cond]
  | .forOp _ _ lower upper step iterValues _ =>
      some <| #[lower, upper, step] ++ iterValues.map LoopCarry.init
  | .yieldOp values => some values
  | .continueOp values => some values
  | .breakOp values => some values
  | _ => none

private def stmtLeafUses (stmt : Stmt) : Array String :=
  match stmtLeafUsesQuery? stmt with
  | some uses => uses
  | none =>
      match stmtLeafUsesAlgebra? stmt with
      | some uses => uses
      | none =>
          match stmtLeafUsesMemory? stmt with
          | some uses => uses
          | none => (stmtLeafUsesStructured? stmt).getD #[]

private def stmtHasLeafEffect : Stmt → Bool
  | .storePtrTko .. => true
  | .storeViewTko .. => true
  | .printTko .. => true
  | .assertOp .. => true
  | _ => false

private def stmtMustKeep : Stmt → Bool
  | .comment _ => true
  | .yieldOp _ => true
  | .continueOp _ => true
  | .breakOp _ => true
  | _ => false

private def stmtIsPureHoistableLeaf : Stmt → Bool
  | .const .. => true
  | .getNumTileBlocks .. => true
  | .getTileBlockId .. => true
  | .iota .. => true
  | .unary .. => true
  | .binary .. => true
  | .cmpf .. => true
  | .cmpi .. => true
  | .cat .. => true
  | .mmaf .. => true
  | .mmai .. => true
  | .cast .. => true
  | .broadcast .. => true
  | .reshape .. => true
  | .permute .. => true
  | .extract .. => true
  | .select .. => true
  | .offset .. => true
  | .makeTensorView .. => true
  | .makePartitionView .. => true
  | .getGlobal .. => true
  | _ => false

private partial def dceBlock (stmts : Array Stmt) (liveAfter : NameSet := {}) : BlockInfo := Id.run do
  let mut live := liveAfter
  let mut keptRev : Array Stmt := #[]
  let mut hasEffect := false
  for stmt in stmts.reverse do
    match stmt with
    | .ifOp results cond thenBody elseBody =>
        let thenInfo := dceBlock thenBody {}
        let elseInfo := dceBlock elseBody {}
        let defs := results.map Binding.name
        let resultsLive := arrayAnyLive live defs
        let nestedEffect := thenInfo.hasEffect || elseInfo.hasEffect
        if resultsLive || nestedEffect then
          keptRev := keptRev.push <| .ifOp results cond thenInfo.stmts elseInfo.stmts
          live := setEraseAll live defs
          live := (live.insert cond)
          live := thenInfo.liveBefore.toList.foldl (fun acc name => acc.insert name) live
          live := elseInfo.liveBefore.toList.foldl (fun acc name => acc.insert name) live
          hasEffect := hasEffect || nestedEffect
    | .forOp results iv lower upper step iterValues body =>
        let bodyInfo := dceBlock body {}
        let mut bound : NameSet := {}
        bound := bound.insert iv.name
        for carry in iterValues do
          bound := bound.insert carry.binder.name
        let freeBody := bodyInfo.liveBefore.toList.foldl (fun acc name =>
          if bound.contains name then acc else acc.insert name
        ) ({} : NameSet)
        let defs := results.map Binding.name
        let resultsLive := arrayAnyLive live defs
        if resultsLive || bodyInfo.hasEffect then
          keptRev := keptRev.push <| .forOp results iv lower upper step iterValues bodyInfo.stmts
          live := setEraseAll live defs
          live := setInsertAll live #[lower, upper, step]
          live := setInsertAll live (iterValues.map LoopCarry.init)
          live := freeBody.toList.foldl (fun acc name => acc.insert name) live
          hasEffect := hasEffect || bodyInfo.hasEffect
    | _ =>
        let defs := stmtDefs stmt
        let keep :=
          stmtMustKeep stmt || stmtHasLeafEffect stmt || arrayAnyLive live defs
        if keep then
          keptRev := keptRev.push stmt
          live := setEraseAll live defs
          live := setInsertAll live (stmtLeafUses stmt)
          hasEffect := hasEffect || stmtHasLeafEffect stmt
  {
    stmts := keptRev.reverse
    liveBefore := live
    hasEffect := hasEffect
  }

private def stmtUsesBoundNames (bound : NameSet) (stmt : Stmt) : Bool :=
  (stmtLeafUses stmt).any bound.contains

private def encodeKey [Repr α] (value : α) : String :=
  reprStr value

private def cseKeyQuery? (stmt : Stmt) : Option String :=
  match stmt with
  | Stmt.const dst value =>
      some s!"const|{encodeKey dst.ty}|{encodeKey value}"
  | Stmt.iota dst =>
      some s!"iota|{encodeKey dst.ty}"
  | Stmt.getGlobal dst globalName =>
      some s!"getGlobal|{encodeKey dst.ty}|{globalName}"
  | _ =>
      none

private def cseKeyAlgebra? (stmt : Stmt) : Option String :=
  match stmt with
  | Stmt.unary dst op src =>
      some s!"unary|{encodeKey dst.ty}|{encodeKey op}|{src}"
  | Stmt.binary dst op lhs rhs =>
      some s!"binary|{encodeKey dst.ty}|{encodeKey op}|{lhs}|{rhs}"
  | Stmt.cmpf dst pred mode lhs rhs lhsTy =>
      some s!"cmpf|{encodeKey dst.ty}|{encodeKey pred}|{encodeKey mode}|{lhs}|{rhs}|{encodeKey lhsTy}"
  | Stmt.cmpi dst pred lhs rhs signedness lhsTy =>
      some s!"cmpi|{encodeKey dst.ty}|{encodeKey pred}|{lhs}|{rhs}|{encodeKey signedness}|{encodeKey lhsTy}"
  | Stmt.cat dst lhs rhs dim lhsTy rhsTy =>
      some s!"cat|{encodeKey dst.ty}|{lhs}|{rhs}|{dim}|{encodeKey lhsTy}|{encodeKey rhsTy}"
  | Stmt.mmaf dst a b c aTy bTy cTy =>
      some s!"mmaf|{encodeKey dst.ty}|{a}|{b}|{c}|{encodeKey aTy}|{encodeKey bTy}|{encodeKey cTy}"
  | Stmt.mmai dst a b c aTy bTy cTy aSigned bSigned =>
      some s!"mmai|{encodeKey dst.ty}|{a}|{b}|{c}|{encodeKey aTy}|{encodeKey bTy}|{encodeKey cTy}|{encodeKey aSigned}|{encodeKey bSigned}"
  | Stmt.cast dst op src srcTy =>
      some s!"cast|{encodeKey dst.ty}|{encodeKey op}|{src}|{encodeKey srcTy}"
  | Stmt.broadcast dst src srcTy =>
      some s!"broadcast|{encodeKey dst.ty}|{src}|{encodeKey srcTy}"
  | Stmt.reshape dst src srcTy =>
      some s!"reshape|{encodeKey dst.ty}|{src}|{encodeKey srcTy}"
  | Stmt.permute dst src permutation srcTy =>
      some s!"permute|{encodeKey dst.ty}|{src}|{encodeKey permutation}|{encodeKey srcTy}"
  | Stmt.extract dst src indices srcTy =>
      some s!"extract|{encodeKey dst.ty}|{src}|{encodeKey indices}|{encodeKey srcTy}"
  | Stmt.select dst cond valIfTrue valIfFalse condTy valueTy =>
      some s!"select|{encodeKey dst.ty}|{cond}|{valIfTrue}|{valIfFalse}|{encodeKey condTy}|{encodeKey valueTy}"
  | _ =>
      none

private def cseKeyView? (stmt : Stmt) : Option String :=
  match stmt with
  | Stmt.offset dst ptr idx ptrTy idxTy =>
      some s!"offset|{encodeKey dst.ty}|{ptr}|{idx}|{encodeKey ptrTy}|{encodeKey idxTy}"
  | Stmt.makeTensorView dst base shape strides =>
      some s!"makeTensorView|{encodeKey dst.ty}|{base}|{encodeKey shape}|{encodeKey strides}"
  | Stmt.makePartitionView dst src =>
      some s!"makePartitionView|{encodeKey dst.ty}|{src}"
  | _ =>
      none

private def cseKey? (stmt : Stmt) : Option String :=
  match cseKeyQuery? stmt with
  | some key => some key
  | none =>
      match cseKeyAlgebra? stmt with
      | some key => some key
      | none => cseKeyView? stmt

private def rewriteStmtTrivial? (subst : NameSubst) (stmt : Stmt) : Option Stmt :=
  let _ := subst
  match stmt with
  | .comment text => some <| .comment text
  | .const dst value => some <| .const dst value
  | .getNumTileBlocks x y z => some <| .getNumTileBlocks x y z
  | .getTileBlockId x y z => some <| .getTileBlockId x y z
  | .iota dst => some <| .iota dst
  | .printTko token message => some <| .printTko token message
  | .getGlobal dst globalName => some <| .getGlobal dst globalName
  | _ => none

private def rewriteStmtAlgebra? (subst : NameSubst) (stmt : Stmt) : Option Stmt :=
  match stmt with
  | .unary dst op src =>
      some <| .unary dst op (resolveSubst subst src)
  | .binary dst op lhs rhs =>
      some <| .binary dst op (resolveSubst subst lhs) (resolveSubst subst rhs)
  | .cmpf dst pred mode lhs rhs lhsTy =>
      some <| .cmpf dst pred mode
        (resolveSubst subst lhs)
        (resolveSubst subst rhs)
        lhsTy
  | .cmpi dst pred lhs rhs signedness lhsTy =>
      some <| .cmpi dst pred
        (resolveSubst subst lhs)
        (resolveSubst subst rhs)
        signedness
        lhsTy
  | .cat dst lhs rhs dim lhsTy rhsTy =>
      some <| .cat dst (resolveSubst subst lhs) (resolveSubst subst rhs) dim lhsTy rhsTy
  | .mmaf dst a b c aTy bTy cTy =>
      some <| .mmaf dst (resolveSubst subst a) (resolveSubst subst b) (resolveSubst subst c) aTy bTy cTy
  | .mmai dst a b c aTy bTy cTy aSigned bSigned =>
      some <| .mmai dst (resolveSubst subst a) (resolveSubst subst b) (resolveSubst subst c) aTy bTy cTy aSigned bSigned
  | .cast dst op src srcTy =>
      some <| .cast dst op (resolveSubst subst src) srcTy
  | .broadcast dst src srcTy =>
      some <| .broadcast dst (resolveSubst subst src) srcTy
  | .reshape dst src srcTy =>
      some <| .reshape dst (resolveSubst subst src) srcTy
  | .permute dst src permutation srcTy =>
      some <| .permute dst (resolveSubst subst src) permutation srcTy
  | .extract dst src indices srcTy =>
      some <| .extract dst (resolveSubst subst src) (indices.map (resolveSubst subst)) srcTy
  | .select dst cond valIfTrue valIfFalse condTy valueTy =>
      some <| .select dst
        (resolveSubst subst cond)
        (resolveSubst subst valIfTrue)
        (resolveSubst subst valIfFalse)
        condTy valueTy
  | .offset dst ptr idx ptrTy idxTy =>
      some <| .offset dst (resolveSubst subst ptr) (resolveSubst subst idx) ptrTy idxTy
  | .makeTensorView dst base shape strides =>
      some <| .makeTensorView dst (resolveSubst subst base) shape strides
  | .makePartitionView dst src =>
      some <| .makePartitionView dst (resolveSubst subst src)
  | .assertOp cond condTy message =>
      some <| .assertOp (resolveSubst subst cond) condTy message
  | _ => none

private def rewriteStmtMemory? (subst : NameSubst) (stmt : Stmt) : Option Stmt :=
  match stmt with
  | .loadPtrTko value token order ptr inputToken ptrTy =>
      some <| .loadPtrTko value token order
        (resolveSubst subst ptr)
        (resolveSubst subst inputToken)
        ptrTy
  | .loadViewTko value token order view indices inputToken viewTy =>
      some <| .loadViewTko value token order
        (resolveSubst subst view)
        (indices.map (resolveSubst subst))
        (resolveSubst subst inputToken)
        viewTy
  | .storePtrTko token order ptr value inputToken ptrTy valueTy =>
      some <| .storePtrTko token order
        (resolveSubst subst ptr)
        (resolveSubst subst value)
        (resolveSubst subst inputToken)
        ptrTy valueTy
  | .storeViewTko token order view indices value inputToken viewTy valueTy =>
      some <| .storeViewTko token order
        (resolveSubst subst view)
        (indices.map (resolveSubst subst))
        (resolveSubst subst value)
        (resolveSubst subst inputToken)
        viewTy valueTy
  | _ => none

private def rewriteLoopCarryInit (subst : NameSubst) (carry : LoopCarry) : LoopCarry :=
  { carry with init := resolveSubst subst carry.init }

mutual

private partial def rewriteStmtInputs (subst : NameSubst) (stmt : Stmt) : Stmt :=
  match rewriteStmtTrivial? subst stmt with
  | some stmt => stmt
  | none =>
      match rewriteStmtAlgebra? subst stmt with
      | some stmt => stmt
      | none =>
          match rewriteStmtMemory? subst stmt with
          | some stmt => stmt
          | none =>
              match rewriteStmtStructured? subst stmt with
              | some stmt => stmt
              | none => panic! "unreachable"

private partial def rewriteBlockInputs (subst : NameSubst) (stmts : Array Stmt) : Array Stmt :=
  stmts.map (fun stmt => rewriteStmtInputs subst stmt)

private partial def rewriteStmtStructured? (subst : NameSubst) (stmt : Stmt) : Option Stmt :=
  match stmt with
  | .ifOp results cond thenBody elseBody =>
      some <| .ifOp results
        (resolveSubst subst cond)
        (rewriteBlockInputs subst thenBody)
        (rewriteBlockInputs subst elseBody)
  | .forOp results iv lower upper step iterValues body =>
      some <| .forOp results iv
        (resolveSubst subst lower)
        (resolveSubst subst upper)
        (resolveSubst subst step)
        (iterValues.map (rewriteLoopCarryInit subst))
        (rewriteBlockInputs subst body)
  | .yieldOp values =>
      some <| .yieldOp (values.map (resolveSubst subst))
  | .continueOp values =>
      some <| .continueOp (values.map (resolveSubst subst))
  | .breakOp values =>
      some <| .breakOp (values.map (resolveSubst subst))
  | _ => none

end

private partial def cseBlock
    (stmts : Array Stmt)
    (substIn : NameSubst := {})
    (seenIn : ExprKeyMap := {})
    : Array Stmt := Id.run do
  let mut subst := substIn
  let mut seen := seenIn
  let mut out : Array Stmt := #[]
  for stmt in stmts do
    let stmt := rewriteStmtInputs subst stmt
    match stmt with
    | .ifOp results cond thenBody elseBody =>
        let thenBody := cseBlock thenBody subst seen
        let elseBody := cseBlock elseBody subst seen
        out := out.push <| .ifOp results cond thenBody elseBody
        subst := eraseNamesFromSubst subst (results.map Binding.name)
    | .forOp results iv lower upper step iterValues body =>
        let bodySubst :=
          eraseNamesFromSubst subst <|
            #[iv.name] ++ iterValues.map (fun carry => carry.binder.name)
        let body := cseBlock body bodySubst seen
        out := out.push <| .forOp results iv lower upper step iterValues body
        subst := eraseNamesFromSubst subst (results.map Binding.name)
    | _ =>
        if let some key := cseKey? stmt then
          let defs := stmtDefs stmt
          if defs.size = 1 then
            let dst := defs[0]!
            match seen.get? key with
            | some prev =>
                subst := eraseNamesFromSubst subst defs
                subst := subst.insert dst prev
            | none =>
                out := out.push stmt
                seen := seen.insert key dst
                subst := eraseNamesFromSubst subst defs
          else
            out := out.push stmt
            subst := eraseNamesFromSubst subst defs
        else
          out := out.push stmt
          subst := eraseNamesFromSubst subst (stmtDefs stmt)
  out

private partial def hoistLoopInvariantsBlock (stmts : Array Stmt) : Array Stmt := Id.run do
  let mut out : Array Stmt := #[]
  for stmt in stmts do
    match stmt with
    | .ifOp results cond thenBody elseBody =>
        out := out.push <| .ifOp results cond
          (hoistLoopInvariantsBlock thenBody)
          (hoistLoopInvariantsBlock elseBody)
    | .forOp results iv lower upper step iterValues body =>
        let body := hoistLoopInvariantsBlock body
        let mut bound : NameSet := {}
        bound := bound.insert iv.name
        for carry in iterValues do
          bound := bound.insert carry.binder.name
        let mut localDefs := bound
        let mut hoisted : Array Stmt := #[]
        let mut keptBody : Array Stmt := #[]
        for bodyStmt in body do
          if stmtIsPureHoistableLeaf bodyStmt && !stmtUsesBoundNames localDefs bodyStmt then
            hoisted := hoisted.push bodyStmt
          else
            keptBody := keptBody.push bodyStmt
            localDefs := setInsertAll localDefs (stmtDefs bodyStmt)
        out := out ++ hoisted
        out := out.push <| .forOp results iv lower upper step iterValues keptBody
    | _ =>
        out := out.push stmt
  out

def deadCodeElimination (mod : Module) : Module :=
  {
    mod with
    entries := mod.entries.map fun entry =>
      { entry with body := (dceBlock entry.body {}).stmts }
  }

def commonSubexpressionElimination (mod : Module) : Module :=
  {
    mod with
    entries := mod.entries.map fun entry =>
      { entry with body := cseBlock entry.body }
  }

def hoistLoopInvariants (mod : Module) : Module :=
  {
    mod with
    entries := mod.entries.map fun entry =>
      { entry with body := hoistLoopInvariantsBlock entry.body }
  }

namespace Pass

def deadCodeElimination : Pass :=
  { name := `dce, run := Tyr.GPU.Codegen.TileIR.deadCodeElimination }

def commonSubexpressionElimination : Pass :=
  { name := `cse, run := Tyr.GPU.Codegen.TileIR.commonSubexpressionElimination }

def hoistLoopInvariants : Pass :=
  { name := `hoistLoopInvariants, run := Tyr.GPU.Codegen.TileIR.hoistLoopInvariants }

def deadCodeEliminationFinal : Pass :=
  { occurrence := 1, name := `dce, run := Tyr.GPU.Codegen.TileIR.deadCodeElimination }

end Pass

namespace PassManager

def run (manager : PassManager) (mod : Module) : Module :=
  manager.passes.foldl (fun acc pass => pass.run acc) mod

def runWithTrace (manager : PassManager) (mod : Module) : Array PassSnapshot := Id.run do
  let mut current := mod
  let mut snapshots : Array PassSnapshot := #[]
  for pass in manager.passes do
    current := pass.run current
    snapshots := snapshots.push {
      name := pass.name
      occurrence := pass.occurrence
      mod := current
    }
  snapshots

end PassManager

def builtinPassManager : PassManager :=
  { passes := #[
      Pass.deadCodeElimination,
      Pass.commonSubexpressionElimination,
      Pass.hoistLoopInvariants,
      Pass.deadCodeEliminationFinal
    ] }

def optimizeModule (mod : Module) : Module :=
  builtinPassManager.run mod

def optimizeModuleWithTrace (mod : Module) : Array PassSnapshot :=
  builtinPassManager.runWithTrace mod

end Tyr.GPU.Codegen.TileIR
