import Tyr.GPU.Codegen.TileIR.Expr

/-!
# Tyr.GPU.Codegen.TileIR.Render

Render Lean TileIR modules to the public `cuda_tile.*` MLIR syntax.
-/

namespace Tyr.GPU.Codegen.TileIR

set_option maxHeartbeats 5000000

private def indent (depth : Nat) : String :=
  String.ofList <| List.replicate (depth * 2) ' '

private def ensureSSA (name : String) : String :=
  if name.startsWith "%" then name else "%" ++ name

private def ensureSymbol (name : String) : String :=
  if name.startsWith "@" then name else "@" ++ name

private def renderParam (param : Param) : String :=
  s!"{ensureSSA param.name}: {param.ty.render}"

private def renderBindings (bindings : Array Binding) : String :=
  String.intercalate ", " <| bindings.toList.map (fun b => ensureSSA b.name)

private def renderBindingTypes (bindings : Array Binding) : String :=
  String.intercalate ", " <| bindings.toList.map (fun b => b.ty.render)

private def renderValues (values : Array String) : String :=
  String.intercalate ", " <| values.toList.map ensureSSA

private def renderIndexList (indices : Array String) : String :=
  String.intercalate ", " <| indices.toList.map ensureSSA

private def renderFloat (value : Float) : String :=
  let text := toString value
  let hasDot := (text.splitOn ".").length > 1
  let hasLowerExp := (text.splitOn "e").length > 1
  let hasUpperExp := (text.splitOn "E").length > 1
  if hasDot || hasLowerExp || hasUpperExp then
    text
  else
    text ++ ".0"

private partial def renderLiteralPayload : Literal → String
  | .int value => toString value
  | .float value => renderFloat value
  | .bool value => if value then "true" else "false"
  | .array items =>
      s!"[{String.intercalate ", " <| items.toList.map renderLiteralPayload}]"

private def renderLiteral (ty : TileType) (value : Literal) : String :=
  match ty.literalScalar? with
  | some scalar =>
      s!"<{scalar.render}: {renderLiteralPayload value}>"
  | none =>
      panic! s!"TileIR constant expects a scalar-backed tile type, got {ty.render}"

private def renderShapeList (shape : Array ShapeDim) : String :=
  String.intercalate ", " <| shape.toList.map ShapeDim.render

private def renderCastStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .cast dst op src srcTy =>
      let suffix :=
        match op with
        | .bitcast => ""
        | .exti signedness => s!" {signedness.render}"
        | .trunci => ""
        | .ftof rounding => s!" {rounding.render}"
        | .ftoi signedness rounding => s!" {signedness.render} {rounding.render}"
        | .itof signedness rounding => s!" {signedness.render} {rounding.render}"
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.{op.render} {ensureSSA src}{suffix} : {srcTy.render} -> {dst.ty.render}"
  | _ =>
      none

private def renderQueryStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .comment text =>
      some s!"{indent depth}// {text}"
  | .const dst value =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.constant {renderLiteral dst.ty value} : {dst.ty.render}"
  | .getNumTileBlocks x y z =>
      some s!"{indent depth}{renderBindings #[x, y, z]} = cuda_tile.get_num_tile_blocks : {renderBindingTypes #[x, y, z]}"
  | .getTileBlockId x y z =>
      some s!"{indent depth}{renderBindings #[x, y, z]} = cuda_tile.get_tile_block_id : {renderBindingTypes #[x, y, z]}"
  | .iota dst =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.iota : {dst.ty.render}"
  | .getGlobal dst globalName =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.get_global {ensureSymbol globalName} : {dst.ty.render}"
  | _ =>
      none

private def renderAlgebraStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .unary dst op src =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.{op.render} {ensureSSA src} : {dst.ty.render}"
  | .binary dst op lhs rhs =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.{op.render} {ensureSSA lhs}, {ensureSSA rhs} : {dst.ty.render}"
  | .cmpf dst pred mode lhs rhs lhsTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.cmpf {pred.render} {mode.render} {ensureSSA lhs}, {ensureSSA rhs} : {lhsTy.render} -> {dst.ty.render}"
  | .cmpi dst pred lhs rhs signedness lhsTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.cmpi {pred.render} {ensureSSA lhs}, {ensureSSA rhs}, {signedness.render} : {lhsTy.render} -> {dst.ty.render}"
  | .cat dst lhs rhs dim lhsTy rhsTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.cat {ensureSSA lhs}, {ensureSSA rhs} dim = {dim} : {lhsTy.render}, {rhsTy.render} -> {dst.ty.render}"
  | .mmaf dst a b c aTy bTy cTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.mmaf {ensureSSA a}, {ensureSSA b}, {ensureSSA c} : {aTy.render}, {bTy.render}, {cTy.render}"
  | .mmai dst a b c aTy bTy cTy aSigned bSigned =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.mmai {ensureSSA a}, {ensureSSA b}, {ensureSSA c} {aSigned.render} {bSigned.render} : {aTy.render}, {bTy.render}, {cTy.render}"
  | .broadcast dst src srcTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.broadcast {ensureSSA src} : {srcTy.render} -> {dst.ty.render}"
  | .reshape dst src srcTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.reshape {ensureSSA src} : {srcTy.render} -> {dst.ty.render}"
  | .permute dst src permutation srcTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.permute {ensureSSA src} [{String.intercalate ", " (permutation.toList.map toString)}] : {srcTy.render} -> {dst.ty.render}"
  | .extract dst src indices srcTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.extract {ensureSSA src}[{renderIndexList indices}] : {srcTy.render} -> {dst.ty.render}"
  | .select dst cond valIfTrue valIfFalse condTy valueTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.select {ensureSSA cond}, {ensureSSA valIfTrue}, {ensureSSA valIfFalse} : {condTy.render}, {valueTy.render} -> {dst.ty.render}"
  | _ =>
      none

private def renderViewStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .offset dst ptr idx ptrTy idxTy =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.offset {ensureSSA ptr}, {ensureSSA idx} : {ptrTy.render}, {idxTy.render} -> {dst.ty.render}"
  | .makeTensorView dst base shape strides =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.make_tensor_view {ensureSSA base}, shape = [{renderShapeList shape}], strides = [{renderShapeList strides}] : {dst.ty.render}"
  | .makePartitionView dst src =>
      some s!"{indent depth}{ensureSSA dst.name} = cuda_tile.make_partition_view {ensureSSA src} : {dst.ty.render}"
  | _ =>
      none

private def renderMemoryStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .loadPtrTko value token order ptr inputToken ptrTy =>
      some s!"{indent depth}{ensureSSA value.name}, {ensureSSA token.name} = cuda_tile.load_ptr_tko {order.render} {ensureSSA ptr} token={ensureSSA inputToken} : {ptrTy.render} -> {value.ty.render}, {token.ty.render}"
  | .loadViewTko value token order view indices inputToken viewTy =>
      some s!"{indent depth}{ensureSSA value.name}, {ensureSSA token.name} = cuda_tile.load_view_tko {order.render} {ensureSSA view}[{renderIndexList indices}] token={ensureSSA inputToken} : {viewTy.render} -> {value.ty.render}, {token.ty.render}"
  | .storePtrTko token order ptr value inputToken ptrTy valueTy =>
      some s!"{indent depth}{ensureSSA token.name} = cuda_tile.store_ptr_tko {order.render} {ensureSSA ptr}, {ensureSSA value} token={ensureSSA inputToken} : {ptrTy.render}, {valueTy.render} -> {token.ty.render}"
  | .storeViewTko token order view indices value inputToken viewTy valueTy =>
      some s!"{indent depth}{ensureSSA token.name} = cuda_tile.store_view_tko {order.render} {ensureSSA view}[{renderIndexList indices}], {ensureSSA value} token={ensureSSA inputToken} : {viewTy.render}, {valueTy.render} -> {token.ty.render}"
  | .printTko token message =>
      some s!"{indent depth}{ensureSSA token.name} = cuda_tile.print_tko \"{message}\" -> {token.ty.render}"
  | .assertOp cond condTy message =>
      some s!"{indent depth}cuda_tile.assert {ensureSSA cond}, \"{message}\" : {condTy.render}"
  | _ =>
      none

private def renderTerminatorStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  match stmt with
  | .yieldOp values =>
      some s!"{indent depth}cuda_tile.yield {renderValues values}"
  | .continueOp values =>
      some s!"{indent depth}cuda_tile.continue {renderValues values}"
  | .breakOp values =>
      some s!"{indent depth}cuda_tile.break {renderValues values}"
  | _ =>
      none

private def renderLeafStmt? (depth : Nat) (stmt : Stmt) : Option String :=
  renderCastStmt? depth stmt <|>
    renderQueryStmt? depth stmt <|>
    renderAlgebraStmt? depth stmt <|>
    renderViewStmt? depth stmt <|>
    renderMemoryStmt? depth stmt <|>
    renderTerminatorStmt? depth stmt <|>
    match stmt with
    | .ifOp .. | .forOp .. =>
        none
    | _ =>
      none

private partial def renderStmt (depth : Nat) (stmt : Stmt) : String :=
  match stmt with
  | .ifOp results cond thenBody elseBody =>
      let lhs := renderBindings results
      let tys := renderBindingTypes results
      let thenText := String.intercalate "\n" <| thenBody.toList.map (renderStmt (depth + 1))
      let elseText := String.intercalate "\n" <| elseBody.toList.map (renderStmt (depth + 1))
      indent depth ++ lhs ++ " = cuda_tile.if " ++ ensureSSA cond ++ " -> (" ++ tys ++ ") {\n" ++
        thenText ++ "\n" ++ indent depth ++ "} else {\n" ++ elseText ++ "\n" ++ indent depth ++ "}"
  | .forOp results iv lower upper step iterValues body =>
      let lhs := renderBindings results
      let tys := renderBindingTypes results
      let iterPart :=
        if iterValues.isEmpty then
          ""
        else
          let parts :=
            iterValues.toList.map fun carry =>
              s!"{ensureSSA carry.binder.name} = {ensureSSA carry.init}"
          s!" iter_values({String.intercalate ", " parts})"
      let bodyText := String.intercalate "\n" <| body.toList.map (renderStmt (depth + 1))
      indent depth ++ lhs ++ " = cuda_tile.for " ++ ensureSSA iv.name ++
        " in (" ++ ensureSSA lower ++ " to " ++ ensureSSA upper ++ ", step " ++ ensureSSA step ++
        ") : " ++ iv.ty.render ++ iterPart ++ " -> (" ++ tys ++ ") {\n" ++
        bodyText ++ "\n" ++ indent depth ++ "}"
  | _ =>
      match renderLeafStmt? depth stmt with
      | some text => text
      | none => panic! "unreachable"

private def renderGlobal (global : Global) : String :=
  s!"  cuda_tile.global {ensureSymbol global.name} {renderLiteral global.ty global.value} : {global.ty.render}"

private def renderEntry (entry : Entry) : String :=
  let params := String.intercalate ", " <| entry.params.toList.map renderParam
  let body := String.intercalate "\n" <| entry.body.toList.map (renderStmt 2)
  "  cuda_tile.entry " ++ ensureSymbol entry.name ++ "(" ++ params ++ ") {\n" ++ body ++ "\n  }"

/-- Render a TileIR type using canonical public syntax. -/
def renderType (ty : TileType) : String :=
  ty.render

/-- Render a complete TileIR module using canonical public syntax. -/
def renderModule (mod : Module) : String :=
  let globals := mod.globals.toList.map renderGlobal
  let entries := mod.entries.toList.map renderEntry
  let items := globals ++ entries
  let body :=
    if items.isEmpty then
      ""
    else
      "\n" ++ String.intercalate "\n\n" items ++ "\n"
  "cuda_tile.module " ++ ensureSymbol mod.name ++ " {" ++ body ++ "}\n"

/-- Render a module after the default TileIR optimization pipeline. -/
def renderOptimizedModule (mod : Module) : String :=
  renderModule mod

end Tyr.GPU.Codegen.TileIR
