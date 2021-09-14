import Lean.Compiler.IR
import Lean.Compiler.IR.Basic
import Lean.Compiler.IR.LiveVars

namespace Lean
namespace IR

def define (f : FunId) (params : Array Param) := Decl.fdecl f params
def var (idx : Index ) := VarId.mk idx
def num (val : Nat) := Expr.lit (LitVal.num 12)

def argToParam : Arg → Param
| Arg.var idx => mkParam idx false IRType.object
| Arg.irrelevant => unreachable!

def argToVarId : Arg → VarId
| Arg.var idx => idx
| Arg.irrelevant => unreachable!

def collectContextExpr : Expr → Array Param
| Expr.fap c ys => ys.map argToParam
| Expr.proj i y => #[mkParam y false IRType.object]
| other => #[]

partial def collectContextFnBody : FnBody → Array Param
| FnBody.vdecl x ty e b => collectContextExpr e ++ collectContextFnBody b
| other => #[]

def collectContext : Decl → Array Param
| Decl.fdecl f xs ty b _  => collectContextFnBody b
| Decl.extern f xs ty _   => #[]

def transposeExpr (arg : Arg) : Expr → Expr
| Expr.fap c ys => Expr.fap (c ++ "T") (ys ++ #[arg])
| Expr.ctor i ys => Expr.ctor i ys
| other => other

partial def transposeConstructor (high : Index) (ty : IRType) (args : Array VarId) (x : VarId) (b : FnBody) : FnBody :=
let rec loop (idx : Nat) (acc : FnBody) :=
  if idx >= args.size then acc else
    let v := var (args[idx].idx + high);
    let dx := var (x.idx + high);
    let acc' := FnBody.vdecl v ty (Expr.proj idx dx)  acc;
    loop (idx + 1) acc'
loop 0 b

partial def transposeFnBody (high : Index) (b : FnBody) : FnBody :=
let rec loop (curr acc : FnBody) : FnBody :=
match curr with
| FnBody.vdecl x ty e b =>
  match e with
  | Expr.proj i y => loop b (FnBody.set y i (Arg.var (var (x.idx + high))) acc)
  | Expr.uproj i y => loop b (FnBody.set y i (Arg.var (var (x.idx + high))) acc)
  | Expr.ctor cinfo args =>
    let vars := args.map (fun x => var ((argToVarId x).idx));
    let acc' := transposeConstructor high ty vars x acc;
    loop b acc'
  | Expr.box t i =>
    let dyId := var (i.idx + high-1);
    let dxId := var (x.idx + high);
    loop b (FnBody.vdecl dyId t (Expr.unbox dxId) acc)
  | other => let dxId := var (x.idx + high - 1);
    loop b (FnBody.vdecl dxId ty (transposeExpr (Arg.var (var (x.idx + high))) e) acc)
| FnBody.ret x => acc
| FnBody.sset x i o y ty b  =>
    let dyId := var (y.idx + high);
    let dxId := var (x.idx + high);
    loop b (FnBody.vdecl dyId ty (Expr.sproj i o dxId) acc)
| other => other
loop b (FnBody.ret (Arg.var (var high)))

partial def returnArg : FnBody → Arg
| FnBody.vdecl x ty e b      => returnArg b
| FnBody.jdecl j xs v b      => unreachable!
| FnBody.set x i y b         => returnArg b
| FnBody.uset x i y b        => returnArg b
| FnBody.sset x i o y ty b   => returnArg b
| FnBody.setTag x cidx b     => returnArg b
| FnBody.inc x n c _ b       => returnArg b
| FnBody.dec x n c _ b       => returnArg b
| FnBody.del x b             => returnArg b
| FnBody.mdata d b           => returnArg b
| FnBody.case tid x xType cs => unreachable!
| FnBody.jmp j ys            => unreachable!
| FnBody.ret x               => x
| FnBody.unreachable         => unreachable!


def transposeDecl : Decl → Decl
| d@(Decl.fdecl f xs ty b m) =>
  let ctx := collectContextFnBody b;
  let high := d.highestId;
  let retvar := returnArg b |> (fun x => match x with | Arg.var idx => idx | other => unreachable!);
  Decl.fdecl (f ++ "T") (ctx ++ #[mkParam (var (retvar.idx + high)) false IRType.object]) ty (transposeFnBody high b) m
| other => other


end IR
end Lean
