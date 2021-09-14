import Tyr
import Lean.Compiler.IR

namespace Lean.IR

def a''' := FnBody.vdecl (var 1) IRType.object (Expr.fap "f1" #[Arg.var (var 0)])
  $ FnBody.vdecl (var 2) IRType.object (Expr.fap "f2" #[Arg.var (var 1)])
  $ FnBody.vdecl (var 3) IRType.object (Expr.fap "f3" #[Arg.var (var 2)])
  $ FnBody.ret (Arg.var (var 3))

def b' := Decl.fdecl
  "test" -- function name
  #[mkParam (var 0) false IRType.object] -- arguments
  IRType.object -- return type
  a'''
  {}


#eval a'''
#eval (collectContext b').map (fun param => param.x)
#eval (collectContextFnBody a''').map (fun param => param.x)
#eval b'
#eval b'
-- def test (x_0 : obj) : obj :=
--   let x_1 : obj := f1 x_0;
--   let x_2 : obj := f2 x_1;
--   let x_3 : obj := f3 x_1;
--   ret x_3
#eval b'.uniqueIds
#eval b'.highestId
#eval transposeDecl b'
-- def test.T (x_0 : obj) (x_1 : obj) (x_2 : obj) (x_6 : obj) : obj :=
--   let x_5 : obj := f3.T x_2 x_6;
--   let x_4 : obj := f2.T x_1 x_5;
--   let x_3 : obj := f1.T x_0 x_4;
--   ret x_3
#eval a'''
#eval (collectLiveVars a''' {}).toList
#eval (returnArg a''') |> (fun x => match x with | Arg.var idx => idx | other => unreachable!)

def c := FnBody.ret (Arg.var (var 1))
def c' := FnBody.vdecl (var 1) IRType.object (Expr.proj 0 (var 0)) c
def c'' := Decl.fdecl
  "test" -- function name
  #[mkParam (var 0) false IRType.object] -- arguments
  IRType.object -- return type
  c'
  {}

#eval (collectContext c'').map (fun param => param.x)
#eval c''
#eval transposeDecl c''


def d' := FnBody.vdecl (var 4) IRType.object (Expr.ctor {name := `C, cidx := 0, size := 0, usize := 0, ssize := 24}  #[])
  $ FnBody.sset (var 4) 0 0 (var 1) IRType.float
  $ FnBody.sset (var 4) 0 8 (var 2) IRType.float
  $ FnBody.sset (var 4) 0 16 (var 3) IRType.float
  $ FnBody.ret (Arg.var (var 4))
def d'' := Decl.fdecl
  "test" -- function name
  #[mkParam (var 1) false IRType.object,
    mkParam (var 2) false IRType.object,
    mkParam (var 3) false IRType.object
  ] -- arguments
  IRType.object -- return type
  d'
  {}

#eval d''
#eval transposeDecl d''


def e' := FnBody.vdecl (var 4) IRType.object (Expr.box IRType.float (var 2))
  $ FnBody.vdecl (var 5) IRType.object (Expr.box IRType.float (var 3))
  $ FnBody.vdecl (var 6) IRType.object (Expr.ctor {name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0}  #[Arg.var (var 4), Arg.var (var 5)])
  $ FnBody.vdecl (var 7) IRType.object (Expr.box IRType.float (var 1))
  $ FnBody.vdecl (var 8) IRType.object (Expr.ctor {name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0}  #[Arg.var (var 7), Arg.var (var 6)])
  $ FnBody.ret (Arg.var (var 8))

def e'' := Decl.fdecl
  "test" -- function name
  #[mkParam (var 1) false IRType.object] -- arguments
  IRType.object -- return type
  e'
  {}



#eval e''
#eval transposeDecl e''

def eee := Decl.fdecl
  "test" -- function name
  #[mkParam (var 1) false IRType.float, mkParam (var 2) false IRType.float] -- arguments
  IRType.object -- return type
  (FnBody.vdecl (var 3) IRType.object (Expr.box IRType.float (var 1))
    $ FnBody.vdecl (var 4) IRType.object (Expr.box IRType.float (var 2))
    $ FnBody.vdecl (var 5) IRType.object (Expr.ctor {name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0}  #[Arg.var (var 3), Arg.var (var 4)])
    $ FnBody.ret (Arg.var (var 5)))
  {}

#eval eee
#eval transposeDecl eee
