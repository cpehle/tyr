#!/usr/bin/env python3
"""PyTorch references for fused residual + LayerNorm at [1,64,1024]."""
import argparse, json, torch
import torch.nn.functional as F


def percentile(xs, pct):
    xs = sorted(xs); return xs[((len(xs)-1)*pct)//100]


def measure(fn, warmup, iters, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); samples=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters): fn()
        b.record(); b.synchronize(); samples.append(a.elapsed_time(b)/iters)
    return samples


def functional(x, residual, weight, bias):
    resid = x + residual
    return F.layer_norm(resid, (1024,), weight, bias, 1e-5), resid


def into(out, out_resid, x, residual, weight, bias):
    y, resid = functional(x, residual, weight, bias)
    out.copy_(y); out_resid.copy_(resid)
    return out, out_resid


def make_row(args, case, backend, route, samples, ok, scope, allocation_free):
    return {"event":"summary","schemaVersion":1,"runId":args.run_id,
      "caseId":case,"backend":backend,"routeActual":route,"timer":"cuda_event", "completionFence": "cudaEventSynchronize(stop)",
      "timingScope":scope,"warmAllocationFree":allocation_free,"compileSetupExcluded":backend=="torch_compile","warmup":args.warmup,
      "iters":args.iters,"repeats":args.repeats,"correctnessOk":ok,
      "latencyMsP10":percentile(samples,10),"latencyMsP50":percentile(samples,50),
      "latencyMsP90":percentile(samples,90),"torchVersion":torch.__version__,
      "cudaVersion":torch.version.cuda}


def run_dtype(args, dtype, label):
    x=torch.rand((1,64,1024),device="cuda",dtype=torch.float32).to(dtype)
    residual=torch.rand_like(x); weight=torch.rand((1024,),device="cuda").to(dtype)
    bias=torch.rand((1024,),device="cuda").to(dtype)
    expected, expected_resid=functional(x,residual,weight,bias)
    out=torch.empty_like(x); out_resid=torch.empty_like(x)
    compiled=torch.compile(into,fullgraph=True)
    compiled(out,out_resid,x,residual,weight,bias); torch.cuda.synchronize()
    tol=2e-2 if dtype==torch.bfloat16 else 5e-3
    compiled_ok=bool(torch.allclose(out,expected,rtol=tol,atol=tol) and torch.allclose(out_resid,expected_resid,rtol=tol,atol=tol))
    eager_out,eager_resid=functional(x,residual,weight,bias)
    eager_ok=bool(torch.allclose(eager_out,expected,rtol=tol,atol=tol) and torch.allclose(eager_resid,expected_resid,rtol=tol,atol=tol))
    eager=measure(lambda:functional(x,residual,weight,bias),args.warmup,args.iters,args.repeats)
    comp=measure(lambda:compiled(out,out_resid,x,residual,weight,bias),args.warmup,args.iters,args.repeats)
    case=f"layernorm_residual_{label}_64x1024"
    return [make_row(args,case,"torch_eager","torch_functional",eager,eager_ok,"end_to_end",False),
      make_row(args,case,"torch_compile","torch_compile_inductor",comp,compiled_ok,"kernel_plus_output_copies",True)]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True)
    p.add_argument("--warmup",type=int,default=20); p.add_argument("--iters",type=int,default=200)
    p.add_argument("--repeats",type=int,default=7); p.add_argument("--jsonl-out",required=True)
    args=p.parse_args(); torch.manual_seed(0)
    rows=run_dtype(args,torch.float32,"f32")+run_dtype(args,torch.bfloat16,"bf16")
    with open(args.jsonl_out,"a",encoding="utf-8") as f:
      for row in rows:
        line=json.dumps(row,sort_keys=True); f.write(line+"\n"); print(line)
    return 0 if all(r["correctnessOk"] for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
