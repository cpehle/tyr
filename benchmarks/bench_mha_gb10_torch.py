#!/usr/bin/env python3
"""PyTorch references for the suite-configured GB10 attention matrix."""
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


def functional(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def into(out, q, k, v):
    out.copy_(functional(q, k, v))
    return out


def make_row(args, case, backend, route, samples, ok, scope, allocation_free,
             work_items=None, work_unit=None):
    latency_p10=percentile(samples,10); latency_p50=percentile(samples,50)
    latency_p90=percentile(samples,90)
    def throughput(latency_ms):
        return None if work_items is None else work_items*1000.0/latency_ms
    return {"event":"summary","schemaVersion":1,"runId":args.run_id,
      "caseId":case,"backend":backend,"routeActual":route,"timer":"cuda_event",
      "completionFence":"cudaEventSynchronize(stop)","timingScope":scope,
      "warmAllocationFree":allocation_free,"compileSetupExcluded":backend=="torch_compile","warmup":args.warmup,"iters":args.iters,
      "repeats":args.repeats,"correctnessOk":ok,"latencyMsP10":latency_p10,
      "latencyMsP50":latency_p50,"latencyMsP90":latency_p90,
      "workItemsPerIteration":work_items,"workItemUnit":work_unit,
      "throughputItemsPerSecondP10":throughput(latency_p90),
      "throughputItemsPerSecondP50":throughput(latency_p50),
      "throughputItemsPerSecondP90":throughput(latency_p10),
      "torchVersion":torch.__version__,"cudaVersion":torch.version.cuda}


CASES = {
    "s64": (1, 1, 64, {"sequence-sweep"}, {"edge", "launch-bound"}),
    "s128": (1, 1, 128, {"quick", "sequence-sweep"}, {"small"}),
    "s256": (1, 1, 256, {"sequence-sweep"}, {"medium"}),
    "s512": (1, 1, 512, {"sequence-sweep"}, {"medium"}),
    "s768": (1, 1, 768, {"sequence-sweep"}, {"training", "micro"}),
    "b1_h16_s768": (1, 16, 768, {"gb10-realistic", "model-shapes", "batch-sweep"}, {"training", "specialized", "multi-head", "latency"}),
    "b2_h16_s768": (2, 16, 768, {"model-shapes", "batch-sweep"}, {"training", "specialized", "multi-head", "throughput"}),
    "b4_h16_s768": (4, 16, 768, {"model-shapes", "batch-sweep"}, {"training", "specialized", "multi-head", "throughput"}),
    "b8_h16_s768": (8, 16, 768, {"model-shapes", "batch-sweep"}, {"training", "specialized", "multi-head", "throughput"}),
    "s1024": (1, 1, 1024, {"sequence-sweep"}, {"large"}),
    "s2048": (1, 1, 2048, {"sequence-sweep"}, {"large"}),
}

def selected_cases(args):
    selected = list(CASES.items())
    if args.case:
        wanted = {part for item in args.case for part in item.split(",")}
        selected = [(name, spec) for name, spec in selected if name in wanted]
    else:
        selected = [(name, spec) for name, spec in selected if args.profile in spec[3]]
    if args.tag:
        selected = [(name, spec) for name, spec in selected if args.tag in spec[4]]
    if not selected:
        raise SystemExit("attention matrix selection is empty")
    return selected

def run_case(args, batch, q_heads, seq_len):
    shape=(batch,q_heads,seq_len,64)
    work_items=batch*seq_len
    q=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    v=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    expected=functional(q,k,v)
    out=torch.empty_like(q)
    compiled=torch.compile(into,fullgraph=True)
    compiled(out,q,k,v); torch.cuda.synchronize()
    tol=3e-2
    compiled_ok=bool(torch.allclose(out,expected,rtol=tol,atol=tol))
    eager_out=functional(q,k,v)
    eager_ok=bool(torch.allclose(eager_out,expected,rtol=tol,atol=tol))
    eager=measure(lambda:functional(q,k,v),args.warmup,args.iters,args.repeats)
    comp=measure(lambda:compiled(out,q,k,v),args.warmup,args.iters,args.repeats)
    case=f"mha_forward_b{batch}_h{q_heads}_s{seq_len}_d64"
    rows=[make_row(args,case,"torch_eager","torch_sdpa",eager,eager_ok,"end_to_end",False,
                   work_items,"tokens"),
      make_row(args,case,"torch_compile","torch_compile_inductor",comp,compiled_ok,
               "kernel_plus_output_copy",True,work_items,"tokens")]

    # A training forward is not equivalent to inference SDPA: PyTorch must
    # retain the softmax state needed by backward, matching Tyr's explicit LSE
    # output. Keep this as a separate row rather than silently changing the
    # established inference comparison.
    qg=q.detach().requires_grad_(True); kg=k.detach().requires_grad_(True); vg=v.detach().requires_grad_(True)
    training_expected=functional(qg,kg,vg)
    training_actual=[None]
    def training_forward():
        training_actual[0]=functional(qg,kg,vg)
        return training_actual[0]
    training=measure(training_forward,args.warmup,args.iters,args.repeats)
    training_ok=bool(torch.allclose(training_actual[0],training_expected,rtol=tol,atol=tol))
    rows.append(make_row(args,case,"torch_eager_training","torch_sdpa_autograd_forward",training,
                         training_ok,"training_forward_graph_and_saved_state",False,work_items,"tokens"))

    # Keep one graph alive so this measures actual backward execution,
    # excluding forward construction and any torch.compile setup.
    og=functional(qg,kg,vg); grad_out=torch.randn_like(og)
    expected_grads=torch.autograd.grad(og,(qg,kg,vg),grad_out,retain_graph=True)
    actual_grads=[None]
    def backward_only():
        actual_grads[0]=torch.autograd.grad(og,(qg,kg,vg),grad_out,retain_graph=True)
        return actual_grads[0]
    backward=measure(backward_only,args.warmup,args.iters,args.repeats)
    backward_ok=all(torch.allclose(a,e,rtol=tol,atol=tol)
                    for a,e in zip(actual_grads[0],expected_grads))
    backward_case=f"mha_backward_b{batch}_h{q_heads}_s{seq_len}_d64"
    rows.append(make_row(args,backward_case,"torch_eager","torch_sdpa_autograd",backward,
                         bool(backward_ok),"backward_only_retained_graph",False,work_items,"tokens"))

    # The primary training metric rebuilds the forward graph and executes all
    # three input gradients inside one CUDA-event interval. It therefore
    # includes real forward/backward GPU work but no torch.compile setup.
    training_step_actual=[None]
    def training_step():
        step_out=functional(qg,kg,vg)
        training_step_actual[0]=torch.autograd.grad(
            step_out,(qg,kg,vg),grad_out,retain_graph=False)
        return training_step_actual[0]
    training_step=measure(training_step,args.warmup,args.iters,args.repeats)
    training_step_ok=all(torch.allclose(a,e,rtol=tol,atol=tol)
                         for a,e in zip(training_step_actual[0],expected_grads))
    training_case=f"mha_training_step_b{batch}_h{q_heads}_s{seq_len}_d64"
    training_row=make_row(args,training_case,"torch_eager_training","torch_sdpa_autograd_full_step",
                          training_step,bool(training_step_ok),"forward_graph_plus_backward",False,
                          work_items,"tokens")
    training_row["gradientOutputDtype"]="bfloat16"
    rows.append(training_row)

    # Tyr currently exposes FP32 gradient buffers. Keep the realistic BF16 row
    # above primary, but also time PyTorch full execution plus materializing
    # the same FP32 output contract inside the CUDA-event interval.
    fp32_expected=tuple(g.float() for g in expected_grads)
    fp32_step_actual=[None]
    def fp32_training_step():
        step_out=functional(qg,kg,vg)
        grads=torch.autograd.grad(step_out,(qg,kg,vg),grad_out,retain_graph=False)
        fp32_step_actual[0]=tuple(g.float() for g in grads)
        return fp32_step_actual[0]
    fp32_step=measure(fp32_training_step,args.warmup,args.iters,args.repeats)
    fp32_step_ok=all(torch.allclose(a,e,rtol=tol,atol=tol)
                     for a,e in zip(fp32_step_actual[0],fp32_expected))
    fp32_row=make_row(args,training_case,"torch_eager_training_fp32_grads",
                      "torch_sdpa_autograd_full_step_fp32_grads",fp32_step,bool(fp32_step_ok),
                      "forward_graph_plus_backward_plus_fp32_gradient_materialization",False,
                      work_items,"tokens")
    fp32_row["gradientOutputDtype"]="float32"
    rows.append(fp32_row)
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True)
    p.add_argument("--warmup",type=int,default=20); p.add_argument("--iters",type=int,default=200)
    p.add_argument("--repeats",type=int,default=7); p.add_argument("--jsonl-out",required=True)
    p.add_argument("--profile",default="gb10-realistic"); p.add_argument("--case",action="append"); p.add_argument("--tag")
    args=p.parse_args(); torch.manual_seed(0)
    rows=[]
    for _, (batch, q_heads, seq_len, _, _) in selected_cases(args):
      rows.extend(run_case(args,batch,q_heads,seq_len))
    with open(args.jsonl_out,"a",encoding="utf-8") as f:
      for row in rows:
        line=json.dumps(row,sort_keys=True); f.write(line+"\n"); print(line)
    return 0 if all(r["correctnessOk"] for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
