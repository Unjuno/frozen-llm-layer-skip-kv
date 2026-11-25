"""Shape-aware BF16/FP32 KV projection microbenchmark, not ω=9/4 implementation.

Compares individual-token GEMV calls, two batched GEMMs, and a single
fused K/V linear projection for model-shaped *rectangular* matrices.
EXCLUDES norms, RoPE, cache write, memory retention, and layer scheduling.
The OpenAI matrix multiplication exponent claim is NOT used as an algorithm.
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SHAPES = {"qwen3-0.6b": (1024, 1024),
          "nanbeige4.3-33b": (7168, 1024)}


def percentile(values, q):
    a = sorted(values)
    t = (len(a) - 1) * q
    l = math.floor(t)
    h = math.ceil(t)
    return a[l] * (h - t) + a[h] * (t - l) if l != h else a[l]


def bench(fn, device, warmups, repeats):
    with torch.no_grad():
        for _ in range(warmups):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    return {"median_s": statistics.median(times),
            "p10_s": percentile(times, 0.1),
            "p90_s": percentile(times, 0.9),
            "trials": repeats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-shape", choices=SHAPES, default="qwen3-0.6b")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--m-values", default="1,2,4,8,16,32,64,128")
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--output", default="results/kv_projection_microbench.json")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; try --device cpu")
    if args.warmups < 0 or args.repeats < 2:
        raise SystemExit("warmups>=0 and repeats>=2 required")
    torch.manual_seed(20261125)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    d, dkv = SHAPES[args.model_shape]
    wk = torch.randn(dkv, d, device=device, dtype=dtype) / math.sqrt(d)
    wv = torch.randn(dkv, d, device=device, dtype=dtype) / math.sqrt(d)
    fused = torch.cat([wk, wv], dim=0).contiguous()
    rows = []
    for n in (int(x) for x in args.m_values.split(",")):
        if n < 1:
            raise ValueError("m-values must be positive")
        state = torch.randn(n, d, device=device, dtype=dtype)
        def individual():
            return [(F.linear(state[i:i + 1], wk), F.linear(state[i:i + 1], wv))
                    for i in range(n)]
        def separate():
            return F.linear(state, wk), F.linear(state, wv)
        def fused_kv():
            return F.linear(state, fused).split(dkv, dim=-1)
        with torch.no_grad():
            a, b = separate()
            c, e = fused_kv()
            max_abs_diff = max((a.float() - c.float()).abs().max().item(),
                               (b.float() - e.float()).abs().max().item())
        individual_times = bench(individual, device, args.warmups, args.repeats)
        separate_times = bench(separate, device, args.warmups, args.repeats)
        fused_times = bench(fused_kv, device, args.warmups, args.repeats)
        rows.append({"M": n,
                     "A_shape": [n, d], "Wkv_fused_shape": [d, 2 * dkv],
                     "max_abs_diff_fused_vs_separate": max_abs_diff,
                     "individual": individual_times, "separate": separate_times,
                     "fused": fused_times})
    result = {
        "model_shape": args.model_shape,
        "model_weights": "random synthetic weights (NOT pretrained)",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_clock": "not measured; record nvidia-smi clocks or Nsight externally",
        "dtype": str(dtype), "torch": torch.__version__,
        "python": platform.python_version(), "os": platform.platform(),
        "seed": 20261125, "warmups": args.warmups,
        "repeats": args.repeats, "batch_requests": 1,
        "rows": rows,
        "scope": "projection only: NOT layer end-to-end latency; NOT omega=9/4",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", out, "rows", len(rows))


if __name__ == "__main__":
    main()
