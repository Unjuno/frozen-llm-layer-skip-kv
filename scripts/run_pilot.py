#!/usr/bin/env python3
"""One-command, gated experiment runner. Run from an installed checkout.

Default smoke stage requires no pretrained model download and uses CPU.
Real and pilot stages require explicit device, immutable model revision,
and a new output directory. No cloud job is created by this script.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B"
# Public HF model commit observed via model_info on 2026-11-25.
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "real", "pilot"], default="smoke")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    p.add_argument("--tap", choices=["pre_residual", "post_residual"], default="post_residual")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--decode-steps", type=int, default=6)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.stage != "smoke" and args.device is None:
        p.error("real/pilot stages require explicit --device cpu or --device cuda")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        p.error("revision must be a full 40-character commit SHA")
    if args.decode_steps < 2 or args.threads < 1:
        p.error("decode-steps>=2 and threads>=1 required")
    device = args.device or "cpu"
    if args.stage == "smoke" and device != "cpu":
        p.error("smoke stage is CPU only")
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    if args.stage == "smoke" and dtype != "float32":
        p.error("smoke stage uses FP32")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.run_dir or (ROOT / "runs" / f"{args.stage}_{stamp}")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    manifest = {"status": "RUNNING", "stage": args.stage, "source_commit": sha,
                "tracked_worktree_dirty": dirty, "device": device, "dtype": dtype,
                "tap": args.tap, "seed": args.seed,
                "model_revision": args.revision if args.stage != "smoke" else None,
                "started_utc": stamp, "commands": []}

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def step(name, command):
        row = {"name": name, "command": command, "status": "RUNNING"}
        manifest["commands"].append(row)
        save()
        print(f"=== {name} ===", flush=True)
        with (out / f"{name}.log").open("w") as f:
            proc = subprocess.run(command, cwd=ROOT, env=env,
                                  stdout=f, stderr=subprocess.STDOUT, text=True)
        row["returncode"] = proc.returncode
        row["status"] = "PASS" if proc.returncode == 0 else "FAIL"
        save()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)

    common = ["--device", device, "--dtype", dtype, "--seed", str(args.seed),
              "--decode-steps", str(args.decode_steps)]
    cli = [sys.executable, "experiments/qwen3/run.py"]
    try:
        if args.stage == "smoke":
            step("pytest", [sys.executable, "-m", "pytest", "-q", "tests/test_qwen3_reference.py"])
            for tap in ("pre_residual", "post_residual"):
                name = "verify_" + tap
                step(name, cli + ["verify", "--model", "synthetic", "--tap", tap]
                     + common + ["--output", str(out / (name + ".json"))])
            step("bench_smoke", cli + ["bench", "--model", "synthetic", "--tap", args.tap,
                 "--warmup", "1", "--repeat", "2"] + common
                 + ["--output", str(out / "bench_smoke.json")])
        else:
            model_args = ["--model", MODEL, "--revision", args.revision, "--tap", args.tap]
            step("verify_real", cli + ["verify"] + model_args + common
                 + ["--output", str(out / "verify_real.json")])
            if args.stage == "pilot":
                ckpt = str(out / "router.pt")
                # All training occurs only after the same model, tap and dtype
                # have passed numerical verification.
                step("train", cli + ["train"] + model_args + common
                     + ["--dataset", "data/pilot/train_multistep.jsonl",
                        "--epochs", "1", "--rollouts", "4", "--max-new-tokens", "16",
                        "--checkpoint", ckpt, "--output", str(out / "train.json")])
                step("dev", [sys.executable, "experiments/qwen3/evaluate.py"]
                     + model_args + ["--device", device, "--dtype", dtype,
                        "--seed", str(args.seed), "--checkpoint", ckpt,
                        "--dataset", "data/pilot/dev_multistep.jsonl",
                        "--output", str(out / "dev.json")])
                step("bench", cli + ["bench"] + model_args + common
                     + ["--warmup", "2", "--repeat", "5",
                        "--output", str(out / "bench.json")])
        manifest["status"] = "COMPLETED_NOT_SCIENTIFIC_VALIDATION"
        save()
    except (subprocess.CalledProcessError, OSError) as e:
        manifest["status"] = "STOPPED"
        manifest["error"] = str(e)
        save()
        print("STOPPED. Inspect", out, file=sys.stderr)
        return e.returncode if isinstance(e, subprocess.CalledProcessError) else 1
    print("COMPLETE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
