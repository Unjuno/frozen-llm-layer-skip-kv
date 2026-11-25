"""Runnable Qwen3 experiment entrypoint: verify, benchmark, and train.

Requires transformers==4.57.3 and a compatible PyTorch installation.
The default 'synthetic' model is random Qwen3 architecture, NOT pretrained.
All task-accuracy/speed claims require running a pinned real checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import json
import math
import platform
import random
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from frozen_skip.qwen3_reference import LinearMaskHead, Qwen3ReferenceRouter


def default_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype(value, device):
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[value]


def load_model(args):
    device = default_device(args.device)
    dtype = default_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    tokenizer = None
    if args.model == "synthetic":
        config = Qwen3Config(
            vocab_size=127, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=1024,
            rms_norm_eps=1e-6, attention_dropout=0.0,
            bos_token_id=2, eos_token_id=1, pad_token_id=0,
            layer_types=["full_attention"] * 4)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
        input_ids = torch.tensor([[2, 14, 9, 17]], device=device)
        revision = "randomly_initialized_no_checkpoint"
    else:
        if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ValueError("Real checkpoint requires --revision with a fixed commit SHA")
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=dtype,
            attn_implementation="eager", trust_remote_code=False)
        model.to(device)
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt").to(device)
        except (ValueError, TypeError):
            input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
        revision = args.revision
        if model.config.model_type != "qwen3" or len(model.model.layers) != 28:
            raise ValueError("This real-model pilot is limited to Qwen3-0.6B")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert getattr(model.config, "_attn_implementation") == "eager"
    assert len(model.model.layers) >= 3
    return model, tokenizer, input_ids, revision, device, dtype


def metadata(model, args, revision, device, dtype):
    return {
        "datetime_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "os": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "model_id": args.model, "model_revision": revision,
        "dtype": str(dtype), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "attention_backend": "eager", "batch": 1,
        "num_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "seed": args.seed, "decode_steps": args.decode_steps,
        "tap": args.tap,
        "method": "correctness reference: Python mask branching / no compile",
    }


def save_json(path, obj):
    if not path:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    print("WROTE", p)


def replay_tokens(model, input_ids, steps):
    # A fixed, valid token sequence independent of experimental branch.
    vocab = model.config.vocab_size
    return [int((input_ids[0, -1].item() + k + 3) % vocab) for k in range(steps)]


def mask_cases(n_middle, n_steps):
    on = [1] * n_middle
    off = [0] * n_middle
    rng = random.Random(731)
    return {
        "all_on": [on.copy() for _ in range(n_steps)],
        "all_off": [off.copy() for _ in range(n_steps)],
        "single_skip": [[0 if i == n_middle // 2 else 1
                         for i in range(n_middle)] for _ in range(n_steps)],
        "alternating": [[(t + i) % 2 for i in range(n_middle)]
                        for t in range(n_steps)],
        "off_then_resume": [off.copy() if t < (n_steps // 2) else on.copy()
                            for t in range(n_steps)],
        "random": [[rng.randrange(2) for _ in range(n_middle)]
                   for _ in range(n_steps)],
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_fixed(model, prompt, tokens, masks=None, mode="eager",
              head_probe=False, tap="post_residual", measure=False):
    """Native path when masks is None; routed path otherwise. Same model weights."""
    cache = DynamicCache(config=model.config)
    router = None
    head = None
    first_mlp_calls = [0]
    def mlp_hook(module, args, output):
        first_mlp_calls[0] += 1
    mlp_handle = model.model.layers[0].mlp.register_forward_hook(mlp_hook)
    if masks is not None:
        if head_probe:
            head = LinearMaskHead(model.config.hidden_size,
                                  len(model.model.layers)).to(
                device=prompt.device, dtype=torch.float32)
            for p in head.parameters():
                p.requires_grad_(False)
        router = Qwen3ReferenceRouter(model, head=head, mode=mode, tap=tap)
    try:
        with torch.no_grad():
            # Prefill all layers once. Do NOT route during prefill.
            _ = model(input_ids=prompt, past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
            sync(prompt.device)
            started = time.perf_counter()
            if router is None:
                outs = []
                for token in tokens:
                    out = model(input_ids=torch.tensor([[token]], device=prompt.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
                    if not measure:
                        if not measure:
                            outs.append(out.logits[:, -1, :].detach().float().cpu())
                stats = {}
            else:
                outs = []
                with router:
                    for token, mask in zip(tokens, masks):
                        router.begin_decode(mask)
                        out = model(input_ids=torch.tensor([[token]], device=prompt.device),
                                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                        router.end_decode()
                        outs.append(out.logits[:, -1, :].detach().float().cpu())
                stats = router.stats_dict()
            sync(prompt.device)
            seconds = time.perf_counter() - started
    finally:
        mlp_handle.remove()
    return (None if measure else torch.stack(outs)), {
        **stats, "first_block_mlp_invocations": first_mlp_calls[0],
        "decode_wall_seconds": seconds if measure else None}


def verify(model, prompt, revision, device, dtype, args):
    tokens = replay_tokens(model, prompt, args.decode_steps)
    nmid = len(model.model.layers) - 2
    tol = args.tolerance if args.tolerance is not None else (
        5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4)
    native, nstats = run_fixed(model, prompt, tokens)
    all_masks = [[1] * nmid for _ in tokens]
    all_on, astats = run_fixed(
        model, prompt, tokens, all_masks, head_probe=True, tap=args.tap)
    full_delta = float((native - all_on).abs().max())
    all_on_pass = (full_delta <= tol and astats["first_attention_calls"] == len(tokens)
                   and astats["head_probe_calls"] == len(tokens)
                   and astats["tap_reads"] == len(tokens))
    mlp_pass = astats["first_block_mlp_invocations"] == len(tokens) + 1
    rows = []
    for name, masks in mask_cases(nmid, len(tokens)).items():
        eager, es = run_fixed(model, prompt, tokens, masks, mode="eager", tap=args.tap)
        lazy, ls = run_fixed(model, prompt, tokens, masks, mode="lazy", tap=args.tap)
        diff = float((eager - lazy).abs().max())
        # Under exact full-context cache semantics, every deferred position
        # before the last ON must be reconstructed; after it may remain pending.
        missing = sum(len(tokens) -
                      max([i + 1 for i, row in enumerate(masks) if row[j]] or [0])
                      for j in range(nmid))
        actual_pending = ls["pending_positions_at_end"]
        row_pass = diff <= tol and actual_pending == missing
        rows.append({"case": name, "pass": row_pass,
                     "max_abs_logit_diff": diff,
                     "eager_skip_projected": es.get("eager_skip_projections", 0),
                     "lazy_reconstructed": ls.get("lazy_reconstructed_positions", 0),
                     "lazy_pending_at_end": actual_pending,
                     "expected_pending": missing})
    result = {
        "status": "PASS" if all_on_pass and mlp_pass and all(x["pass"] for x in rows)
                  else "FAIL",
        "environment": metadata(model, args, revision, device, dtype),
        "native_vs_masked_all_on": {"pass": all_on_pass,
                                    "max_abs_logits": full_delta,
                                    "first_attention_calls": astats.get("first_attention_calls", 0),
                                    "first_mlp_calls": astats["first_block_mlp_invocations"],
                                    "head_probe_calls": astats.get("head_probe_calls", 0)},
        "first_layer_mlp_pass": mlp_pass,
        "eager_vs_lazy_fixed_replay": rows,
        "tolerance": tol,
        "claim_scope": "numeric replay correctness only: NOT accuracy or GPU speed",
    }
    save_json(args.output, result)
    if result["status"] != "PASS":
        raise SystemExit(2)


def benchmark(model, prompt, revision, device, dtype, args):
    # This is a REFERENCE Python dispatcher; includes CPU branching overhead.
    tokens = replay_tokens(model, prompt, args.decode_steps)
    nmid = len(model.model.layers) - 2
    cases = mask_cases(nmid, len(tokens))
    result_rows = []
    for label, masks, mode in [
        ("native", None, "eager"),
        ("all_on", cases["all_on"], "eager"),
        ("fixed_skip", cases["single_skip"], "eager"),
        ("dynamic_eager", cases["alternating"], "eager"),
        ("dynamic_lazy", cases["alternating"], "lazy"),
    ]:
        for _ in range(args.warmup):
            run_fixed(model, prompt, tokens, masks, mode=mode, tap=args.tap,
                      head_probe=masks is not None, measure=True)
        times = []
        for _ in range(args.repeat):
            _, stats = run_fixed(model, prompt, tokens, masks, mode=mode,
                                 tap=args.tap, measure=True, head_probe=masks is not None)
            times.append(stats["decode_wall_seconds"])
        result_rows.append({
            "mode": label, "median_s": statistics.median(times),
            "min_s": min(times), "max_s": max(times),
            "individual_wall_seconds": times})
    save_json(args.output, {
        "status": "MEASURED_CPU_WALL" if device.type == "cpu" else "MEASURED_GPU_WALL",
        "environment": metadata(model, args, revision, device, dtype),
        "warmup": args.warmup, "repeat": args.repeat,
        "results": result_rows,
        "important": "Python reference, fixed masks and linear head probe, batch 1. Not optimized serving or sampled routing."
    })


def normalize(s):
    return " ".join(str(s).strip().lower().split())


def rollout(model, tokenizer, prompt, target, head, args, device):
    # Each rollout executes frozen Qwen3 autoregressively with sampled LAYER
    # masks, not a teacher-forced search for a fixed layer subset.
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(device)
    cache = DynamicCache(config=model.config)
    controller = Qwen3ReferenceRouter(
        model, head=head, mode="eager", tap=args.tap, exploration=args.exploration)
    ids = []
    with torch.no_grad():
        initial = model(input_ids=input_ids, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        token = int(initial.logits[0, -1].argmax())
        ids.append(token)
        with controller:
            for _ in range(args.max_new_tokens - 1):
                if token == tokenizer.eos_token_id:
                    break
                controller.begin_decode()
                output = model(input_ids=torch.tensor([[token]], device=device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                controller.end_decode()
                token = int(output.logits[0, -1].argmax())
                ids.append(token)
    generated = tokenizer.decode(ids, skip_special_tokens=True)
    exact = float(normalize(generated) == normalize(target))
    # Partial similarity is an auxiliary TRAINING proxy, not task accuracy.
    similarity = SequenceMatcher(None, normalize(generated), normalize(target)).ratio()
    quality_proxy = 0.8 * exact + 0.2 * similarity
    mean_active = (sum(sum(int(v) for v in row) for row in controller.masks)
                   / max(1, len(controller.masks) * controller.n_middle))
    reward = quality_proxy - args.cost_weight * mean_active
    if not controller.log_probs:
        return None
    return {"reward": float(reward), "quality_proxy": quality_proxy,
            "exact": exact, "mean_active": mean_active,
            "logp": torch.stack(controller.log_probs).sum(),
            "generated": generated, "mask": controller.masks}


def train(model, tokenizer, revision, device, dtype, args):
    if tokenizer is None:
        raise ValueError("Training requires a real checkpoint and tokenizer")
    if not args.dataset:
        raise ValueError("Training requires --dataset train.jsonl")
    if args.rollouts < 2:
        raise ValueError("RLOO needs >=2 independent rollouts per prompt")
    if args.max_new_tokens < 2:
        raise ValueError("Need >=2 generated tokens for decode routing")
    if args.epochs < 1 or not 0.5 < args.initial_on < 1.0:
        raise ValueError("epochs>=1 and 0.5<initial-on<1 required")
    data_bytes = Path(args.dataset).read_bytes()
    examples = [json.loads(line) for line in data_bytes.decode('utf-8').splitlines()
                if line.strip()]
    if not examples:
        raise ValueError("Empty training dataset")
    for row in examples:
        if "prompt" not in row or "answer" not in row:
            raise ValueError("Each JSONL row must have prompt and answer")
    head = LinearMaskHead(model.config.hidden_size, len(model.model.layers)).to(
        device=device, dtype=torch.float32)
    # Start close to all-ON without changing the exploration distribution.
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(math.log(args.initial_on / (1.0 - args.initial_on)))
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    records = []
    updates = 0
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for epoch in range(args.epochs):
        for idx, item in enumerate(examples):
            group = [rollout(model, tokenizer, item["prompt"], item["answer"],
                             head, args, device) for _ in range(args.rollouts)]
            group = [g for g in group if g is not None]
            if len(group) < 2:
                records.append({"epoch": epoch, "sample": idx, "skipped": "insufficient routed rollouts"})
                continue
            qvals = [g["quality_proxy"] for g in group]
            # When every output is equally wrong, a cost-only update risks
            # selecting fast but systematically wrong paths.
            if max(qvals) - min(qvals) < 1e-6 and not any(g["exact"] for g in group) and not args.allow_cost_only:
                records.append({"epoch": epoch, "sample": idx,
                                "skipped": "no task-quality discrimination"})
                continue
            rewards = [g["reward"] for g in group]
            losses = []
            for n, g in enumerate(group):
                baseline = sum(rewards[j] for j in range(len(group)) if j != n) / (len(group) - 1)
                advantage = g["reward"] - baseline
                losses.append(-g["logp"] * advantage)
            loss = torch.stack(losses).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).detach().cpu())
            if not math.isfinite(grad_norm):
                raise RuntimeError("Non-finite policy gradient")
            if grad_norm == 0:
                records.append({"epoch": epoch, "sample": idx, "skipped": "zero gradient"})
                continue
            optim.step()
            updates += 1
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen model received a gradient")
            records.append({"epoch": epoch, "sample": idx,
                            "mean_quality_proxy": statistics.mean(qvals),
                            "mean_exact": statistics.mean(g["exact"] for g in group),
                            "mean_active": statistics.mean(g["mean_active"] for g in group),
                            "loss": float(loss.detach().cpu()), "gradient_norm": grad_norm,
                            "rollouts": [{k: v for k, v in g.items() if k != "logp"} for g in group]})
    changed = any(not torch.equal(v, initial[k]) for k, v in head.state_dict().items())
    result = {"environment": metadata(model, args, revision, device, dtype),
              "status": "TRAINED_NOT_VALIDATED" if updates and changed else "NO_EFFECTIVE_UPDATE",
              "updates": updates, "head_changed": changed, "records": records,
              "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
              "initial_on_before_exploration": args.initial_on,
              "quality_gate_heuristic": not args.allow_cost_only,
              "no_speed_or_quality_claim": True}
    if not updates or not changed:
        save_json(args.output, result)
        raise SystemExit(3)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(),
                "num_layers": len(model.model.layers), "tap": args.tap,
                "model_id": args.model, "model_revision": revision,
                "training": "synthetic arithmetic debug dataset / on-policy mask RLOO",
                "seed": args.seed, "epoch_count": args.epochs,
                "updates": updates, "dataset_sha256": result["dataset_sha256"]}, checkpoint)
    result["checkpoint"] = str(checkpoint)
    save_json(args.output, result)


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["verify", "bench", "train"])
    p.add_argument("--model", default="synthetic",
                   help="synthetic or Qwen/Qwen3-0.6B")
    p.add_argument("--revision", help="REQUIRED immutable commit SHA for real model")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--tap", default="post_residual",
                   choices=["pre_residual", "post_residual"])
    p.add_argument("--prompt", default="Reply with one integer only. What is 3 + 4?")
    p.add_argument("--decode-steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--output", default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--exploration", type=float, default=0.05)
    p.add_argument("--initial-on", type=float, default=0.95,
                   help="Initial per-layer Bernoulli p, before exploration (pilot default)")
    p.add_argument("--cost-weight", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--allow-cost-only", action="store_true")
    p.add_argument("--checkpoint", default="checkpoints/router_pilot.pt")
    return p


def main():
    p = make_parser()
    args = p.parse_args()
    if args.decode_steps < 2 or args.repeat < 1:
        p.error("decode-steps must be >=2, repeat must be >=1")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tok, prompt, rev, device, dtype = load_model(args)
    if args.command == "verify":
        verify(model, prompt, rev, device, dtype, args)
    elif args.command == "bench":
        benchmark(model, prompt, rev, device, dtype, args)
    else:
        train(model, tok, rev, device, dtype, args)


if __name__ == "__main__":
    main()
