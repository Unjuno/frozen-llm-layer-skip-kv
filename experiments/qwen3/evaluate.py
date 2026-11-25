"""Held-out smoke evaluation of a trained Qwen3 linear layer-mask head.

Runs native full model, simple fixed masks, and deterministic learned head
on identical *prompts*. They generate their own outputs, unlike fixed-token
KV correctness replay. The supplied debug dev set is NOT a scientific test.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from frozen_skip.qwen3_reference import LinearMaskHead, Qwen3ReferenceRouter
from run import default_device, default_dtype, load_model, normalize, save_json


def generate(model, tokenizer, prompt, mode, head, max_tokens, tap, device):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt").to(device)
    cache = DynamicCache(config=model.config)
    controller = None
    nmid = len(model.model.layers) - 2
    if mode != "native":
        controller = Qwen3ReferenceRouter(model, head=head if mode == "learned" else None,
                                           mode="eager", tap=tap, exploration=0.0,
                                           deterministic=True)
    generated = []
    with torch.no_grad():
        prefill = model(input_ids=ids, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        tok = int(prefill.logits[0, -1].argmax())
        generated.append(tok)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if controller is not None:
            controller.__enter__()
        try:
            for _ in range(max_tokens - 1):
                if tok == tokenizer.eos_token_id:
                    break
                if controller is not None:
                    if mode == "learned":
                        controller.begin_decode()  # feature-selected mask
                    elif mode == "every_other":
                        controller.begin_decode([int(i % 2 == 0) for i in range(nmid)])
                    elif mode == "every_fourth":
                        controller.begin_decode([int(i % 4 == 0) for i in range(nmid)])
                    else:
                        raise ValueError(mode)
                out = model(input_ids=torch.tensor([[tok]], device=device),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                if controller is not None:
                    controller.end_decode()
                tok = int(out.logits[0, -1].argmax())
                generated.append(tok)
        finally:
            if controller is not None:
                controller.__exit__(None, None, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    masks = controller.masks if controller is not None else None
    active_rate = (sum(sum(x) for x in masks) / (len(masks) * nmid)
                   if masks else 1.0)
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed, active_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--revision", required=True, help="Immutable checkpoint SHA")
    p.add_argument("--checkpoint", required=True, help="Head-only pilot checkpoint")
    p.add_argument("--dataset", default="data/pilot/dev_arithmetic.jsonl")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--tap", default="post_residual",
                   choices=["pre_residual", "post_residual"])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--output", default="results/qwen3_pilot_dev.json")
    args = p.parse_args()
    if args.revision in {"main", "master"} or len(args.revision) < 10:
        p.error("Real pretrained checkpoint needs an immutable commit SHA")
    device = default_device(args.device)
    dtype = default_dtype(args.dtype, device)
    # load_model needs these two run-specific fields even in evaluation.
    args.prompt = "Return only the integer answer."
    args.decode_steps = args.max_new_tokens
    model, tokenizer, _, revision, device, dtype = load_model(args)
    if tokenizer is None:
        p.error("Evaluation requires a pretrained checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_id", "model_revision", "tap", "num_layers"):
        if key not in checkpoint:
            raise ValueError("Invalid head checkpoint: missing " + key)
    if (checkpoint["model_id"] != args.model
            or checkpoint["model_revision"] != revision
            or checkpoint["tap"] != args.tap
            or checkpoint["num_layers"] != len(model.model.layers)):
        raise ValueError("Head checkpoint/model revision/tap mismatch")
    head = LinearMaskHead(model.config.hidden_size, len(model.model.layers)).to(
        device=device, dtype=torch.float32)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    for q in head.parameters():
        q.requires_grad_(False)
    examples = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()
                if line.strip()]
    rows = []
    modes = ["native", "every_other", "every_fourth", "learned"]
    for item in examples:
        output = {"prompt": item["prompt"], "expected": item["answer"], "modes": {}}
        for mode in modes:
            text, seconds, rate = generate(
                model, tokenizer, item["prompt"], mode, head, args.max_new_tokens,
                args.tap, device)
            output["modes"][mode] = {
                "generated": text, "exact": int(normalize(text) == normalize(item["answer"])),
                "decode_wall_s": seconds, "mean_active_middle": rate}
        rows.append(output)
    summaries = {}
    for mode in modes:
        summaries[mode] = {
            "exact_accuracy": sum(x["modes"][mode]["exact"] for x in rows) / len(rows),
            "median_decode_wall_s": statistics.median(x["modes"][mode]["decode_wall_s"]
                                                      for x in rows),
            "mean_active_middle": statistics.mean(x["modes"][mode]["mean_active_middle"]
                                                  for x in rows)}
    result = {
        "status": "DEV_SMOKE_NOT_SCIENTIFIC_BENCHMARK",
        "model": args.model, "revision": revision, "tap": args.tap,
        "device": str(device), "dtype": str(dtype), "examples": len(rows),
        "modes": summaries, "per_example": rows,
        "limitations": [
            "Tiny arithmetic dev set; no held-out publication benchmark",
            "every_other/every_fourth are not the best fixed masks",
            "Python reference dispatcher; batch=1 and eager cache only",
            "Decode wall time includes CPU synchronization and varies by answer length",
            "No statistical power or quality noninferiority conclusion"]
    }
    save_json(args.output, result)


if __name__ == "__main__":
    main()
