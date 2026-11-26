"""Bounded diagnostic: is early attention informative for equal-depth path choice?

This is a one-step counterfactual probe, NOT the proposed online RLOO policy.
For each held-in/held-out arithmetic prompt, keep a common gold answer prefix
and an identical current whitespace token; evaluate three prespecified masks,
each skipping exactly three intermediate blocks. Fit a regularized linear
readout on train prompts only; compare against the train-best fixed mask.
No test targets are used in fitting or deployment selection. Oracle uses test
targets only for a diagnostic upper bound. Layer count is matched, not latency.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from frozen_skip.qwen3_reference import Qwen3ReferenceRouter

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
# Human-readable 1-based layer numbers, each middle-only, chosen before evaluation.
CANDIDATES = [[5, 13, 21], [8, 16, 24], [11, 19, 26]]


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def collect(model, tok, data, split):
    rows, inputs = [], []
    device = next(model.parameters()).device
    for i, item in enumerate(data):
        # The current token is deliberately held constant across prompts.
        answer = tok.encode(item["answer"], add_special_tokens=False)
        candidates = [j for j in range(1, len(answer)-1)
                      if tok.decode([answer[j]]) == " "
                      and tok.decode([answer[j+1]]).isdigit()]
        if not candidates:
            raise ValueError("No space followed by an integer token in answer")
        j = candidates[0]
        cur, target = answer[j], answer[j+1]
        pre = tok.apply_chat_template([{"role":"user","content":item["prompt"]}],
                  tokenize=True, add_generation_prompt=True, enable_thinking=False,
                  return_tensors="pt").to(device)
        pre = torch.cat([pre, torch.tensor([answer[:j]], device=device)], dim=1)
        cache = DynamicCache(config=model.config)
        inp = torch.tensor([[cur]], device=device)
        with torch.no_grad():
            model(input_ids=pre, past_key_values=cache, use_cache=True, logits_to_keep=1)
            features = {}
            def capture_pre(m, a, output):
                features["attention"] = output[0][0,-1].detach().cpu().clone()
            def capture_post(m, a):
                features["residual"] = a[0][0,-1].detach().cpu().clone()
            def capture_block(m, a, output):
                features["block"] = output[0,-1].detach().cpu().clone()
            h1 = model.model.layers[0].self_attn.register_forward_hook(capture_pre)
            h2 = model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(capture_post)
            h3 = model.model.layers[0].register_forward_hook(capture_block)
            try:
                native = model(input_ids=inp, past_key_values=copy.deepcopy(cache),
                               use_cache=True, logits_to_keep=1).logits[0,-1].float()
                features["embedding"] = model.model.embed_tokens(inp)[0,-1].detach().cpu().clone()
                full_loss = float(-native.log_softmax(dim=-1)[target])
            finally:
                for handle in (h1,h2,h3):
                    handle.remove()
            losses, correct = [], []
            for removed in CANDIDATES:
                mask = [layer not in removed for layer in range(2,28)]
                with Qwen3ReferenceRouter(model, mode="eager", tap="post_residual") as r:
                    r.begin_decode(mask)
                    logits = model(input_ids=inp, past_key_values=copy.deepcopy(cache),
                                   use_cache=True, logits_to_keep=1).logits[0,-1].float()
                    r.end_decode()
                    assert torch.equal(r.current_tap.cpu()[0], features["residual"])
                    assert r.stats["skipped_middle"] == len(removed)
                    losses.append(float(-logits.log_softmax(dim=-1)[target]))
                    correct.append(int(logits.argmax() == target))
        inputs.append(features)
        rows.append({"split":split,"index":i,"prompt":item["prompt"],
                     "answer":item["answer"],"current_token_id":cur,
                     "current_token":tok.decode([cur]),"target_token_id":target,
                     "target_token":tok.decode([target]),"native_nll":full_loss,
                     "candidate_nll":losses,"candidate_top1_correct":correct})
        print(f"{split} {i+1}/{len(data)} nll={losses}",flush=True)
    return rows, inputs


def fit_and_predict(train, test, y, alpha=0.1):
    # Center and L2 normalize using training-only centering, then solve dual
    # ridge for multi-output losses (equivalent to a linear readout + bias).
    x = torch.stack(train).double()
    z = torch.stack(test).double()
    x = torch.nn.functional.normalize(x,dim=1)
    z = torch.nn.functional.normalize(z,dim=1)
    mean = x.mean(dim=0,keepdim=True)
    x, z = x-mean,z-mean
    ym = y.mean(dim=0,keepdim=True)
    a = torch.linalg.solve(x@x.T+alpha*torch.eye(len(x),dtype=x.dtype),y-ym)
    prediction = z@x.T@a+ym
    return prediction.argmin(dim=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",default="data/pilot/train_multistep.jsonl")
    p.add_argument("--test",default="data/pilot/dev_multistep.jsonl")
    p.add_argument("--n-train",type=int,default=6)
    p.add_argument("--n-test",type=int,default=4)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    torch.manual_seed(20261126)
    train=read_rows(args.train)[:args.n_train]
    test=read_rows(args.test)[:args.n_test]
    if not train or not test or {x["prompt"] for x in train}&{x["prompt"] for x in test}:
        raise ValueError("Require nonempty disjoint prompt sets")
    start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,
               attn_implementation="eager",trust_remote_code=False).to(args.device).eval()
    model.requires_grad_(False)
    assert len(model.model.layers)==28
    tr, ft=collect(model,tok,train,"train")
    te, fe=collect(model,tok,test,"test")
    y=torch.tensor([r["candidate_nll"] for r in tr],dtype=torch.float64)
    yt=torch.tensor([r["candidate_nll"] for r in te],dtype=torch.float64)
    if len({r["current_token_id"] for r in tr+te})!=1:
        raise RuntimeError("Current token identity was not controlled")
    fixed=int(y.mean(dim=0).argmin())
    oracle=yt.argmin(dim=1)
    predictions={"train_best_fixed":torch.full((len(te),),fixed),
                 "oracle_test_only":oracle}
    for key in ("embedding","attention","residual","block"):
        predictions[key+"_linear"]=fit_and_predict([d[key] for d in ft],
                                                    [d[key] for d in fe],y)
    scores={}
    ix=torch.arange(len(te))
    for key, pred in predictions.items():
        losses=yt[ix,pred]
        scores[key]={"mean_nll_nat":float(losses.mean()),
                     "selected_mask_indices":pred.tolist(),
                     "oracle_matches":int((pred==oracle).sum()),
                     "per_example_nll":losses.tolist()}
    scores["uniform_random_expectation"]={"mean_nll_nat":float(yt.mean())}
    scores["native_all_on"]={"mean_nll_nat":sum(r["native_nll"] for r in te)/len(te)}
    result={"status":"MEASURED_DIAGNOSTIC_NOT_POLICY_VALIDATION",
            "model":MODEL,"revision":REVISION,"torch":torch.__version__,
            "transformers":__import__("transformers").__version__,"python":platform.python_version(),
            "os":platform.platform(),"device":args.device,"dtype":"float32",
            "threads":torch.get_num_threads(),"batch":1,"seed":20261126,
            "candidate_skips_1_based":CANDIDATES,"active_blocks_per_candidate":25,
            "ridge_alpha":0.1,"n_train":len(tr),"n_test":len(te),"train_fixed_index":fixed,
            "train_rows":tr,"test_rows":te,"scores":scores,
            "elapsed_s_including_load":time.monotonic()-start,
            "data_sha256":{"train":hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                           "test":hashlib.sha256(Path(args.test).read_bytes()).hexdigest()},
            "scope":"One gold-prefix decode position per prompt. Same whitespace token. "
                    "Train-only linear candidate loss readout; not full 26-bit RLOO. "
                    "Equal block count not measured GPU cost; tiny arithmetic sample."}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("WROTE",out,flush=True)

if __name__=="__main__":
    main()
