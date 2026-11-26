"""Pre-specified, matched-depth early-activation diagnostic, v2.

Not full 26-bit RLOO, not end-to-end generation, not a GPU benchmark.
Native Qwen3 processes a full chat prefill and a common answer prefix.
At the final single-token decode step, each candidate runs the same number
of blocks, skipping four middle blocks. Skipped terminal-step K/V are not
materialized, because no future step is evaluated; the native cache prefix
is restored after each candidate. This has the same final-step hidden-state
semantics as exact Eager project-only for the skipped blocks, but is NOT a
general KV executor. Every candidate sees identical prefix state/position.

We use a fresh deterministic, disjoint synthetic MCQ dataset. Scores are
conditional on four answer labels, not free-text generation accuracy.
Native restricted accuracy >=80% is a predeclared task-viability gate.
The primary post-residual linear readout must improve conditional NLL by
>=0.02 nat over the fixed mask chosen on train+dev, with paired bootstrap
95% interval for improvement above zero and no lost test correct answers.
Even PASS is exploratory, not a publication claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 2026112602
# 1-based layers. Prespecified patterns, including spread and clustered masks.
MASKS = [
    [4,10,16,22], [5,11,17,23], [6,12,18,24], [7,13,19,25],
    [8,14,20,26], [2,3,4,5], [9,10,11,12],
    [16,17,18,19], [24,25,26,27], [5,13,21,27],
    [3,8,16,24], [4,11,19,26],
]
ALPHAS = (0.01, 0.1, 1.0, 10.0)
FEATURES = ("embedding", "attention", "residual", "block")
SPLITS = {"train":24, "dev":12, "test":24}
LETTERS = "ABCD"


def dataset():
    rng = random.Random(SEED)
    result = []
    arithmetic = [(a,b) for a in range(2,20) for b in range(a,20)]
    compare = [(a,b) for a in range(10,99) for b in range(a+1,100)]
    rng.shuffle(arithmetic); rng.shuffle(compare)
    ai = ci = li = 0
    for split,n in SPLITS.items():
        for i in range(n):
            kind = ("addition", "compare", "lookup")[i % 3]
            if kind == "addition":
                a,b = arithmetic[ai]; ai += 1
                correct = str(a+b)
                other = [str(a+b-1),str(a+b+1),str(a+b+3)]
                question = f"What is {a} + {b}?"
            elif kind == "compare":
                a,b = compare[ci]; ci += 1
                correct = str(b)
                other = [str(a),str(a-1),str(a-2)]
                question = "Which number is largest?"
            else:
                li += 1
                vals = rng.sample(list(range(10,99)),4)
                keys = ["red","blue","green","yellow"]
                key = rng.randrange(4)
                correct = str(vals[key])
                other = [str(v) for j,v in enumerate(vals) if j != key]
                question = (f"In record {li}, " + ", ".join(
                    f"{k}={v}" for k,v in zip(keys,vals)) +
                    f". What is the value of {keys[key]}?")
            # Balanced answer labels within each split, independent of task.
            label = (i // 3 + i % 3) % 4
            rng.shuffle(other)
            options = other.copy()
            options.insert(label,correct)
            prompt = ("Choose the correct option. Reply with its letter only.\n"
                      + question + "\n" + "\n".join(
                          f"{l}. {o}" for l,o in zip(LETTERS,options)))
            result.append({"id":f"{split}_{i:02d}","split":split,"kind":kind,
                           "prompt":prompt,"answer":LETTERS[label],
                           "options":options,"answer_index":label})
    assert len({r["prompt"] for r in result}) == len(result)
    return result


def inputs(tok, rows, device):
    # No answer text is supplied. The common "Answer:" prefix is a teacher-
    # forced response prefix, not part of the evaluation label.
    answer_prefix = tok.encode("Answer:", add_special_tokens=False)
    if len(answer_prefix)<2:
        raise RuntimeError("Unexpected answer prefix tokenization")
    last = answer_prefix[-1]
    all_ids=[]
    for r in rows:
        ids=tok.apply_chat_template([{"role":"user","content":r["prompt"]}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        all_ids.append(ids+answer_prefix[:-1])
    width=max(map(len,all_ids))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids=torch.tensor([[pad]*(width-len(x))+x for x in all_ids],device=device)
    mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in all_ids],device=device)
    pos=mask.cumsum(-1)-1
    pos.masked_fill_(mask==0,0)
    next_ids=torch.full((len(rows),1),last,dtype=torch.long,device=device)
    mask_next=torch.cat([mask,torch.ones_like(next_ids)],dim=1)
    pos_next=mask.sum(-1,keepdim=True)
    return ids,mask,pos,next_ids,mask_next,pos_next,last


def label_tokens(tok):
    variants=[]
    for letter in LETTERS:
        candidates={tuple(tok.encode(s,add_special_tokens=False)) for s in (letter," "+letter)}
        ids=sorted({v[0] for v in candidates if len(v)==1})
        if not ids: raise RuntimeError("Label lacks a single-token representation")
        variants.append(ids)
    assert len(set(sum(variants,[]))) == len(sum(variants,[]))
    return variants


def score(logits, labels, targets):
    # Group bare and space-prefixed label variants; normalize across labels.
    logits=logits.float()
    all_lse=logits.logsumexp(-1)
    label_logits=torch.stack([logits[:,x].logsumexp(-1) for x in labels],-1)
    lp=label_logits.log_softmax(-1)
    i=torch.arange(len(targets),device=logits.device)
    nll=-lp[i,targets]
    pred=label_logits.argmax(-1)
    full=all_lse-label_logits[i,targets]
    top1=logits.argmax(-1).cpu().tolist()
    direct=[int(t in labels[int(y)]) for t,y in zip(top1,targets.cpu())]
    return nll.cpu(), (pred==targets).cpu(),full.cpu(),direct


def collect(model,tok,rows,labels,batch_size):
    device=next(model.parameters()).device
    feat={k:[] for k in FEATURES}
    result=[]
    for lo in range(0,len(rows),batch_size):
        group=rows[lo:lo+batch_size]
        ids,mask,pos,cur,mask_next,pos_next,last=inputs(tok,group,device)
        cache=DynamicCache(config=model.config)
        width=ids.shape[1]
        targets=torch.tensor([r["answer_index"] for r in group],device=device)
        fs={}
        with torch.no_grad():
            model(input_ids=ids,attention_mask=mask,position_ids=pos,
                  past_key_values=cache,use_cache=True,logits_to_keep=1)
            def run():
                return model(input_ids=cur,attention_mask=mask_next,
                    position_ids=pos_next,past_key_values=cache,use_cache=True,
                    cache_position=torch.tensor([width],device=device),
                    logits_to_keep=1).logits[:,0,:]
            block=model.model.layers[0]
            def a(m,x,o): fs["attention"]=o[0][:,-1,:].detach().cpu().clone()
            def b(m,x): fs["residual"]=x[0][:,-1,:].detach().cpu().clone()
            def c(m,x,o): fs["block"]=o[:,-1,:].detach().cpu().clone()
            hooks=[block.self_attn.register_forward_hook(a),
                   block.post_attention_layernorm.register_forward_pre_hook(b),
                   block.register_forward_hook(c)]
            try:
                native=score(run(),labels,targets)
                fs["embedding"]=model.model.embed_tokens(cur)[:,0,:].detach().cpu().clone()
            finally:
                for h in hooks:h.remove()
            cache.crop(width)
            cand=[]
            for removed in MASKS:
                original=[]
                try:
                    for j in removed:
                        layer=model.model.layers[j-1]
                        original.append((layer,layer.forward))
                        layer.forward=lambda hidden_states,*args,**kwargs:hidden_states
                    cand.append(score(run(),labels,targets))
                finally:
                    for layer,fn in original: layer.forward=fn
                    cache.crop(width)
                assert all(x.get_seq_length()==width for x in cache.layers)
        for k in FEATURES:feat[k].append(fs[k])
        for i,r in enumerate(group):
            result.append({**r,"current_token_id":last,
                "native_nll":float(native[0][i]),"native_correct":int(native[1][i]),
                "native_full_label_nll":float(native[2][i]),"native_direct_correct":native[3][i],
                "candidate_nll":[float(s[0][i]) for s in cand],
                "candidate_correct":[int(s[1][i]) for s in cand],
                "candidate_full_label_nll":[float(s[2][i]) for s in cand]})
        print(f"collected {lo+len(group)}/{len(rows)} rows; native correct {sum(r['native_correct'] for r in result)}",flush=True)
    return result,{k:torch.cat(v,dim=0).double() for k,v in feat.items()}


def predict(x,z,y,alpha):
    x=torch.nn.functional.normalize(x,dim=1)
    z=torch.nn.functional.normalize(z,dim=1)
    mean=x.mean(0,keepdim=True); x=x-mean;z=z-mean
    # Remove prompt difficulty common to all mask alternatives.
    y=y-y.mean(1,keepdim=True)
    ym=y.mean(0,keepdim=True)
    a=torch.linalg.solve(x@x.T+alpha*torch.eye(len(x),dtype=x.dtype),y-ym)
    return (z@x.T@a+ym).argmin(1)


def matrix(rows,key):
    return torch.tensor([r[key] for r in rows],dtype=torch.float64)


def bootstrap(diffs):
    gen=torch.Generator().manual_seed(SEED)
    draws=torch.randint(len(diffs),(10000,len(diffs)),generator=gen)
    means=diffs[draws].mean(1)
    return [float(x) for x in torch.quantile(means,torch.tensor([0.025,0.975],dtype=means.dtype))]


def evaluate(rows,feat):
    tr=[i for i,r in enumerate(rows) if r["split"]=="train"]
    de=[i for i,r in enumerate(rows) if r["split"]=="dev"]
    te=[i for i,r in enumerate(rows) if r["split"]=="test"]
    fit=tr+de
    y=matrix(rows,"candidate_nll")
    acc=matrix(rows,"candidate_correct")
    full=matrix(rows,"candidate_full_label_nll")
    k=int(y[fit].mean(0).argmin())
    best_test=int(y[te].mean(0).argmin())
    preds={"fixed_train_dev":torch.full((len(te),),k),
           "hindsight_fixed_test_only":torch.full((len(te),),best_test),
           "oracle_test_only":y[te].argmin(1)}
    # This diagnostic baseline knows the prompt family, a privileged signal.
    preds["fixed_per_task_train_dev"]=torch.tensor([
        int(y[[j for j in fit if rows[j]["kind"]==rows[i]["kind"]]].mean(0).argmin())
        for i in te])
    tuned={}
    for name in FEATURES:
        x=feat[name]
        choices=[]
        for alpha in ALPHAS:
            p=predict(x[tr],x[de],y[tr],alpha)
            choices.append(float(y[de,p].mean()))
        a=ALPHAS[min(range(len(ALPHAS)),key=lambda i:choices[i])]
        tuned[name]={"alpha":a,"dev_nll_by_alpha":choices}
        preds[name+"_linear"]=predict(x[fit],x[te],y[fit],a)
    scores={}
    fixed_loss=y[te,k]
    for name,p in preds.items():
        ls=y[te,p];good=acc[te,p]
        diffs=fixed_loss-ls
        scores[name]={"nll":float(ls.mean()),"correct":int(good.sum()),
                      "full_label_nll":float(full[te,p].mean()),
                      "selected":p.tolist(),"per_row_nll":ls.tolist(),
                      "gain_vs_fixed":float(diffs.mean()),"paired_bootstrap_95":bootstrap(diffs)}
    scores["uniform_random_expectation"]={"nll":float(y[te].mean()),
                        "expected_correct":float(acc[te].mean(1).sum())}
    scores["native"]={"nll":sum(rows[i]["native_nll"] for i in te)/len(te),
                       "correct":sum(rows[i]["native_correct"] for i in te),
                       "direct_correct":sum(rows[i]["native_direct_correct"] for i in te),
                       "full_label_nll":sum(rows[i]["native_full_label_nll"] for i in te)/len(te)}
    primary=scores["residual_linear"]
    viable=scores["native"]["correct"]/len(te)>=0.8
    pass_=viable and primary["gain_vs_fixed"]>=0.02 and primary["paired_bootstrap_95"][0]>0 and primary["correct"]>=scores["fixed_train_dev"]["correct"]
    return {"native_viable":viable,"primary_gate":"PASS_EXPLORATORY" if pass_ else "NOT_DEMONSTRATED",
            "fixed_index":k,"hindsight_fixed_index":best_test,
            "train_dev_candidate_mean_nll":y[fit].mean(0).tolist(),
            "test_candidate_mean_nll":y[te].mean(0).tolist(),
            "tuned":tuned,"scores":scores}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size",type=int,default=12)
    p.add_argument("--output",required=True)
    p.add_argument("--local-snapshot",required=True)
    args=p.parse_args()
    torch.manual_seed(SEED)
    start=time.monotonic()
    snapshot=Path(args.local_snapshot)
    if snapshot.name!=REV:raise ValueError("Unexpected snapshot revision")
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
        dtype=torch.float32,attn_implementation="eager",trust_remote_code=False).eval()
    model.requires_grad_(False)
    assert model.config.num_hidden_layers==28
    rows=dataset()
    labels=label_tokens(tok)
    print("DESIGN",json.dumps({"splits":SPLITS,"masks":MASKS,"labels":labels,"batch":args.batch_size}),flush=True)
    collected,feat=collect(model,tok,rows,labels,args.batch_size)
    assert len({r["current_token_id"] for r in collected})==1
    result={"scope":"teacher-forced one-token four-choice diagnostic; not generation, not full mask training, not GPU",
        "model":MODEL,"revision":REV,"torch":torch.__version__,"transformers":__import__("transformers").__version__,
        "python":platform.python_version(),"os":platform.platform(),"device":"cpu","dtype":"float32","threads":torch.get_num_threads(),
        "seed":SEED,"batch":args.batch_size,"splits":SPLITS,"candidate_skips_1_based":MASKS,"active_blocks":24,
        "label_token_ids":labels,"data_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        "rows":collected,"evaluation":evaluate(collected,feat),"elapsed_s":time.monotonic()-start}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("RESULT",json.dumps(result["evaluation"],separators=(",",":")),flush=True)
    print("WROTE",out,flush=True)

if __name__=="__main__":
    main()
