"""Within-prompt routing pilot: preservation of native next-token distributions.

Pre-specified 24 prompts (8 train / 4 dev / 12 test), 6 native greedy decode
positions, 8 candidate masks skipping exactly four middle blocks. All masks
are drawn from the previous 12-mask pilot; the previous best fixed mask and
late-block mask are retained along with other spread/clustered masks.
The full native history is fixed for each counterfactual single-token step.
Terminal-step skipped K/V are not generated; candidate caches are cropped
and the native K/V tail restored before advancing the native trajectory.
This is NOT deployment with persistent skipped histories, GPU timing, or
task accuracy. The target is the native distribution, not a gold answer.

Primary: residual-based linear mask-candidate readout (diagnostic, NOT the
full 26-bit Bernoulli policy) improves full-vocabulary KL by >=0.02 nat
relative to BOTH train+dev-selected fixed and prompt-conditioned readouts,
paired prompt-bootstrap lower bounds >0, and improves over within-prompt
mask permutation expectation (lower bound >0). This is exploratory only.
Oracle uses test losses to select a mask; not deployable.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REV="c1899de289a04d12100db370d81485cdf75e47ca"
MODEL="Qwen/Qwen3-0.6B"
SEED=2026112603
STEPS=6
MASKS=[[4,10,16,22],[5,11,17,23],[6,12,18,24],[8,14,20,26],
       [9,10,11,12],[16,17,18,19],[24,25,26,27],[3,8,16,24]]
ALPHAS=(0.01,0.1,1.0,10.0)
FEATURES=("embedding","attention","residual","block")

# Fixed, fresh prompts. Not selected by model performance.
TASKS={
    "explain":[
        "Explain in two short sentences why ice floats on water.",
        "Explain in two short sentences why plants need sunlight.",
        "Explain in two short sentences why a bicycle stays upright when moving.",
        "Explain in two short sentences why a mirror reflects light.",
        "Explain in two short sentences why salt dissolves in water.",
        "Explain in two short sentences why we hear thunder after seeing lightning.",
    ],
    "arithmetic":[
        "Calculate 17 plus 26 and briefly explain the addition.",
        "Calculate 12 times 7 and briefly explain the multiplication.",
        "Calculate 84 minus 39 and briefly explain the subtraction.",
        "Calculate 16 times 8 and briefly explain the multiplication.",
        "Calculate 93 minus 47 and briefly explain the subtraction.",
        "Calculate 28 plus 35 and briefly explain the addition.",
    ],
    "lookup":[
        "Record: Alice has a green notebook, Ben a red pen, Cara a blue bag. Describe what Ben and Cara have in a full sentence.",
        "Record: Tokyo has rain, Osaka has sunshine, Kyoto has snow. Describe the weather in Kyoto and Tokyo in a full sentence.",
        "Record: the oak is tall, the maple is short, the pine is medium. Describe the oak and maple in a full sentence.",
        "Record: Tuesday is music, Wednesday is chess, Thursday is art. Describe the activities on Tuesday and Thursday in a full sentence.",
        "Record: Lina owns a cat, Omar owns a dog, Mei owns a rabbit. Describe Omar and Mei's pets in a full sentence.",
        "Record: box A holds rice, box B holds beans, box C holds apples. Describe what boxes B and C hold in a full sentence.",
    ],
    "list":[
        "List five common kitchen utensils with a short description.",
        "List five colors found in nature and examples of each.",
        "List five animals that live in the ocean.",
        "List five things a student might carry to school.",
        "List five foods commonly eaten for breakfast.",
        "List five objects that can be made from wood.",
    ],
}

def dataset():
    rows=[]
    for i in range(6):
        split="train" if i<2 else "dev" if i==2 else "test"
        for kind,prompts in TASKS.items():
            rows.append({"id":f"{kind}_{i}","kind":kind,"split":split,"prompt":prompts[i]})
    assert len({r["prompt"] for r in rows})==24
    return rows

def native_inputs(tok,rows):
    ids=[tok.apply_chat_template([{"role":"user","content":r["prompt"]}],
          tokenize=True,add_generation_prompt=True,enable_thinking=False) for r in rows]
    w=max(map(len,ids))
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokens=torch.tensor([[pad]*(w-len(x))+x for x in ids])
    mask=torch.tensor([[0]*(w-len(x))+[1]*len(x) for x in ids])
    pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
    return tokens,mask,pos

def collect(model,tok,rows):
    ids,mask,pos=native_inputs(tok,rows)
    cache=DynamicCache(config=model.config)
    first=model.model.layers[0]
    fs={}
    def capture_attn(m,a,o):fs["attention"]=o[0][:,-1,:].detach().cpu().clone()
    def capture_resid(m,a):fs["residual"]=a[0][:,-1,:].detach().cpu().clone()
    def capture_block(m,a,o):fs["block"]=o[:,-1,:].detach().cpu().clone()
    records=[];features={k:[] for k in FEATURES}
    eos=set([tok.eos_token_id]) if tok.eos_token_id is not None else set()
    e=getattr(model.generation_config,"eos_token_id",None)
    if isinstance(e,list):eos.update(e)
    elif isinstance(e,int):eos.add(e)
    checks=[]
    with torch.no_grad():
        out=model(input_ids=ids,attention_mask=mask,position_ids=pos,
             past_key_values=cache,use_cache=True,logits_to_keep=1)
        current=out.logits[:,0,:].argmax(-1,keepdim=True)
        ended=torch.tensor([int(t) in eos for t in current[:,0]])
        generated=[[int(x)] for x in current[:,0]]
        for step in range(STEPS):
            width=mask.shape[1]
            eligible=(~ended).clone()
            if not bool(eligible.any()):break
            current_mask=torch.cat([mask,torch.ones_like(current)],dim=1)
            current_pos=mask.sum(-1,keepdim=True)
            def run():
                return model(input_ids=current,attention_mask=current_mask,
                    position_ids=current_pos,cache_position=torch.tensor([width]),
                    past_key_values=cache,use_cache=True,logits_to_keep=1).logits[:,0,:].float()
            hooks=[first.self_attn.register_forward_hook(capture_attn),
                   first.post_attention_layernorm.register_forward_pre_hook(capture_resid),
                   first.register_forward_hook(capture_block)]
            try:ref=run()
            finally:
                for h in hooks:h.remove()
            fs["embedding"]=model.model.embed_tokens(current)[:,0,:].detach().cpu().clone()
            ref_lp=ref.log_softmax(-1);ref_p=ref_lp.exp()
            next_token=ref.argmax(-1,keepdim=True)
            # Store exactly the native appended K/V pair, never the candidate tail.
            tail=[(l.keys[...,-1:,:].clone(),l.values[...,-1:,:].clone()) for l in cache.layers]
            cache.crop(width)
            losses=[];matches=[];token_nll=[]
            for removed in MASKS:
                orig=[]
                try:
                    for j in removed:
                        layer=model.model.layers[j-1]
                        orig.append((layer,layer.forward))
                        layer.forward=lambda hidden_states,*a,**kw:hidden_states
                    z=run()
                    lp=z.log_softmax(-1)
                    kl=(ref_p*(ref_lp-lp)).sum(-1)
                    if float(kl.min()) < -1e-5:raise RuntimeError("Negative KL outside tolerance")
                    losses.append(kl.clamp_min(0).cpu())
                    matches.append((z.argmax(-1)==next_token[:,0]).cpu())
                    token_nll.append((-lp.gather(1,next_token)[:,0]).cpu())
                finally:
                    for l,fn in orig:l.forward=fn
                    cache.crop(width)
                assert all(l.get_seq_length()==width for l in cache.layers)
            if step in (0,STEPS-1):
                replay=run()
                delta=float((ref-replay).abs().max())
                checks.append({"step":step,"max_abs_logits_after_candidate_sweep":delta})
                if delta>2e-4:raise RuntimeError(f"Native replay changed: {delta}")
                cache.crop(width)
            for j,(k,v) in enumerate(tail):
                cache.update(k,v,j)
                assert cache.layers[j].get_seq_length()==width+1
                assert torch.equal(cache.layers[j].keys[...,-1:,:],k)
                assert torch.equal(cache.layers[j].values[...,-1:,:],v)
            for i,r in enumerate(rows):
                if not bool(eligible[i]):continue
                rec={"id":r["id"],"kind":r["kind"],"split":r["split"],"step":step,
                     "current_token_id":int(current[i,0]),"current_token":tok.decode([int(current[i,0])]),
                     "native_next_token_id":int(next_token[i,0]),"native_next_token":tok.decode([int(next_token[i,0])]),
                     "candidate_kl":[float(a[i]) for a in losses],
                     "candidate_top1_match":[int(a[i]) for a in matches],
                     "candidate_native_token_nll":[float(a[i]) for a in token_nll]}
                records.append(rec)
                for k in FEATURES:features[k].append(fs[k][i])
                generated[i].append(int(next_token[i,0]))
            print(f"step {step+1}/{STEPS}: {int(eligible.sum())} eligible prompts; mean candidate KL={float(torch.stack(losses)[:,eligible].mean()):.6f}",flush=True)
            ended=ended|torch.tensor([int(t) in eos for t in next_token[:,0]])
            current=next_token
            mask=current_mask
    return records,{k:torch.stack(v).double() for k,v in features.items()},checks,{r["id"]:tok.decode(generated[i],skip_special_tokens=True) for i,r in enumerate(rows)}

def groups(records,split=None):
    g={}
    for i,r in enumerate(records):
        if split is None or r["split"]==split:g.setdefault(r["id"],[]).append(i)
    return g

def readout(x,z,y,alpha):
    # Global TRAIN-only centering and scalar scale: affine scores on raw taps.
    # No per-token normalization or additional learned network.
    mean=x.mean(0,keepdim=True)
    scale=(x-mean).square().sum(1).mean().sqrt().clamp_min(1e-12)
    x=(x-mean)/scale;z=(z-mean)/scale
    y=y-y.mean(1,keepdim=True)
    ym=y.mean(0,keepdim=True)
    a=torch.linalg.solve(x@x.T+alpha*torch.eye(len(x),dtype=x.dtype),y-ym)
    return (z@x.T@a+ym).argmin(1)

def avg_group(loss,gs):
    return torch.stack([loss[ids].mean(0) for ids in gs.values()])

def bootstrap(delta):
    gen=torch.Generator().manual_seed(SEED)
    ind=torch.randint(len(delta),(10000,len(delta)),generator=gen)
    return [float(v) for v in torch.quantile(delta[ind].mean(1),torch.tensor([.025,.975],dtype=torch.float64))]

def evaluate(records,feat):
    y=torch.tensor([r["candidate_kl"] for r in records],dtype=torch.float64)
    match=torch.tensor([r["candidate_top1_match"] for r in records],dtype=torch.float64)
    ng={s:groups(records,s) for s in ("train","dev","test")}
    allg=groups(records)
    expected={"train":8,"dev":4,"test":12}
    viable=all(len(ng[s])==expected[s] and min(map(len,ng[s].values()))>=4 for s in expected)
    tr=[i for ids in ng["train"].values() for i in ids]
    de=[i for ids in ng["dev"].values() for i in ids]
    te=[i for ids in ng["test"].values() for i in ids]
    fit=tr+de; fg={**ng["train"],**ng["dev"]};tg=ng["test"]
    fidx=int(avg_group(y,fg).mean(0).argmin())
    preds={"fixed_train_dev":torch.full((len(te),),fidx),
           "token_oracle_test_only":y[te].argmin(1)}
    ph={pid:int(y[ids].mean(0).argmin()) for pid,ids in tg.items()}
    preds["prompt_oracle_test_only"]=torch.tensor([ph[records[i]["id"]] for i in te])
    hidx=int(avg_group(y,tg).mean(0).argmin())
    preds["hindsight_fixed_test_only"]=torch.full((len(te),),hidx)
    fam={}
    for k in TASKS:
        subset={pid:ids for pid,ids in fg.items() if records[ids[0]]["kind"]==k}
        fam[k]=int(avg_group(y,subset).mean(0).argmin())
    preds["task_fixed_train_dev"]=torch.tensor([fam[records[i]["kind"]] for i in te])
    # Choose hyperparameters on dev only, with prompt-weighted loss.
    tuning={}
    local={p:[te.index(i) for i in ids] for p,ids in tg.items()}
    devlocal={p:[de.index(i) for i in ids] for p,ids in ng["dev"].items()}
    for name in FEATURES:
        x=feat[name]
        dev_losses=[]
        for alpha in ALPHAS:
            pd=readout(x[tr],x[de],y[tr],alpha)
            dev_losses.append(float(avg_group(y[de,pd],devlocal).mean()))
        a=ALPHAS[min(range(len(ALPHAS)),key=lambda j:dev_losses[j])]
        tuning[name]={"alpha":a,"dev_kl":dev_losses}
        preds[name+"_linear"]=readout(x[fit],x[te],y[fit],a)
    # Deployable prompt-conditioned control sees first decode activation only.
    x=feat["residual"]
    xtr=torch.stack([x[v[0]] for v in ng["train"].values()])
    xde=torch.stack([x[v[0]] for v in ng["dev"].values()])
    ytr=avg_group(y,ng["train"]);yde=avg_group(y,ng["dev"])
    dl=[]
    for a in ALPHAS:
        p=readout(xtr,xde,ytr,a)
        dl.append(float(yde[torch.arange(len(p)),p].mean()))
    a=ALPHAS[min(range(len(ALPHAS)),key=lambda j:dl[j])]
    tuning["prompt_residual"]={"alpha":a,"dev_kl":dl}
    pf=readout(torch.stack([x[v[0]] for v in fg.values()]),
         torch.stack([x[v[0]] for v in tg.values()]),avg_group(y,fg),a)
    pm=dict(zip(tg,pf.tolist()))
    preds["prompt_residual_linear"]=torch.tensor([pm[records[i]["id"]] for i in te])
    scores={}
    per={}
    for name,p in preds.items():
        loss=y[te,p]
        per[name]=avg_group(loss,local)
        scores[name]={"kl":float(per[name].mean()),"top1_match":float(avg_group(match[te,p],local).mean()),
            "selected":p.tolist(),"per_prompt_kl":per[name].tolist()}
    # Exact expectation for independent uniform within-prompt permutation of
    # selected mask multiset: preserves each prompt's mask usage frequency.
    p=preds["residual_linear"];shuffle=[]
    for ids in local.values():
        masks=p[ids];shuffle.append(y[torch.tensor(te)[ids]][:,masks].mean())
    per["within_prompt_permutation_expectation"]=torch.stack(shuffle)
    scores["within_prompt_permutation_expectation"]={"kl":float(per["within_prompt_permutation_expectation"].mean()),
      "per_prompt_kl":per["within_prompt_permutation_expectation"].tolist()}
    d=per["residual_linear"]
    comparisons={}
    for key in ("fixed_train_dev","prompt_residual_linear","task_fixed_train_dev","within_prompt_permutation_expectation"):
        delta=per[key]-d
        comparisons[key]={"gain":float(delta.mean()),"paired_prompt_bootstrap_95":bootstrap(delta)}
    op=per["prompt_oracle_test_only"]-per["token_oracle_test_only"]
    oracle_change={p:len(set(y[ids].argmin(1).tolist())) for p,ids in tg.items()}
    dynamic_change={p:len(set(preds["residual_linear"][ids].tolist())) for p,ids in local.items()}
    enough=viable and all(comparisons[k]["gain"]>=.02 and comparisons[k]["paired_prompt_bootstrap_95"][0]>0
                  for k in ("fixed_train_dev","prompt_residual_linear")) and comparisons["within_prompt_permutation_expectation"]["paired_prompt_bootstrap_95"][0]>0
    return {"primary_gate":"PASS_EXPLORATORY" if enough else "NOT_DEMONSTRATED",
      "coverage_gate":viable,"positions_per_prompt":{p:len(ids) for p,ids in allg.items()},
      "global_fixed_index":fidx,"hindsight_fixed_index":hidx,"task_fixed_indices":fam,"tuning":tuning,
      "test_prompt_ids":list(tg),"test_record_indices":te,"scores":scores,"comparisons":comparisons,
      "oracle_within_prompt_gain":{"gain":float(op.mean()),"paired_prompt_bootstrap_95":bootstrap(op)},
      "oracle_unique_masks_per_prompt":oracle_change,"dynamic_unique_masks_per_prompt":dynamic_change,
      "train_dev_fixed_mean_kl":avg_group(y,fg).mean(0).tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-snapshot",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    start=time.monotonic();torch.manual_seed(SEED)
    snapshot=Path(args.local_snapshot)
    if snapshot.name!=REV:raise ValueError("Wrong model revision")
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32,
             attn_implementation="eager",trust_remote_code=False).eval()
    model.requires_grad_(False)
    assert model.config.num_hidden_layers==28
    rows=dataset()
    print("DESIGN",json.dumps({"prompts":24,"steps":STEPS,"masks":MASKS,"train":8,"dev":4,"test":12}),flush=True)
    records,features,checks,gen=collect(model,tok,rows)
    result={"model":MODEL,"revision":REV,"dtype":"float32","device":"cpu","torch":torch.__version__,
      "transformers":__import__("transformers").__version__,"python":platform.python_version(),"os":platform.platform(),
      "threads":torch.get_num_threads(),"batch":len(rows),"seed":SEED,"steps":STEPS,
      "candidate_skips_1_based":MASKS,"active_blocks":24,
      "scope":"native greedy trajectory, exact full-vocabulary KL on one-step counterfactuals; no persistent skip history or accuracy claims",
      "prompts":rows,"data_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
      "native_generation":gen,"cache_replay_checks":checks,"records":records,
      "evaluation":evaluate(records,features),"elapsed_s":time.monotonic()-start}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    e=result["evaluation"]
    print("RESULT",json.dumps({k:v for k,v in e.items() if k not in ("scores","tuning","test_record_indices")}),flush=True)
    print("SCORES",json.dumps({k:{a:b for a,b in v.items() if a not in ("selected","per_prompt_kl")} for k,v in e["scores"].items()}),flush=True)
    print("WROTE",out,flush=True)

if __name__=="__main__":
    main()
