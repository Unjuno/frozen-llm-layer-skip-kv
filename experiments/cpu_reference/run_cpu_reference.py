#!/usr/bin/env python3
"""CPU reference experiment: shared 1st-attention tap, dynamic masks, Eager/Lazy KV.

This is a randomly initialized *synthetic* five-layer decoder. It is NOT Qwen3.
The experiment establishes implementation invariants, not pretrained model
quality, GPU performance, or BF16/numerical fidelity to Qwen3/Nanbeige.
Run: python experiments/cpu_reference/run_cpu_reference.py --output results/cpu_reference.json
Requires: torch (tested with torch 2.10.0+cpu), Python 3.13.
"""
import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

D = 16
HEADS = 4
KV_HEADS = 2
HEAD_DIM = D // HEADS
N_LAYERS = 5
VOCAB = 29
DTYPE = torch.float64


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=DTYPE))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def rope(x, position):
    """A small positional rotation, consistently applied to ready and rebuilt KV."""
    inv = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, dtype=DTYPE) / HEAD_DIM)
    a = position * inv
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * a.cos() - x[..., 1::2] * a.sin()
    out[..., 1::2] = x[..., 0::2] * a.sin() + x[..., 1::2] * a.cos()
    return out


class FrozenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.q = nn.Linear(D, HEADS * HEAD_DIM, bias=False, dtype=DTYPE)
        self.k = nn.Linear(D, KV_HEADS * HEAD_DIM, bias=False, dtype=DTYPE)
        self.v = nn.Linear(D, KV_HEADS * HEAD_DIM, bias=False, dtype=DTYPE)
        self.o = nn.Linear(D, D, bias=False, dtype=DTYPE)
        self.mlp_up = nn.Linear(D, 2 * D, bias=False, dtype=DTYPE)
        self.mlp_down = nn.Linear(2 * D, D, bias=False, dtype=DTYPE)

    def kv_from_state(self, state, pos):
        z = self.norm1(state)
        k = rope(self.k(z).reshape(KV_HEADS, HEAD_DIM), pos)
        v = self.v(z).reshape(KV_HEADS, HEAD_DIM)
        return k, v

    def run(self, x, pos, kv_by_pos):
        assert sorted(kv_by_pos) == list(range(pos + 1)), "KV must be complete before attention"
        z = self.norm1(x)
        q = rope(self.q(z).reshape(HEADS, HEAD_DIM), pos)
        keys = torch.stack([kv_by_pos[t][0] for t in range(pos + 1)])
        values = torch.stack([kv_by_pos[t][1] for t in range(pos + 1)])
        keys = keys.repeat_interleave(HEADS // KV_HEADS, dim=1)
        values = values.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = torch.einsum('hd,thd->ht', q, keys) / math.sqrt(HEAD_DIM)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.einsum('ht,thd->hd', weights, values).reshape(D)
        a = self.o(attention)
        u = x + a  # tap: 1st attention + residual, BEFORE the 1st block's MLP
        h = u + self.mlp_down(torch.nn.functional.silu(self.mlp_up(self.norm2(u))))
        return h, u


class FrozenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D, dtype=DTYPE)
        self.layers = nn.ModuleList(FrozenBlock() for _ in range(N_LAYERS))
        self.norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, VOCAB, bias=False, dtype=DTYPE)
        for p in self.parameters():
            p.requires_grad_(False)


@dataclass
class KVState:
    mode: str
    ready: list = field(default_factory=lambda: [dict() for _ in range(N_LAYERS)])
    deferred: list = field(default_factory=lambda: [dict() for _ in range(N_LAYERS)])
    stats: Counter = field(default_factory=Counter)

    def prepare(self, layer_idx, block, x, pos, execute):
        if not execute:
            if self.mode == 'eager':
                self.ready[layer_idx][pos] = block.kv_from_state(x, pos)
                self.stats['eager_skip_kv_pairs'] += 1
                self.stats['total_kv_projection_pairs'] += 1
            else:
                self.deferred[layer_idx][pos] = x.detach().clone()
                self.stats['lazy_saved_inputs'] += 1
            return
        # Fill ALL missing prior positions before this layer executes attention.
        if self.mode == 'lazy':
            for old_pos, old_state in sorted(self.deferred[layer_idx].items()):
                assert old_pos < pos
                self.ready[layer_idx][old_pos] = block.kv_from_state(old_state, old_pos)
                self.stats['lazy_reconstructed_kv_pairs'] += 1
                self.stats['total_kv_projection_pairs'] += 1
            self.deferred[layer_idx].clear()
        self.ready[layer_idx][pos] = block.kv_from_state(x, pos)
        self.stats['total_kv_projection_pairs'] += 1
        self.stats['executed_attention_calls'] += 1
        if layer_idx == 0:
            self.stats['first_attention_calls'] += 1


def one_step(model, kv, token, pos, forced_mask=None, policy=None, sampling_seed=None):
    assert (forced_mask is None) != (policy is None), 'Provide fixed mask OR linear policy'
    h = model.embed(torch.tensor(token))
    first = model.layers[0]
    kv.prepare(0, first, h, pos, execute=True)
    h, tap = first.run(h, pos, kv.ready[0])
    kv.stats['shared_tap_reads'] += int(policy is not None)
    selected_log_prob = None
    if policy is not None:
        logits = policy(tap.detach())
        probs = torch.sigmoid(logits)
        eps = 0.05
        sampling_p = eps + (1 - 2 * eps) * probs
        dist = torch.distributions.Bernoulli(probs=sampling_p)
        act = dist.sample()
        mask = [bool(x) for x in act.tolist()]
        selected_log_prob = dist.log_prob(act).sum()
    else:
        mask = list(map(bool, forced_mask))
    assert len(mask) == N_LAYERS - 2
    for layer_idx in range(1, N_LAYERS - 1):
        block = model.layers[layer_idx]
        execute = mask[layer_idx - 1]
        kv.prepare(layer_idx, block, h, pos, execute)
        if execute:
            h, _ = block.run(h, pos, kv.ready[layer_idx])
    last_idx = N_LAYERS - 1
    last = model.layers[last_idx]
    kv.prepare(last_idx, last, h, pos, execute=True)
    h, _ = last.run(h, pos, kv.ready[last_idx])
    out = model.lm_head(model.norm(h))
    return out, tap, mask, selected_log_prob


def fixed_replay(model, tokens, masks, mode, with_head=False):
    kv = KVState(mode)
    outputs, taps = [], []
    head = nn.Linear(D, N_LAYERS - 2, dtype=DTYPE) if with_head else None
    with torch.no_grad():
        for pos, token in enumerate(tokens):
            # The same computed tap is consumed by the head; its prediction
            # cannot affect the all-ON comparison when masks are forced.
            logits, tap, _, _ = one_step(model, kv, token, pos, forced_mask=masks[pos])
            if head is not None:
                _ = head(tap)
                kv.stats['shared_tap_reads'] += 1
            outputs.append(logits)
            taps.append(tap)
    return torch.stack(outputs), torch.stack(taps), dict(kv.stats)


def policy_rollout(model, policy, initial_token, n_steps):
    kv = KVState('eager')
    token = initial_token
    logs, decisions, generated = [], [], []
    for pos in range(n_steps):
        logits, _, mask, log_prob = one_step(model, kv, token, pos, policy=policy)
        logs.append(log_prob)
        decisions.append(mask)
        token = int(logits.argmax().item())
        generated.append(token)
    # Artificial exact-match task: the next-token sequence must end at a
    # predetermined token; this is a synthetic test, NOT LLM accuracy.
    correct = int(generated[-1] == ((initial_token + 1) % VOCAB))
    cost = sum(sum(m) + 2 for m in decisions) / (N_LAYERS * n_steps)
    return {'reward': correct - 0.05 * cost, 'correct': correct,
            'cost': cost, 'log_prob': torch.stack(logs).sum(),
            'generated': generated, 'decisions': decisions}


def model_digest(model):
    values = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def run():
    torch.manual_seed(20261118)
    torch.set_num_threads(1)
    model = FrozenDecoder()
    all_on_tokens = [2, 3, 7, 11, 5, 23, 4, 1]
    masks_on = [[1] * (N_LAYERS - 2) for _ in all_on_tokens]
    first_mlp_counter = Counter()
    hook = model.layers[0].mlp_down.register_forward_hook(
        lambda *_: first_mlp_counter.update({'calls': 1}))
    native, tap_native, native_stats = fixed_replay(model, all_on_tokens, masks_on, 'eager', with_head=False)
    masked, tap_masked, mask_stats = fixed_replay(model, all_on_tokens, masks_on, 'eager', with_head=True)
    hook.remove()
    assert first_mlp_counter['calls'] == 2 * len(all_on_tokens)
    native_all_on_max_abs = float((native - masked).abs().max())
    tap_max_abs = float((tap_native - tap_masked).abs().max())
    assert native_all_on_max_abs == 0 and tap_max_abs == 0
    assert native_stats['first_attention_calls'] == len(all_on_tokens)
    assert mask_stats['first_attention_calls'] == len(all_on_tokens)
    assert mask_stats['shared_tap_reads'] == len(all_on_tokens)

    # Cover single-skip, alternating, all-off, skip->resume and mixed masks.
    torch.manual_seed(444)
    cases = {
        'alternating': [[(pos + layer) % 2 for layer in range(3)] for pos in range(8)],
        'all_middle_off': [[0, 0, 0] for _ in range(8)],
        'off_then_resume': [[0, 0, 0] for _ in range(6)] + [[1, 1, 1] for _ in range(2)],
        'mixed': [[int(x) for x in torch.randint(0, 2, (3,)).tolist()] for _ in range(8)],
        'all_on': [[1, 1, 1] for _ in range(8)],
    }
    for trial in range(24):
        cases[f'random_{trial:02d}'] = [
            torch.randint(0, 2, (3,)).tolist() for _ in range(8)]
    case_rows = []
    for name, masks in cases.items():
        eager_logits, _, eager_stats = fixed_replay(model, all_on_tokens, masks, 'eager')
        lazy_logits, _, lazy_stats = fixed_replay(model, all_on_tokens, masks, 'lazy')
        delta = float((eager_logits - lazy_logits).abs().max())
        assert delta <= 1e-12, (name, delta)
        eager_proj = eager_stats.get('total_kv_projection_pairs', 0)
        lazy_proj = lazy_stats.get('total_kv_projection_pairs', 0)
        assert lazy_proj <= eager_proj
        expected_saved = sum(8 - max([i + 1 for i, row in enumerate(masks) if row[j]] or [0]) for j in range(3))
        assert eager_proj - lazy_proj == expected_saved, (name, eager_proj, lazy_proj, expected_saved)
        case_rows.append({'case': name, 'max_abs_logits': delta,
                          'eager_kv_pairs': eager_proj, 'lazy_kv_pairs': lazy_proj,
                          'avoided_kv_pairs': eager_proj - lazy_proj,
                          'first_attention_calls_each': eager_stats['first_attention_calls']})

    # Negative control: intentionally wrong positions during lazy KV replay
    # must fail the same-logits criterion, proving the test can catch bad RoPE.
    original_projector = model.layers[1].kv_from_state
    def incorrect_kv_position(state, pos):
        return original_projector(state, pos + 1)
    try:
        model.layers[1].kv_from_state = incorrect_kv_position
        wrong_logits, _, _ = fixed_replay(
            model, all_on_tokens, cases['off_then_resume'], 'lazy')
    finally:
        model.layers[1].kv_from_state = original_projector
    control_logits, _, _ = fixed_replay(
        model, all_on_tokens, cases['off_then_resume'], 'eager')
    negative_control_delta = float((wrong_logits - control_logits).abs().max())
    assert negative_control_delta > 1e-8, negative_control_delta

    torch.manual_seed(37)
    policy = nn.Linear(D, N_LAYERS - 2, dtype=DTYPE)
    before_model = model_digest(model)
    before_w = policy.weight.detach().clone()
    n_rollouts = 6
    rollouts = [policy_rollout(model, policy, initial_token=7, n_steps=5)
                for _ in range(n_rollouts)]
    loss = torch.zeros((), dtype=DTYPE)
    for idx, row in enumerate(rollouts):
        mean_others = sum(x['reward'] for k, x in enumerate(rollouts) if k != idx) / (n_rollouts - 1)
        loss = loss - (row['reward'] - mean_others) * row['log_prob'] / n_rollouts
    loss.backward()
    grad_norm = float(policy.weight.grad.norm())
    assert grad_norm > 0 and math.isfinite(grad_norm)
    opt = torch.optim.SGD(policy.parameters(), lr=0.05)
    opt.step()
    head_change = float((policy.weight.detach() - before_w).abs().max())
    assert head_change > 0
    assert model_digest(model) == before_model
    assert all(p.grad is None for p in model.parameters())

    return {
        'description': 'CPU synthetic frozen 5-layer GQA-like decoder correctness experiment; NOT pretrained Qwen3',
        'date_utc': '2026-11-18',
        'environment': {'os': platform.platform(), 'python': platform.python_version(),
                        'torch': torch.__version__, 'gpu_available': torch.cuda.is_available(),
                        'device': 'CPU', 'threads': torch.get_num_threads(),
                        'dtype': str(DTYPE), 'seed_model': 20261118, 'seed_masks': 444,
                        'seed_router': 37},
        'synthetic_architecture': {'layers': N_LAYERS, 'hidden_size': D, 'q_heads': HEADS,
                                   'kv_heads': KV_HEADS, 'head_dim': HEAD_DIM, 'vocab': VOCAB,
                                   'first_and_last_always_executed': True},
        'tests': {
            'native_vs_masked_all_on': {'status': 'PASS', 'max_abs_logits': native_all_on_max_abs,
                                       'max_abs_first_attention_tap': tap_max_abs,
                                       'first_attention_calls': native_stats['first_attention_calls'],
                                       'router_consumptions': mask_stats['shared_tap_reads'],
                                       'first_mlp_calls_native_plus_masked': first_mlp_counter['calls']},
            'negative_control_wrong_kv_position': {'status': 'PASS',
                          'max_abs_logits_when_position_corrupted': negative_control_delta,
                          'expected_difference_detected': True},
            'fixed_replay_eager_vs_lazy': {'status': 'PASS', 'cases': case_rows,
                                          'tolerance_max_abs': 1e-12},
            'head_only_reinforce_gradient': {'status': 'PASS', 'rollouts': n_rollouts,
                                            'synthetic_correct_count': sum(r['correct'] for r in rollouts),
                                            'gradient_norm': grad_norm,
                                            'max_head_weight_change': head_change,
                                            'frozen_model_sha256_unchanged': True,
                                            'rewards': [round(r['reward'], 6) for r in rollouts]}
        },
        'limitations': ['No actual pretrained Qwen3 checkpoint, tokenizer or generation benchmarks',
                        'No CUDA; GPU latency / VRAM / speedup not measured',
                        'Synthetic RoPE-like rotation, architecture not numerically identical to Qwen3',
                        'Only 8-token fixed replay sequences and one synthetic policy-gradient step',
                        'No meaningful task-quality or model-improvement claim']
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/cpu_reference.json')
    args = ap.parse_args()
    report = run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('RESULT_FILE:', output)
