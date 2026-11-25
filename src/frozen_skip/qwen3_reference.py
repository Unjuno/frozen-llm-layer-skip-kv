"""Qwen3 v4.57.3 token-wise layer-mask reference executor.

This is a correctness-first, batch=1, eager-attention reference. It reuses the
EXISTING first-layer self-attention output through hooks, keeps the first
layer's MLP and the last decoder layer active, and trains only a Linear head.

KV semantics: Eager means project K/V even when the block is bypassed.
Lazy means store its true input and reconstruct missing K/V BEFORE the next
execution of that layer, using the SAME frozen K-projection, K norm, RoPE,
V-projection and positions. CPU synchronizations are deliberately present:
this reference is NOT a production GPU-fast serving implementation.

Tested source contract: transformers==4.57.3 Qwen3DecoderLayer/Attention.
Real pretrained Qwen3 integration remains unverified until run externally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class LinearMaskHead(nn.Module):
    """All intermediate masks at once; no layerwise decision network."""

    def __init__(self, hidden_size: int, num_hidden_layers: int):
        super().__init__()
        if num_hidden_layers < 3:
            raise ValueError("Requires first, intermediate, and last decoder layers")
        self.linear = nn.Linear(hidden_size, num_hidden_layers - 2, bias=True)

    def act(self, tap: torch.Tensor, exploration: float = 0.0):
        if not 0.0 <= exploration < 0.5:
            raise ValueError("exploration must be in [0,0.5)")
        # Called inside torch.enable_grad to retain gradients only for head.
        logits = self.linear(tap.detach().float())
        probs = torch.sigmoid(logits)
        probs = exploration + (1.0 - 2.0 * exploration) * probs
        dist = torch.distributions.Bernoulli(probs=probs)
        sample = dist.sample()
        return sample[0].to(dtype=torch.bool), dist.log_prob(sample).sum()


@dataclass
class Pending:
    hidden: torch.Tensor
    position_ids: torch.Tensor
    cache_pos: torch.Tensor


class Qwen3ReferenceRouter:
    """Context manager patching only decoder-layer forward methods.

    One invocation processes one request (batch=1, decode length=1).
    Prefill is native and unconditional. A new DynamicCache is required for
    every replay. No torch.compile, sliding caches, beam search, or padding.
    """

    def __init__(self, model: nn.Module, head: Optional[LinearMaskHead] = None,
                 mode: str = "eager", tap: str = "post_residual",
                 exploration: float = 0.05):
        if mode not in {"eager", "lazy"}:
            raise ValueError("mode must be eager or lazy")
        if tap not in {"pre_residual", "post_residual"}:
            raise ValueError("tap must be pre_residual or post_residual")
        self.model = model
        self.head = head
        self.mode = mode
        self.tap = tap
        self.exploration = exploration
        self.layers = model.model.layers
        self.n_middle = len(self.layers) - 2
        if self.n_middle <= 0:
            raise ValueError("No intermediate layers")
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("Use attn_implementation='eager' for reference correctness")
        if any(t != "full_attention" for t in getattr(model.config, "layer_types",
                                                      ["full_attention"] * len(self.layers))):
            raise ValueError("This reference does not support sliding-window layers")
        self.active = False
        self.current_mask = None
        self.current_log_prob = None
        self.current_tap = None
        self.first_input = None
        self.forced_mask = None
        self.pending = {j: [] for j in range(1, len(self.layers) - 1)}
        self.stats = Counter()
        self.masks = []
        self.log_probs = []
        self._orig = []
        self._hooks = []

    def __enter__(self):
        def on_first_enter(module, args):
            if self.active:
                self.first_input = args[0]

        def on_attention_exit(module, args, output):
            if not self.active:
                return
            self.stats["first_attention_calls"] += 1
            if self.first_input is None:
                raise RuntimeError("First-layer input capture was missing")
            attn_output = output[0] if isinstance(output, tuple) else output
            tensor = attn_output if self.tap == "pre_residual" else self.first_input + attn_output
            if tensor.shape[0] != 1 or tensor.shape[1] != 1:
                raise ValueError("Only batch=1, one decode token is supported")
            self.current_tap = tensor[:, -1, :].detach()
            if self.forced_mask is None:
                if self.head is None:
                    raise ValueError("No head is available for policy sampling")
                # Model can be under torch.no_grad(); head's policy score must not be.
                with torch.enable_grad():
                    action, lp = self.head.act(self.current_tap, self.exploration)
                # CPU synchronization here is intentional and NOT a speed technique.
                self.current_mask = [bool(v) for v in action.tolist()]
                self.current_log_prob = lp
            else:
                self.current_mask = self.forced_mask.copy()
                if self.head is not None:
                    with torch.no_grad():
                        _ = self.head.linear(self.current_tap.float())
                    self.stats["head_probe_calls"] += 1

        self._hooks.append(self.layers[0].register_forward_pre_hook(on_first_enter))
        self._hooks.append(self.layers[0].self_attn.register_forward_hook(on_attention_exit))
        for idx in range(1, len(self.layers) - 1):
            layer = self.layers[idx]
            original = layer.forward
            self._orig.append((layer, original))

            def routed(hidden_states, *args, _idx=idx, _orig=original, **kwargs):
                if not self.active:
                    return _orig(hidden_states, *args, **kwargs)
                if self.current_mask is None:
                    raise RuntimeError("Head must select a mask during first attention")
                cache = kwargs.get("past_key_values")
                cp = kwargs.get("cache_position")
                if not isinstance(cache, DynamicCache) or cp is None or cp.numel() != 1:
                    raise ValueError("Requires DynamicCache and 1-token decoding")
                if self.current_mask[_idx - 1]:
                    self._materialize(_idx, cache)
                    actual = cache.layers[_idx].get_seq_length()
                    expected = int(cp[0].item())
                    if actual != expected:
                        raise RuntimeError(f"Layer {_idx} KV length {actual} != {expected}")
                    self.stats["executed_middle"] += 1
                    return _orig(hidden_states, *args, **kwargs)
                self.stats["skipped_middle"] += 1
                if self.mode == "eager":
                    self._project_and_append(_idx, hidden_states, kwargs["position_ids"],
                                             cp, kwargs["position_embeddings"], cache)
                    self.stats["eager_skip_projections"] += int(cp.numel())
                else:
                    self.pending[_idx].append(Pending(
                        hidden_states.detach().clone(),
                        kwargs["position_ids"].detach().clone(),
                        cp.detach().clone()))
                    self.stats["lazy_deferred_positions"] += 1
                return hidden_states  # EXACT identity bypass, no attention or MLP
            layer.forward = routed
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active = False
        for layer, original in self._orig:
            layer.forward = original
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._orig.clear()

    def begin_decode(self, forced_mask=None):
        if forced_mask is not None:
            if len(forced_mask) != self.n_middle:
                raise ValueError("Mask width differs from number of intermediate layers")
            forced_mask = [bool(x) for x in forced_mask]
        self.forced_mask = forced_mask
        self.active = True
        self.current_mask = None
        self.current_log_prob = None
        self.current_tap = None
        self.first_input = None

    def end_decode(self):
        if self.current_mask is None:
            raise RuntimeError("Decode step finished without choosing a mask")
        if self.stats["first_attention_calls"] != len(self.masks) + 1:
            raise RuntimeError("First Attention was not invoked exactly once per decode step")
        self.masks.append(self.current_mask.copy())
        if self.current_log_prob is not None:
            self.log_probs.append(self.current_log_prob)
        self.active = False
        return self.current_mask

    def _project_and_append(self, idx, hidden, position_ids, cache_pos, embeddings, cache):
        attn = self.layers[idx].self_attn
        normalized = self.layers[idx].input_layernorm(hidden)
        batch, seq, _ = normalized.shape
        shape = (batch, seq, -1, attn.head_dim)
        k = attn.k_norm(attn.k_proj(normalized).view(shape)).transpose(1, 2)
        v = attn.v_proj(normalized).view(shape).transpose(1, 2)
        cos, sin = embeddings
        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
        cache.update(k, v, idx, {"cos": cos, "sin": sin, "cache_position": cache_pos})
        self.stats["kv_projected_positions"] += seq

    def _materialize(self, idx, cache):
        rows = self.pending[idx]
        if not rows:
            return
        previous_length = cache.layers[idx].get_seq_length()
        if int(rows[0].cache_pos[0].item()) != previous_length:
            raise RuntimeError("Pending KV does not start at current cached prefix")
        hidden = torch.cat([r.hidden for r in rows], dim=1)
        posids = torch.cat([r.position_ids for r in rows], dim=1)
        cps = torch.cat([r.cache_pos for r in rows], dim=0)
        expected = torch.arange(previous_length, previous_length + len(rows), device=cps.device)
        if not torch.equal(cps, expected):
            raise RuntimeError("Deferred positions are not contiguous or ordered")
        cos_sin = self.model.model.rotary_emb(hidden, posids)
        self._project_and_append(idx, hidden, posids, cps, cos_sin, cache)
        self.stats["lazy_reconstructed_positions"] += len(rows)
        self.stats["lazy_batched_projection_calls"] += 1
        self.pending[idx] = []

    def stats_dict(self):
        return {**dict(self.stats),
                "pending_positions_at_end": sum(len(x) for x in self.pending.values()),
                "decode_steps": len(self.masks),
                "executed_masks": self.masks}
