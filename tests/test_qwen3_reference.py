"""CPU tests for a randomly initialized, small but *real* Qwen3 architecture.

Pretrained checkpoint integration, numerical BF16, and GPU performance need
their own later runs. This suite never downloads pretrained model weights.
"""
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from frozen_skip.qwen3_reference import LinearMaskHead, Qwen3ReferenceRouter
from experiments.qwen3.run import run_fixed, mask_cases


@pytest.fixture
def tiny():
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=127, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, attention_dropout=0.0,
        layer_types=["full_attention"] * 4, bos_token_id=2, eos_token_id=1,
        pad_token_id=0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.mark.parametrize("tap", ["pre_residual", "post_residual"])
def test_all_on_does_not_change_native_logits_or_first_mlp(tiny, tap):
    prompt = torch.tensor([[2, 4, 5]], dtype=torch.long)
    tokens = [6, 7, 8, 9, 10]
    native, _ = run_fixed(tiny, prompt, tokens)
    masks = [[True] * (len(tiny.model.layers) - 2) for _ in tokens]
    outputs, stats = run_fixed(tiny, prompt, tokens, masks, head_probe=True, tap=tap)
    assert torch.allclose(native, outputs, atol=1e-5, rtol=0)
    assert stats["first_attention_calls"] == len(tokens)
    assert stats["head_probe_calls"] == len(tokens)
    assert stats["tap_reads"] == len(tokens)
    # One prefill invocation + one first-layer MLP invocation per decode step.
    assert stats["first_block_mlp_invocations"] == len(tokens) + 1


@pytest.mark.parametrize("name", ["all_on", "all_off", "single_skip",
                                  "alternating", "off_then_resume", "random"])
def test_eager_vs_lazy_fixed_replay(tiny, name):
    prompt = torch.tensor([[2, 4, 5]], dtype=torch.long)
    tokens = [6, 7, 8, 9, 10, 11]
    masks = mask_cases(len(tiny.model.layers) - 2, len(tokens))[name]
    eager, es = run_fixed(tiny, prompt, tokens, masks, mode="eager")
    lazy, ls = run_fixed(tiny, prompt, tokens, masks, mode="lazy")
    assert torch.allclose(eager, lazy, atol=2e-4, rtol=0)
    expected_pending = sum(len(tokens) -
                           max([idx + 1 for idx, row in enumerate(masks) if row[j]] or [0])
                           for j in range(len(tiny.model.layers) - 2))
    assert ls["pending_positions_at_end"] == expected_pending
    assert es["first_attention_calls"] == len(tokens)


def test_head_policy_gradients_do_not_touch_frozen_llm(tiny):
    torch.manual_seed(147)
    head = LinearMaskHead(tiny.config.hidden_size, len(tiny.model.layers))
    cache = DynamicCache(config=tiny.config)
    with torch.no_grad():
        tiny(input_ids=torch.tensor([[2, 4]]), past_key_values=cache, use_cache=True)
    with Qwen3ReferenceRouter(tiny, head=head, mode="eager") as runner:
        with torch.no_grad():
            for tok in (6, 7, 8):
                runner.begin_decode()
                tiny(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
                runner.end_decode()
    assert len(runner.log_probs) == 3
    loss = -torch.stack(runner.log_probs).sum()
    loss.backward()
    assert head.linear.weight.grad is not None
    assert torch.isfinite(head.linear.weight.grad).all()
    assert head.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny.parameters())


def test_mask_width_is_guarded(tiny):
    with Qwen3ReferenceRouter(tiny) as runner:
        with pytest.raises(ValueError):
            runner.begin_decode([True])


@pytest.mark.parametrize("tap", ["pre_residual", "post_residual"])
def test_tap_is_actual_native_activation(tiny, tap):
    """Read the native tensor; no extra residual or attention computation."""
    prompt = torch.tensor([[2, 4, 5]])
    cache = DynamicCache(config=tiny.config)
    with torch.no_grad():
        tiny(input_ids=prompt, past_key_values=cache, use_cache=True)
    seen = []
    block = tiny.model.layers[0]
    if tap == "pre_residual":
        handle = block.self_attn.register_forward_hook(
            lambda module, args, result: seen.append(result[0]))
    else:
        handle = block.post_attention_layernorm.register_forward_pre_hook(
            lambda module, args: seen.append(args[0]))
    try:
        with Qwen3ReferenceRouter(tiny, tap=tap) as runner, torch.no_grad():
            runner.begin_decode([True] * runner.n_middle)
            tiny(input_ids=torch.tensor([[6]]), past_key_values=cache, use_cache=True)
            runner.end_decode()
            assert len(seen) == 1
            assert runner.current_tap.data_ptr() == seen[0][:, -1, :].data_ptr()
            assert torch.equal(runner.current_tap, seen[0][:, -1, :])
    finally:
        handle.remove()


def test_project_only_matches_native_kv(tiny):
    """Independent native Attention cache vs manual projection on same input."""
    prompt = torch.tensor([[2, 4, 5]])
    a = DynamicCache(config=tiny.config)
    b = DynamicCache(config=tiny.config)
    with torch.no_grad():
        tiny(input_ids=prompt, past_key_values=a, use_cache=True)
        tiny(input_ids=prompt, past_key_values=b, use_cache=True)
        tiny(input_ids=torch.tensor([[6]]), past_key_values=a, use_cache=True)
        with Qwen3ReferenceRouter(tiny, mode="eager") as runner:
            runner.begin_decode([False] + [True] * (runner.n_middle - 1))
            tiny(input_ids=torch.tensor([[6]]), past_key_values=b, use_cache=True)
            runner.end_decode()
    # Layer index 1 receives the same native first-block output in both paths.
    # Compare its KV independently of downstream changes caused by skipping it.
    assert torch.allclose(a.layers[1].keys, b.layers[1].keys, atol=1e-6, rtol=0)
    assert torch.allclose(a.layers[1].values, b.layers[1].values, atol=1e-6, rtol=0)
