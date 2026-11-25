# Pilot-only arithmetic prompts

These examples are **debug inputs** for checking whether a Qwen3 router can update via actual autoregressive rollouts. They are **not** a scientifically meaningful evaluation benchmark. The training script does not read the dev set. Do not use these samples to claim quality improvements, broad generalization, or success over a fixed-skip baseline.

Any publishable evaluation requires a separate held-out, versioned benchmark, independent model baselines, multiple seeds, and noninferiority criteria.

## Current pilot inputs

`train_multistep.jsonl` (8 prompts) and `dev_multistep.jsonl` (4 disjoint prompts) require multi-token answers. The first generated token is produced by the mandatory full prefill; subsequent tokens are routed. The earlier single-integer files are retained as debug fixtures only. The multistep prompts are not a publishable accuracy benchmark.
