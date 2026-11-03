# Integration notes

[← Project overview](../README.md) · [Usage](usage.md)

This repository keeps upstream-derived changes in **forked submodules**, with only the A2C2-specific training and online wrapper code maintained under `a2c2/`. The following commits were recorded as gitlinks in the outer project for the published source version.

| Submodule | Recorded commit | Purpose |
|:--|:--|:--|
| [`openpi-comet`](https://github.com/ljc-1222/openpi-comet) | [`6a8329f`](https://github.com/ljc-1222/openpi-comet/tree/6a8329fcccd9d7289ffba6c62cf6093f593888b7) | π₀.₅ policy and A2C2 latent-return APIs |
| [`BEHAVIOR-1K`](https://github.com/ljc-1222/BEHAVIOR-1K) | [`37a4910`](https://github.com/ljc-1222/BEHAVIOR-1K/tree/37a4910a2545f1d91ecf7eff45a603c420ba98c6) | Evaluation wrappers and per-rollout diagnostics |

## Base-policy latent interface

The OpenPI fork extends `src/openpi/models/pi0.py` so `sample_actions(..., return_prefix_z=True)` can return both the action chunk and a mask-pooled prefix feature. `src/openpi/policies/policy.py` adds `infer_with_prefix_z` to the policy interface. This allows the source-frame latent and predicted chunk to be acquired together without independently repeating the prefix calculation.

The repository's own [`a2c2/src/online.py`](../a2c2/src/online.py) consumes those outputs, caches the current chunk and source latent, and predicts correction residuals from new observations at execution time. The websocket server starts from [`a2c2/scripts/serve_a2c2_b1k.py`](../a2c2/scripts/serve_a2c2_b1k.py).

## BEHAVIOR evaluation

The BEHAVIOR fork includes its A2C2 observation wrapper and [online metrics accumulator](https://github.com/ljc-1222/BEHAVIOR-1K/blob/37a4910a2545f1d91ecf7eff45a603c420ba98c6/OmniGibson/omnigibson/learning/a2c2_online_metrics.py). The wrapper supplies modalities and task information expected by the selected A2C2 model config; the evaluator stores residual, action clipping, gripper timing and smoothness data in per-rollout JSON.

The saved checkpoint configuration specifies the enabled depth, camera-relative pose, privileged task-info and inference-timing inputs.

## Source and attribution

The underlying [Comet project](https://github.com/mli0603/openpi-comet) was **second on the official 2025 BEHAVIOR held-out-test leaderboard** (overall Q-score **0.2514**); its competition score is distinct from this repository's Task 1-only score. The A2C2 mechanism originates from [Sendai et al.](https://arxiv.org/abs/2509.23224). The [BEHAVIOR-1K simulator](https://github.com/StanfordVL/BEHAVIOR-1K) and [OpenPI](https://github.com/Physical-Intelligence/openpi) are upstream projects.

The fork submodules preserve their respective upstream licenses. The outer repository currently has no separate root license. Dataset, model-weight, and simulator-asset terms are determined by their respective sources.
