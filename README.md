# Residual Action Chunk Correction for OpenPI-Comet

**Multimodal residual correction on BEHAVIOR-1K Task 1**

This project adapts [Asynchronous Action Chunk Correction (A2C2)](https://arxiv.org/abs/2509.23224) to a frozen [OpenPI-Comet](https://github.com/mli0603/openpi-comet) vision-language-action (VLA) policy. We built an aligned training-data pipeline, a multimodal correction head, and per-step integration with the BEHAVIOR-1K evaluator.

On **Task 1 — Picking Up Trash**, all four loss variants reduced validation mean squared error (MSE) against expert actions. Every corrected policy nevertheless scored below frozen Comet in closed-loop evaluation. Diagnostics of clipping, gripper timing, and action smoothness characterize this **offline–online gap**.

[Method](docs/method.md) · [Results and rollout videos](docs/results.md) · [Setup and usage](docs/usage.md) · [Dataset](https://huggingface.co/datasets/ljc-1222/a2c2_dataset) · [Evaluation artifacts](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd)

## Approach and implementation

Comet generates a **32-step chunk of 23-dimensional robot actions**. While that chunk is executed, a separate model uses the latest observation to predict a residual for the current action. The base VLA stays frozen.

![Frozen Comet generates an action chunk and prefix latent; a trainable head uses the latest observations to correct each selected action before clipping and execution.](assets/a2c2_pipeline.svg)

We implemented four components:

- **Aligned correction data:** Built a pipeline that stores frozen-policy chunks, valid-offset masks, and mask-pooled prefix latents. It pairs each action at offset $k$ with the observation and expert action at execution time $t+k$.
- **Multimodal correction head:** Implemented a **6-layer, 8-head Transformer** that fuses visual features, robot state, task language, policy context, and chunk position. The head uses frozen ResNet-18 visual encoders, a language GRU, and a residual MLP.
- **Online integration:** Added a wrapper that caches the source chunk and latent, predicts a residual at each control step, and applies residual scaling and action-range clipping.
- **Loss ablations and diagnostics:** Compared four losses to test action-scale normalization, robust regression, and extra gripper weighting. Recorded correction magnitude, clipping, gripper-event timing, and action smoothness during rollouts.

For a chunk generated at time $t$ and executed at offset $k$:

```math
a_{t+k}^{\mathrm{exec}} = \mathrm{clip}\left(a_{t,k}^{\mathrm{base}} + \lambda\,\Delta a_{t,k}\right).
```

Here $\lambda$ scales the residual. The wrapper refreshes chunks synchronously. [Method](docs/method.md) details the aligned targets, token inputs, loss definitions, and optional modalities.

## Task 1 results

The four correction models were trained for **200,000 steps** and evaluated against frozen Comet on **10 fixed-seed instances**. Q-score gives partial credit for task-goal completion. Nonzero instances count runs with some progress; SD is the population standard deviation across instances.

| Policy | Mean final Q-score ↑ | SD | Nonzero instances |
|:--|--:|--:|--:|
| **Frozen OpenPI-Comet** | **0.6667** | 0.3944 | 8/10 |
| + A2C2, raw MSE | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized MSE | **0.2333** | 0.2134 | 6/10 |
| + A2C2, normalized Huber | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized Huber + gripper weighting | 0.1000 | 0.2134 | 2/10 |

**Normalized MSE reduced clipping while increasing action jerk.** The fraction of clipped action dimensions fell from **4.735% to 2.329%**, while discrete action jerk rose from **0.040 to 0.160**. All four correction variants increased jerk by approximately **4–7×**. These diagnostics show that fewer action-range violations did not ensure better task performance.

**Rollout comparison (instance 155):** [Frozen Comet, Q = 1.0000](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/baseline/task0001_pi05_b1kpt50_cs32/online_eval/20260610_221456/videos/picking_up_trash_155_0.mp4) · [Normalized-MSE correction, Q = 0.3333](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_mse/online_eval/20260610_120432/videos/picking_up_trash_155_0.mp4). Full diagnostics and per-instance metrics are in [Results and analysis](docs/results.md).

## Analysis and proposed experiments

The residual losses do not explicitly supervise gripper events or enforce chunk-level smoothness. Three candidate explanations motivate the experiments below. Their causal contributions have not been isolated, and the experiments remain proposals.

| Candidate mechanism | Interpretation | Proposed experiment |
|:--|:--|:--|
| **Expert-to-policy state shift** | Training uses expert trajectories; rollouts encounter missed grasps, contact errors, and shifted objects, where the learned residual may be unreliable. | Replay failed rollout states and sweep residual scaling, group-wise clipping, and action-range projection to test conservative correction. |
| **Task-specialization mismatch** | A Task 1 correction head may pull a broadly trained multi-task policy toward a narrower action distribution and disturb useful base behavior. | Compare a mixed-task head, target-task specialists, and task-conditioned residual heads. |
| **Temporal and multimodal inconsistency** | Pointwise deterministic regression can pull a coherent sampled chunk toward a different demonstrated action mode. The higher jerk is consistent with this concern. | Add chunk-level smoothness/jerk regularization and test selection among multiple sampled base chunks. |

## Get started

```bash
git clone --recurse-submodules https://github.com/ljc-1222/behavior-a2c2.git
cd behavior-a2c2
bash setup.sh --skip-submodule-update --no-behavior-sparse-checkout
```

The full workflow requires Linux, a compatible GPU environment, and BEHAVIOR simulator assets. The guides below cover data preparation, training, evaluation, and the interfaces used by this implementation.

| Guide | Contents |
|:--|:--|
| [Method](docs/method.md) | Data alignment, correction-head architecture, and loss equations |
| [Results and analysis](docs/results.md) | Q-scores, diagnostic definitions, hypotheses, and rollout artifacts |
| [Setup and usage](docs/usage.md) | Dependencies and commands for the complete workflow |
| [Integration notes](docs/integration.md) | Pinned forks, latent-return API, and evaluation hooks |

Core code lives in [`a2c2/src/`](a2c2/src); dataset, training, evaluation, and server entrypoints are in [`a2c2/scripts/`](a2c2/scripts).

## Team and acknowledgments

**Jin-Cheng Lin, Ting-Wei Lin, I-Ting Chu** · Deep Learning Final Project · June 2026

The correction mechanism follows [Sendai et al., *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*](https://arxiv.org/abs/2509.23224). The base policy comes from [Team Comet's OpenPI-Comet](https://github.com/mli0603/openpi-comet), the second-place solution to the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html). Upstream sources and licensing are recorded in [Integration notes](docs/integration.md).

