# Residual Action Chunk Correction for OpenPI-Comet

**Multimodal residual correction on BEHAVIOR-1K Task 1**

This project adapts [Asynchronous Action Chunk Correction (A2C2)](https://arxiv.org/abs/2509.23224) to a frozen [OpenPI-Comet](https://github.com/mli0603/openpi-comet) vision-language-action (VLA) policy. We built an aligned training-data pipeline, a multimodal correction head, and per-step integration with the BEHAVIOR-1K evaluator.

On **Task 1 — Picking Up Trash**, all four loss variants reduced validation mean squared error (MSE) against expert actions. Every corrected policy nevertheless scored below frozen Comet in closed-loop evaluation. This is the central **offline–online gap** examined in the study.

[Method](docs/method.md) · [Results](#task-1-results) · [Analysis](#analysis-and-next-steps) · [Setup and usage](docs/usage.md)

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

The four correction models were trained for **200,000 steps** and evaluated against frozen Comet on **10 fixed-seed instances**. All four reduced corrected-action validation MSE relative to the base actions. The table shows the closed-loop evaluation results.

Q-score gives partial credit for task-goal completion. Nonzero instances count runs with some progress; SD is the population standard deviation across instances.

| Policy | Mean final Q-score ↑ | SD | Nonzero instances |
|:--|--:|--:|--:|
| **Frozen OpenPI-Comet** | **0.6667** | 0.3944 | 8/10 |
| + A2C2, raw MSE | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized MSE | **0.2333** | 0.2134 | 6/10 |
| + A2C2, normalized Huber | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized Huber + gripper weighting | 0.1000 | 0.2134 | 2/10 |

## Analysis and next steps

The following explanations connect the observed control behavior and training setup to proposed follow-up experiments. They remain hypotheses; the experiments have not been run.

**Chunk consistency.** All four correction variants increased discrete action jerk by approximately **4–7×**. The normalized-MSE policy clipped fewer action dimensions yet still scored below frozen Comet. Pointwise residual regression may pull a sampled chunk toward a different demonstrated action mode, disrupting its temporal coherence. Proposed tests add chunk-level smoothness regularization and evaluate selection among multiple sampled base chunks.

**Expert-to-policy state shift.** The correction head learns from expert trajectories, while online execution encounters missed grasps, contact errors, and shifted object poses. Corrections may be unreliable on these policy-induced states. Replaying failed rollout states would help examine that mismatch. Sweeps of residual scale, group-wise clipping, and action-range projection would test whether more conservative corrections improve task scores.

**Task specialization.** Comet is trained across tasks, whereas the correction head fits Task 1 residuals. This narrower supervision may disturb useful behavior from the base policy. Comparing a broad mixed-task head, target-task specialists, and task-conditioned residual heads would help distinguish over-specialization from negative transfer.

The [evaluation diagnostics](docs/results.md) provide the complete control-metric table, metric definitions, and limits of these interpretations.

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
| [Evaluation diagnostics](docs/results.md) | Control metrics, definitions, and evidence limits |
| [Setup and usage](docs/usage.md) | Dependencies and commands for the complete workflow |
| [Integration notes](docs/integration.md) | Pinned forks, latent-return API, and evaluation hooks |

Core code lives in [`a2c2/src/`](a2c2/src); dataset, training, evaluation, and server entrypoints are in [`a2c2/scripts/`](a2c2/scripts).

## Team and acknowledgments

**Jin-Cheng Lin, Ting-Wei Lin, I-Ting Chu** · Deep Learning Final Project · June 2026

The correction mechanism follows [Sendai et al., *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*](https://arxiv.org/abs/2509.23224). The base policy comes from [Team Comet's OpenPI-Comet](https://github.com/mli0603/openpi-comet), the second-place solution to the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html). Upstream sources and licensing are recorded in [Integration notes](docs/integration.md).
