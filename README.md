# Residual Action Chunk Correction for OpenPI-Comet

**Multimodal feedback for long-horizon robot control on BEHAVIOR-1K**

Can fresh observations improve an action chunk that a vision-language-action policy has already generated? This project adapts [Asynchronous Action Chunk Correction (A2C2)](https://arxiv.org/abs/2509.23224) to a frozen [OpenPI-Comet](https://github.com/mli0603/openpi-comet) policy. We built the aligned training-data pipeline, a multimodal residual model, and the BEHAVIOR-1K rollout integration, then compared four correction objectives on **Task 1 — Picking Up Trash**.

The experiments expose an **offline–online gap**: closer imitation of demonstrated actions did not improve task completion. Action diagnostics and the discussion below identify possible mechanisms and concrete follow-up experiments.

[Method](docs/method.md) · [Results & rollout videos](docs/results.md) · [Setup & usage](docs/usage.md) · [Dataset](https://huggingface.co/datasets/ljc-1222/a2c2_dataset) · [Evaluation artifacts](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd)

## Approach and implementation

Comet generates a **32-step chunk of 23-dimensional robot actions**. While that chunk is executed, a separate model uses the latest observation to predict a residual for the current action. The base VLA stays frozen.

![Frozen Comet generates an action chunk and prefix latent; a trainable head uses fresh observations to correct each selected action before clipping and execution.](assets/a2c2_pipeline.svg)

Our implementation includes:

- **Aligned correction data:** extend expert episodes with frozen-policy chunks, valid-offset masks, and mask-pooled prefix latents; pair the action at offset $k$ with the observation and expert action at execution time $t+k$.
- **Multimodal correction head:** fuse visual features, robot state, task language, policy context, and chunk position using a **6-layer, 8-head Transformer** and a residual MLP. The model includes frozen ResNet-18 visual encoders and a language GRU.
- **Online integration:** cache the source chunk and latent, infer a correction at each control step, scale the residual, and clip the composed action to robot limits through the patched OpenPI and BEHAVIOR interfaces.
- **Controlled ablations and diagnostics:** compare four residual losses and record correction magnitude, clipping, gripper-event timing, and action smoothness alongside task scores.

For a chunk generated at time $t$ and executed at offset $k$:

```math
a_{t+k}^{\mathrm{exec}} = \mathrm{clip}\left(a_{t,k}^{\mathrm{base}} + \lambda\,\Delta a_{t,k}\right).
```

Here $\lambda$ scales the predicted correction. [Method](docs/method.md) explains the aligned targets, token inputs, loss definitions, and optional modalities. The online implementation refreshes chunks synchronously; this study evaluates per-step correction without a real-time scheduling benchmark.

## Task 1 results

The four correction models were trained for **200,000 steps** and evaluated against frozen Comet on **10 fixed-seed instances**. Q-score measures partial task-goal completion; a nonzero score indicates progress, rather than full success. SD is the population standard deviation across instances.

| Policy | Mean final Q-score ↑ | SD | Nonzero instances |
|:--|--:|--:|--:|
| **Frozen OpenPI-Comet** | **0.6667** | 0.3944 | 8/10 |
| + A2C2, raw MSE | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized MSE | **0.2333** | 0.2134 | 6/10 |
| + A2C2, normalized Huber | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized Huber + gripper weighting | 0.1000 | 0.2134 | 2/10 |

All four models reduced corrected-action validation MSE in the project experiments, yet none matched the frozen policy online. **Range normalization helped, but was insufficient:** normalized MSE was the strongest correction variant and reduced clipped action dimensions from **4.735% to 2.329%**, while its discrete action jerk rose from **0.040 to 0.160**. Across all variants, jerk increased by approximately **4–7×**. Smaller prediction errors and fewer range violations therefore did not ensure coherent closed-loop behavior.

[Results & analysis](docs/results.md) contains the full diagnostics and links to the original per-instance metrics and summaries. For a qualitative comparison, watch **instance 155**: [frozen Comet](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/baseline/task0001_pi05_b1kpt50_cs32/online_eval/20260610_221456/videos/picking_up_trash_155_0.mp4) (**Q = 1.0000**) and [normalized-MSE correction](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_mse/online_eval/20260610_120432/videos/picking_up_trash_155_0.mp4) (**Q = 0.3333**). This illustrates one instance; the table summarizes all ten.

## Why could correction reduce performance?

The report discusses three explanations for the offline–online gap. These are **hypotheses**, rather than experimentally isolated causes; the proposed follow-ups have not been evaluated.

| Possible mechanism | Why it matters | Follow-up proposed in the report |
|:--|:--|:--|
| **Expert-to-policy state shift** | Training uses expert trajectories; rollouts encounter missed grasps, contact errors, and shifted objects, where the learned residual may be unreliable. | Replay failed rollout states and sweep residual scaling, group-wise clipping, and action-range projection to test conservative correction. |
| **Task-specialization mismatch** | A Task 1 correction head may pull a broadly trained multi-task policy toward a narrower action distribution and disturb useful base behavior. | Compare a mixed-task head, target-task specialists, and task-conditioned residual heads. |
| **Loss of temporal and multimodal consistency** | Pointwise deterministic regression can pull a coherent sampled chunk toward a different demonstrated action mode. The higher jerk is consistent with this concern. | Add chunk-level smoothness/jerk regularization and test selection among multiple sampled base chunks. |

Action-scale imbalance and event-like gripper control motivated the normalization, Huber, and weighting ablations already tested above. Their limited recovery suggests that changing a pointwise loss alone did not resolve the control problem. The resulting research direction is to **preserve the base policy's coherent behavior while making corrections conservative and sensitive to rollout state**, then validate changes using task scores as well as action-level error.

## Getting started

```bash
git clone --recurse-submodules https://github.com/ljc-1222/behavior-a2c2.git
cd behavior-a2c2
bash setup.sh --skip-submodule-update --no-behavior-sparse-checkout
```

The full workflow requires Linux, a compatible GPU environment, and BEHAVIOR simulator assets. [Setup & usage](docs/usage.md) covers Task 1 data download, RGB feature-cache generation, training, offline evaluation, and online rollout. Commands are usage examples; the results above are the original June 2026 experiments.

| Guide | Contents |
|:--|:--|
| [Method](docs/method.md) | Data alignment, correction-head architecture, and loss equations |
| [Results & analysis](docs/results.md) | Q-scores, diagnostic definitions, hypotheses, and rollout artifacts |
| [Setup & usage](docs/usage.md) | Dependencies and commands for the complete workflow |
| [Integration notes](docs/integration.md) | Pinned forks, latent-return API, and evaluation hooks |

Core code lives in [`a2c2/src/`](a2c2/src); dataset, training, evaluation, and server entrypoints are in [`a2c2/scripts/`](a2c2/scripts).

## Team and acknowledgments

**Jin-Cheng Lin, Ting-Wei Lin, I-Ting Chu** · Deep Learning Final Project · June 2026

The correction mechanism follows [Sendai et al., *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*](https://arxiv.org/abs/2509.23224). The base policy comes from [Team Comet's OpenPI-Comet](https://github.com/mli0603/openpi-comet), the second-place solution to the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html). Our results cover the local Task 1 study. Upstream sources and licensing are recorded in [Integration notes](docs/integration.md).
