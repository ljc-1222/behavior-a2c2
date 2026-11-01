# Task 1 results and analysis

[← Project overview](../README.md) · [Method](method.md) · [Usage](usage.md)

## Experimental setup and outcome

The June 2026 project evaluated **Task 1 — Picking Up Trash** (`picking_up_trash`) on **10 fixed-seed BEHAVIOR instances**. Four correction-head variants, each trained for **200,000 steps**, were compared with frozen OpenPI-Comet.

| Policy | Mean final Q-score ↑ | Population SD | Nonzero instances |
|:--|--:|--:|--:|
| **Frozen Comet** | **0.6667** | 0.3944 | 8/10 |
| + A2C2, raw MSE | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized MSE | **0.2333** | 0.2134 | 6/10 |
| + A2C2, normalized Huber | 0.0000 | 0.0000 | 0/10 |
| + A2C2, normalized Huber + gripper weighting | 0.1000 | 0.2134 | 2/10 |

BEHAVIOR's final Q-score permits partial credit for task-goal conditions. **Nonzero instances count progress, not full task success.** The linked summaries below contain the per-instance final scores, mean, and both population and sample SD; this table uses population SD, matching the project report. All five policies were evaluated on instance IDs **67, 106, 155, 156, 161, 162, 171, 196, 245, and 246**, with rollout suffix `0`.

All four correction variants were reported to lower validation corrected-action MSE relative to the base actions. Nevertheless, every corrected policy underperformed the frozen baseline online. The report does not provide a numerical validation-MSE table, so no numerical offline improvement is claimed here.

## Published evaluation artifacts

The [Hugging Face archive](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd) contains the original online evaluation outputs. Links below pin the archive revision to preserve the correspondence with the reported results.

| Policy | Evaluation run | Final-score summary |
|:--|:--|:--|
| Frozen Comet | [20260610_221456](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd/baseline/task0001_pi05_b1kpt50_cs32/online_eval/20260610_221456) | [JSON](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/baseline/task0001_pi05_b1kpt50_cs32/online_eval/20260610_221456/summary.json) |
| Raw MSE | [20260610_162839](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_raw_mse/online_eval/20260610_162839) | [JSON](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_raw_mse/online_eval/20260610_162839/summary.json) |
| Normalized MSE | [20260610_120432](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_mse/online_eval/20260610_120432) | [JSON](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_mse/online_eval/20260610_120432/summary.json) |
| Normalized Huber | [20260610_ql3j06](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_huber/online_eval/20260610_ql3j06) | [JSON](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_huber/online_eval/20260610_ql3j06/summary.json) |
| Normalized Huber + gripper weighting | [20260610_003834](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/tree/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_picking_up_trash_stage1_norm_huber_gripw/online_eval/20260610_003834) | [JSON](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_picking_up_trash_stage1_norm_huber_gripw/online_eval/20260610_003834/summary.json) |

Each run includes per-instance metrics and a summary. Baseline, raw MSE, normalized MSE, and gripper-weighted Huber runs also include videos; the normalized-Huber run has metrics and logs without a video directory. The archive contains evaluation outputs, rather than correction-head checkpoints.

For **instance 155**, compare [frozen Comet (Q = 1.0000)](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/baseline/task0001_pi05_b1kpt50_cs32/online_eval/20260610_221456/videos/picking_up_trash_155_0.mp4) with [normalized-MSE correction (Q = 0.3333)](https://huggingface.co/ljc-1222/a2c2-task0001-ablation/blob/669e984b98b7828c72e805be76e97d767b08b8dd/step%3D200k/task0001_norm_mse/online_eval/20260610_120432/videos/picking_up_trash_155_0.mp4). These are complete, separately timed rollouts from the same instance ID; they are not synchronized clips or an isolated causal test. The scores come from the matching JSON summaries.

## Online action diagnostics

The project also reported the following rollout diagnostics. They characterize the changed control behavior without establishing a single cause of the Q-score drop.

| Policy | Mean residual L2 | Clipped dimensions (%) | Gripper Acc@5 | Mean gripper delay (steps) | Discrete action jerk |
|:--|--:|--:|--:|--:|--:|
| Frozen Comet | 0.000 | 4.735 | 0.713 | 510.454 | **0.040** |
| + A2C2, raw MSE | 0.383 | 9.189 | 0.000 | 178.875 | 0.269 |
| + A2C2, normalized MSE | 0.335 | 2.329 | 0.703 | 482.033 | 0.160 |
| + A2C2, normalized Huber | 0.115 | 1.695 | 0.268 | 615.282 | 0.174 |
| + A2C2, normalized Huber + gripper weighting | 0.113 | 2.357 | 0.328 | 513.311 | 0.176 |

Definitions follow the pinned [BEHAVIOR metrics accumulator](https://github.com/ljc-1222/BEHAVIOR-1K/blob/37a4910a2545f1d91ecf7eff45a603c420ba98c6/OmniGibson/omnigibson/learning/a2c2_online_metrics.py):

- **Residual L2:** mean norm of the scaled residual before action clipping.
- **Clipped dimensions:** fraction of scalar action dimensions flagged by the clipping mask, rather than the fraction of control steps with any clipping.
- **Gripper Acc@5:** fraction of matched command/realization events whose delay is at most five steps. Unmatched commands are tracked separately, so this is not an all-event success rate.
- **Gripper delay:** mean delay over matched events, without restricting the average to five-step matches.
- **Discrete action jerk:** mean norm of the action vector's third finite difference. The implementation does not divide by the control time interval; this is a smoothness proxy, rather than physical jerk in SI units.

**Temporal instability is the clearest shared diagnostic signal.** Every correction variant increased jerk by approximately 4–7×. Normalized MSE cut clipping below the baseline level and nearly retained its matched gripper-event accuracy, but its task score remained lower. Thus, fewer range violations and good matched-event timing were insufficient. A smaller mean residual was also insufficient: both Huber variants had lower residual norms than normalized MSE but poorer task scores. Jerk is correlated with the degradation here; causation was not isolated.

## Interpretation and proposed experiments

**Expert-to-policy state shift.** The correction head learns from expert demonstrations, while online execution visits states affected by failed contacts, missed grasps, and accumulated pose errors. Replay of failed rollout states can reveal where corrections become unreliable. The report proposes sweeping residual scale, group-wise clipping, and action-range projection to test whether conservative correction improves Q-score while reducing correction magnitude, clipping, delay, and jerk. This is a diagnostic and mitigation experiment, not a demonstrated solution to distribution shift.

**Task-specialization mismatch.** A Task 1 specialist may perturb useful behavior encoded by the multi-task Comet policy. The proposed comparison is a broad mixed-task correction head versus task specialists and task-conditioned residual MLPs, to distinguish over-specialization from negative transfer. No such comparison is reported in the current results.

**Temporal and multimodal inconsistency.** The flow policy can sample different plausible, coherent chunks. A deterministic pointwise residual target may pull an action toward a demonstrated mode that differs from the sampled base chunk, disrupting temporal consistency. The elevated jerk supports investigating this possibility. Proposed tests add chunk-level smoothness/jerk regularization and select among multiple sampled base chunks.

Action heterogeneity motivated the four loss ablations: raw error can underweight small-range grippers, and continuous regression does not explicitly model open/close event timing. Normalization, robust regression, and gripper weighting were implemented and tested; chunk-level regularization, mixed-task comparisons, and multi-sample selection remain proposals. The evidence supports an offline–online mismatch in this setting, without establishing that A2C2 fails on other tasks or that any proposed change would recover performance.

The numerical results and discussion originate in the [June 2026 project source](https://github.com/DennisRapheal/2026-DLP-Final-Project-Report/blob/bf1978bb87dc140d5e63809c281bf9b1b4b44b14/main.tex). Final Q-score aggregates also match the published summaries linked above. No new training or simulator evaluation is included in this documentation revision.
