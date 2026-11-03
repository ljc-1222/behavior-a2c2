# Task 1 evaluation diagnostics

[← Project overview](../README.md) · [Task 1 scores](../README.md#task-1-results) · [Analysis and next steps](../README.md#analysis-and-next-steps) · [Method](method.md)

## Evaluation scope

The Task 1 study compared frozen OpenPI-Comet with four correction heads trained for **200,000 steps**, using **10 fixed-seed BEHAVIOR instances**. The complete Q-score, population SD, and nonzero-instance results are in the [main results table](../README.md#task-1-results). Q-score permits partial task-goal completion; nonzero instances indicate progress.

The diagnostics quantify correction magnitude, clipping, gripper-event timing and temporal smoothness.

## Control measurements

| Policy | Residual L₂ | Clipped dimensions (%) | Gripper Acc@5 | Mean gripper delay (steps) | Discrete action jerk |
|:--|--:|--:|--:|--:|--:|
| Frozen Comet | 0.000 | 4.735 | 0.713 | 510.454 | **0.040** |
| + A2C2, raw MSE | 0.383 | 9.189 | 0.000 | 178.875 | 0.269 |
| + A2C2, normalized MSE | 0.335 | 2.329 | 0.703 | 482.033 | 0.160 |
| + A2C2, normalized Huber | 0.115 | 1.695 | 0.268 | 615.282 | 0.174 |
| + A2C2, normalized Huber + gripper weighting | 0.113 | 2.357 | 0.328 | 513.311 | 0.176 |

Definitions follow the pinned [BEHAVIOR metrics accumulator](https://github.com/ljc-1222/BEHAVIOR-1K/blob/37a4910a2545f1d91ecf7eff45a603c420ba98c6/OmniGibson/omnigibson/learning/a2c2_online_metrics.py):

- **Residual L₂:** mean norm of the scaled residual before action clipping.
- **Clipped dimensions:** fraction of scalar action dimensions flagged by the clipping mask.
- **Gripper Acc@5:** fraction of matched command/realization events whose delay is at most five steps.
- **Gripper delay:** mean delay over all matched events.
- **Discrete action jerk:** mean norm of the executed action vector's third finite difference, without control-interval normalization. It serves as a discrete smoothness proxy.

## Diagnostic comparisons

**Fewer range violations did not restore task performance.** The normalized-MSE policy reduced the clipping fraction from **4.735% to 2.329%** and retained near-baseline matched-event Acc@5 (**0.703 versus 0.713**). Its mean Q-score was nevertheless **0.2333**, compared with **0.6667** for frozen Comet. The corresponding mean gripper delay was **482.033 steps**.

**Small residuals were also insufficient.** The two Huber variants had lower residual norms (**0.115 and 0.113**) than normalized MSE (**0.335**), yet lower task scores.

**Temporal instability was shared across variants.** Every correction variant increased discrete action jerk by approximately **4–7×**. This association motivates the proposed chunk-consistency experiments.

## Interpreting the comparisons

The normalized losses combine per-dimension scaling with group aggregation, so the comparison measures their joint effect. Gripper weighting increases the regression emphasis on gripper dimensions; event timing remains outside the loss objective.

The project reported lower corrected-action validation MSE for all four variants; per-variant numerical values were not reported.

Proposed residual-scaling sweeps would test sensitivity to correction strength, while replaying failed rollout states would examine correction behavior on policy-induced states. The [README analysis](../README.md#analysis-and-next-steps) pairs the candidate explanations with proposed follow-up experiments.
