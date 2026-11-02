# Task 1 evaluation diagnostics

[← Project overview](../README.md) · [Task 1 scores](../README.md#task-1-results) · [Analysis and next steps](../README.md#analysis-and-next-steps) · [Method](method.md)

## Evaluation scope

The Task 1 study compared frozen OpenPI-Comet with four correction heads trained for **200,000 steps**, using **10 fixed-seed BEHAVIOR instances**. The complete Q-score, population SD, and nonzero-instance results are in the [main results table](../README.md#task-1-results). Q-score permits partial task-goal completion; nonzero instances indicate progress.

The control diagnostics below describe how the executed actions changed. They support the analysis without identifying a single cause of the Q-score decline.

## Control measurements

| Policy | Residual L₂ | Clipped dimensions (%) | Gripper Acc@5 | Mean gripper delay (steps) | Discrete action jerk |
|:--|--:|--:|--:|--:|--:|
| Frozen Comet | 0.000 | 4.735 | 0.713 | 510.454 | **0.040** |
| + A2C2, raw MSE | 0.383 | 9.189 | 0.000 | 178.875 | 0.269 |
| + A2C2, normalized MSE | 0.335 | 2.329 | 0.703 | 482.033 | 0.160 |
| + A2C2, normalized Huber | 0.115 | 1.695 | 0.268 | 615.282 | 0.174 |
| + A2C2, normalized Huber + gripper weighting | 0.113 | 2.357 | 0.328 | 513.311 | 0.176 |

Definitions follow the pinned [BEHAVIOR metrics accumulator](https://github.com/ljc-1222/BEHAVIOR-1K/blob/37a4910a2545f1d91ecf7eff45a603c420ba98c6/OmniGibson/omnigibson/learning/a2c2_online_metrics.py):

- **Residual L₂:** mean norm of the scaled residual before action clipping. This is not the difference between the clipped executed action and the base action.
- **Clipped dimensions:** fraction of scalar action dimensions flagged by the clipping mask, rather than the fraction of control steps with any clipping.
- **Gripper Acc@5:** fraction of matched command/realization events whose delay is at most five steps. Unmatched commands are tracked separately and excluded from this fraction.
- **Gripper delay:** mean delay over matched events, without restricting the average to five-step matches.
- **Discrete action jerk:** mean norm of the executed action vector's third finite difference. The implementation does not divide by the control time interval, so this is a smoothness proxy rather than physical jerk in SI units.

## Diagnostic comparisons

**Fewer range violations did not restore task performance.** The normalized-MSE policy reduced the clipping fraction from **4.735% to 2.329%** and retained near-baseline matched-event Acc@5 (**0.703 versus 0.713**). Its mean Q-score was nevertheless **0.2333**, compared with **0.6667** for frozen Comet. Matched-event accuracy should not be interpreted as generally good timing: mean gripper delay remained **482.033 steps**, and unmatched events are excluded from Acc@5.

**Small residuals were also insufficient.** The two Huber variants had lower residual norms (**0.115 and 0.113**) than normalized MSE (**0.335**), yet lower task scores. Residual magnitude alone is therefore a poor substitute for evaluating the complete policy.

**Temporal instability was shared across variants.** Every correction variant increased discrete action jerk by approximately **4–7×**. This motivates testing chunk consistency, but does not prove that jerk caused failure or that the base policy's action modes collapsed.

## Evidence limits

The normalized losses change both per-dimension scaling and group aggregation. Their performance cannot be attributed to range normalization alone. Gripper weighting increases the regression emphasis on gripper dimensions without explicitly supervising open/close events or their timing.

The project reported lower corrected-action validation MSE for all four variants, but did not provide a numerical offline-MSE table. No precise improvement percentage is claimed. The evidence concerns this Task 1 setting and ten evaluation instances; it does not establish how A2C2 would perform across other tasks.

Expert-state training, task specialization, and chunk consistency remain candidate explanations. A mixed-task-versus-specialist comparison and direct action-mode analysis were not performed. Residual-scaling sweeps can test conservative correction, but cannot by themselves isolate the causal effect of state-distribution shift. The [README analysis](../README.md#analysis-and-next-steps) pairs these hypotheses with the proposed follow-up directions.
