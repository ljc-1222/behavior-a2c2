# Beyond MSE: Revising A2C2 Residual Correction for Flow-Model Action Chunks

## 1. Motivation

The current A2C2 setup can be viewed as a lightweight residual correction module placed on top of a frozen OpenPI-COMET / π0.5-style base policy. The base policy first produces an action chunk:

```text
A_base = [a_base_0, a_base_1, ..., a_base_31] ∈ R^{32 × 23}
```

At each control step, A2C2 selects one base action from the chunk:

```text
a_base = A_base[k] ∈ R^{23}
```

Then the correction head predicts a residual action:

```text
Δa_pred ∈ R^{23}
```

The executed action is:

```text
a_exec = a_base + Δa_pred
```

The naive supervised target is:

```text
Δa_target = a_expert - a_base
```

and the simplest training loss is raw residual MSE:

```text
L = ||Δa_pred - Δa_target||²
```

This is a reasonable first baseline, but it is too weak for the current problem setting because the 23-D R1Pro / BEHAVIOR action space is heterogeneous, bounded, temporally structured, and partially event-like. The revision below keeps the A2C2 idea intact, but replaces naive raw MSE with a more robot-control-aware objective.

---

## 2. Why Raw Residual MSE Is Not Enough

### 2.1 The 23-D action vector is heterogeneous

The action vector is not a homogeneous Euclidean control vector. It is a concatenation of different controller commands:

```text
[0:3]    base command
[3:7]    torso joint positions
[7:14]   left arm 7-DoF joint positions
[14]     left gripper width
[15:22]  right arm 7-DoF joint positions
[22]     right gripper width
```

So the action groups are:

```text
base:          3-D
torso:         4-D
left_arm:      7-D
left_gripper:  1-D
right_arm:     7-D
right_gripper: 1-D
```

These dimensions have different units, scales, physical effects, and task relevance. Therefore, treating all 23 dimensions equally under raw MSE implicitly assumes that all dimensions have the same numerical and behavioral meaning, which is not true.

---

### 2.2 Scale imbalance

Raw MSE is dominated by large-range dimensions. For example, if the gripper range is approximately:

```text
[0.00, 0.05]
```

then even a completely wrong gripper command contributes at most:

```text
0.05² = 0.0025
```

But an arm joint error of only `0.2 rad` contributes:

```text
0.2² = 0.04
```

which is already 16 times larger than the maximum gripper squared error. As a result, raw MSE may optimize arm and torso residuals while underweighting gripper timing, even though gripper timing can determine whether the task succeeds or fails.

---

### 2.3 Gripper control is event-like

Although gripper width is represented as a continuous scalar, its task-level meaning is often closer to a discrete event:

```text
open
close
hold
release
```

A pure MSE regression objective tends to learn conditional averages. If some demonstrations close the gripper at a certain moment while others do not, the MSE optimum may become a half-open value such as:

```text
gripper = 0.025
```

This value may be numerically reasonable but behaviorally invalid. For grasping tasks, a slightly mistimed close/open event can cause complete task failure.

---

### 2.4 Residual addition can produce invalid actions

A2C2 executes:

```text
a_corr = a_base + Δa_pred
```

But each action dimension has valid physical or controller bounds. Without range awareness, A2C2 can output:

```text
left_gripper < 0
right_gripper > max_width
joint position beyond joint limit
base command beyond controller limit
```

Even if the simulator or controller clamps the action at execution time, the training loss is still computed in unconstrained residual space. This creates a mismatch:

```text
training:    unconstrained residual regression
execution:   bounded controller action
```

---

### 2.5 Joint/action-space error is not task-space error

Manipulation success is usually determined by:

```text
end-effector position
end-effector orientation
object contact
gripper timing
object displacement
task completion predicates
```

not by raw joint-space MSE alone. The same numerical joint error may produce very different end-effector errors depending on which joint is affected and the current robot configuration.

---

### 2.6 The base policy is generative

The base policy is a flow-model / VLA-style policy that generates action chunks from a distribution. In the same observation, different samples may produce different plausible action chunks. A deterministic residual MSE target:

```text
Δa_target = a_expert - a_base
```

forces the correction head to map each sampled base action toward a single expert action. This may reduce the multimodal behavior of the base policy if the correction is too strong or too globally defined.

Therefore, the goal should not be to replace the base policy with a deterministic expert regressor. The goal should be:

```text
preserve the base policy's competence
while locally correcting action chunks using the latest observation
```

---

## 3. Revised Pipeline Overview

The revised pipeline keeps the base flow policy frozen and trains A2C2 as a small per-step residual correction head.

```text
observation_t, language, robot_state_t
        ↓
frozen base flow policy
        ↓
base action chunk A_base ∈ R^{32 × 23}
        ↓
select chunk action a_base_k
        ↓
A2C2 correction head
        ↓
Δa_pred
        ↓
a_corr = a_base_k + Δa_pred
        ↓
range projection / gripper event decision
        ↓
a_exec
```

The main revision is the training objective:

```text
raw residual MSE
        ↓
range-normalized, group-aware, robust residual objective
+ gripper event supervision
+ action-limit regularization
+ temporal smoothness regularization
```

---

## 4. Offline Correction Dataset Construction

For each demonstration trajectory, first run the frozen base policy to generate action chunks.

```text
A_base_t = π_base(obs_t, language)
```

For each chunk index `k`, select:

```text
a_base_{t,k} = A_base_t[k]
```

Then align the selected base action with the expert action at the corresponding future step:

```text
a_expert_{t+k}
```

The residual target is:

```text
Δa_target_{t,k} = a_expert_{t+k} - a_base_{t,k}
```

Each A2C2 training sample contains:

```text
obs_{t+k}
state_{t+k}
language instruction
a_base_{t,k}
chunk index embedding τ_k
base policy latent z_t
optional full base action chunk A_base_t
Δa_target_{t,k}
```

The chunk index embedding can follow the original A2C2 idea:

```text
τ_k = [sin(2πk/H), cos(2πk/H)]
```

where `H = 32` is the action chunk length.

---

## 5. Action Statistics Preprocessing

Before training the correction head, compute dimension-wise statistics from the training data.

For each action dimension `i`:

```text
q01_i = 1% quantile of action_i
q99_i = 99% quantile of action_i
scale_i = q99_i - q01_i
```

This is preferred over raw min/max because it is less sensitive to outliers.

Then normalize residual errors by dimension:

```text
e_i = (Δa_pred_i - Δa_target_i) / scale_i
```

This prevents large-range dimensions from dominating the loss and gives small-range but task-critical dimensions, such as grippers, a meaningful training signal.

---

## 6. Group-Aware Residual Loss

Split the residual into semantic action groups:

```text
G = {
  base,
  torso,
  left_arm,
  right_arm,
  left_gripper,
  right_gripper
}
```

For each group:

```text
L_g = mean_{i ∈ g} Huber(e_i)
```

where:

```text
e_i = (Δa_pred_i - Δa_target_i) / scale_i
```

Then combine group losses:

```text
L_res = Σ_g w_g L_g
```

A simple initial setting is:

```text
w_base          = 1.0
w_torso         = 1.0
w_left_arm      = 1.0
w_right_arm     = 1.0
w_left_gripper  = 2.0
w_right_gripper = 2.0
```

The gripper weight can be increased because gripper errors are numerically small but task-critical.

A more conservative version is:

```text
w_g = 1.0 for all groups
```

and rely only on range normalization. This is a cleaner first ablation.

---

## 7. Robust Residual Regression with Huber Loss

Instead of MSE, use Huber / Smooth L1 on normalized residual errors.

```text
Huber(e) = 0.5 e²                  if |e| ≤ δ
         = δ(|e| - 0.5δ)           otherwise
```

The reason is that A2C2 residual targets can contain outliers:

```text
large residual = expert action - bad base-policy sample
```

If raw MSE is used, a small number of very bad base samples can dominate training. Huber loss keeps local regression behavior near zero but becomes less aggressive for large residuals.

Recommended first setting:

```text
δ = 1.0
```

because the error is already normalized by `scale_i`.

---

## 8. Gripper Event Supervision

For each gripper, define a binary open/close target.

If smaller width means closed:

```text
close_target = 1 if expert_gripper_width < threshold
close_target = 0 otherwise
```

A simple threshold is the midpoint:

```text
threshold = (gripper_low + gripper_high) / 2
```

If the data distribution is bimodal, a better threshold can be found from the empirical gripper-width histogram.

Add two classification heads to A2C2:

```text
left_close_logit
right_close_logit
```

The gripper event loss is:

```text
L_gripper_event =
    BCE(left_close_logit, left_close_target)
  + BCE(right_close_logit, right_close_target)
```

The final gripper command can be handled in two ways.

### Option A: Classification overrides gripper width

```text
if close_prob > 0.5:
    gripper = closed_value
else:
    gripper = open_value
```

This is simple and event-aligned, but may be too discrete.

### Option B: Blend classification and continuous width

```text
gripper_cont = a_base_gripper + Δa_pred_gripper

if close_prob > 0.5:
    gripper = blend(gripper_cont, closed_value)
else:
    gripper = blend(gripper_cont, open_value)
```

This preserves continuous control while improving event timing.

For the first implementation, Option A is easier to debug.

---

## 9. Action-Limit Regularization

After residual prediction:

```text
a_corr = a_base + Δa_pred
```

Add a soft limit penalty during training:

```text
L_limit =
    ||ReLU(a_corr - high)||²
  + ||ReLU(low - a_corr)||²
```

This encourages the correction head to stay within valid action bounds.

At inference time, apply hard projection:

```text
a_exec = clamp(a_corr, low, high)
```

Recommended usage:

```text
training:    soft action-limit penalty
inference:   hard clamp
```

This avoids relying only on controller-side clipping, which can hide training-time mistakes.

---

## 10. Temporal Smoothness Regularization

Because A2C2 corrects actions inside a chunk, the residual should not introduce high-frequency jitter.

The simplest smoothness loss is:

```text
L_smooth = ||Δa_pred_{t,k} - Δa_pred_{t,k-1}||²
```

This penalizes abrupt changes in the residual itself.

A more expert-aligned version is:

```text
L_smooth =
|| (a_corr_{t,k} - a_corr_{t,k-1})
 - (a_expert_{t+k} - a_expert_{t+k-1}) ||²
```

The first version is simpler and more stable. The second version more directly imitates expert action dynamics.

Recommended first implementation:

```text
L_smooth = ||Δa_pred_{t,k} - Δa_pred_{t,k-1}||²
```

computed only when consecutive chunk positions are available in the same training batch.

---

## 11. Full Revised Objective

The proposed objective is:

```text
L_total =
    L_res
  + λ_grip L_gripper_event
  + λ_limit L_limit
  + λ_smooth L_smooth
```

where:

```text
L_res = Σ_g w_g mean_{i ∈ g} Huber(
    (Δa_pred_i - Δa_target_i) / scale_i
)
```

Recommended first hyperparameters:

```text
λ_grip   = 1.0
λ_limit  = 0.1
λ_smooth = 0.05
```

If training becomes too conservative, reduce:

```text
λ_limit
λ_smooth
```

If gripper timing remains poor, increase:

```text
λ_grip
w_left_gripper
w_right_gripper
```

---

## 12. Revised Training Algorithm

```text
Input:
  demonstration dataset D
  frozen base policy π_base
  correction head π_A2C2
  action bounds low, high
  dimension-wise scales scale_i

For each training epoch:
  For each trajectory in D:
    1. Run frozen base policy:
         A_base_t = π_base(obs_t, language)

    2. For sampled chunk index k:
         a_base = A_base_t[k]
         a_expert = expert_action_{t+k}
         Δa_target = a_expert - a_base

    3. Build A2C2 input:
         x = {
           obs_{t+k},
           state_{t+k},
           language,
           a_base,
           τ_k,
           z_t,
           optional A_base_t
         }

    4. Predict:
         Δa_pred, close_logits = π_A2C2(x)

    5. Correct:
         a_corr = a_base + Δa_pred

    6. Compute losses:
         L_res      = range-normalized group Huber residual loss
         L_grip     = gripper open/close BCE
         L_limit    = action bound violation penalty
         L_smooth   = residual smoothness penalty

    7. Optimize:
         L_total = L_res + λ_grip L_grip + λ_limit L_limit + λ_smooth L_smooth
```

---

## 13. Revised Inference Algorithm

```text
Input:
  latest observation obs_now
  latest robot state state_now
  language instruction
  current base action chunk A_base
  current chunk index k
  base latent z

1. Select base action:
     a_base = A_base[k]

2. Build A2C2 input:
     x = {
       obs_now,
       state_now,
       language,
       a_base,
       τ_k,
       z,
       optional A_base
     }

3. Predict correction:
     Δa_pred, close_logits = π_A2C2(x)

4. Add residual:
     a_corr = a_base + Δa_pred

5. Apply gripper event decision:
     close_prob = sigmoid(close_logits)
     update left/right gripper according to close_prob

6. Clamp to valid action range:
     a_exec = clamp(a_corr, low, high)

7. Send a_exec to controller.
```

---

## 14. Recommended Ablation Plan

Do not enable all tricks at once without ablation. Use the following order:

### Baseline

```text
Raw residual MSE
```

### Ablation 1: Range normalization

```text
Raw MSE → range-normalized MSE
```

Purpose:

```text
Check whether scale imbalance is hurting training.
```

### Ablation 2: Robust regression

```text
Range-normalized MSE → range-normalized Huber
```

Purpose:

```text
Check whether large residual outliers from bad base-policy samples dominate training.
```

### Ablation 3: Group-aware weighting

```text
Range-normalized Huber → group-weighted normalized Huber
```

Purpose:

```text
Check whether gripper/base/arm groups need different task-level weights.
```

### Ablation 4: Gripper event loss

```text
+ BCE open/close supervision for left and right grippers
```

Purpose:

```text
Check whether gripper timing improves beyond continuous width regression.
```

### Ablation 5: Action-limit penalty

```text
+ soft bound violation penalty
```

Purpose:

```text
Check whether corrected actions become more physically valid.
```

### Ablation 6: Temporal smoothness

```text
+ residual smoothness loss
```

Purpose:

```text
Check whether correction becomes less jittery and more stable in rollout.
```

---

## 15. Evaluation Metrics

Do not evaluate only by validation residual loss. The revised objective is designed to better align with robot behavior, so rollout-level metrics are necessary.

Recommended metrics:

```text
1. rollout success rate
2. BEHAVIOR Q-score
3. validation residual loss
4. per-group residual error
5. gripper open/close classification accuracy
6. gripper timing error
7. out-of-range action ratio
8. mean action-limit violation magnitude
9. end-effector position error, if FK is available
10. end-effector orientation error, if FK is available
11. action smoothness / jerk
12. correction magnitude ||Δa_pred||
```

Important diagnostic symptoms:

```text
If residual loss improves but success does not:
  action-space objective is still misaligned with task success.

If arm error improves but grasping does not:
  gripper event/timing is likely underweighted.

If correction magnitude becomes too large:
  A2C2 may be replacing the base policy instead of correcting it.

If out-of-range ratio is high:
  action-limit regularization or inference clamp is necessary.

If rollout becomes jittery:
  increase temporal smoothness or reduce correction magnitude.
```

---

## 16. Practical First Version

For the first implementation, use this minimal revised loss:

```text
L_total =
    L_res_norm_huber
  + λ_grip L_gripper_event
  + λ_limit L_limit
```

where:

```text
L_res_norm_huber =
Σ_g w_g mean_{i ∈ g} Huber(
    (Δa_pred_i - Δa_target_i) / scale_i
)
```

Start with:

```text
w_g = 1.0 for all non-gripper groups
w_gripper = 2.0
λ_grip = 1.0
λ_limit = 0.1
```

Do not add temporal smoothness until you have checked whether correction is actually jittery.

---

## 17. Suggested Method Paragraph

The method can be described as follows:

```text
We freeze the base flow-matching VLA policy and train A2C2 as a lightweight per-step residual correction head. Given the latest observation, the selected base action from the current action chunk, a chunk-position embedding, and base-policy features, the correction head predicts a residual action that is added to the base action. Instead of training this residual with naive raw MSE, we use a range-normalized, group-aware robust regression objective to account for the heterogeneous controller space. We further add gripper event supervision to handle the open/close nature of grasping, and an action-limit penalty to keep corrected actions within valid physical bounds. At inference time, the corrected action is clamped to the legal action range before execution. This design preserves the competence of the frozen flow policy while making the residual correction more aligned with manipulation-relevant behavior.
```

---

## 18. One-Sentence Summary

```text
Beyond naive residual MSE, A2C2 should be trained with a range-normalized, group-aware, event-aware, and bounded residual objective so that it corrects the base flow-model action chunk in a way that is numerically stable, physically valid, and better aligned with manipulation success.
```
