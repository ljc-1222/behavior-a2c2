# Method

[← Project overview](../README.md) · [Evaluation diagnostics](results.md) · [Usage](usage.md)

## Motivation and execution path

OpenPI-Comet generates a chunk of **32 actions** for the R1Pro robot; each executed action contains **23 control dimensions**. A chunk can become stale while the scene changes. Following the [A2C2 method](https://arxiv.org/abs/2509.23224), we hold Comet fixed and train an auxiliary model to correct the single action selected at each execution step.

For a chunk generated at source time $t$ and executed at offset $k$, the frozen policy supplies a base action chunk $A_t$, valid-action mask $M_t$, and prefix latent $z_t$. The head observes the latest state $o_{t+k}$ and predicts a correction:

```math
(A_t,z_t)=\pi_{\mathrm{Comet}}(o_t,\ell),
```

```math
\Delta a_{t,k}=f_{\theta}(o_{t+k},A_t,M_t,a_{t,k}^{\mathrm{base}},z_t,\ell,\tau(k)),
```

```math
a_{t+k}^{\mathrm{exec}}=
\mathrm{clip}\left(a_{t,k}^{\mathrm{base}}+\lambda\,\Delta a_{t,k}\right).
```

Here $\ell$ is the task instruction, $\tau(k)$ encodes position in the action chunk, and $\lambda$ is the correction scale (the online default is $1$). Clipping occurs after residual addition. This implementation runs per-step residual inference on a receding-horizon chunk; it **does not establish real-time latency guarantees or reproduce an asynchronous scheduling benchmark**. The architecture diagram appears on the [project overview](../README.md).

## Training data and target alignment

`a2c2/scripts/create_dataset.py` augments BEHAVIOR expert episodes with baseline action chunks and a prefix latent:

- Each source frame $t$ receives a frozen-policy chunk $A_t$ with 32 actions, a valid-offset mask, and policy timing.
- A companion latent parquet stores the mask-pooled output of the COMET/PaliGemma **prefix forward pass**. This is **not** a representation from the action denoising loop.
- For each valid $(t,k)$, the execution frame is $t+k$; the supervised residual is $\Delta a_{t,k}^{*}=a_{t+k}^{\mathrm{expert}}-a_{t,k}^{\mathrm{base}}$.

The dataset builder preserves the original episode metadata and video modalities. The dataset loader samples valid source/offset pairs within each episode. Scripts are configured for **Task 1, `picking_up_trash`, `task-0001`** by default. The serialized model config controls which additional modalities a particular checkpoint expects.

## Correction head

The implementation in [`a2c2/src/model.py`](../a2c2/src/model.py) uses token projections followed by a **6-layer, 8-head Transformer encoder** (default hidden dimension **512**) and an MLP that regresses a **23-D residual**. Its feature groups include:

| Group | Implementation |
|:--|:--|
| Base policy | Full action chunk and validity mask, selected base action, and cached prefix latent $z_t$ (2048-D) |
| Robot | Proprioception (256-D) and chunk-offset time encoding |
| Visual | Three RGB camera views; frozen ResNet-18 features or decoded frames |
| Language | Task instruction represented by token embeddings and a GRU |
| Optional or config-dependent | Depth cameras, camera-relative poses, task-info vector, policy inference timing |

RGB and task language are required by current checkpoint validation; **depth is configurable**, not universally required. The training CLI defaults to depth, camera-relative poses, task information and inference timing enabled. Exact enabled features must be read from a saved checkpoint's configuration rather than inferred from the filename.

The frozen-base hooks live in the [OpenPI-Comet fork](https://github.com/ljc-1222/openpi-comet). [Integration notes](integration.md) identify the relevant entrypoints.

## Loss ablations

The loss implementation is in [`a2c2/src/loss.py`](../a2c2/src/loss.py). Four available `--loss-preset` values were used in the Task 1 comparison:

| Preset | Meaning |
|:--|:--|
| `raw_mse` | Ordinary mean squared residual error in the original action units |
| `norm_mse` | Per-dimension scale-normalized squared error, averaged within each of six action groups |
| `norm_huber` | Scale-normalized Huber error within each group |
| `norm_huber_gripw` | Same normalized Huber loss with double weight on the two gripper groups |

For each action dimension $j$, a robust scale is estimated from the training split, and the normalized residual error is defined as

```math
s_j=\max\!\left(Q_{0.99}(a_j)-Q_{0.01}(a_j),\varepsilon\right),
\qquad
e_j=\frac{\widehat{\Delta a}_j-\Delta a_j^{*}}{s_j}.
```

The normalized loss aggregates **group-wise means**:

```math
\mathcal{L}=\sum_{g\in\mathcal{G}} w_g\,
\frac{1}{|g|}\sum_{j\in g}\rho(e_j).
```

For normalized MSE, $\rho(e)=e^2$; for the Huber variants, $\rho$ is the Huber penalty with the configured $\delta$.

The six groups are base, torso, left arm, left gripper, right arm and right gripper. Defaults are Huber delta **1.0**, gripper weight **2.0** in the weighted preset, and minimum scale $10^{-4}$. These are **sums of group means**, not a uniform average over all 23 dimensions.

## Evaluation interfaces

Offline evaluation measures residual, corrected-action, and base-action errors on episode data. Online evaluation compares the frozen baseline with the residual-corrected wrapper in OmniGibson.

See the [Task 1 results](../README.md#task-1-results) for the primary scores, [analysis and next steps](../README.md#analysis-and-next-steps) for the proposed explanations, and [evaluation diagnostics](results.md) for the complete control measurements and metric definitions.
