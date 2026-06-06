# A2C2 Correction Head

This directory contains the A2C2 dataset tooling, model code, training script,
evaluation script, and reference OpenPI patch notes for the BEHAVIOR-1K task18
experiment.

Task18 is:

```text
task-0018 / tidying_bedroom
```

The released dataset variant is:

```text
tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

## Directory Layout

```text
a2c2/
  README.md
  scripts/
    create_dataset.py
    train.py
    eval.py
    serve_a2c2_b1k.py
    test_online_eval.py
  src/
    dataset.py
    model.py
    online.py
  openpi_modification/
    pi0.py
    policy.py
```

`openpi_modification/` keeps reference copies of the OpenPI changes needed to
return PI0.5 prefix latents. The active online path uses the patched
`../openpi-comet` submodule plus `src/online.py`.

## Method Summary

A2C2 follows "Leave No Observation Behind: Real-time Correction for VLA Action
Chunks" (arXiv:2509.23224, https://arxiv.org/abs/2509.23224). It trains a
correction head on top of a frozen baseline policy instead of retraining the
whole policy. For source frame `t` and action chunk offset `k`:

```text
a_exec[t+k] = a_base[t, k] + delta_a[t, k]
delta_a[t, k] = correction_head(o[t+k], a_base[t, k], tau[k], z[t], language)
```

Where:

- `a_base[t, k]` is the PI0.5 OpenPI-COMET baseline action at chunk offset `k`.
- `z[t]` is the baseline policy latent extracted from the same source frame.
- `o[t+k]` and `expert_action[t+k]` come from BEHAVIOR demonstrations.
- `delta_a[t, k] = expert_action[t+k] - a_base[t, k]` is the supervised target.

## Dataset Source

The A2C2 dataset is derived from BEHAVIOR-1K 2025 challenge demos:

```text
BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos
```

The released task18 variant was generated with:

```text
OpenPI/COMET config:      pi05_b1k-base
OpenPI/COMET checkpoint:  pi05-b1kpt50-cs32
Action horizon:           32
Action dim:               23
Task index:               18
Task name:                tidying_bedroom
```

Generation steps:

1. Select BEHAVIOR task18 (`task-0018`, `tidying_bedroom`).
2. Keep the original BEHAVIOR/LeRobot parquet files, metadata, videos, and
   annotations.
3. Run fused baseline inference for every source frame.
4. Save `a2c2.base_action_chunk` and `a2c2.valid_action_mask` in the main
   parquet files.
5. Save the mask-pooled baseline policy latent in sidecar latent parquet files.

## Policy Latent Definition

`z` is not produced by the action denoising loop. It is the mask-pooled final
hidden state from the PI0.5 COMET prefix forward pass:

```python
prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
(prefix_out, _), _ = module.PaliGemma.llm(
    [prefix_tokens, None],
    mask=prefix_attn_mask,
    positions=positions,
)
z = sum(prefix_out * prefix_mask) / sum(prefix_mask)
```

The latent summarizes image tokens, prompt tokens, and discretized PI0.5 state
tokens after contextualization by the PaliGemma language model. It uses the same
checkpoint, prompt, and input transforms as the baseline action chunk.

When both action chunks and latents are needed, `create_dataset.py --build` uses
a fused runner so the prefix forward pass is computed once and reused for both
the action denoising cache and pooled `z`.

## Dataset Layout

After downloading, the dataset root is:

```text
a2c2_dataset/
  README.md
  tidying_bedroom_pi05-b1kpt50-cs32_h32_v1/
    manifest.json
    data/task-0018/episode_XXXXXXXX.parquet
    latent/data/task-0018/episode_XXXXXXXX.parquet
    latent/meta/manifest.json
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/episodes/task-0018/episode_XXXXXXXX.json
    videos/task-0018/<video_key>/episode_XXXXXXXX.mp4
    annotations/task-0018/episode_XXXXXXXX.json
```

The main parquet files keep the original BEHAVIOR/LeRobot fields, including:

```text
index
episode_index
task_index
timestamp
observation.state
observation.cam_rel_poses
observation.task_info
action
```

A2C2 adds:

```text
a2c2.base_action_chunk  # fixed-size float32 [32, 23]
a2c2.valid_action_mask  # fixed-size bool [32]
a2c2.policy_infer_ms    # float32
```

Latents are stored separately:

```text
latent/data/task-0018/episode_XXXXXXXX.parquet
```

Each latent parquet contains:

```text
a2c2.base_policy_z      # fixed-size float32 [latent_dim]
```

The main parquet and latent parquet are aligned by episode and row order.

## Training Alignment

For each source row `source_idx` and chunk offset `k`:

```text
valid         = main[source_idx]["a2c2.valid_action_mask"][k]
z             = latent[source_idx]["a2c2.base_policy_z"]
base_action   = main[source_idx]["a2c2.base_action_chunk"][k]
target_idx    = source_idx + k
expert_action = main[target_idx]["action"]
delta_target  = expert_action - base_action
```

Only samples with `valid == True` are used, because action chunks near the end
of an episode may point beyond the demonstration length.

## Download The Released Dataset

The public Hugging Face dataset is:

```text
https://huggingface.co/datasets/ljc-1222/a2c2_dataset
```

From the outer repository root:

```bash
python a2c2/scripts/create_dataset.py
```

By default, this command downloads only the task18 variant:

```text
tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

It is equivalent to:

```bash
python a2c2/scripts/create_dataset.py \
  --download-variant tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

The downloader uses Hugging Face snapshot allow patterns for repo metadata plus
that variant directory, so other variants are not downloaded by default.

Download destination:

```text
./a2c2_dataset/
```

Custom destination:

```bash
python a2c2/scripts/create_dataset.py --download-root /path/to/a2c2_dataset
```

Common download options:

```text
--repo-id ljc-1222/a2c2_dataset
--download-root PATH
--download-variant tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
--download-all-variants  # explicit opt-in for every variant
--revision REV
--force-download
--local-files-only
--max-workers N
```

Install the minimal download dependency if needed:

```bash
python -m pip install huggingface_hub
```

## Rebuild The Dataset

Most users should download the released dataset. Rebuild only when changing the
input demos, baseline checkpoint, OpenPI config, task selection, or feature
extraction code.

For real OpenPI/COMET inference, run the build through the `openpi-comet` `uv`
environment created by the outer `setup.sh`:

```bash
cd "$B1K_ROOT"
bash setup.sh --download-challenge-demos

cd "$B1K_ROOT/openpi-comet"
UV_CACHE_DIR="$B1K_ROOT/.uv-cache" uv run --no-sync python ../a2c2/scripts/create_dataset.py \
  --build \
  --source-root "$B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos" \
  --openpi-root "$B1K_ROOT/openpi-comet" \
  --checkpoint-dir "$B1K_ROOT/openpi-comet/checkpoints/pi05-b1kpt50-cs32" \
  --output-root "$B1K_ROOT/a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1" \
  --config-name pi05_b1k-base \
  --task-index 18 \
  --cache-seed 42
```

If `--output-root` is omitted, task18 defaults to:

```text
./a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

Resume behavior:

- Complete episode files are skipped after row-count and schema checks.
- Incomplete writes use temporary files and atomic replacement.
- Per-episode progress is stored under `meta/a2c2_progress/`.
- `--overwrite` removes the output root before rebuilding.

Mock-policy schema checks do not need the OpenPI model:

```bash
python a2c2/scripts/create_dataset.py \
  --build \
  --mock-policy \
  --max-episodes 1 \
  --max-frames-per-episode 4 \
  --output-root /tmp/a2c2_schema_check \
  --overwrite
```

## Train

Default task18 training:

```bash
python a2c2/scripts/train.py \
  --dataset-root a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1 \
  --output-dir a2c2/runs/task18_wandb_bs128_w8_bpe4 \
  --task-dir task-0018 \
  --steps 400000 \
  --batch-size 128 \
  --num-workers 8 \
  --batches-per-episode 4 \
  --lr 1e-5 \
  --weight-decay 1e-5 \
  --log-every 20 \
  --save-every 5000 \
  --eval-every 1000 \
  --eval-batch-size 128 \
  --wandb \
  --wandb-project a2c2 \
  --wandb-run-name task18_wandb_bs128_w8_bpe4
```

Small local smoke test:

```bash
python a2c2/scripts/train.py \
  --dataset-root a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1 \
  --output-dir /tmp/a2c2_smoke_train \
  --task-dir task-0018 \
  --max-episodes 2 \
  --steps 10 \
  --batch-size 4 \
  --num-workers 0 \
  --eval-every 0 \
  --rgb-backbone small-cnn \
  --depth-backbone small-cnn \
  --no-pretrained-rgb \
  --no-pretrained-depth
```

## Offline Evaluate

```bash
python a2c2/scripts/eval.py \
  --dataset-root a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1 \
  --checkpoint a2c2/runs/task18_wandb_bs128_w8_bpe4/latest.pt \
  --task-dir task-0018 \
  --split val \
  --num-samples 10000 \
  --batch-size 16
```

## Online BEHAVIOR Evaluate

Run online evaluation with the normal BEHAVIOR websocket client and the A2C2
server. The A2C2 server calls the base OpenPI-COMET policy for a full action
chunk, caches the chunk plus `prefix_z`, then corrects the selected chunk action
at every environment step with the latest observation.

Terminal 1:

```bash
cd "$B1K_ROOT/openpi-comet"
export PATH="$HOME/.local/bin:$CONDA_DIR/bin:$PATH"
export UV_CACHE_DIR="$B1K_ROOT/.uv-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$B1K_ROOT/.cache/jax"
export A2C2_CHECKPOINT="${A2C2_CHECKPOINT:-$B1K_ROOT/a2c2/ckpt/model_latent.pt}"

uv run --no-sync ../a2c2/scripts/serve_a2c2_b1k.py \
  --task-name="$B1K_TASK_NAME" \
  --control-mode=receeding_horizon \
  --max-len=32 \
  --a2c2-checkpoint="$A2C2_CHECKPOINT" \
  policy:checkpoint \
  --policy.config=pi05_b1k-base \
  --policy.dir="./checkpoints/$B1K_CHECKPOINT_NAME"
```

Terminal 2 is the same BEHAVIOR command documented in the outer `README.md`
baseline evaluation section:

```bash
cd "$B1K_ROOT/BEHAVIOR-1K"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$B1K_CONDA_ENV"

RUN_LOG="$B1K_ROOT/BEHAVIOR-1K/output/${B1K_TASK_NAME}_a2c2_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_LOG"

xvfb-run -a -s "-screen 0 1280x720x24" python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name="$B1K_TASK_NAME" \
  log_path="$RUN_LOG" \
  model.host=127.0.0.1 \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  eval_instance_ids="[0]" \
  write_video=true
```

Online data flow:

```text
source frame t:
  OpenPI-COMET infer_with_prefix_z(obs[t])
  cache actions[0:32], valid mask, prefix_z, log1p(policy_infer_ms)

each environment step t+k:
  read latest robot_r1::proprio and optional online features
  selected_base_action = cached actions[k]
  time_feature = [sin(2*pi*k/(H-1)), cos(2*pi*k/(H-1))]
  residual = A2C2(obs[t+k], selected_base_action, cached chunk, prefix_z, time_feature)
  execute selected_base_action + residual
```

`serve_a2c2_b1k.py` supports both task18 checkpoints currently present under
`a2c2/ckpt/`:

- `model_latent.pt` uses `prefix_z` and requires the active OpenPI patches.
- `model_no_latent.pt` sets `use_base_policy_z=False` and calls normal
  `Policy.infer(...)`.

The current task18 checkpoints use state, base action chunk, selected base
action, time feature, and optionally `prefix_z`; they do not require online
RGB/depth/camera/task/language tensors. If a future checkpoint enables those
flags, the online wrapper will read them from the current BEHAVIOR observation
and fail fast when a required feature is missing. Use
`--allow-missing-online-features` only for zero-filled smoke tests.

Fast online smoke test:

```bash
cd "$B1K_ROOT/openpi-comet"
uv run --no-sync python ../a2c2/scripts/test_online_eval.py
```

`test_online_eval.py` first checks that the pinned `openpi-comet` exposes
`Policy.infer_with_prefix_z(...)` and `Pi0.sample_actions(...,
return_prefix_z=True)`, then runs a fake BEHAVIOR environment for five steps.
It checks that base chunks are fetched at environment steps 0 and 3, correction
offsets are 0, 1, 2, 0, 1, the correction head receives tensors with the online
evaluation shapes, and the fake environment receives `base_action + residual`
for every step.

## OpenPI Integration Notes

The active `../openpi-comet` submodule now contains the two runtime changes
needed by online latent checkpoints:

- `src/openpi/models/pi0.py` exposes `return_prefix_z` from PI0 action sampling.
- `src/openpi/policies/policy.py` exposes `Policy.infer_with_prefix_z(...)`.

`openpi_modification/` remains as a small reference copy of those changes for
review and future rebases.
