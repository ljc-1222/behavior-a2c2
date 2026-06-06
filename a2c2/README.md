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
  src/
    dataset.py
    model.py
  openpi_modification/
    pi0.py
    policy.py
```

`openpi_modification/` is reference-only. It is not imported by the current
training code and is not applied by `../setup.sh`.

## Method Summary

A2C2 trains a correction head on top of a frozen baseline policy instead of
retraining the whole policy. For source frame `t` and action chunk offset `k`:

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

## Evaluate

```bash
python a2c2/scripts/eval.py \
  --dataset-root a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1 \
  --checkpoint a2c2/runs/task18_wandb_bs128_w8_bpe4/latest.pt \
  --task-dir task-0018 \
  --split val \
  --num-samples 10000 \
  --batch-size 16
```

## Reference OpenPI Patches

`openpi_modification/` contains reference copies for a future online A2C2
integration path:

- `pi0.py` shows how to expose `return_prefix_z` from PI0 action sampling.
- `policy.py` shows how to add `Policy.infer_with_prefix_z(...)`.

These files are not active code in the current workspace. The pinned
`openpi-comet` submodule currently contains only the baseline B1K websocket
compatibility patches:

```text
scripts/serve_b1k.py
src/openpi/policies/b1k_policy.py
src/openpi/shared/b1k_network_utils.py
```

The pinned submodule does not include an online A2C2 websocket server. The
dataset builder extracts `a2c2.base_policy_z` directly from the loaded
OpenPI/COMET model internals, so dataset extraction and correction-head training
do not require applying the reference files.

To make online A2C2 evaluation real later:

1. Port the reference changes into `openpi-comet`.
2. Add and test an online A2C2 websocket server or wrapper.
3. Commit and push those changes in the `openpi-comet` submodule.
4. Update the outer b1k gitlink.
5. Document the new online evaluation command in the outer `README.md`.
