# Setup and usage

[← Project overview](../README.md) · [Method](method.md) · [Integration](integration.md)

This page describes the **Task 1 (`picking_up_trash`)** workspace: environment setup, local correction data, training, and offline/online evaluation. BEHAVIOR/OmniGibson requires a compatible Linux GPU environment and simulator assets.

## 1. Checkout and dependencies

```bash
git clone --recurse-submodules https://github.com/ljc-1222/behavior-a2c2.git
cd behavior-a2c2
git submodule status
```

The gitlinks in this repository record specific OpenPI and BEHAVIOR fork commits. To preserve those commits, **do not run `git submodule update --remote`**. The setup helper otherwise tracks its configured branches, so initialize submodules first and use:

```bash
bash setup.sh --skip-submodule-update --no-behavior-sparse-checkout
```

The script prepares the separate OpenPI (`uv`) and BEHAVIOR (`conda`) environments, datasets, and baseline checkpoint. It can download sizeable dependencies. See `bash setup.sh --help` for options such as `--skip-system-packages`, `--skip-behavior-dataset`, `--skip-checkpoint`, and `--download-conda`.

In later shells:

```bash
export B1K_ROOT="$(pwd -P)"
# Point this at the conda root selected by setup.sh.
source /path/to/conda/etc/profile.d/conda.sh
conda activate behavior
```

## 2. Prepare local Task 1 correction data

Training requires a local correction dataset containing expert actions and observations, frozen-policy chunks, valid-offset masks, and prefix latents. Set the dataset root from the workspace shell:

```bash
export DATASET="$B1K_ROOT/a2c2_dataset/task1"
```

Expected layout:

```text
$DATASET/
  data/task-0001/
  latent/data/task-0001/
  videos/task-0001/
  meta/
```

The existing builder constructs this dataset from BEHAVIOR expert episodes and a frozen Comet checkpoint. It requires the patched OpenPI environment, source demonstrations, and baseline checkpoint prepared during setup:

```bash
(
  cd "$B1K_ROOT/openpi-comet"
  uv run --no-sync ../a2c2/scripts/create_dataset.py \
    --build \
    --task-index 1 \
    --output-root "$DATASET"
)
```

The builder's `--source-root` and `--checkpoint-dir` flags select non-default local inputs. Dataset construction generates base chunks and prefix latents; it does not train the VLA or the correction head. An existing dataset with the same schema can be used instead.

## 3. Generate the frozen RGB feature cache

Training defaults to `--rgb-cache-kind resnet18-features` and expects a local ResNet-18 feature cache. If the selected dataset root does not already contain this cache, generate it before training:

```bash
python a2c2/scripts/mp4_to_parquet.py "$DATASET/videos/task-0001" \
  --output-kind resnet18-features \
  --output-root "$DATASET/rgb_features_resnet18/task-0001" \
  --image-size 224 \
  --no-use-depth
```

This creates `$DATASET/rgb_features_resnet18/task-0001/` containing one feature-parquet per RGB camera/video. The converter defaults to 128-pixel resizing if unspecified, so the explicit 224-pixel setting above matches the training default. Feature extraction requires PyTorch, torchvision and a pretrained ResNet-18 (unless overridden). Alternatively, use `--rgb-cache-kind none` in training to decode RGB frames directly, at greater runtime cost.

Depth remains a separate video input controlled by `--use-depth` / `--no-use-depth`. RGB and task language remain mandatory in the current model loader.

## 4. Train a correction head

The following Task 1 example uses normalized MSE, the strongest correction variant in the reported comparison:

```bash
python a2c2/scripts/train.py \
  --dataset-root "$DATASET" \
  --task-dir task-0001 \
  --output-dir a2c2/runs/task1_norm_mse \
  --loss-preset norm_mse \
  --steps 200000 \
  --seed 42
```

Change `--loss-preset` to one of `raw_mse`, `norm_mse`, `norm_huber`, or `norm_huber_gripw` for the other objectives. The script saves its model config, selected CLI arguments and action statistics in the checkpoint; use these values when interpreting or loading a particular run.

## 5. Offline evaluation

```bash
python a2c2/scripts/eval.py \
  --dataset-root "$DATASET" \
  --task-dir task-0001 \
  --checkpoint a2c2/runs/task1_norm_mse/latest.pt \
  --split val \
  --seed 42
```

Evaluation reconstructs the episode split from dataset order, seed, validation ratio and maximum episode count. Use the training values for `--seed`, `--val-ratio` and `--max-episodes`. Evaluation rejects an empty validation split.

This offline command measures action-regression error. BEHAVIOR Q-scores come from online simulator rollouts.

## 6. Online Task 1 evaluation

Online rollout needs the full OpenPI-Comet and BEHAVIOR/OmniGibson environments, the frozen Comet checkpoint, a trained A2C2 checkpoint, and BEHAVIOR task instances. Run the base policy server from `openpi-comet/`, for example:

```bash
cd "$B1K_ROOT/openpi-comet"
uv run --no-sync ../a2c2/scripts/serve_a2c2_b1k.py \
  --task-name=picking_up_trash \
  --a2c2-checkpoint=../a2c2/runs/task1_norm_mse/latest.pt \
  policy:checkpoint \
  --policy.config=pi05_b1k-base \
  --policy.dir=./checkpoints/pi05-b1kpt50-cs32
```

In a separate shell with the BEHAVIOR environment active, use the project's websocket evaluator (the default WebSocket host must match the server). A2C2 requires the wrapper exposing depth/task-info when enabled:

```bash
cd "$B1K_ROOT/BEHAVIOR-1K"
xvfb-run -a python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=picking_up_trash \
  model.host=127.0.0.1 \
  env_wrapper._target_=omnigibson.learning.wrappers.A2C2Wrapper \
  eval_instance_ids="[0]" \
  write_video=false
```

For a **frozen baseline** comparison, serve the uncorrected `openpi-comet/scripts/serve_b1k.py` and select `omnigibson.learning.wrappers.RGBWrapper` in BEHAVIOR. The online diagnostic JSON is recorded as `a2c2_online` in the patched evaluator. Use the same task instances, seed, and chunk length for baseline and corrected-policy comparisons.

## Checkpoint and integration notes

- Checkpoint-dependent commands require a locally trained or otherwise available correction-head checkpoint. Training saves `latest.pt` and numbered checkpoints under the selected output directory.
- `a2c2/scripts/test_online_eval.py` is an integration smoke script requiring the fork dependencies.
