# Setup and usage

[← Project overview](../README.md) · [Method](method.md) · [Integration](integration.md)

This page records the code entrypoints for the **Task 1 (`picking_up_trash`)** workspace. It is not a claim that the commands below were executed again for the portfolio release. BEHAVIOR/OmniGibson uses a substantial Linux GPU environment and simulator assets.

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

## 2. Download the Task 1 A2C2 dataset

The dataset is released separately at [Hugging Face](https://huggingface.co/datasets/ljc-1222/a2c2_dataset). It contains expert demonstration metadata, videos, base policy chunks and prefix latents.

```bash
python -m pip install huggingface_hub pyarrow tqdm
python a2c2/scripts/create_dataset.py \
  --download-variant picking_up_trash_pi05-b1kpt50-cs32_h32_v1
```

Expected selected root:

```text
a2c2_dataset/picking_up_trash_pi05-b1kpt50-cs32_h32_v1/
  data/task-0001/
  latent/data/task-0001/
  videos/task-0001/
  meta/
```

The build path (`create_dataset.py --build --task-index 1`) instead regenerates chunks/latents from BEHAVIOR demonstrations using a frozen Comet checkpoint. This requires both patched submodules and the OpenPI environment; download mode is the lower-effort route.

## 3. Generate the frozen RGB feature cache

Training defaults to `--rgb-cache-kind resnet18-features`, but downloaded data does **not** include this local preprocessing cache. Generate it before using the default training path:

```bash
DATASET=a2c2_dataset/picking_up_trash_pi05-b1kpt50-cs32_h32_v1
python a2c2/scripts/mp4_to_parquet.py "$DATASET/videos/task-0001" \
  --output-kind resnet18-features \
  --image-size 224 \
  --no-use-depth
```

This creates `$DATASET/rgb_features_resnet18/task-0001/` containing one feature-parquet per RGB camera/video. The converter defaults to 128-pixel resizing if unspecified, so the explicit 224-pixel setting above matches the training default. Feature extraction requires PyTorch, torchvision and a pretrained ResNet-18 (unless overridden). Alternatively, use `--rgb-cache-kind none` in training to decode RGB frames directly, at greater runtime cost.

Depth remains a separate video input controlled by `--use-depth` / `--no-use-depth`. RGB and task language remain mandatory in the current model loader.

## 4. Train a correction head

The following is a **Task 1 example** using the best-performing loss *from the original comparison*, not a promise of reproducing that exact historical checkpoint:

```bash
python a2c2/scripts/train.py \
  --dataset-root "$DATASET" \
  --task-dir task-0001 \
  --output-dir a2c2/runs/task1_norm_mse \
  --loss-preset norm_mse \
  --steps 200000 \
  --seed 42
```

Change `--loss-preset` to one of `raw_mse`, `norm_mse`, `norm_huber`, or `norm_huber_gripw` for the other objectives. The script saves its model config, chosen CLI args and action statistics in the checkpoint; use those values when interpreting or loading a particular run.

## 5. Offline evaluation

```bash
python a2c2/scripts/eval.py \
  --dataset-root "$DATASET" \
  --task-dir task-0001 \
  --checkpoint a2c2/runs/task1_norm_mse/latest.pt \
  --split val \
  --seed 42
```

The separate `eval.py` currently regenerates the episode split using the dataset order, validation ratio and seed. Set `--seed`, `--val-ratio` and `--max-episodes` consistently with the training run; mismatching split settings risks evaluating training episodes. If the validation set is empty, the updated script now fails rather than silently evaluating training data under the `val` label.

This *offline* command measures action-regression error; it does not run the simulator or compute the BEHAVIOR Q-score.

## 6. Online Task 1 evaluation

Online rollout needs the full OpenPI-Comet and BEHAVIOR/OmniGibson environments, the frozen Comet checkpoint, a trained A2C2 checkpoint, and BEHAVIOR task instances. Run the base policy server from `openpi-comet/`, for example:

```bash
cd openpi-comet
uv run --no-sync ../a2c2/scripts/serve_a2c2_b1k.py \
  --task-name=picking_up_trash \
  --a2c2-checkpoint=../a2c2/runs/task1_norm_mse/latest.pt \
  policy:checkpoint \
  --policy.config=pi05_b1k-base \
  --policy.dir=./checkpoints/pi05-b1kpt50-cs32
```

In the BEHAVIOR environment, use the project's websocket evaluator (the default WebSocket host must match the server). A2C2 requires the wrapper exposing depth/task-info when enabled:

```bash
cd BEHAVIOR-1K
xvfb-run -a python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=picking_up_trash \
  model.host=127.0.0.1 \
  env_wrapper._target_=omnigibson.learning.wrappers.A2C2Wrapper \
  eval_instance_ids="[0]" \
  write_video=false
```

For a **frozen baseline** comparison, serve the uncorrected `openpi-comet/scripts/serve_b1k.py` and select `omnigibson.learning.wrappers.RGBWrapper` in BEHAVIOR. The online diagnostic JSON is recorded as `a2c2_online` in the patched evaluator. The commands above illustrate the available integration, not a recently validated full-stack run.

## Operational limits

- An A2C2 checkpoint is **not** bundled with this repository. The public results are historical aggregates, not turnkey precomputed predictions.
- The project does not promise exact package/platform compatibility with future versions of OpenPI, OmniGibson or CUDA.
- A schema-level smoke script exists at `a2c2/scripts/test_online_eval.py`, but it still needs fork dependencies; it was **not rerun** for this README update.
