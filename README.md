# b1k A2C2 Task18 Workspace

This repository is the outer workspace for the BEHAVIOR-1K task18
(`tidying_bedroom`) A2C2 experiment. The project uses the `openpi-comet`
PI0.5 baseline policy to extract base action chunks and policy latents, then
trains an A2C2 correction head to predict the residual between the expert
action and the baseline action.

The outer `b1k` repository owns the setup workflow, documentation, A2C2
training code, and submodule pins. The forked upstream projects remain
submodules.

## Repository Layout

```text
b1k/
  README.md
  setup.sh
  .gitmodules
  openpi-comet/            # submodule: ljc-1222/openpi-comet, dev/ljc-1222
  BEHAVIOR-1K/             # submodule: ljc-1222/BEHAVIOR-1K, dev/ljc-1222
  a2c2/
    README.md              # dataset, training, eval, and reference notes
    scripts/
      serve_a2c2_b1k.py    # online BEHAVIOR websocket server with A2C2 residuals
      test_online_eval.py  # fake-env online residual smoke test
    src/
      online.py            # online B1K/OpenPI A2C2 wrapper
    openpi_modification/   # reference-only OpenPI A2C2 patches
```

Pinned submodules:

```text
openpi-comet  ec1dfe54757a731123869f6e5fe16e4a0a1cea0c
BEHAVIOR-1K   398ff024db4c5b5e8be0fd38e632bc00579eb470
```

## Requirements

Recommended runtime:

- Ubuntu 22.04 LTS
- NVIDIA GPU with at least 24GB VRAM
- 64GB RAM or more
- 150GB disk minimum; 200GB+ is more practical when keeping datasets, caches,
  videos, and checkpoints
- Network access to GitHub, Hugging Face, PyTorch wheels, and Ubuntu apt
  mirrors

`setup.sh` can install missing apt runtime packages. Use
`--skip-system-packages` if the image already provides git, git-lfs, xvfb, GL
runtime libraries, and related dependencies.

## Clone With Submodules

Recommended:

```bash
git clone --filter=blob:none https://github.com/ljc-1222/b1k.git
cd b1k
git submodule update --init --recursive --depth 1 --filter=blob:none
```

One-command clone:

```bash
git clone \
  --filter=blob:none \
  --also-filter-submodules \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/ljc-1222/b1k.git
cd b1k
```

`setup.sh` also checks the submodules and initializes them when needed:

```bash
git submodule update --init --recursive --depth 1 --filter=blob:none openpi-comet BEHAVIOR-1K
```

For `BEHAVIOR-1K`, setup uses sparse checkout by default to keep only the files
needed for installation and experiments:

```text
setup.sh README.md OmniGibson bddl3 joylo eval-jobqueue
```

## Environment Setup

Run from the repository root:

```bash
bash setup.sh
```

Common options:

```bash
# Skip apt package installation when the image already has the runtime packages.
bash setup.sh --skip-system-packages

# Non-interactive install: download Miniforge if no conda root is found.
bash setup.sh --download-conda

# Use a specific conda root.
bash setup.sh --conda-dir "$HOME/.local/share/b1k/miniforge3"

# Skip BEHAVIOR runtime assets and task instances.
bash setup.sh --skip-behavior-dataset

# Download the 2025 challenge demo subset for task18 dataset rebuilding.
bash setup.sh --download-challenge-demos

# Skip the OpenPI-COMET policy checkpoint.
bash setup.sh --skip-checkpoint
```

Conda resolution order:

1. `--conda-dir` or `B1K_CONDA_DIR`
2. `B1K_CONDA_EXE`
3. Parent directories outside `B1K_ROOT`, looking for `miniforge3`,
   `miniconda3`, `anaconda3`, `mambaforge`, `conda`, `.conda`, or `opt/conda`
4. `conda` on `PATH`, while ignoring project-local `./miniconda3` unless it was
   selected explicitly
5. If no conda root is found, prompt for an existing root or download
   Miniforge; use `--download-conda` in non-interactive shells

Useful environment overrides:

```bash
export B1K_ROOT="$(pwd -P)"
export B1K_CONDA_ENV=behavior
export B1K_TASK_NAME=tidying_bedroom
export B1K_TASK_DIR=task-0018
export B1K_CHECKPOINT_NAME=pi05-b1kpt50-cs32
```

Generated setup output:

```text
setup_run.log
```

This file is ignored by git.

For a new shell after setup, set:

```bash
cd /path/to/b1k
export B1K_ROOT="$(pwd -P)"
export CONDA_DIR="${B1K_CONDA_DIR:-$(awk '/Using conda:/ {print $NF}' setup_run.log | tail -1)}"
test -n "$CONDA_DIR" && test -f "$CONDA_DIR/etc/profile.d/conda.sh"
```

## Hugging Face Token

Most downloads are public, but setting `HF_TOKEN` avoids rate limits and
authorization surprises:

```bash
read -rsp "Paste HF_TOKEN: " HF_TOKEN
echo
export HF_TOKEN
```

Verify:

```bash
curl -sS \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami-v2
```

## A2C2 Dataset

The released dataset is hosted at:

```text
https://huggingface.co/datasets/ljc-1222/a2c2_dataset
```

Download it in the project environment:

```bash
cd "$B1K_ROOT"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate behavior
python -m pip install huggingface_hub pyarrow tqdm

python a2c2/scripts/create_dataset.py
```

By default, this downloads only the task18 dataset variant:

```text
tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

The downloader limits the Hugging Face snapshot to the repo metadata and that
variant directory. Other variants are excluded unless `--download-all-variants`
is passed explicitly.

The selected files are downloaded under:

```text
a2c2_dataset/
```

The default task18 training dataset root inside that download is:

```text
a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1
```

For dataset schema, alignment details, rebuild commands, and training/eval
entrypoints, see:

```text
a2c2/README.md
```

## Rebuild The Dataset

Most users should download the released dataset. Rebuild only when changing the
source demos, OpenPI checkpoint, model config, task selection, or latent/action
extraction code.

```bash
cd "$B1K_ROOT"
bash setup.sh --download-challenge-demos

cd "$B1K_ROOT/openpi-comet"
UV_CACHE_DIR="$B1K_ROOT/.uv-cache" uv run --no-sync python ../a2c2/scripts/create_dataset.py \
  --build \
  --source-root "$B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos" \
  --openpi-root "$B1K_ROOT/openpi-comet" \
  --checkpoint-dir "$B1K_ROOT/openpi-comet/checkpoints/pi05-b1kpt50-cs32" \
  --config-name pi05_b1k-base \
  --task-index 18 \
  --cache-seed 42
```

Fast schema smoke test:

```bash
cd "$B1K_ROOT"
python a2c2/scripts/create_dataset.py \
  --build \
  --mock-policy \
  --max-episodes 1 \
  --max-frames-per-episode 4 \
  --output-root /tmp/a2c2_schema_check \
  --overwrite
```

## Train The Correction Head

Use the `behavior` conda environment created by `setup.sh`:

```bash
cd "$B1K_ROOT"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate behavior
python -m pip install pyarrow tqdm wandb
```

Full task18 training example:

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

Small local training smoke test:

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

## Evaluate The Correction Head

Offline dataset evaluation:

```bash
python a2c2/scripts/eval.py \
  --dataset-root a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1 \
  --checkpoint a2c2/runs/task18_wandb_bs128_w8_bpe4/latest.pt \
  --task-dir task-0018 \
  --split val \
  --num-samples 10000 \
  --batch-size 16
```

## Baseline BEHAVIOR Evaluation

Run the baseline websocket evaluation in two terminals after `setup.sh`
finishes. These commands assume the task18 defaults:

```text
tidying_bedroom / task-0018
```

Prepare shared variables in each terminal:

```bash
cd /path/to/b1k
export B1K_ROOT="$(pwd -P)"
export CONDA_DIR="${B1K_CONDA_DIR:-$(awk '/Using conda:/ {print $NF}' setup_run.log | tail -1)}"
export B1K_TASK_NAME="${B1K_TASK_NAME:-tidying_bedroom}"
export B1K_CONDA_ENV="${B1K_CONDA_ENV:-behavior}"
export B1K_CHECKPOINT_NAME="${B1K_CHECKPOINT_NAME:-pi05-b1kpt50-cs32}"
```

Terminal 1 starts the OpenPI-COMET websocket policy server:

```bash
cd "$B1K_ROOT/openpi-comet"
export PATH="$HOME/.local/bin:$CONDA_DIR/bin:$PATH"
export UV_CACHE_DIR="$B1K_ROOT/.uv-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="$B1K_ROOT/.cache/jax"

uv run --no-sync scripts/serve_b1k.py \
  --task_name="$B1K_TASK_NAME" \
  --control_mode=receeding_horizon \
  --max_len=32 \
  policy:checkpoint \
  --policy.config=pi05_b1k-base \
  --policy.dir="./checkpoints/$B1K_CHECKPOINT_NAME"
```

Terminal 2 runs BEHAVIOR evaluation against that server:

```bash
cd "$B1K_ROOT/BEHAVIOR-1K"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$B1K_CONDA_ENV"

RUN_LOG="$B1K_ROOT/BEHAVIOR-1K/output/${B1K_TASK_NAME}_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_LOG"

export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH="$B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets"
export OMNIGIBSON_APPDATA_PATH="$B1K_ROOT/og-appdata"
export TMPDIR="$B1K_ROOT/tmp"

xvfb-run -a -s "-screen 0 1280x720x24" python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name="$B1K_TASK_NAME" \
  log_path="$RUN_LOG" \
  model.host=127.0.0.1 \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  eval_instance_ids="[0]" \
  write_video=true
```

Output videos are written under `$RUN_LOG/videos/`.

## Online A2C2 BEHAVIOR Evaluation

Online A2C2 uses the same BEHAVIOR websocket client command as the baseline
section. Replace only Terminal 1 with the A2C2 server below. The server keeps
the OpenPI-COMET base policy in the loop, caches each base action chunk and
base-policy latent, then runs the A2C2 residual head at every environment step
using the latest observation before returning `base_action + residual`.

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

Use `a2c2/ckpt/model_no_latent.pt` with `--a2c2-checkpoint` to run a checkpoint
that does not require `prefix_z`. The latent checkpoint requires the active
`openpi-comet` patches in `src/openpi/models/pi0.py` and
`src/openpi/policies/policy.py`.

The checked-in task18 checkpoints use state, base action chunk, selected base
action, time feature, and optionally `prefix_z`, so `RGBWrapper` is sufficient
for the BEHAVIOR command above. Future checkpoints trained with online RGB,
depth, camera-pose, task-info, language, or policy-timing features must be run
with matching online observations. `--allow-missing-online-features` exists only
for zero-filled smoke runs.

Fast online smoke test:

```bash
cd "$B1K_ROOT/openpi-comet"
uv run --no-sync python ../a2c2/scripts/test_online_eval.py
```

The smoke test uses a fake BEHAVIOR environment, a fake base policy, and a
deterministic A2C2 head. It verifies that a new base chunk is requested at the
correct steps, each residual sees the latest observation in the expected tensor
shapes, and the action sent to the environment is exactly `base_action +
residual`.

## Submodule Maintenance

When changing a submodule:

```bash
git -C openpi-comet switch dev/ljc-1222
git -C BEHAVIOR-1K switch dev/ljc-1222
```

Commit and push inside the submodule first. Then update the outer gitlink:

```bash
cd "$B1K_ROOT"
git add openpi-comet BEHAVIOR-1K .gitmodules setup.sh README.md
git commit -m "chore: update submodule pins"
```

Check submodule state:

```bash
git submodule status --recursive
git submodule foreach 'git status --short --branch'
```

The outer `b1k` repository uses `main` for these setup/documentation commits.
The nested forks use their `dev/ljc-1222` branches.

## Ignored Local Outputs

Do not commit:

```text
a2c2_dataset/
a2c2/runs/
openpi-comet/checkpoints/
BEHAVIOR-1K/OmniGibson/datasets/
setup_run.log
.cache/
.uv-cache/
.uv-python/
tmp/
miniconda3/
og-appdata/
```

Common cleanup:

```bash
rm -rf "$B1K_ROOT"/.cache "$B1K_ROOT"/.uv-cache
rm -rf "$B1K_ROOT"/a2c2/runs
rm -rf "$B1K_ROOT"/BEHAVIOR-1K/output
rm -f "$B1K_ROOT"/setup_run.log
```
