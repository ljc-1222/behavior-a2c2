# A2C2 Dataset Commands

These commands create a BEHAVIOR/LeRobot-style dataset rooted under the current
workspace, without modifying `openpi-comet/` or `BEHAVIOR-1K/`.

## Environment

```bash
cd /root/b1k
export PATH="$HOME/.local/bin:/root/b1k/miniconda3/bin:$PATH"
export UV_CACHE_DIR="/root/b1k/.uv-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="/root/b1k/.cache/jax"
```

## Fast Schema Smoke Test

This uses a deterministic mock policy and only writes a few frames, so it checks
the dataset layout and parquet schema without loading the 12G checkpoint.

```bash
/root/b1k/openpi-comet/.venv/bin/python /root/b1k/a2c2_create_lerobot_dataset.py \
  --mock-policy \
  --max-episodes=1 \
  --max-frames-per-episode=8 \
  --output-root=/root/b1k/a2c2_dataset/smoke_picking_up_trash_h32_v1 \
  --overwrite

/root/b1k/openpi-comet/.venv/bin/python /root/b1k/a2c2_validate_dataset.py \
  --dataset-root=/root/b1k/a2c2_dataset/smoke_picking_up_trash_h32_v1 \
  --allow-truncated
```

## Full A2C2 Dataset Creation

This runs the real `pi05-b1kpt12-cs32` policy once per source frame and writes
one augmented parquet per episode.

```bash
/root/b1k/openpi-comet/.venv/bin/python /root/b1k/a2c2_create_lerobot_dataset.py \
  --source-root=/root/b1k/BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos \
  --openpi-root=/root/b1k/openpi-comet \
  --checkpoint-dir=/root/b1k/openpi-comet/checkpoints/pi05-b1kpt12-cs32 \
  --output-root=/root/b1k/a2c2_dataset/picking_up_trash_pi05-b1kpt12-cs32_h32_v1 \
  --config-name=pi05_b1k-base \
  --task-name=picking_up_trash \
  --cache-seed=42
```

Resume after an interrupted run:

```bash
/root/b1k/openpi-comet/.venv/bin/python /root/b1k/a2c2_create_lerobot_dataset.py \
  --output-root=/root/b1k/a2c2_dataset/picking_up_trash_pi05-b1kpt12-cs32_h32_v1 \
  --skip-existing
```

## Full Validation

```bash
/root/b1k/openpi-comet/.venv/bin/python /root/b1k/a2c2_validate_dataset.py \
  --dataset-root=/root/b1k/a2c2_dataset/picking_up_trash_pi05-b1kpt12-cs32_h32_v1 \
  --num-random-samples=100
```

The output dataset keeps the original columns:

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

and appends:

```text
a2c2.base_action_chunk  # fixed-size float32 [32, 23]
a2c2.valid_action_mask  # fixed-size bool [32]
a2c2.policy_infer_ms    # float32
```
