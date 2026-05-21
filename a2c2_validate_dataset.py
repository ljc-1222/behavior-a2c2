#!/usr/bin/env python3
"""Validate a BEHAVIOR/LeRobot-style A2C2 dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "a2c2_dataset/picking_up_trash_pi05-b1kpt12-cs32_h32_v1"
DEFAULT_OPENPI_ROOT = WORKSPACE_ROOT / "openpi-comet"

REQUIRED_ORIGINAL_COLUMNS = (
    "index",
    "episode_index",
    "task_index",
    "timestamp",
    "observation.state",
    "observation.cam_rel_poses",
    "observation.task_info",
    "action",
)

A2C2_COLUMNS = (
    "a2c2.base_action_chunk",
    "a2c2.valid_action_mask",
    "a2c2.policy_infer_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=23)
    parser.add_argument("--num-random-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--skip-full-numeric-scan",
        action="store_true",
        help="Skip per-episode all-value finite checks for faster validation.",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Allow parquet row counts to be shorter than meta/episodes.jsonl lengths.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def fixed_or_variable_list_to_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        outer_size = array.type.list_size
        inner = array.values
        if pa.types.is_fixed_size_list(inner.type):
            inner_size = inner.type.list_size
            flat = inner.values.to_numpy(zero_copy_only=False)
            return np.asarray(flat, dtype=dtype).reshape(len(array), outer_size, inner_size)
        flat = inner.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=dtype).reshape(len(array), outer_size)
    return np.asarray(array.to_pylist(), dtype=dtype)


def validate_layout(root: Path, task_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(root.exists(), f"Dataset root does not exist: {root}")
    require((root / "data" / f"task-{task_index:04d}").is_dir(), "Missing task data directory.")
    require((root / "meta/info.json").is_file(), "Missing meta/info.json.")
    require((root / "meta/tasks.jsonl").is_file(), "Missing meta/tasks.jsonl.")
    require((root / "meta/episodes.jsonl").is_file(), "Missing meta/episodes.jsonl.")
    require((root / "meta/episodes_stats.jsonl").is_file(), "Missing meta/episodes_stats.jsonl.")
    require((root / "meta/episodes").exists(), "Missing meta/episodes path.")
    require((root / "videos").exists(), "Missing videos path.")
    require((root / "annotations").exists(), "Missing annotations path.")

    with (root / "meta/info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    for column in A2C2_COLUMNS:
        require(column in info["features"], f"meta/info.json missing feature {column}")

    episodes = [
        row for row in load_jsonl(root / "meta/episodes.jsonl") if int(row["episode_index"]) // 10_000 == task_index
    ]
    require(bool(episodes), f"No task-{task_index:04d} episodes found in meta/episodes.jsonl")
    return episodes, info


def validate_parquet_schema(path: Path, action_horizon: int, action_dim: int) -> pa.Table:
    table = pq.read_table(path)
    for column in REQUIRED_ORIGINAL_COLUMNS + A2C2_COLUMNS:
        require(column in table.column_names, f"{path} missing column {column}")

    chunk_type = table.schema.field("a2c2.base_action_chunk").type
    mask_type = table.schema.field("a2c2.valid_action_mask").type
    require(pa.types.is_fixed_size_list(chunk_type), "a2c2.base_action_chunk should be a fixed-size list.")
    require(chunk_type.list_size == action_horizon, f"base chunk horizon should be {action_horizon}.")
    require(pa.types.is_fixed_size_list(chunk_type.value_type), "base chunk inner action dim should be fixed-size.")
    require(chunk_type.value_type.list_size == action_dim, f"base chunk action dim should be {action_dim}.")
    require(pa.types.is_float32(chunk_type.value_type.value_type), "base chunk values should be float32.")
    require(pa.types.is_fixed_size_list(mask_type), "a2c2.valid_action_mask should be a fixed-size list.")
    require(mask_type.list_size == action_horizon, f"mask horizon should be {action_horizon}.")
    require(pa.types.is_boolean(mask_type.value_type), "valid mask values should be bool.")
    require(pa.types.is_float32(table.schema.field("a2c2.policy_infer_ms").type), "policy_infer_ms should be float32.")
    return table


def validate_hf_read(path: Path, openpi_root: Path, action_horizon: int, action_dim: int) -> None:
    dataset = load_dataset("parquet", data_files=[str(path)], split="train")
    row = dataset[0]
    require(len(row["action"]) == action_dim, "HuggingFace read returned unexpected action shape.")
    require(len(row["a2c2.base_action_chunk"]) == action_horizon, "HuggingFace read returned unexpected chunk horizon.")
    require(len(row["a2c2.base_action_chunk"][0]) == action_dim, "HuggingFace read returned unexpected action dim.")

    sys.path.insert(0, str(openpi_root / "src"))
    sys.path.insert(0, str(WORKSPACE_ROOT / "BEHAVIOR-1K/OmniGibson"))
    try:
        from omnigibson.learning.utils.lerobot_utils import hf_transform_to_torch

        transformed = hf_transform_to_torch({key: [value] for key, value in row.items()})
    except Exception as exc:
        print(f"hf_transform_to_torch import/use skipped: {exc}")
        return

    chunk = transformed["a2c2.base_action_chunk"][0]
    action = transformed["action"][0]
    require(tuple(chunk.shape) == (action_horizon, action_dim), "hf_transform_to_torch chunk shape mismatch.")
    require(tuple(action.shape) == (action_dim,), "hf_transform_to_torch action shape mismatch.")


def validate_episode(
    root: Path,
    task_index: int,
    episode: dict[str, Any],
    action_horizon: int,
    action_dim: int,
    rng: random.Random,
    samples_per_episode: int,
    full_numeric_scan: bool,
    allow_truncated: bool,
) -> dict[str, Any]:
    ep_idx = int(episode["episode_index"])
    path = root / "data" / f"task-{task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    require(path.is_file(), f"Missing episode parquet: {path}")
    table = validate_parquet_schema(path, action_horizon, action_dim)
    expected_len = int(episode["length"])
    if allow_truncated:
        require(table.num_rows <= expected_len, f"{path} has more rows than metadata length.")
    else:
        require(table.num_rows == expected_len, f"{path} row count {table.num_rows} != metadata length {expected_len}.")

    first_video = (
        root
        / "videos"
        / f"task-{task_index:04d}"
        / "observation.images.rgb.head"
        / f"episode_{ep_idx:08d}.mp4"
    )
    require(first_video.exists(), f"Missing linked video: {first_video}")

    chunks = fixed_or_variable_list_to_numpy(table.column("a2c2.base_action_chunk"), np.float32)
    masks = fixed_or_variable_list_to_numpy(table.column("a2c2.valid_action_mask"), np.bool_)
    actions = fixed_or_variable_list_to_numpy(table.column("action"), np.float32)
    timestamps = table.column("timestamp").combine_chunks().to_numpy(zero_copy_only=False)

    require(chunks.shape == (table.num_rows, action_horizon, action_dim), f"Chunk shape mismatch in {path}")
    require(masks.shape == (table.num_rows, action_horizon), f"Mask shape mismatch in {path}")
    require(actions.shape == (table.num_rows, action_dim), f"Action shape mismatch in {path}")

    if full_numeric_scan:
        require(np.isfinite(chunks).all(), f"Non-finite base chunks in {path}")
        require(np.isfinite(actions).all(), f"Non-finite expert actions in {path}")
        require(np.isfinite(timestamps).all(), f"Non-finite timestamps in {path}")

    if table.num_rows > 0 and samples_per_episode > 0:
        for _ in range(samples_per_episode):
            source_idx = rng.randrange(table.num_rows)
            offset = rng.randrange(action_horizon)
            expected_valid = source_idx + offset < table.num_rows
            require(
                bool(masks[source_idx, offset]) == expected_valid,
                f"Mask mismatch for episode {ep_idx}, source {source_idx}, offset {offset}",
            )
            base = chunks[source_idx, offset]
            require(base.shape == (action_dim,), "Base action shape mismatch.")
            if expected_valid:
                expert = actions[source_idx + offset]
                require(expert.shape == (action_dim,), "Expert action shape mismatch.")
                require(timestamps[source_idx + offset] >= timestamps[source_idx], "Timestamp alignment is not monotonic.")

    return {
        "episode_index": ep_idx,
        "rows": table.num_rows,
        "chunk_shape": list(chunks.shape[1:]),
        "mean_abs_base_action": float(np.mean(np.abs(chunks))),
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    openpi_root = args.openpi_root.resolve()
    episodes, info = validate_layout(root, args.task_index)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_path = root / "data" / f"task-{args.task_index:04d}" / f"episode_{int(episodes[0]['episode_index']):08d}.parquet"
    validate_hf_read(first_path, openpi_root, args.action_horizon, args.action_dim)

    rng = random.Random(args.seed)
    per_episode_samples = max(1, args.num_random_samples // max(1, len(episodes)))
    summaries = []
    for episode in tqdm(episodes, desc="validating episodes"):
        summaries.append(
            validate_episode(
                root,
                args.task_index,
                episode,
                args.action_horizon,
                args.action_dim,
                rng,
                per_episode_samples,
                full_numeric_scan=not args.skip_full_numeric_scan,
                allow_truncated=args.allow_truncated,
            )
        )

    expected_frames = int(info["total_frames"])
    actual_frames = sum(summary["rows"] for summary in summaries)
    if args.max_episodes is None and not args.allow_truncated:
        require(actual_frames == expected_frames, f"Total rows {actual_frames} != info total_frames {expected_frames}.")

    print(
        json.dumps(
            {
                "dataset_root": str(root),
                "episodes_validated": len(summaries),
                "frames_validated": actual_frames,
                "first_episode": summaries[0],
            },
            indent=4,
        )
    )
    print("A2C2 dataset validation passed.")


if __name__ == "__main__":
    main()
