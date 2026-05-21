#!/usr/bin/env python3
"""Create a BEHAVIOR/LeRobot-style A2C2 dataset.

This script writes a new dataset root that keeps the original BEHAVIOR parquet
columns and layout, then appends A2C2-specific base-policy action chunks to each
frame. It intentionally lives at the workspace root so the source repositories do
not need to be modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos"
DEFAULT_OPENPI_ROOT = WORKSPACE_ROOT / "openpi-comet"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OPENPI_ROOT / "checkpoints/pi05-b1kpt12-cs32"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "a2c2_dataset/picking_up_trash_pi05-b1kpt12-cs32_h32_v1"

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

RGB_VIDEO_KEYS = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


@dataclasses.dataclass(frozen=True)
class TaskInfo:
    task_index: int
    task_name: str
    task_prompt: str
    raw_task_record: dict[str, Any]


class MockPolicy:
    """Fast deterministic policy used only for schema smoke tests."""

    def __init__(self, horizon: int, action_dim: int) -> None:
        self.horizon = horizon
        self.action_dim = action_dim

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        seed = int(np.nan_to_num(np.abs(state[:8]).sum() * 1_000_000)) % (2**32)
        rng = np.random.default_rng(seed)
        return {
            "actions": rng.normal(0.0, 0.01, size=(self.horizon, self.action_dim)).astype(np.float32),
        }


class EpisodeRgbReader:
    def __init__(self, source_root: Path, task_index: int, episode_index: int) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._paths: dict[str, Path] = {}
        for camera, video_key in RGB_VIDEO_KEYS.items():
            path = source_root / "videos" / f"task-{task_index:04d}" / video_key / f"episode_{episode_index:08d}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open RGB video: {path}")
            self._caps[camera] = cap
            self._paths[camera] = path

    def read(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera, cap in self._caps.items():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Unexpected end of video while reading {self._paths[camera]}")
            frames[camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()

    def __enter__(self) -> "EpisodeRgbReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-name", default="pi05_b1k-base")
    parser.add_argument("--task-name", default="picking_up_trash")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=23)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--cache-seed", type=int, default=0)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated absolute episode indices, e.g. 10010,10020. Defaults to all task episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Debug-only truncation. Produces a readable smoke-test dataset, not a full training dataset.",
    )
    parser.add_argument("--mock-policy", action="store_true", help="Use deterministic fake chunks for fast schema tests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def add_repo_paths(openpi_root: Path) -> None:
    sys.path.insert(0, str(openpi_root / "src"))
    sys.path.insert(0, str(openpi_root / "packages/openpi-client/src"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link_path, target_is_directory=target.is_dir())


def get_task_info(source_root: Path, task_name: str) -> TaskInfo:
    for record in load_jsonl(source_root / "meta/tasks.jsonl"):
        if record["task_name"] == task_name:
            return TaskInfo(
                task_index=int(record["task_index"]),
                task_name=record["task_name"],
                task_prompt=record["task"],
                raw_task_record=record,
            )
    raise ValueError(f"Task {task_name!r} not found in {source_root / 'meta/tasks.jsonl'}")


def parse_episode_filter(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    result: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            result.add(int(item))
    return result


def select_episodes(
    source_root: Path,
    task_index: int,
    episodes_filter: set[int] | None,
    max_episodes: int | None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(source_root / "meta/episodes.jsonl")
        if int(row["episode_index"]) // 10_000 == task_index
    ]
    if episodes_filter is not None:
        rows = [row for row in rows if int(row["episode_index"]) in episodes_filter]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("No episodes selected.")
    return rows


def prepare_output_root(args: argparse.Namespace) -> None:
    output_root = args.output_root
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif args.skip_existing:
            return
        else:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite or --skip-existing.")
    output_root.mkdir(parents=True, exist_ok=True)


def build_info_json(
    source_root: Path,
    selected_episodes: list[dict[str, Any]],
    task_info: TaskInfo,
    action_horizon: int,
    action_dim: int,
    max_frames_per_episode: int | None,
) -> dict[str, Any]:
    with (source_root / "meta/info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    lengths = [int(row["length"]) for row in selected_episodes]
    if max_frames_per_episode is not None:
        lengths = [min(length, max_frames_per_episode) for length in lengths]

    info["total_episodes"] = len(selected_episodes)
    info["total_frames"] = int(sum(lengths))
    info["total_tasks"] = 1
    info["total_videos"] = len(selected_episodes) * 9
    info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    info["a2c2"] = {
        "task_name": task_info.task_name,
        "task_index": task_info.task_index,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "base_action_chunk_column": "a2c2.base_action_chunk",
        "valid_action_mask_column": "a2c2.valid_action_mask",
        "policy_infer_ms_column": "a2c2.policy_infer_ms",
    }
    info["features"]["a2c2.base_action_chunk"] = {
        "dtype": "float32",
        "shape": [action_horizon, action_dim],
        "names": ["action_horizon", "action_dim"],
    }
    info["features"]["a2c2.valid_action_mask"] = {
        "dtype": "bool",
        "shape": [action_horizon],
        "names": ["action_horizon"],
    }
    info["features"]["a2c2.policy_infer_ms"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    return info


def prepare_metadata(
    args: argparse.Namespace,
    task_info: TaskInfo,
    selected_episodes: list[dict[str, Any]],
) -> None:
    meta_root = args.output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    task_root = args.output_root / "data" / f"task-{task_info.task_index:04d}"
    task_root.mkdir(parents=True, exist_ok=True)

    info = build_info_json(
        args.source_root,
        selected_episodes,
        task_info,
        args.action_horizon,
        args.action_dim,
        args.max_frames_per_episode,
    )
    with (meta_root / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    write_jsonl(meta_root / "tasks.jsonl", [task_info.raw_task_record])

    episodes = []
    for row in selected_episodes:
        row = dict(row)
        if args.max_frames_per_episode is not None:
            row["length"] = min(int(row["length"]), args.max_frames_per_episode)
        episodes.append(row)
    write_jsonl(meta_root / "episodes.jsonl", episodes)

    selected_indices = {int(row["episode_index"]) for row in selected_episodes}
    stats_rows = [
        row
        for row in load_jsonl(args.source_root / "meta/episodes_stats.jsonl")
        if int(row["episode_index"]) in selected_indices
    ]
    write_jsonl(meta_root / "episodes_stats.jsonl", stats_rows)

    if args.max_frames_per_episode is None:
        atomic_symlink(args.source_root / "meta/episodes", meta_root / "episodes")
    else:
        write_truncated_episode_meta(args, task_info, selected_episodes)

    atomic_symlink(args.source_root / "videos", args.output_root / "videos")
    atomic_symlink(args.source_root / "annotations", args.output_root / "annotations")


def write_truncated_episode_meta(
    args: argparse.Namespace,
    task_info: TaskInfo,
    selected_episodes: list[dict[str, Any]],
) -> None:
    dst_dir = args.output_root / "meta/episodes" / f"task-{task_info.task_index:04d}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for row in selected_episodes:
        ep_idx = int(row["episode_index"])
        src = args.source_root / "meta/episodes" / f"task-{task_info.task_index:04d}" / f"episode_{ep_idx:08d}.json"
        dst = dst_dir / src.name
        with src.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        length = min(int(row["length"]), args.max_frames_per_episode)
        meta["n_steps"] = length
        meta["num_samples"] = length
        with dst.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)


def load_policy(args: argparse.Namespace, task_prompt: str) -> Any:
    if args.mock_policy:
        return MockPolicy(args.action_horizon, args.action_dim)

    add_repo_paths(args.openpi_root)
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config_name)
    return policy_config.create_trained_policy(train_config, args.checkpoint_dir, default_prompt=task_prompt)


def get_policy_noise_dim(policy: Any, default_dim: int) -> int:
    model = getattr(policy, "_model", None)
    return int(getattr(model, "action_dim", default_dim))


def deterministic_noise(
    cache_seed: int,
    episode_index: int,
    local_index: int,
    action_horizon: int,
    model_action_dim: int,
) -> np.ndarray:
    seed_sequence = np.random.SeedSequence([cache_seed, episode_index, local_index])
    rng = np.random.default_rng(seed_sequence)
    return rng.standard_normal((action_horizon, model_action_dim)).astype(np.float32)


def fixed_size_chunk_array(values: np.ndarray, horizon: int, action_dim: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    inner = pa.FixedSizeListArray.from_arrays(flat, action_dim)
    return pa.FixedSizeListArray.from_arrays(inner, horizon)


def fixed_size_mask_array(values: np.ndarray, horizon: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.bool_).reshape(-1), type=pa.bool_())
    return pa.FixedSizeListArray.from_arrays(flat, horizon)


def ensure_original_columns(table: pa.Table, path: Path) -> None:
    missing = [col for col in REQUIRED_ORIGINAL_COLUMNS if col not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing required BEHAVIOR columns: {missing}")


def create_episode_table(
    args: argparse.Namespace,
    policy: Any,
    task_info: TaskInfo,
    episode: dict[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    ep_idx = int(episode["episode_index"])
    task_index = task_info.task_index
    src_path = args.source_root / "data" / f"task-{task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    table = pq.read_table(src_path)
    ensure_original_columns(table, src_path)

    if args.max_frames_per_episode is not None:
        table = table.slice(0, min(table.num_rows, args.max_frames_per_episode))

    num_rows = table.num_rows
    states = table.column("observation.state").combine_chunks()
    chunks = np.empty((num_rows, args.action_horizon, args.action_dim), dtype=np.float32)
    masks = np.zeros((num_rows, args.action_horizon), dtype=np.bool_)
    infer_ms = np.empty((num_rows,), dtype=np.float32)
    policy_noise_dim = get_policy_noise_dim(policy, args.model_action_dim)

    with EpisodeRgbReader(args.source_root, task_index, ep_idx) as reader:
        for local_idx in tqdm(range(num_rows), desc=f"episode_{ep_idx:08d}", leave=False):
            frames = reader.read()
            state = np.asarray(states[local_idx].as_py(), dtype=np.float32)
            obs = {
                "observation/egocentric_camera": frames["head"],
                "observation/wrist_image_left": frames["left_wrist"],
                "observation/wrist_image_right": frames["right_wrist"],
                "observation/state": state,
                "prompt": task_info.task_prompt,
            }
            noise = deterministic_noise(
                args.cache_seed,
                ep_idx,
                local_idx,
                args.action_horizon,
                policy_noise_dim,
            )
            start = time.monotonic()
            action_chunk = np.asarray(policy.infer(obs, noise=noise)["actions"], dtype=np.float32)
            infer_ms[local_idx] = np.float32((time.monotonic() - start) * 1000)
            if action_chunk.shape[0] < args.action_horizon or action_chunk.shape[1] < args.action_dim:
                raise ValueError(
                    f"Policy returned action chunk shape {action_chunk.shape}; "
                    f"expected at least ({args.action_horizon}, {args.action_dim})."
                )
            chunks[local_idx] = action_chunk[: args.action_horizon, : args.action_dim]
            valid_len = min(args.action_horizon, num_rows - local_idx)
            masks[local_idx, :valid_len] = True

    if not np.isfinite(chunks).all():
        raise ValueError(f"Non-finite values found in generated action chunks for episode {ep_idx}.")
    if not np.isfinite(infer_ms).all():
        raise ValueError(f"Non-finite policy timings found for episode {ep_idx}.")

    table = table.append_column(
        "a2c2.base_action_chunk",
        fixed_size_chunk_array(chunks, args.action_horizon, args.action_dim),
    )
    table = table.append_column("a2c2.valid_action_mask", fixed_size_mask_array(masks, args.action_horizon))
    table = table.append_column("a2c2.policy_infer_ms", pa.array(infer_ms, type=pa.float32()))

    summary = {
        "episode_index": ep_idx,
        "rows": num_rows,
        "base_action_chunk_shape": [args.action_horizon, args.action_dim],
        "mean_policy_infer_ms": float(np.mean(infer_ms)),
        "max_policy_infer_ms": float(np.max(infer_ms)),
    }
    return table, summary


def write_manifest(
    args: argparse.Namespace,
    task_info: TaskInfo,
    selected_episodes: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "openpi_root": str(args.openpi_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "config_name": args.config_name,
        "task_name": task_info.task_name,
        "task_index": task_info.task_index,
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
        "model_action_dim": args.model_action_dim,
        "cache_seed": args.cache_seed,
        "mock_policy": bool(args.mock_policy),
        "max_frames_per_episode": args.max_frames_per_episode,
        "num_episodes": len(selected_episodes),
        "num_frames": int(sum(summary["rows"] for summary in episode_summaries)),
        "columns_added": [
            "a2c2.base_action_chunk",
            "a2c2.valid_action_mask",
            "a2c2.policy_infer_ms",
        ],
        "episodes": episode_summaries,
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.openpi_root = args.openpi_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_root = args.output_root.resolve()

    task_info = get_task_info(args.source_root, args.task_name)
    selected_episodes = select_episodes(
        args.source_root,
        task_info.task_index,
        parse_episode_filter(args.episodes),
        args.max_episodes,
    )

    prepare_output_root(args)
    prepare_metadata(args, task_info, selected_episodes)
    policy = load_policy(args, task_info.task_prompt)

    episode_summaries: list[dict[str, Any]] = []
    task_data_dir = args.output_root / "data" / f"task-{task_info.task_index:04d}"
    for episode in tqdm(selected_episodes, desc="episodes"):
        ep_idx = int(episode["episode_index"])
        out_path = task_data_dir / f"episode_{ep_idx:08d}.parquet"
        if out_path.exists() and args.skip_existing:
            table = pq.read_table(out_path, columns=["index"])
            episode_summaries.append({"episode_index": ep_idx, "rows": table.num_rows, "skipped_existing": True})
            continue
        table, summary = create_episode_table(args, policy, task_info, episode)
        pq.write_table(table, out_path, compression=args.compression)
        episode_summaries.append(summary)

    write_manifest(args, task_info, selected_episodes, episode_summaries)
    print(f"Wrote A2C2 dataset to: {args.output_root}")


if __name__ == "__main__":
    main()
