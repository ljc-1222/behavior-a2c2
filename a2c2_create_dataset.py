#!/usr/bin/env python3
"""Download or build the integrated A2C2 dataset.

By default this script downloads the published Hugging Face dataset so a user
only needs this one entrypoint to get the full A2C2 data locally.

For reproducibility, pass --build to recreate the dataset from BEHAVIOR-1K
source demos and an OpenPI/COMET checkpoint. The built dataset keeps the
BEHAVIOR/LeRobot parquet layout and appends A2C2 base-action fields. The COMET
z-layer is stored as a sidecar under <output-root>/latent so the large latent
vectors can be joined by row order without bloating the primary robot tables.
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
from typing import Any, Callable, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_REPO_ID = "ljc-1222/a2c2_dataset"
DEFAULT_DOWNLOAD_ROOT = WORKSPACE_ROOT / "a2c2_dataset"
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "BEHAVIOR-1K/OmniGibson/datasets/2025-challenge-demos"
DEFAULT_OPENPI_ROOT = WORKSPACE_ROOT / "openpi-comet"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OPENPI_ROOT / "checkpoints/pi05-b1kpt50-cs32"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1"
DEFAULT_CONFIG_NAME = "pi05_b1k-base"
DEFAULT_TASK_NAME = "tidying_bedroom"
DEFAULT_VARIANT_SUFFIX = "pi05-b1kpt50-cs32_h32_v1"
DEFAULT_LATENT_COLUMN = "a2c2.base_policy_z"
PROGRESS_DIRNAME = "a2c2_progress"

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

A2C2_MAIN_COLUMNS = (
    "a2c2.base_action_chunk",
    "a2c2.valid_action_mask",
    "a2c2.policy_infer_ms",
)

RGB_VIDEO_KEYS = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}

Z_SOURCE_DESCRIPTION = (
    "mask-pooled prefix_out from COMET/OpenPI PI0.5 PaliGemma prefix forward "
    "over image, prompt, and discrete state tokens"
)


def import_build_dependencies() -> None:
    global cv2, jax, jnp, np, pa, pq, tqdm
    try:
        import cv2 as cv2_module
        import jax as jax_module
        import jax.numpy as jnp_module
        import numpy as np_module
        import pyarrow as pa_module
        import pyarrow.parquet as pq_module
        from tqdm import tqdm as tqdm_function
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing a build dependency. Download mode only needs huggingface_hub, "
            "but --build requires cv2, jax, numpy, pyarrow, and tqdm in the OpenPI environment."
        ) from exc

    cv2 = cv2_module
    jax = jax_module
    jnp = jnp_module
    np = np_module
    pa = pa_module
    pq = pq_module
    tqdm = tqdm_function


@dataclasses.dataclass(frozen=True)
class TaskInfo:
    task_index: int
    task_name: str
    task_prompt: str
    raw_task_record: dict[str, Any]


class MockPolicy:
    """Fast deterministic policy used only for schema and resume checks."""

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
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild from BEHAVIOR-1K demos and OpenPI/COMET instead of downloading the published dataset.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID, help="Hugging Face dataset repo to download.")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Local directory for the downloaded dataset repo. Used unless --build is set.",
    )
    parser.add_argument("--revision", default=None, help="Hugging Face revision/branch/commit to download.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Public downloads do not need one.")
    parser.add_argument("--force-download", action="store_true", help="Redownload files even if cached locally.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only the local Hugging Face cache.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers for Hugging Face files.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to ./a2c2_dataset/<task-name>_pi05-b1kpt50-cs32_h32_v1.",
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--task-name",
        default=None,
        help=f"BEHAVIOR task name. Defaults to {DEFAULT_TASK_NAME!r} unless --task-index is set.",
    )
    parser.add_argument("--task-index", type=int, default=None, help="BEHAVIOR task index, e.g. 1 for task-0001.")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=23)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--latent-column", default=DEFAULT_LATENT_COLUMN)
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=8,
        help="Batch size for latent extraction and fused action+latent inference.",
    )
    parser.add_argument("--mock-latent-dim", type=int, default=2048)
    parser.add_argument("--cache-seed", type=int, default=42)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated absolute episode indices, e.g. 180020,180070. Defaults to all task episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Debug-only truncation. Produces a readable partial dataset, not a full training dataset.",
    )
    parser.add_argument("--mock-policy", action="store_true", help="Use deterministic fake chunks and latents.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output root before starting.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume by reusing complete per-episode main and latent files. This is the default.",
    )
    parser.add_argument("--skip-existing", dest="resume", action="store_true", help="Alias for --resume.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Fail if output root already exists.")
    parser.add_argument("--compression", default="snappy")
    args = parser.parse_args()
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.action_dim <= 0:
        raise ValueError("--action-dim must be positive.")
    if args.model_action_dim <= 0:
        raise ValueError("--model-action-dim must be positive.")
    if args.latent_batch_size <= 0:
        raise ValueError("--latent-batch-size must be positive.")
    if args.mock_latent_dim <= 0:
        raise ValueError("--mock-latent-dim must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if args.task_index is not None and args.task_index < 0:
        raise ValueError("--task-index must be non-negative.")
    if args.task_name is None and args.task_index is None:
        args.task_name = DEFAULT_TASK_NAME
    return args


def download_published_dataset(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it with "
            "`python -m pip install huggingface_hub`, or run from the project environment."
        ) from exc

    download_root = args.download_root.expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading A2C2 dataset from Hugging Face: {args.repo_id}")
    print(f"Destination: {download_root}")

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=download_root,
        token=args.token,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    dataset_root = Path(local_path).resolve()
    summarize_downloaded_dataset(dataset_root)
    return dataset_root


def summarize_downloaded_dataset(dataset_root: Path) -> None:
    info_paths = sorted(dataset_root.rglob("meta/info.json"))
    if not info_paths:
        print("Downloaded files, but no dataset root with meta/info.json was found.")
        return

    print("Downloaded dataset roots:")
    for info_path in info_paths:
        root = info_path.parent.parent
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        rel = root.relative_to(dataset_root).as_posix()
        if rel == ".":
            rel = dataset_root.name
        data_parquets = sum(1 for path in (root / "data").rglob("*.parquet") if path.is_file())
        latent_parquets = sum(1 for path in (root / "latent" / "data").rglob("*.parquet") if path.is_file())
        videos = sum(1 for path in (root / "videos").rglob("*.mp4") if path.is_file())
        annotations = sum(1 for path in (root / "annotations").rglob("*.json") if path.is_file())
        print(f"  - {rel}")
        print(f"    robot:       {info.get('robot_type', 'unknown')}")
        print(f"    episodes:    {info.get('total_episodes', 'unknown')}")
        print(f"    frames:      {info.get('total_frames', 'unknown')}")
        print(f"    data:        {data_parquets} parquet files")
        print(f"    latent:      {latent_parquets} parquet files")
        print(f"    videos:      {videos} mp4 files")
        print(f"    annotations: {annotations} json files")


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_parquet_atomic(table: pa.Table, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(table, tmp_path, compression=compression)
    os.replace(tmp_path, path)


def atomic_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link_path, target_is_directory=target.is_dir())


def sanitize_variant_component(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def default_output_root_for_task(task_name: str) -> Path:
    return WORKSPACE_ROOT / "a2c2_dataset" / f"{sanitize_variant_component(task_name)}_{DEFAULT_VARIANT_SUFFIX}"


def get_task_info(source_root: Path, task_name: str | None, task_index: int | None) -> TaskInfo:
    for record in load_jsonl(source_root / "meta/tasks.jsonl"):
        record_task_index = int(record["task_index"])
        if task_index is not None and record_task_index != task_index:
            continue
        if task_name is not None and record["task_name"] != task_name:
            continue
        if task_index is not None and task_name is not None:
            print(f"Selected task {record_task_index:04d}: {record['task_name']}")
        elif task_index is not None:
            print(f"Selected task {record_task_index:04d}: {record['task_name']} (--task-index {task_index})")
        elif task_name is not None:
            print(f"Selected task {record_task_index:04d}: {record['task_name']} (--task-name {task_name})")
        else:
            raise ValueError("Either task_name or task_index must be provided.")
        return TaskInfo(
            task_index=record_task_index,
            task_name=record["task_name"],
            task_prompt=record["task"],
            raw_task_record=record,
        )
    selector = []
    if task_index is not None:
        selector.append(f"task_index={task_index}")
    if task_name is not None:
        selector.append(f"task_name={task_name!r}")
    raise ValueError(f"Task ({', '.join(selector)}) not found in {source_root / 'meta/tasks.jsonl'}")


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


def expected_episode_rows(episode: dict[str, Any], max_frames_per_episode: int | None) -> int:
    length = int(episode["length"])
    if max_frames_per_episode is not None:
        return min(length, max_frames_per_episode)
    return length


def ensure_safe_output_root(source_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root == source_root:
        raise ValueError("--output-root must not be the BEHAVIOR source root.")
    if overwrite and source_root.is_relative_to(output_root):
        raise ValueError("--overwrite would remove the source root. Choose an output path outside the source tree.")


def prepare_output_root(args: argparse.Namespace) -> None:
    ensure_safe_output_root(args.source_root, args.output_root, args.overwrite)
    if args.output_root.exists():
        if args.overwrite:
            shutil.rmtree(args.output_root)
        elif not args.resume:
            raise FileExistsError(f"Output root already exists: {args.output_root}. Use --resume or --overwrite.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "data").mkdir(parents=True, exist_ok=True)
    (args.output_root / "latent" / "data").mkdir(parents=True, exist_ok=True)
    (args.output_root / "latent" / "meta").mkdir(parents=True, exist_ok=True)
    (args.output_root / "meta" / PROGRESS_DIRNAME).mkdir(parents=True, exist_ok=True)


def build_info_json(
    args: argparse.Namespace,
    selected_episodes: list[dict[str, Any]],
    task_info: TaskInfo,
) -> dict[str, Any]:
    with (args.source_root / "meta/info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    lengths = [expected_episode_rows(row, args.max_frames_per_episode) for row in selected_episodes]

    info["total_episodes"] = len(selected_episodes)
    info["total_frames"] = int(sum(lengths))
    info["total_tasks"] = 1
    info["total_videos"] = len(selected_episodes) * 9
    info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    info["a2c2"] = {
        "task_name": task_info.task_name,
        "task_index": task_info.task_index,
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
        "base_action_chunk_column": "a2c2.base_action_chunk",
        "valid_action_mask_column": "a2c2.valid_action_mask",
        "policy_infer_ms_column": "a2c2.policy_infer_ms",
        "latent_column": args.latent_column,
        "latent_root": "latent",
        "latent_storage": "latent/data/task-XXXX/episode_XXXXXXXX.parquet",
        "fused_action_latent_inference": True,
        "z_source": Z_SOURCE_DESCRIPTION,
    }
    info["features"]["a2c2.base_action_chunk"] = {
        "dtype": "float32",
        "shape": [args.action_horizon, args.action_dim],
        "names": ["action_horizon", "action_dim"],
    }
    info["features"]["a2c2.valid_action_mask"] = {
        "dtype": "bool",
        "shape": [args.action_horizon],
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
    (args.output_root / "data" / f"task-{task_info.task_index:04d}").mkdir(parents=True, exist_ok=True)
    (args.output_root / "latent" / "data" / f"task-{task_info.task_index:04d}").mkdir(parents=True, exist_ok=True)

    atomic_write_json(meta_root / "info.json", build_info_json(args, selected_episodes, task_info))
    write_jsonl(meta_root / "tasks.jsonl", [task_info.raw_task_record])

    episodes = []
    for row in selected_episodes:
        row = dict(row)
        row["length"] = expected_episode_rows(row, args.max_frames_per_episode)
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
        length = expected_episode_rows(row, args.max_frames_per_episode)
        meta["n_steps"] = length
        meta["num_samples"] = length
        atomic_write_json(dst, meta)


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


def fixed_size_latent_array(values: np.ndarray, latent_dim: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, latent_dim)


def latent_table(values: np.ndarray, latent_column: str) -> pa.Table:
    if values.ndim != 2:
        raise ValueError(f"Expected latent array with shape [num_rows, latent_dim], got {values.shape}")
    return pa.table({latent_column: fixed_size_latent_array(values, values.shape[1])})


def ensure_original_columns(table: pa.Table, path: Path) -> None:
    missing = [col for col in REQUIRED_ORIGINAL_COLUMNS if col not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing required BEHAVIOR columns: {missing}")


def main_episode_complete(path: Path, expected_rows: int, action_horizon: int, action_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        schema = pq.read_schema(path)
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows != expected_rows:
            return False
        for column in REQUIRED_ORIGINAL_COLUMNS + A2C2_MAIN_COLUMNS:
            if column not in schema.names:
                return False
        chunk_type = schema.field("a2c2.base_action_chunk").type
        mask_type = schema.field("a2c2.valid_action_mask").type
        if not pa.types.is_fixed_size_list(chunk_type) or chunk_type.list_size != action_horizon:
            return False
        if not pa.types.is_fixed_size_list(chunk_type.value_type) or chunk_type.value_type.list_size != action_dim:
            return False
        if not pa.types.is_fixed_size_list(mask_type) or mask_type.list_size != action_horizon:
            return False
        return pa.types.is_float32(schema.field("a2c2.policy_infer_ms").type)
    except Exception:
        return False


def latent_episode_complete(path: Path, expected_rows: int, latent_column: str) -> bool:
    if not path.exists():
        return False
    try:
        schema = pq.read_schema(path)
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows != expected_rows or latent_column not in schema.names:
            return False
        return pa.types.is_fixed_size_list(schema.field(latent_column).type)
    except Exception:
        return False


def latent_dim_from_file(path: Path, latent_column: str) -> int | None:
    field_type = pq.read_schema(path).field(latent_column).type
    if pa.types.is_fixed_size_list(field_type):
        return int(field_type.list_size)
    return None


def stack_transformed_inputs(items: list[dict[str, Any]]) -> dict[str, Any]:
    def stack_leaf(*values: Any) -> np.ndarray:
        return np.stack([np.asarray(value) for value in values], axis=0)

    return jax.tree_util.tree_map(stack_leaf, *items)


def raw_obs_to_model_inputs(policy: Any, raw_batch: list[dict[str, Any]]) -> dict[str, Any]:
    transformed = [policy._input_transform(obs) for obs in raw_batch]  # noqa: SLF001 - local dataset tool.
    return stack_transformed_inputs(transformed)


def raw_obs_to_model_observation(policy: Any, raw_batch: list[dict[str, Any]]) -> Any:
    from openpi.models import model as openpi_model

    stacked = raw_obs_to_model_inputs(policy, raw_batch)
    return openpi_model.Observation.from_dict(stacked)


def apply_policy_output_transform_batch(policy: Any, model_inputs: dict[str, Any], model_actions: np.ndarray) -> np.ndarray:
    transformed_actions = []
    states = np.asarray(model_inputs["state"])
    for idx in range(model_actions.shape[0]):
        outputs = {
            "state": states[idx],
            "actions": np.asarray(model_actions[idx]),
        }
        outputs = policy._output_transform(outputs)  # noqa: SLF001 - local dataset tool.
        transformed_actions.append(np.asarray(outputs["actions"], dtype=np.float32))
    return np.stack(transformed_actions, axis=0)


def build_mock_latent_runner(latent_dim: int) -> Callable[[list[dict[str, Any]]], np.ndarray]:
    def run(raw_batch: list[dict[str, Any]]) -> np.ndarray:
        latents = np.empty((len(raw_batch), latent_dim), dtype=np.float32)
        for idx, obs in enumerate(raw_batch):
            state = np.asarray(obs["observation/state"], dtype=np.float32)
            seed = int(np.nan_to_num(np.abs(state[:16]).sum() * 1_000_000)) % (2**32)
            rng = np.random.default_rng(seed)
            latents[idx] = rng.normal(0.0, 0.01, size=(latent_dim,)).astype(np.float32)
        return latents

    return run


def build_real_latent_runner(policy: Any) -> Callable[[list[dict[str, Any]]], np.ndarray]:
    if getattr(policy, "_is_pytorch_model", False):
        raise NotImplementedError(
            "This script currently targets the JAX/Orbax COMET checkpoint path. "
            "The local pi05-b1kpt50-cs32 checkpoint uses params/, not model.safetensors."
        )

    from flax import nnx
    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    model = getattr(policy, "_model")
    graphdef, state = nnx.split(model)

    def extract_pooled_z(model_state: nnx.State, observation: openpi_model.Observation) -> jax.Array:
        module = nnx.merge(graphdef, model_state)
        observation = openpi_model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = module.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        weights = prefix_mask.astype(jnp.float32)
        pooled = jnp.sum(prefix_out.astype(jnp.float32) * weights[..., None], axis=1)
        pooled = pooled / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)
        return pooled

    jitted_extract = jax.jit(extract_pooled_z)

    def run(raw_batch: list[dict[str, Any]]) -> np.ndarray:
        observation = raw_obs_to_model_observation(policy, raw_batch)
        return np.asarray(jitted_extract(state, observation), dtype=np.float32)

    return run


def build_latent_runner(args: argparse.Namespace, policy: Any) -> Callable[[list[dict[str, Any]]], np.ndarray]:
    if args.mock_policy:
        return build_mock_latent_runner(args.mock_latent_dim)
    return build_real_latent_runner(policy)


def build_mock_fused_runner(
    args: argparse.Namespace,
    policy: Any,
) -> Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    latent_runner = build_mock_latent_runner(args.mock_latent_dim)

    def run(raw_batch: list[dict[str, Any]], _noise_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = time.monotonic()
        actions = np.stack(
            [np.asarray(policy.infer(obs)["actions"], dtype=np.float32) for obs in raw_batch],
            axis=0,
        )
        latents = latent_runner(raw_batch)
        per_row_ms = np.full((len(raw_batch),), (time.monotonic() - start) * 1000 / len(raw_batch), dtype=np.float32)
        return actions, latents, per_row_ms

    return run


def build_real_fused_runner(
    policy: Any,
) -> Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if getattr(policy, "_is_pytorch_model", False):
        raise NotImplementedError(
            "Fused action+latent inference currently targets the JAX/Orbax COMET checkpoint path. "
            "The local pi05-b1kpt50-cs32 checkpoint uses params/, not model.safetensors."
        )

    from flax import nnx
    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    sample_kwargs = dict(getattr(policy, "_sample_kwargs", {}))
    num_steps = int(sample_kwargs.pop("num_steps", 10))
    if sample_kwargs:
        raise NotImplementedError(f"Unsupported fused sample kwargs: {sorted(sample_kwargs)}")

    model = getattr(policy, "_model")
    graphdef, state = nnx.split(model)

    def run_model(
        model_state: nnx.State,
        observation: openpi_model.Observation,
        noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        module = nnx.merge(graphdef, model_state)
        observation = openpi_model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps

        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = module.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        weights = prefix_mask.astype(jnp.float32)
        pooled_z = jnp.sum(prefix_out.astype(jnp.float32) * weights[..., None], axis=1)
        pooled_z = pooled_z / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)

        def step(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            x_t, time_value = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = module.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time_value, batch_size),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1]),
            )
            full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_prefix_out, suffix_out), _ = module.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = module.action_out_proj(suffix_out[:, -module.action_horizon :])
            return x_t + dt * v_t, time_value + dt

        def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            _x_t, time_value = carry
            return time_value >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0, dtype=noise.dtype)))
        return actions, pooled_z

    jitted_run_model = jax.jit(run_model)

    def run(raw_batch: list[dict[str, Any]], noise_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_inputs = raw_obs_to_model_inputs(policy, raw_batch)
        model_inputs_jax = jax.tree_util.tree_map(jnp.asarray, model_inputs)
        observation = openpi_model.Observation.from_dict(model_inputs_jax)

        start = time.monotonic()
        model_actions, latents = jitted_run_model(state, observation, jnp.asarray(noise_batch))
        model_actions_np = np.asarray(model_actions, dtype=np.float32)
        latents_np = np.asarray(latents, dtype=np.float32)
        elapsed_ms = (time.monotonic() - start) * 1000

        actions = apply_policy_output_transform_batch(policy, model_inputs, model_actions_np)
        per_row_ms = np.full((len(raw_batch),), elapsed_ms / len(raw_batch), dtype=np.float32)
        return actions, latents_np, per_row_ms

    return run


def build_fused_runner(
    args: argparse.Namespace,
    policy: Any,
) -> Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if args.mock_policy:
        return build_mock_fused_runner(args, policy)
    return build_real_fused_runner(policy)


def flush_latent_batch(
    raw_batch: list[dict[str, Any]],
    latent_batches: list[np.ndarray],
    latent_runner: Callable[[list[dict[str, Any]]], np.ndarray],
) -> None:
    if not raw_batch:
        return
    latent_batches.append(latent_runner(raw_batch))
    raw_batch.clear()


def flush_fused_batch(
    raw_batch: list[dict[str, Any]],
    noise_batch: list[np.ndarray],
    local_indices: list[int],
    chunks: np.ndarray,
    infer_ms: np.ndarray,
    latent_batches: list[np.ndarray],
    fused_runner: Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    action_horizon: int,
    action_dim: int,
) -> None:
    if not raw_batch:
        return

    actions, latents, batch_infer_ms = fused_runner(raw_batch, np.stack(noise_batch, axis=0))
    if actions.shape[0] != len(raw_batch) or latents.shape[0] != len(raw_batch):
        raise ValueError(
            f"Fused runner returned {actions.shape[0]} action rows and {latents.shape[0]} latent rows "
            f"for {len(raw_batch)} inputs."
        )
    if actions.ndim != 3 or actions.shape[1] < action_horizon or actions.shape[2] < action_dim:
        raise ValueError(
            f"Fused runner returned action shape {actions.shape}; "
            f"expected at least ({len(raw_batch)}, {action_horizon}, {action_dim})."
        )
    if batch_infer_ms.shape[0] != len(raw_batch):
        raise ValueError(f"Fused runner returned timing shape {batch_infer_ms.shape}; expected ({len(raw_batch)},).")

    for batch_idx, local_idx in enumerate(local_indices):
        chunks[local_idx] = actions[batch_idx, :action_horizon, :action_dim]
        infer_ms[local_idx] = batch_infer_ms[batch_idx]
    latent_batches.append(np.asarray(latents, dtype=np.float32))

    raw_batch.clear()
    noise_batch.clear()
    local_indices.clear()


def create_episode_artifacts(
    args: argparse.Namespace,
    policy: Any,
    latent_runner: Callable[[list[dict[str, Any]]], np.ndarray],
    fused_runner: Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    task_info: TaskInfo,
    episode: dict[str, Any],
    need_main: bool,
    need_latent: bool,
) -> tuple[pa.Table | None, pa.Table | None, dict[str, Any]]:
    ep_idx = int(episode["episode_index"])
    task_index = task_info.task_index
    src_path = args.source_root / "data" / f"task-{task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    table = pq.read_table(src_path)
    ensure_original_columns(table, src_path)

    expected_rows = expected_episode_rows(episode, args.max_frames_per_episode)
    if table.num_rows < expected_rows:
        raise ValueError(f"{src_path} has {table.num_rows} rows, expected at least {expected_rows}.")
    if table.num_rows != expected_rows:
        table = table.slice(0, expected_rows)

    num_rows = table.num_rows
    states = table.column("observation.state").combine_chunks()
    chunks = np.empty((num_rows, args.action_horizon, args.action_dim), dtype=np.float32) if need_main else None
    masks = np.zeros((num_rows, args.action_horizon), dtype=np.bool_) if need_main else None
    infer_ms = np.empty((num_rows,), dtype=np.float32) if need_main else None
    policy_noise_dim = get_policy_noise_dim(policy, args.model_action_dim) if need_main else args.model_action_dim
    raw_batch: list[dict[str, Any]] = []
    fused_noise_batch: list[np.ndarray] = []
    fused_indices: list[int] = []
    latent_batches: list[np.ndarray] = []
    used_fused_inference = bool(need_main and need_latent and fused_runner is not None)

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

            if need_main:
                assert masks is not None
                noise = deterministic_noise(
                    args.cache_seed,
                    ep_idx,
                    local_idx,
                    args.action_horizon,
                    policy_noise_dim,
                )
                valid_len = min(args.action_horizon, num_rows - local_idx)
                masks[local_idx, :valid_len] = True

                if used_fused_inference:
                    raw_batch.append(obs)
                    fused_noise_batch.append(noise)
                    fused_indices.append(local_idx)
                    if len(raw_batch) == args.latent_batch_size:
                        assert chunks is not None and infer_ms is not None and fused_runner is not None
                        flush_fused_batch(
                            raw_batch,
                            fused_noise_batch,
                            fused_indices,
                            chunks,
                            infer_ms,
                            latent_batches,
                            fused_runner,
                            args.action_horizon,
                            args.action_dim,
                        )
                else:
                    start = time.monotonic()
                    action_chunk = np.asarray(policy.infer(obs, noise=noise)["actions"], dtype=np.float32)
                    assert chunks is not None and infer_ms is not None
                    infer_ms[local_idx] = np.float32((time.monotonic() - start) * 1000)
                    if action_chunk.shape[0] < args.action_horizon or action_chunk.shape[1] < args.action_dim:
                        raise ValueError(
                            f"Policy returned action chunk shape {action_chunk.shape}; "
                            f"expected at least ({args.action_horizon}, {args.action_dim})."
                        )
                    chunks[local_idx] = action_chunk[: args.action_horizon, : args.action_dim]

            if need_latent and not used_fused_inference:
                raw_batch.append(obs)
                if len(raw_batch) == args.latent_batch_size:
                    flush_latent_batch(raw_batch, latent_batches, latent_runner)

        if used_fused_inference:
            assert chunks is not None and infer_ms is not None and fused_runner is not None
            flush_fused_batch(
                raw_batch,
                fused_noise_batch,
                fused_indices,
                chunks,
                infer_ms,
                latent_batches,
                fused_runner,
                args.action_horizon,
                args.action_dim,
            )
        elif need_latent:
            flush_latent_batch(raw_batch, latent_batches, latent_runner)

    main_table: pa.Table | None = None
    if need_main:
        assert chunks is not None and masks is not None and infer_ms is not None
        if not np.isfinite(chunks).all():
            raise ValueError(f"Non-finite values found in generated action chunks for episode {ep_idx}.")
        if not np.isfinite(infer_ms).all():
            raise ValueError(f"Non-finite policy timings found for episode {ep_idx}.")
        main_table = table.append_column(
            "a2c2.base_action_chunk",
            fixed_size_chunk_array(chunks, args.action_horizon, args.action_dim),
        )
        main_table = main_table.append_column("a2c2.valid_action_mask", fixed_size_mask_array(masks, args.action_horizon))
        main_table = main_table.append_column("a2c2.policy_infer_ms", pa.array(infer_ms, type=pa.float32()))

    latent_sidecar: pa.Table | None = None
    latent_dim: int | None = None
    if need_latent:
        latents = np.concatenate(latent_batches, axis=0) if latent_batches else np.empty((0, 0), dtype=np.float32)
        if latents.shape[0] != num_rows:
            raise ValueError(f"Latent row count {latents.shape[0]} does not match episode rows {num_rows}.")
        if not np.isfinite(latents).all():
            raise ValueError(f"Non-finite latent values found for episode {ep_idx}.")
        latent_dim = int(latents.shape[1])
        latent_sidecar = latent_table(latents, args.latent_column)

    summary = {
        "episode_index": ep_idx,
        "rows": int(num_rows),
        "main_generated": bool(need_main),
        "latent_generated": bool(need_latent),
        "fused_action_latent_inference": used_fused_inference,
        "base_action_chunk_shape": [args.action_horizon, args.action_dim],
    }
    if infer_ms is not None:
        summary["mean_policy_infer_ms"] = float(np.mean(infer_ms))
        summary["max_policy_infer_ms"] = float(np.max(infer_ms))
    if latent_dim is not None:
        summary["latent_dim"] = latent_dim
    return main_table, latent_sidecar, summary


def progress_marker_path(args: argparse.Namespace, episode_index: int) -> Path:
    return args.output_root / "meta" / PROGRESS_DIRNAME / f"episode_{episode_index:08d}.json"


def episode_needs_work(args: argparse.Namespace, task_info: TaskInfo, episode: dict[str, Any]) -> bool:
    ep_idx = int(episode["episode_index"])
    expected_rows = expected_episode_rows(episode, args.max_frames_per_episode)
    main_path = args.output_root / "data" / f"task-{task_info.task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    latent_path = args.output_root / "latent" / "data" / f"task-{task_info.task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    return not (
        main_episode_complete(main_path, expected_rows, args.action_horizon, args.action_dim)
        and latent_episode_complete(latent_path, expected_rows, args.latent_column)
    )


def noop_latent_runner(_raw_batch: list[dict[str, Any]]) -> np.ndarray:
    raise RuntimeError("No latent runner was created because all selected episode outputs already exist.")


def process_episode(
    args: argparse.Namespace,
    policy: Any,
    latent_runner: Callable[[list[dict[str, Any]]], np.ndarray],
    fused_runner: Callable[[list[dict[str, Any]], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    task_info: TaskInfo,
    episode: dict[str, Any],
) -> dict[str, Any]:
    ep_idx = int(episode["episode_index"])
    expected_rows = expected_episode_rows(episode, args.max_frames_per_episode)
    main_path = args.output_root / "data" / f"task-{task_info.task_index:04d}" / f"episode_{ep_idx:08d}.parquet"
    latent_path = args.output_root / "latent" / "data" / f"task-{task_info.task_index:04d}" / f"episode_{ep_idx:08d}.parquet"

    main_done = main_episode_complete(main_path, expected_rows, args.action_horizon, args.action_dim)
    latent_done = latent_episode_complete(latent_path, expected_rows, args.latent_column)
    need_main = not main_done
    need_latent = not latent_done

    if not args.resume and (main_path.exists() or latent_path.exists()):
        raise FileExistsError(f"Episode output already exists for {ep_idx}; use --resume or --overwrite.")

    if not need_main and not need_latent:
        summary = {
            "episode_index": ep_idx,
            "rows": expected_rows,
            "main_path": str(main_path),
            "latent_path": str(latent_path),
            "latent_dim": latent_dim_from_file(latent_path, args.latent_column),
            "skipped_existing": True,
        }
        atomic_write_json(progress_marker_path(args, ep_idx), summary)
        return summary

    main_table, latent_sidecar, summary = create_episode_artifacts(
        args,
        policy,
        latent_runner,
        fused_runner,
        task_info,
        episode,
        need_main,
        need_latent,
    )

    if main_table is not None:
        write_parquet_atomic(main_table, main_path, args.compression)
    if latent_sidecar is not None:
        write_parquet_atomic(latent_sidecar, latent_path, args.compression)
        summary["latent_dim"] = latent_dim_from_file(latent_path, args.latent_column)

    summary["main_path"] = str(main_path)
    summary["latent_path"] = str(latent_path)
    summary["main_skipped_existing"] = not need_main
    summary["latent_skipped_existing"] = not need_latent
    atomic_write_json(progress_marker_path(args, ep_idx), summary)
    return summary


def write_manifests(
    args: argparse.Namespace,
    task_info: TaskInfo,
    selected_episodes: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
) -> None:
    latent_dims = sorted({summary.get("latent_dim") for summary in episode_summaries if summary.get("latent_dim")})
    total_frames = int(sum(int(summary["rows"]) for summary in episode_summaries))
    root_manifest = {
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
        "resume_enabled": bool(args.resume),
        "num_episodes": len(selected_episodes),
        "num_frames": total_frames,
        "columns_added_to_main_parquet": list(A2C2_MAIN_COLUMNS),
        "latent_root": str(args.output_root / "latent"),
        "latent_column": args.latent_column,
        "latent_dtype": "float32",
        "latent_dim": latent_dims[0] if len(latent_dims) == 1 else None,
        "latent_storage_layout": "one parquet per episode under latent/data/task-XXXX; each file has only the z column",
        "row_alignment": "Main parquet rows and latent parquet rows match by episode and row order.",
        "z_source": Z_SOURCE_DESCRIPTION,
        "full_action_inference_run": True,
        "fused_action_latent_inference": True,
        "episodes": episode_summaries,
    }
    atomic_write_json(args.output_root / "manifest.json", root_manifest)

    latent_manifest = {
        "created_at_utc": root_manifest["created_at_utc"],
        "dataset_root": str(args.output_root),
        "latent_root": str(args.output_root / "latent"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "config_name": args.config_name,
        "task_name": task_info.task_name,
        "task_index": task_info.task_index,
        "latent_column": args.latent_column,
        "latent_dtype": "float32",
        "latent_dim": root_manifest["latent_dim"],
        "storage_layout": root_manifest["latent_storage_layout"],
        "row_alignment": root_manifest["row_alignment"],
        "z_source": Z_SOURCE_DESCRIPTION,
        "episodes": episode_summaries,
    }
    atomic_write_json(args.output_root / "latent" / "meta" / "manifest.json", latent_manifest)


def main() -> None:
    args = parse_args()
    if not args.build:
        dataset_root = download_published_dataset(args)
        print(f"A2C2 dataset is ready at: {dataset_root}")
        return

    import_build_dependencies()

    args.source_root = args.source_root.resolve()
    args.openpi_root = args.openpi_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()

    task_info = get_task_info(args.source_root, args.task_name, args.task_index)
    if args.output_root is None:
        args.output_root = default_output_root_for_task(task_info.task_name)
    args.output_root = args.output_root.resolve()

    selected_episodes = select_episodes(
        args.source_root,
        task_info.task_index,
        parse_episode_filter(args.episodes),
        args.max_episodes,
    )

    prepare_output_root(args)
    prepare_metadata(args, task_info, selected_episodes)

    if any(episode_needs_work(args, task_info, episode) for episode in selected_episodes):
        policy = load_policy(args, task_info.task_prompt)
        latent_runner = build_latent_runner(args, policy)
        fused_runner = build_fused_runner(args, policy)
    else:
        policy = None
        latent_runner = noop_latent_runner
        fused_runner = None

    episode_summaries: list[dict[str, Any]] = []
    for episode in tqdm(selected_episodes, desc="episodes"):
        episode_summaries.append(process_episode(args, policy, latent_runner, fused_runner, task_info, episode))

    write_manifests(args, task_info, selected_episodes, episode_summaries)
    print(f"Wrote integrated A2C2 dataset to: {args.output_root}")
    print(f"Wrote z-layer sidecars to: {args.output_root / 'latent'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - command-line tool should fail with a concise message.
        raise SystemExit(f"ERROR: {exc}") from exc
