"""Dataset utilities for online A2C2 BEHAVIOR/OpenPI parquet/video exports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterator, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


RGB_VIDEO_COLUMNS = (
    "observation.images.rgb.head",
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
)
DEPTH_VIDEO_COLUMNS = (
    "observation.images.depth.head",
    "observation.images.depth.left_wrist",
    "observation.images.depth.right_wrist",
)
DEPTH_CAMERA_INTRINSICS: dict[str, tuple[np.ndarray, tuple[int, int]]] = {
    "observation.images.depth.head": (
        np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (720, 720),
    ),
    "observation.images.depth.left_wrist": (
        np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (480, 480),
    ),
    "observation.images.depth.right_wrist": (
        np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (480, 480),
    ),
}
DEFAULT_DEPTH_MAX_M = 10.0
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
PAD_TOKEN_ID = 0
EMPTY_TOKEN_ID = 1


class EpisodePair(NamedTuple):
    data_path: Path
    latent_path: Path


def resolve_dataset_root(path: Path) -> Path:
    """Resolve either an A2C2 root or its parent directory."""

    path = path.expanduser().resolve()
    if (path / "data").is_dir() and (path / "latent" / "data").is_dir():
        return path

    candidates = [p for p in path.iterdir() if (p / "data").is_dir() and (p / "latent" / "data").is_dir()]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No A2C2 dataset root found under {path}")
    names = ", ".join(str(p) for p in candidates)
    raise ValueError(f"Multiple dataset roots found under {path}; pass one explicitly: {names}")


def discover_episode_pairs(dataset_root: Path, task_dir: str | None = None) -> list[EpisodePair]:
    """Find matching data/latent parquet pairs."""

    dataset_root = resolve_dataset_root(dataset_root)
    pattern = f"{task_dir}/episode_*.parquet" if task_dir else "task-*/episode_*.parquet"
    data_paths = sorted((dataset_root / "data").glob(pattern))
    pairs: list[EpisodePair] = []
    for data_path in data_paths:
        rel = data_path.relative_to(dataset_root / "data")
        latent_path = dataset_root / "latent" / "data" / rel
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing latent parquet for {data_path}: {latent_path}")
        pairs.append(EpisodePair(data_path=data_path, latent_path=latent_path))
    if not pairs:
        raise FileNotFoundError(f"No episode parquet files found in {dataset_root / 'data'}")
    return pairs


def split_episode_pairs(
    pairs: list[EpisodePair],
    val_ratio: float,
    seed: int,
    max_episodes: int | None = None,
) -> tuple[list[EpisodePair], list[EpisodePair]]:
    """Shuffle episode pairs and split into train/validation subsets."""

    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_episodes is not None:
        pairs = pairs[:max_episodes]
    val_count = int(round(len(pairs) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(pairs) > 1:
        val_count = 1
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    if not train_pairs:
        raise ValueError("No training episodes left after split.")
    return train_pairs, val_pairs


def dataset_root_from_pair(pair: EpisodePair) -> Path:
    return pair.data_path.parents[2]


def task_dir_from_pair(pair: EpisodePair) -> str:
    return pair.data_path.parent.name


def resolve_language_instruction(dataset_root: Path, task_dir: str | None = None) -> str:
    """Resolve the natural-language task instruction from dataset metadata."""

    dataset_root = resolve_dataset_root(dataset_root)
    task_index = None
    if task_dir and task_dir.startswith("task-"):
        task_index = int(task_dir.split("-", maxsplit=1)[1])

    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if task_index is None or int(row.get("task_index", -1)) == task_index:
                    return str(row.get("task") or row.get("task_name") or "")

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if task_index is not None and int(row.get("episode_index", -1)) // 10_000 != task_index:
                    continue
                tasks = row.get("tasks") or []
                if tasks:
                    return str(tasks[0])

    raise FileNotFoundError(f"Could not resolve a language instruction under {dataset_root / 'meta'}")


def tokenize_language_instruction(text: str, *, max_length: int, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize an instruction into stable hashed token ids plus a non-padding mask."""

    if max_length <= 0:
        raise ValueError("language max length must be positive.")
    if vocab_size <= EMPTY_TOKEN_ID + 1:
        raise ValueError("language vocab size must be greater than 2.")

    raw_tokens = TOKEN_RE.findall(text.lower())
    token_ids = np.full((max_length,), PAD_TOKEN_ID, dtype=np.int64)
    token_mask = np.zeros((max_length,), dtype=np.bool_)
    if not raw_tokens:
        token_ids[0] = EMPTY_TOKEN_ID
        token_mask[0] = True
        return token_ids, token_mask

    for idx, token in enumerate(raw_tokens[:max_length]):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        token_ids[idx] = 2 + (value % (vocab_size - 2))
        token_mask[idx] = True
    return token_ids, token_mask


def _scaled_intrinsics(column: str, *, image_size: int) -> np.ndarray:
    if column not in DEPTH_CAMERA_INTRINSICS:
        raise KeyError(f"No camera intrinsics registered for depth column: {column}")

    intrinsics, (native_h, native_w) = DEPTH_CAMERA_INTRINSICS[column]
    scaled = intrinsics.astype(np.float32, copy=True)
    scale_x = float(image_size) / float(native_w)
    scale_y = float(image_size) / float(native_h)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _depth_frame_to_meters(frame: np.ndarray, *, depth_max_m: float) -> np.ndarray:
    if frame.ndim == 3:
        if frame.shape[-1] == 1:
            raw = frame[..., 0]
        elif np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2]):
            raw = frame[..., 0]
        else:
            raw = frame.astype(np.float32).mean(axis=-1)
    else:
        raw = frame

    values = np.asarray(raw, dtype=np.float32)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(values, dtype=np.float32)

    if np.issubdtype(raw.dtype, np.floating):
        if max_value <= 1.5:
            depth_m = values * depth_max_m
        elif max_value <= depth_max_m * 1.5:
            depth_m = values
        else:
            depth_m = values / 1000.0
    elif max_value > 255.0:
        depth_m = values / 1000.0
    else:
        depth_m = values / 255.0 * depth_max_m

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_m = np.where(valid, depth_m, 0.0).astype(np.float32, copy=False)
    return np.clip(depth_m, 0.0, depth_max_m).astype(np.float32, copy=False)


def _depth_to_hha_like(
    cv2_module: Any,
    frame: np.ndarray,
    *,
    column: str,
    image_size: int,
    depth_max_m: float,
) -> np.ndarray:
    """Convert a decoded depth frame into HHA-style CHW values in [-1, 1]."""

    if depth_max_m <= 0.0:
        raise ValueError("depth_max_m must be positive.")

    depth_m = _depth_frame_to_meters(frame, depth_max_m=depth_max_m)
    depth_m = cv2_module.resize(depth_m, (image_size, image_size), interpolation=cv2_module.INTER_NEAREST)
    intrinsics = _scaled_intrinsics(column, image_size=image_size)

    valid = np.isfinite(depth_m) & (depth_m > 1e-6)
    safe_depth = np.where(valid, np.clip(depth_m, 1e-3, depth_max_m), depth_max_m).astype(np.float32, copy=False)

    min_depth_m = 0.1
    inv_depth = 1.0 / safe_depth
    inv_min = 1.0 / depth_max_m
    inv_max = 1.0 / min_depth_m
    disparity = (inv_depth - inv_min) / max(inv_max - inv_min, 1e-6)
    disparity = np.clip(disparity, 0.0, 1.0) * 2.0 - 1.0

    rows, _cols = np.meshgrid(
        np.arange(image_size, dtype=np.float32),
        np.arange(image_size, dtype=np.float32),
        indexing="ij",
    )
    camera_y = (rows - intrinsics[1, 2]) * safe_depth / max(float(intrinsics[1, 1]), 1e-6)
    camera_y = np.clip(camera_y / depth_max_m, -1.0, 1.0)

    grad_y, grad_x = np.gradient(safe_depth)
    normal_x = -grad_x * float(intrinsics[0, 0])
    normal_y = -grad_y * float(intrinsics[1, 1])
    normal_z = np.ones_like(safe_depth, dtype=np.float32)
    normal_norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal_z = np.clip(normal_z / np.maximum(normal_norm, 1e-6), 0.0, 1.0)
    angle_proxy = normal_z * 2.0 - 1.0

    values = np.stack([disparity, camera_y, angle_proxy], axis=0).astype(np.float32, copy=False)
    values[:, ~valid] = -1.0
    return values.copy()


def preprocess_video_frame(
    cv2_module: Any,
    frame: np.ndarray,
    *,
    image_size: int,
    is_depth: bool,
    column: str | None = None,
    depth_preprocess: str = "normalized",
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
) -> np.ndarray:
    """Resize a decoded video frame into float32 CHW values in [-1, 1]."""

    if is_depth and depth_preprocess == "hha":
        if column is None:
            raise ValueError("column is required for HHA-style depth preprocessing.")
        return _depth_to_hha_like(cv2_module, frame, column=column, image_size=image_size, depth_max_m=depth_max_m)
    if is_depth and depth_preprocess != "normalized":
        raise ValueError(f"Unknown depth_preprocess: {depth_preprocess!r}")

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]
    frame = cv2_module.resize(frame, (image_size, image_size), interpolation=cv2_module.INTER_AREA)
    if not is_depth:
        frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        values = frame.astype(np.float32) / 127.5 - 1.0
    else:
        values = frame.astype(np.float32)
        scale = 65535.0 if frame.dtype == np.uint16 else 255.0
        if float(values.max(initial=0.0)) > scale:
            scale = max(float(values.max()), 1.0)
        values = np.clip(values / scale, 0.0, 1.0) * 2.0 - 1.0
    return np.transpose(values.astype(np.float32, copy=False), (2, 0, 1)).copy()


def fixed_or_variable_list_to_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    """Convert Arrow list/fixed-size-list columns to dense numpy arrays."""

    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        outer_size = array.type.list_size
        inner = array.values
        if pa.types.is_fixed_size_list(inner.type):
            inner_size = inner.type.list_size
            flat = inner.values.to_numpy(zero_copy_only=False)
            return np.asarray(flat, dtype=dtype).reshape(len(array), outer_size, inner_size).copy()
        flat = inner.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=dtype).reshape(len(array), outer_size).copy()
    return np.asarray(array.to_pylist(), dtype=dtype).copy()


def list_column_to_fixed_numpy(column: pa.ChunkedArray, dtype: np.dtype, dim: int) -> np.ndarray:
    """Convert an Arrow list column to [N, dim], padding or truncating if needed."""

    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type) and array.type.list_size == dim:
        flat = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=dtype).reshape(len(array), dim).copy()

    values = np.zeros((len(array), dim), dtype=dtype)
    for idx, item in enumerate(array.to_pylist()):
        if item is None:
            continue
        raw = np.asarray(item, dtype=dtype).reshape(-1)
        count = min(dim, raw.shape[0])
        values[idx, :count] = raw[:count]
    return values


def load_episode(pair: EpisodePair, *, cam_rel_pose_dim: int = 21, task_info_dim: int = 82) -> dict[str, np.ndarray]:
    """Load one episode's tabular A2C2 fields plus aligned base-policy latents."""

    required_columns = [
        "observation.state",
        "action",
        "a2c2.base_action_chunk",
        "a2c2.valid_action_mask",
    ]
    optional_columns = [
        "observation.cam_rel_poses",
        "observation.task_info",
        "a2c2.policy_infer_ms",
    ]
    schema_names = set(pq.read_schema(pair.data_path).names)
    columns = required_columns + [column for column in optional_columns if column in schema_names]
    data = pq.read_table(pair.data_path, columns=columns)
    latent = pq.read_table(pair.latent_path, columns=["a2c2.base_policy_z"])
    if data.num_rows != latent.num_rows:
        raise ValueError(f"Row mismatch: {pair.data_path} has {data.num_rows}, {pair.latent_path} has {latent.num_rows}")

    rows = data.num_rows
    episode = {
        "states": fixed_or_variable_list_to_numpy(data.column("observation.state"), np.float32),
        "actions": fixed_or_variable_list_to_numpy(data.column("action"), np.float32),
        "chunks": fixed_or_variable_list_to_numpy(data.column("a2c2.base_action_chunk"), np.float32),
        "masks": fixed_or_variable_list_to_numpy(data.column("a2c2.valid_action_mask"), np.bool_),
        "zs": fixed_or_variable_list_to_numpy(latent.column("a2c2.base_policy_z"), np.float32),
        "cam_rel_poses": np.zeros((rows, cam_rel_pose_dim), dtype=np.float32),
        "task_infos": np.zeros((rows, task_info_dim), dtype=np.float32),
        "policy_infer_ms": np.zeros((rows, 1), dtype=np.float32),
    }
    if "observation.cam_rel_poses" in data.column_names:
        episode["cam_rel_poses"] = list_column_to_fixed_numpy(
            data.column("observation.cam_rel_poses"),
            np.float32,
            cam_rel_pose_dim,
        )
    if "observation.task_info" in data.column_names:
        episode["task_infos"] = list_column_to_fixed_numpy(data.column("observation.task_info"), np.float32, task_info_dim)
    if "a2c2.policy_infer_ms" in data.column_names:
        infer_ms = np.asarray(data.column("a2c2.policy_infer_ms").combine_chunks().to_numpy(), dtype=np.float32)
        episode["policy_infer_ms"] = np.log1p(np.maximum(infer_ms, 0.0)).reshape(rows, 1).astype(np.float32)
    return episode


class EpisodeVideoReader:
    """Short-lived RGB/depth mp4 reader for one episode batch."""

    def __init__(
        self,
        pair: EpisodePair,
        *,
        use_rgb: bool,
        use_depth: bool,
        image_size: int,
        depth_preprocess: str = "normalized",
        depth_max_m: float = DEFAULT_DEPTH_MAX_M,
    ) -> None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("RGB/depth training requires opencv-python (`cv2`).") from exc

        self.cv2 = cv2_module
        self.image_size = image_size
        self.depth_preprocess = depth_preprocess
        self.depth_max_m = depth_max_m
        self.caps: dict[str, Any] = {}
        self.is_depth: dict[str, bool] = {}
        dataset_root = dataset_root_from_pair(pair)
        task_dir = task_dir_from_pair(pair)
        episode_name = pair.data_path.name.replace(".parquet", ".mp4")
        for column in RGB_VIDEO_COLUMNS if use_rgb else ():
            self._open(dataset_root / "videos" / task_dir / column / episode_name, column, is_depth=False)
        for column in DEPTH_VIDEO_COLUMNS if use_depth else ():
            self._open(dataset_root / "videos" / task_dir / column / episode_name, column, is_depth=True)

    def _open(self, path: Path, column: str, *, is_depth: bool) -> None:
        cap = self.cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for {column}: {path}")
        self.caps[column] = cap
        self.is_depth[column] = is_depth

    def read_group_batch(self, columns: tuple[str, ...], frame_indices: np.ndarray) -> np.ndarray:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        batch = np.empty(
            (frame_indices.shape[0], len(columns), 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for view_idx, column in enumerate(columns):
            frames = self._read_frames(column, frame_indices)
            batch[:, view_idx] = frames
        return batch

    def _read_frames(self, column: str, frame_indices: np.ndarray) -> np.ndarray:
        loaded: dict[int, np.ndarray] = {}
        for frame_idx in sorted({int(idx) for idx in frame_indices.tolist()}):
            loaded[frame_idx] = self._read_frame(column, frame_idx)
        return np.stack([loaded[int(idx)] for idx in frame_indices.tolist()], axis=0)

    def _read_frame(self, column: str, frame_idx: int) -> np.ndarray:
        cap = self.caps[column]
        cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {column}")
        return preprocess_video_frame(
            self.cv2,
            frame,
            image_size=self.image_size,
            is_depth=self.is_depth[column],
            column=column,
            depth_preprocess=self.depth_preprocess,
            depth_max_m=self.depth_max_m,
        )

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()

    def __enter__(self) -> "EpisodeVideoReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class A2C2RandomSampleDataset(IterableDataset):
    """Yield pre-batched examples, with every batch sampled from a single episode."""

    def __init__(
        self,
        episode_pairs: list[EpisodePair],
        action_horizon: int,
        batch_size: int,
        seed: int,
        total_samples: int | None = None,
        *,
        batches_per_episode: int = 1,
        use_rgb: bool = True,
        use_depth: bool = True,
        image_size: int = 224,
        depth_preprocess: str = "normalized",
        depth_max_m: float = DEFAULT_DEPTH_MAX_M,
        use_language: bool = True,
        language_instruction: str | None = None,
        language_vocab_size: int = 4096,
        language_max_length: int = 32,
        use_cam_rel_poses: bool = False,
        cam_rel_pose_dim: int = 21,
        use_task_info: bool = False,
        task_info_dim: int = 82,
        use_policy_infer_ms: bool = False,
    ) -> None:
        super().__init__()
        self.episode_pairs = list(episode_pairs)
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.seed = seed
        self.total_samples = total_samples
        self.batches_per_episode = batches_per_episode
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.image_size = image_size
        self.depth_preprocess = depth_preprocess
        self.depth_max_m = depth_max_m
        self.use_language = use_language
        self.language_instruction = language_instruction
        self.language_vocab_size = language_vocab_size
        self.language_max_length = language_max_length
        self.use_cam_rel_poses = use_cam_rel_poses
        self.cam_rel_pose_dim = cam_rel_pose_dim
        self.use_task_info = use_task_info
        self.task_info_dim = task_info_dim
        self.use_policy_infer_ms = use_policy_infer_ms
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.batches_per_episode <= 0:
            raise ValueError("batches_per_episode must be positive.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.depth_preprocess not in {"normalized", "hha"}:
            raise ValueError("depth_preprocess must be either 'normalized' or 'hha'.")
        if self.depth_max_m <= 0.0:
            raise ValueError("depth_max_m must be positive.")
        missing_required = []
        if not self.use_rgb:
            missing_required.append("RGB")
        if not self.use_depth:
            missing_required.append("depth")
        if not self.use_language:
            missing_required.append("task language")
        if missing_required:
            names = ", ".join(missing_required)
            raise ValueError(
                f"A2C2 datasets must include RGBD and task-language inputs; disabled required feature(s): {names}."
            )
        if self.use_language and not self.language_instruction:
            raise ValueError("use_language=True requires a language_instruction.")
        if self.use_language:
            self.language_tokens, self.language_token_mask = tokenize_language_instruction(
                self.language_instruction or "",
                max_length=self.language_max_length,
                vocab_size=self.language_vocab_size,
            )
        else:
            self.language_tokens = None
            self.language_token_mask = None

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        pairs = self.episode_pairs[worker_id::num_workers]
        if not pairs:
            return
        total_samples = self.total_samples
        if total_samples is not None and num_workers > 1:
            base = total_samples // num_workers
            remainder = total_samples % num_workers
            total_samples = base + (1 if worker_id < remainder else 0)

        rng = np.random.default_rng(self.seed + worker_id)
        yielded = 0
        while total_samples is None or yielded < total_samples:
            order = rng.permutation(len(pairs))
            for episode_idx in order:
                pair = pairs[int(episode_idx)]
                episode = load_episode(
                    pair,
                    cam_rel_pose_dim=self.cam_rel_pose_dim,
                    task_info_dim=self.task_info_dim,
                )
                reader = None
                if self.use_rgb or self.use_depth:
                    reader = EpisodeVideoReader(
                        pair,
                        use_rgb=self.use_rgb,
                        use_depth=self.use_depth,
                        image_size=self.image_size,
                        depth_preprocess=self.depth_preprocess,
                        depth_max_m=self.depth_max_m,
                    )
                try:
                    for _ in range(self.batches_per_episode):
                        if total_samples is not None and yielded >= total_samples:
                            return
                        current_batch_size = self.batch_size
                        if total_samples is not None:
                            current_batch_size = min(current_batch_size, total_samples - yielded)
                        if current_batch_size <= 0:
                            return
                        batch = self._build_episode_batch(episode, rng, current_batch_size, reader)
                        yielded += int(batch["base_action"].shape[0])
                        yield batch
                finally:
                    if reader is not None:
                        reader.close()

    def _draw_source_offsets(
        self,
        episode: dict[str, np.ndarray],
        rng: np.random.Generator,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks = np.asarray(episode["masks"], dtype=np.bool_)
        rows, horizon = masks.shape
        offsets = np.arange(horizon, dtype=np.int64)
        in_bounds = np.arange(rows, dtype=np.int64)[:, None] + offsets[None, :] < rows
        valid = masks & in_bounds
        valid_sources = np.flatnonzero(valid.any(axis=1))
        if valid_sources.size == 0:
            raise ValueError("Episode has no valid source/action-offset pairs.")

        source_indices = np.empty((batch_size,), dtype=np.int64)
        chunk_offsets = np.empty((batch_size,), dtype=np.int64)
        for idx in range(batch_size):
            source_idx = int(rng.choice(valid_sources))
            valid_offsets = np.flatnonzero(valid[source_idx])
            k = int(rng.choice(valid_offsets))
            source_indices[idx] = source_idx
            chunk_offsets[idx] = k
        return source_indices, chunk_offsets

    def _build_episode_batch(
        self,
        episode: dict[str, np.ndarray],
        rng: np.random.Generator,
        batch_size: int,
        reader: EpisodeVideoReader | None,
    ) -> dict[str, np.ndarray]:
        source_indices, chunk_offsets = self._draw_source_offsets(episode, rng, batch_size)
        target_indices = source_indices + chunk_offsets
        base_action_chunks = np.asarray(episode["chunks"][source_indices], dtype=np.float32).copy()
        base_actions = np.asarray(base_action_chunks[np.arange(batch_size), chunk_offsets], dtype=np.float32).copy()
        expert_actions = np.asarray(episode["actions"][target_indices], dtype=np.float32).copy()
        denom = max(self.action_horizon - 1, 1)
        phase = 2.0 * math.pi * chunk_offsets.astype(np.float32) / float(denom)

        batch = {
            "observation_state": np.asarray(episode["states"][target_indices], dtype=np.float32).copy(),
            "base_action_chunk": base_action_chunks,
            "base_policy_z": np.asarray(episode["zs"][source_indices], dtype=np.float32).copy(),
            "time_feature": np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32),
            "valid_action_mask": np.asarray(episode["masks"][source_indices], dtype=np.bool_).copy(),
            "base_action": base_actions,
            "target_delta": expert_actions - base_actions,
            "expert_action": expert_actions,
        }
        if self.use_cam_rel_poses:
            batch["cam_rel_poses"] = np.asarray(episode["cam_rel_poses"][target_indices], dtype=np.float32).copy()
        if self.use_task_info:
            batch["task_info"] = np.asarray(episode["task_infos"][target_indices], dtype=np.float32).copy()
        if self.use_policy_infer_ms:
            batch["policy_infer_ms"] = np.asarray(episode["policy_infer_ms"][source_indices], dtype=np.float32).copy()
        if self.use_language:
            assert self.language_tokens is not None
            assert self.language_token_mask is not None
            batch["language_tokens"] = np.broadcast_to(
                self.language_tokens,
                (batch_size, self.language_tokens.shape[0]),
            ).copy()
            batch["language_token_mask"] = np.broadcast_to(
                self.language_token_mask,
                (batch_size, self.language_token_mask.shape[0]),
            ).copy()
        if reader is not None and self.use_rgb:
            batch["rgb_images"] = reader.read_group_batch(RGB_VIDEO_COLUMNS, target_indices)
        if reader is not None and self.use_depth:
            batch["depth_images"] = reader.read_group_batch(DEPTH_VIDEO_COLUMNS, target_indices)
        return batch


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def pick_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
