#!/usr/bin/env python3
"""Convert BEHAVIOR mp4 videos into resized frame parquet sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RGB_VIDEO_KEYS = (
    "observation.images.rgb.head",
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
)
DEPTH_VIDEO_KEYS = (
    "observation.images.depth.head",
    "observation.images.depth.left_wrist",
    "observation.images.depth.right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Path to a task video folder, e.g. dataset/videos/task-0018.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root. Defaults to <dataset_root>/video_frames/<task-dir> for frames, "
            "or <dataset_root>/rgb_features_resnet18/<task-dir> for ResNet18 features."
        ),
    )
    parser.add_argument(
        "--output-kind",
        choices=("frames", "resnet18-features"),
        default="frames",
        help="Write resized image frames or frozen ResNet18 RGB features.",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Resize frames to image_size x image_size.")
    parser.add_argument(
        "--use-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert depth videos too. Pass --no-use-depth to convert only RGB videos.",
    )
    parser.add_argument(
        "--video-keys",
        nargs="+",
        default=None,
        help=(
            "Explicit video subdirectories to convert. If omitted, converts RGB plus depth unless "
            "--no-use-depth is passed."
        ),
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec, e.g. zstd, snappy, none.")
    parser.add_argument("--row-group-size", type=int, default=256, help="Frames per parquet row group.")
    parser.add_argument("--feature-batch-size", type=int, default=256, help="ResNet18 feature extraction batch size.")
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="auto", help="Feature extraction device: auto, cuda, cpu, etc.")
    parser.add_argument("--pretrained-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=None, help="Optional debug limit on number of mp4 files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquet files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing files.")
    args = parser.parse_args()

    args.video_folder = args.video_folder.expanduser().resolve()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.row_group_size <= 0:
        raise ValueError("--row-group-size must be positive.")
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be positive.")
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max-videos must be positive when provided.")
    if args.compression.lower() == "none":
        args.compression = None
    if args.video_keys is None:
        args.video_keys = list(RGB_VIDEO_KEYS)
        if args.use_depth and args.output_kind == "frames":
            args.video_keys.extend(DEPTH_VIDEO_KEYS)
    if args.output_kind == "resnet18-features":
        non_rgb_keys = [key for key in args.video_keys if key not in RGB_VIDEO_KEYS]
        if non_rgb_keys:
            names = ", ".join(non_rgb_keys)
            raise ValueError(f"--output-kind resnet18-features only supports RGB video keys. Got: {names}")
        args.use_depth = False

    if args.output_root is None:
        args.output_root = default_output_root(args.video_folder, output_kind=args.output_kind)
    else:
        args.output_root = args.output_root.expanduser().resolve()
    return args


def default_output_root(video_folder: Path, *, output_kind: str) -> Path:
    dirname = "rgb_features_resnet18" if output_kind == "resnet18-features" else "video_frames"
    if video_folder.parent.name == "videos":
        dataset_root = video_folder.parent.parent
        return dataset_root / dirname / video_folder.name
    return Path.cwd() / dirname / video_folder.name


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("mp4_to_parquet.py requires opencv-python (`cv2`).") from exc
    return cv2


def discover_videos(video_folder: Path, video_keys: list[str]) -> list[tuple[str, Path]]:
    if not video_folder.is_dir():
        raise FileNotFoundError(f"Video folder does not exist: {video_folder}")

    videos: list[tuple[str, Path]] = []
    for video_key in video_keys:
        view_dir = video_folder / video_key
        if not view_dir.is_dir():
            raise FileNotFoundError(f"Missing video key directory: {view_dir}")
        videos.extend((video_key, path) for path in sorted(view_dir.glob("episode_*.mp4")))
    if not videos:
        keys = ", ".join(video_keys)
        raise FileNotFoundError(f"No episode_*.mp4 files found under {video_folder} for keys: {keys}")
    return videos


def output_path_for(output_root: Path, video_key: str, mp4_path: Path) -> Path:
    return output_root / video_key / mp4_path.with_suffix(".parquet").name


def is_depth_key(video_key: str) -> bool:
    return ".depth." in video_key


def frame_schema(*, image_size: int, video_key: str) -> pa.Schema:
    frame_size = image_size * image_size * 3
    is_depth = is_depth_key(video_key)
    metadata = {
        b"height": str(image_size).encode("ascii"),
        b"width": str(image_size).encode("ascii"),
        b"channels": b"3",
        b"color_order": b"decoded" if is_depth else b"RGB",
        b"layout": b"HWC",
        b"dtype": b"uint8",
        b"is_depth": b"true" if is_depth else b"false",
        b"video_key": video_key.encode("utf-8"),
    }
    return pa.schema(
        [
            ("frame_index", pa.int32()),
            ("frame", pa.list_(pa.uint8(), frame_size)),
        ],
        metadata=metadata,
    )


def feature_schema(*, image_size: int, video_key: str, feature_dtype: str, pretrained_rgb: bool) -> pa.Schema:
    feature_type = pa.float16() if feature_dtype == "float16" else pa.float32()
    metadata = {
        b"feature_model": b"resnet18",
        b"feature_dim": b"512",
        b"feature_dtype": feature_dtype.encode("ascii"),
        b"image_size": str(image_size).encode("ascii"),
        b"pretrained_rgb": b"true" if pretrained_rgb else b"false",
        b"color_order": b"RGB",
        b"video_key": video_key.encode("utf-8"),
    }
    return pa.schema(
        [
            ("frame_index", pa.int32()),
            ("feature", pa.list_(feature_type, 512)),
        ],
        metadata=metadata,
    )


def make_frame_table(indices: list[int], frames: list[np.ndarray], schema: pa.Schema) -> pa.Table:
    if not frames:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.int32()),
                pa.FixedSizeListArray.from_arrays(pa.array([], type=pa.uint8()), schema.field("frame").type.list_size),
            ],
            schema=schema,
        )

    frame_size = schema.field("frame").type.list_size
    values = np.stack(frames, axis=0).reshape(-1)
    value_array = pa.array(values, type=pa.uint8())
    frame_array = pa.FixedSizeListArray.from_arrays(value_array, frame_size)
    return pa.Table.from_arrays([pa.array(indices, type=pa.int32()), frame_array], schema=schema)


def make_feature_table(indices: list[int], features: list[np.ndarray], schema: pa.Schema) -> pa.Table:
    if not features:
        feature_type = schema.field("feature").type
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.int32()),
                pa.FixedSizeListArray.from_arrays(pa.array([], type=feature_type.value_type), feature_type.list_size),
            ],
            schema=schema,
        )

    feature_type = schema.field("feature").type
    values = np.concatenate(features, axis=0).reshape(-1)
    value_array = pa.array(values, type=feature_type.value_type)
    feature_array = pa.FixedSizeListArray.from_arrays(value_array, feature_type.list_size)
    return pa.Table.from_arrays([pa.array(indices, type=pa.int32()), feature_array], schema=schema)


def convert_one_video_to_frames(
    *,
    cv2_module,
    video_key: str,
    mp4_path: Path,
    output_path: Path,
    image_size: int,
    row_group_size: int,
    compression: str | None,
) -> int:
    cap = cv2_module.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    schema = frame_schema(image_size=image_size, video_key=video_key)
    writer = pq.ParquetWriter(tmp_path, schema=schema, compression=compression)
    frame_count = 0
    batch_indices: list[int] = []
    batch_frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            resized = cv2_module.resize(bgr, (image_size, image_size), interpolation=cv2_module.INTER_AREA)
            frame = normalize_decoded_frame(cv2_module, resized, is_depth=is_depth_key(video_key))
            batch_indices.append(frame_count)
            batch_frames.append(frame)
            frame_count += 1

            if len(batch_frames) >= row_group_size:
                writer.write_table(make_frame_table(batch_indices, batch_frames, schema))
                batch_indices.clear()
                batch_frames.clear()

        if batch_frames:
            writer.write_table(make_frame_table(batch_indices, batch_frames, schema))
    finally:
        writer.close()
        cap.release()

    tmp_path.replace(output_path)
    return frame_count


def convert_one_video_to_resnet18_features(
    *,
    cv2_module,
    feature_extractor,
    torch_module,
    device,
    video_key: str,
    mp4_path: Path,
    output_path: Path,
    image_size: int,
    feature_batch_size: int,
    row_group_size: int,
    compression: str | None,
    feature_dtype: str,
    pretrained_rgb: bool,
) -> int:
    cap = cv2_module.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    schema = feature_schema(
        image_size=image_size,
        video_key=video_key,
        feature_dtype=feature_dtype,
        pretrained_rgb=pretrained_rgb,
    )
    writer = pq.ParquetWriter(tmp_path, schema=schema, compression=compression)
    frame_count = 0
    pending_indices: list[int] = []
    pending_frames: list[np.ndarray] = []
    output_indices: list[int] = []
    output_features: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            resized = cv2_module.resize(bgr, (image_size, image_size), interpolation=cv2_module.INTER_AREA)
            rgb = cv2_module.cvtColor(resized, cv2_module.COLOR_BGR2RGB)
            pending_indices.append(frame_count)
            pending_frames.append(np.asarray(rgb, dtype=np.uint8))
            frame_count += 1

            if len(pending_frames) >= feature_batch_size:
                features = extract_resnet18_features(
                    torch_module,
                    feature_extractor,
                    device,
                    pending_frames,
                    feature_dtype=feature_dtype,
                )
                output_indices.extend(pending_indices)
                output_features.append(features)
                pending_indices.clear()
                pending_frames.clear()

            if len(output_indices) >= row_group_size:
                writer.write_table(make_feature_table(output_indices, output_features, schema))
                output_indices.clear()
                output_features.clear()

        if pending_frames:
            features = extract_resnet18_features(
                torch_module,
                feature_extractor,
                device,
                pending_frames,
                feature_dtype=feature_dtype,
            )
            output_indices.extend(pending_indices)
            output_features.append(features)

        if output_indices:
            writer.write_table(make_feature_table(output_indices, output_features, schema))
    finally:
        writer.close()
        cap.release()

    tmp_path.replace(output_path)
    return frame_count


def normalize_decoded_frame(cv2_module, frame: np.ndarray, *, is_depth: bool) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]

    if not is_depth:
        frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)

    if frame.dtype == np.uint8:
        return np.asarray(frame, dtype=np.uint8)

    values = frame.astype(np.float32)
    max_value = float(values.max(initial=0.0))
    scale = 65535.0 if max_value > 255.0 else max(max_value, 1.0)
    return np.clip(values / scale * 255.0, 0.0, 255.0).astype(np.uint8)


def import_feature_dependencies():
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ModuleNotFoundError as exc:
        raise RuntimeError("ResNet18 feature export requires torch and torchvision.") from exc
    return torch, ResNet18_Weights, resnet18


def pick_torch_device(torch_module, raw: str):
    if raw != "auto":
        return torch_module.device(raw)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def build_resnet18_feature_extractor(*, torch_module, weights_cls, resnet18_fn, pretrained_rgb: bool, device):
    weights = weights_cls.DEFAULT if pretrained_rgb else None
    model = resnet18_fn(weights=weights)
    model = torch_module.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_resnet18_features(
    torch_module,
    feature_extractor,
    device,
    frames: list[np.ndarray],
    *,
    feature_dtype: str,
) -> np.ndarray:
    values = np.stack(frames, axis=0)
    tensor = torch_module.from_numpy(values).to(device=device, dtype=torch_module.float32)
    tensor = tensor.permute(0, 3, 1, 2).contiguous().div_(255.0)
    mean = torch_module.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch_module.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch_module.no_grad():
        features = feature_extractor(tensor).flatten(1).detach().cpu().numpy()
    if feature_dtype == "float16":
        return features.astype(np.float16, copy=False)
    return features.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.video_folder, args.video_keys)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    print(f"Video folder: {args.video_folder}")
    print(f"Output root:  {args.output_root}")
    print(f"Output kind:  {args.output_kind}")
    print(f"Image size:   {args.image_size}x{args.image_size}")
    print(f"Use depth:    {args.use_depth}")
    print(f"Videos:       {len(videos)}")
    if args.dry_run:
        for video_key, mp4_path in videos:
            print(f"DRY {mp4_path} -> {output_path_for(args.output_root, video_key, mp4_path)}")
        return

    cv2_module = import_cv2()
    torch_module = None
    feature_extractor = None
    device = None
    if args.output_kind == "resnet18-features":
        torch_module, weights_cls, resnet18_fn = import_feature_dependencies()
        device = pick_torch_device(torch_module, args.device)
        feature_extractor = build_resnet18_feature_extractor(
            torch_module=torch_module,
            weights_cls=weights_cls,
            resnet18_fn=resnet18_fn,
            pretrained_rgb=args.pretrained_rgb,
            device=device,
        )
        print(f"Feature model: resnet18 pretrained={args.pretrained_rgb} dtype={args.feature_dtype} device={device}")

    converted = 0
    skipped = 0
    total_frames = 0
    for index, (video_key, mp4_path) in enumerate(videos, start=1):
        out_path = output_path_for(args.output_root, video_key, mp4_path)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(videos)}] skip existing {out_path}")
            continue
        if args.output_kind == "resnet18-features":
            assert torch_module is not None
            assert feature_extractor is not None
            assert device is not None
            frames = convert_one_video_to_resnet18_features(
                cv2_module=cv2_module,
                feature_extractor=feature_extractor,
                torch_module=torch_module,
                device=device,
                video_key=video_key,
                mp4_path=mp4_path,
                output_path=out_path,
                image_size=args.image_size,
                feature_batch_size=args.feature_batch_size,
                row_group_size=args.row_group_size,
                compression=args.compression,
                feature_dtype=args.feature_dtype,
                pretrained_rgb=args.pretrained_rgb,
            )
        else:
            frames = convert_one_video_to_frames(
                cv2_module=cv2_module,
                video_key=video_key,
                mp4_path=mp4_path,
                output_path=out_path,
                image_size=args.image_size,
                row_group_size=args.row_group_size,
                compression=args.compression,
            )
        converted += 1
        total_frames += frames
        print(f"[{index}/{len(videos)}] wrote {frames} frames -> {out_path}", flush=True)

    print(f"done: converted={converted} skipped={skipped} frames={total_frames}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
