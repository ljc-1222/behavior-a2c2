#!/usr/bin/env python3
"""Evaluate an A2C2 correction head checkpoint on parquet data and mp4 videos."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from dataset import (  # noqa: E402
    A2C2RandomSampleDataset,
    discover_episode_pairs,
    move_batch_to_device,
    pick_device,
    resolve_dataset_root,
    resolve_language_instruction,
    split_episode_pairs,
)
from model import A2C2CorrectionHead, A2C2CorrectionHeadConfig  # noqa: E402


DEFAULT_TASK18_DATASET_ROOT = Path("a2c2_dataset/tidying_bedroom_pi05-b1kpt50-cs32_h32_v1")
DEFAULT_TASK18_TASK_DIR = "task-0018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_TASK18_DATASET_ROOT,
        help="Dataset root. Defaults to the task18 A2C2 dataset under the b1k workspace.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", default=DEFAULT_TASK18_TASK_DIR, help="Task directory filter.")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batches-per-episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="RGB/depth resize size. Defaults to the checkpoint arg, then 224.",
    )
    parser.add_argument("--language-instruction", default=None, help="Override the dataset metadata instruction.")
    return parser.parse_args()


def config_from_checkpoint(payload: dict) -> A2C2CorrectionHeadConfig:
    raw = payload.get("config", {})
    valid_keys = {field.name for field in fields(A2C2CorrectionHeadConfig)}
    filtered = {key: value for key, value in raw.items() if key in valid_keys}
    return A2C2CorrectionHeadConfig(**filtered)


def image_size_from_checkpoint(payload: dict, raw_image_size: int | None) -> int:
    if raw_image_size is not None:
        return raw_image_size
    checkpoint_args = payload.get("args", {})
    if "image_size" in checkpoint_args:
        return int(checkpoint_args["image_size"])
    return 224


def build_dataset_kwargs(
    cfg: A2C2CorrectionHeadConfig,
    image_size: int,
    language_instruction: str | None,
    batches_per_episode: int,
) -> dict:
    return {
        "batches_per_episode": batches_per_episode,
        "use_rgb": cfg.use_rgb,
        "use_depth": cfg.use_depth,
        "image_size": image_size,
        "depth_preprocess": cfg.depth_preprocess,
        "depth_max_m": float(getattr(cfg, "depth_max_m", 10.0)),
        "use_language": cfg.use_language,
        "language_instruction": language_instruction,
        "language_vocab_size": cfg.language_vocab_size,
        "language_max_length": cfg.language_max_length,
        "use_cam_rel_poses": cfg.use_cam_rel_poses,
        "cam_rel_pose_dim": cfg.cam_rel_pose_dim,
        "use_task_info": cfg.use_task_info,
        "task_info_dim": cfg.task_info_dim,
        "use_policy_infer_ms": cfg.use_policy_infer_ms,
    }


def predict_delta(model: A2C2CorrectionHead, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["observation_state"],
        batch["base_action"],
        batch["base_action_chunk"],
        batch["base_policy_z"],
        batch["time_feature"],
        batch["valid_action_mask"],
        rgb_images=batch.get("rgb_images"),
        depth_images=batch.get("depth_images"),
        language_tokens=batch.get("language_tokens"),
        language_token_mask=batch.get("language_token_mask"),
        cam_rel_poses=batch.get("cam_rel_poses"),
        task_info=batch.get("task_info"),
        policy_infer_ms=batch.get("policy_infer_ms"),
    )


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    payload = torch.load(args.checkpoint.expanduser(), map_location=device)
    cfg = config_from_checkpoint(payload)
    image_size = image_size_from_checkpoint(payload, args.image_size)
    model = A2C2CorrectionHead(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dataset_root = resolve_dataset_root(args.dataset_root)
    checkpoint_args = payload.get("args", {})
    pairs = discover_episode_pairs(dataset_root, args.task_dir)
    language_instruction = args.language_instruction
    if cfg.use_language and language_instruction is None:
        language_instruction = checkpoint_args.get("language_instruction")
        if language_instruction is None:
            language_instruction = resolve_language_instruction(dataset_root, args.task_dir)
    dataset_kwargs = build_dataset_kwargs(
        cfg,
        image_size,
        language_instruction,
        args.batches_per_episode,
    )
    train_pairs, val_pairs = split_episode_pairs(pairs, args.val_ratio, args.seed, args.max_episodes)
    if args.split == "train":
        eval_pairs = train_pairs
    elif args.split == "val":
        eval_pairs = val_pairs if val_pairs else train_pairs
    else:
        eval_pairs = train_pairs + val_pairs

    dataset = A2C2RandomSampleDataset(
        eval_pairs,
        action_horizon=cfg.action_horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        total_samples=args.num_samples,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    total = 0
    residual_mse_sum = 0.0
    residual_mae_sum = 0.0
    corrected_mse_sum = 0.0
    base_mse_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred_delta = predict_delta(model, batch)
            target_delta = batch["target_delta"]
            base_action = batch["base_action"]
            expert_action = batch["expert_action"]
            corrected_action = base_action + pred_delta

            batch_size = target_delta.shape[0]
            total += batch_size
            residual_mse_sum += F.mse_loss(pred_delta, target_delta, reduction="sum").item()
            residual_mae_sum += F.l1_loss(pred_delta, target_delta, reduction="sum").item()
            corrected_mse_sum += F.mse_loss(corrected_action, expert_action, reduction="sum").item()
            base_mse_sum += F.mse_loss(base_action, expert_action, reduction="sum").item()
            if total >= args.num_samples:
                break

    denom = max(total * cfg.action_dim, 1)
    print(f"dataset_root: {dataset_root}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"image_size: {image_size}")
    if cfg.use_language:
        print(f"language_instruction: {language_instruction}")
    print(f"split: {args.split}")
    print(f"episodes: {len(eval_pairs)}")
    print(f"samples: {total}")
    print(f"residual_mse: {residual_mse_sum / denom:.8f}")
    print(f"residual_mae: {residual_mae_sum / denom:.8f}")
    print(f"corrected_action_mse: {corrected_mse_sum / denom:.8f}")
    print(f"base_action_mse: {base_mse_sum / denom:.8f}")


if __name__ == "__main__":
    main()
