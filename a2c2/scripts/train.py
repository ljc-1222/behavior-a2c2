#!/usr/bin/env python3
"""Train an A2C2 correction head on BEHAVIOR/OpenPI parquet data and mp4 videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
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
    parser.add_argument("--output-dir", type=Path, default=Path("a2c2/runs/task18"))
    parser.add_argument("--task-dir", default=DEFAULT_TASK18_TASK_DIR, help="Task directory filter.")
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batches-per-episode", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000, help="Run validation every N steps. 0 disables validation.")
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rgb-backbone", choices=("resnet18", "swin_t", "small-cnn"), default="resnet18")
    parser.add_argument("--depth-backbone", choices=("resnet18", "swin_t", "small-cnn"), default="resnet18")
    parser.add_argument("--depth-preprocess", choices=("hha", "normalized"), default="hha")
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--pretrained-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-vocab-size", type=int, default=4096)
    parser.add_argument("--language-token-dim", type=int, default=128)
    parser.add_argument("--language-hidden-dim", type=int, default=256)
    parser.add_argument("--language-max-length", type=int, default=32)
    parser.add_argument("--cam-rel-pose-dim", type=int, default=21)
    parser.add_argument("--task-info-dim", type=int, default=82)
    parser.add_argument(
        "--use-latent",
        dest="use_latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use base-policy latent z during training. Pass --no-use-latent to train without latent.",
    )
    parser.add_argument("--use-cam-rel-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-task-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-policy-infer-ms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--language-instruction",
        default=None,
        help="Override the task instruction. Defaults to the dataset metadata instruction.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="a2c2")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.set_defaults(use_rgb=True, use_depth=True, use_language=True)
    return parser.parse_args()


def build_dataset_kwargs(
    args: argparse.Namespace,
    cfg: A2C2CorrectionHeadConfig,
    language_instruction: str | None,
) -> dict:
    return {
        "batches_per_episode": args.batches_per_episode,
        "use_rgb": cfg.use_rgb,
        "use_depth": cfg.use_depth,
        "image_size": args.image_size,
        "depth_preprocess": cfg.depth_preprocess,
        "depth_max_m": cfg.depth_max_m,
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


def save_checkpoint(
    output_dir: Path,
    model: A2C2CorrectionHead,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{step:06d}.pt"
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(model.config),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(payload, path)
    torch.save(payload, output_dir / "latest.pt")
    return path


def init_wandb(
    args: argparse.Namespace,
    cfg: A2C2CorrectionHeadConfig,
    dataset_root: Path,
    train_episodes: int,
    val_episodes: int,
    num_parameters: int,
):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging was requested. Install it with `pip install wandb`.") from exc

    run_config = {
        "dataset_root": str(dataset_root),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "num_parameters": num_parameters,
        "model_config": asdict(cfg),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=run_config,
        dir=str(args.output_dir),
    )


@torch.no_grad()
def evaluate_model(
    model: A2C2CorrectionHead,
    val_pairs,
    cfg: A2C2CorrectionHeadConfig,
    dataset_kwargs: dict,
    device: torch.device,
    batch_size: int,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    if not val_pairs:
        return {}

    was_training = model.training
    model.eval()
    dataset = A2C2RandomSampleDataset(
        val_pairs,
        action_horizon=cfg.action_horizon,
        batch_size=batch_size,
        seed=seed,
        total_samples=num_samples,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    total = 0
    residual_mse_sum = 0.0
    residual_mae_sum = 0.0
    corrected_mse_sum = 0.0
    base_mse_sum = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_delta = predict_delta(model, batch)
        target_delta = batch["target_delta"]
        base_action = batch["base_action"]
        expert_action = batch["expert_action"]
        corrected_action = base_action + pred_delta

        batch_size_actual = target_delta.shape[0]
        total += batch_size_actual
        residual_mse_sum += F.mse_loss(pred_delta, target_delta, reduction="sum").item()
        residual_mae_sum += F.l1_loss(pred_delta, target_delta, reduction="sum").item()
        corrected_mse_sum += F.mse_loss(corrected_action, expert_action, reduction="sum").item()
        base_mse_sum += F.mse_loss(base_action, expert_action, reduction="sum").item()
        if total >= num_samples:
            break

    if was_training:
        model.train()

    denom = max(total * cfg.action_dim, 1)
    return {
        "val/residual_mse": residual_mse_sum / denom,
        "val/residual_mae": residual_mae_sum / denom,
        "val/corrected_action_mse": corrected_mse_sum / denom,
        "val/base_action_mse": base_mse_sum / denom,
        "val/samples": float(total),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_root = resolve_dataset_root(args.dataset_root)
    pairs = discover_episode_pairs(dataset_root, args.task_dir)
    train_pairs, val_pairs = split_episode_pairs(pairs, args.val_ratio, args.seed, args.max_episodes)
    print(f"Dataset root: {dataset_root}")
    print(f"Episodes: train={len(train_pairs)} val={len(val_pairs)}")
    print(f"Image size: {args.image_size}")
    print(f"Episode batches: batch_size={args.batch_size} batches_per_episode={args.batches_per_episode}")
    language_instruction = args.language_instruction or resolve_language_instruction(dataset_root, args.task_dir)
    args.language_instruction = language_instruction
    print(f"Language instruction: {language_instruction}")

    cfg = A2C2CorrectionHeadConfig(
        use_base_policy_z=args.use_latent,
        use_rgb=args.use_rgb,
        use_depth=args.use_depth,
        use_language=args.use_language,
        use_cam_rel_poses=args.use_cam_rel_poses,
        use_task_info=args.use_task_info,
        use_policy_infer_ms=args.use_policy_infer_ms,
        rgb_backbone=args.rgb_backbone,
        depth_backbone=args.depth_backbone,
        depth_preprocess=args.depth_preprocess,
        depth_max_m=args.depth_max_m,
        pretrained_rgb=args.pretrained_rgb,
        pretrained_depth=args.pretrained_depth,
        freeze_rgb=args.freeze_rgb,
        freeze_depth=args.freeze_depth,
        language_vocab_size=args.language_vocab_size,
        language_token_dim=args.language_token_dim,
        language_hidden_dim=args.language_hidden_dim,
        language_max_length=args.language_max_length,
        cam_rel_pose_dim=args.cam_rel_pose_dim,
        task_info_dim=args.task_info_dim,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    device = pick_device(args.device)
    model = A2C2CorrectionHead(cfg).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable A2C2 parameters were found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    num_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    dataset_kwargs = build_dataset_kwargs(args, cfg, language_instruction)
    print(f"Model parameters: total={num_parameters:,} trainable={trainable_parameters:,}")

    train_dataset = A2C2RandomSampleDataset(
        train_pairs,
        action_horizon=cfg.action_horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_iter = iter(train_loader)
    if args.eval_every > 0 and not val_pairs:
        print("WARNING: --eval-every was set, but validation split is empty. Validation will be skipped.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "language_instruction": language_instruction,
                "model_config": asdict(cfg),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            f,
            indent=2,
        )

    wandb_run = init_wandb(
        args=args,
        cfg=cfg,
        dataset_root=dataset_root,
        train_episodes=len(train_pairs),
        val_episodes=len(val_pairs),
        num_parameters=num_parameters,
    )

    model.train()
    running_loss = 0.0
    start = time.time()
    for step in range(1, args.steps + 1):
        batch = move_batch_to_device(next(train_iter), device)
        pred_delta = predict_delta(model, batch)
        loss = F.mse_loss(pred_delta, batch["target_delta"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": loss_value,
                    "train/grad_norm": float(grad_norm.detach().cpu()),
                    "train/lr": args.lr,
                },
                step=step,
            )

        if step % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - start
            print(f"step={step} loss={avg:.6f} lr={args.lr:.2e} elapsed_s={elapsed:.1f}", flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_avg": avg,
                        "train/steps_per_second": args.log_every / max(elapsed, 1e-8),
                        "train/elapsed_s_per_log_window": elapsed,
                    },
                    step=step,
                )
            running_loss = 0.0
            start = time.time()

        if args.eval_every > 0 and val_pairs and step % args.eval_every == 0:
            metrics = evaluate_model(
                model=model,
                val_pairs=val_pairs,
                cfg=cfg,
                dataset_kwargs=dataset_kwargs,
                device=device,
                batch_size=args.eval_batch_size,
                num_samples=args.eval_samples,
                seed=args.seed + step,
            )
            if metrics:
                print(
                    "eval "
                    f"step={step} "
                    f"residual_mse={metrics['val/residual_mse']:.8f} "
                    f"corrected_action_mse={metrics['val/corrected_action_mse']:.8f} "
                    f"base_action_mse={metrics['val/base_action_mse']:.8f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

        if step % args.save_every == 0:
            path = save_checkpoint(args.output_dir, model, optimizer, step, args)
            print(f"saved {path}", flush=True)
            if wandb_run is not None:
                wandb_run.summary["latest_checkpoint"] = str(path)
                wandb_run.summary["latest_step"] = step

    path = save_checkpoint(args.output_dir, model, optimizer, args.steps, args)
    if wandb_run is not None:
        wandb_run.summary["final_checkpoint"] = str(path)
        wandb_run.summary["final_step"] = args.steps
        wandb_run.finish()
    print(f"training complete: {path}")


if __name__ == "__main__":
    main()
