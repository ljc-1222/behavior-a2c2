"""Stage-1 A2C2 residual losses beyond raw MSE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "base": (0, 1, 2),
    "torso": (3, 4, 5, 6),
    "left_arm": (7, 8, 9, 10, 11, 12, 13),
    "left_gripper": (14,),
    "right_arm": (15, 16, 17, 18, 19, 20, 21),
    "right_gripper": (22,),
}

LOSS_PRESETS = ("raw_mse", "norm_mse", "norm_huber", "norm_huber_gripw")


@dataclass(frozen=True)
class A2C2LossConfig:
    preset: str = "raw_mse"
    huber_delta: float = 1.0
    gripper_weight: float = 2.0
    min_action_scale: float = 1e-4

    def __post_init__(self) -> None:
        if self.preset not in LOSS_PRESETS:
            raise ValueError(f"Unknown loss preset {self.preset!r}; expected one of {LOSS_PRESETS}.")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if self.gripper_weight <= 0.0:
            raise ValueError("gripper_weight must be positive.")
        if self.min_action_scale <= 0.0:
            raise ValueError("min_action_scale must be positive.")


def loss_config_to_dict(config: A2C2LossConfig) -> dict[str, Any]:
    return asdict(config)


def loss_config_from_dict(raw: dict[str, Any] | None) -> A2C2LossConfig:
    if not raw:
        return A2C2LossConfig()
    valid_keys = A2C2LossConfig.__dataclass_fields__.keys()
    return A2C2LossConfig(**{key: value for key, value in raw.items() if key in valid_keys})


class A2C2ResidualLoss(nn.Module):
    """Configurable residual loss for A2C2 stage-1 ablations."""

    def __init__(
        self,
        config: A2C2LossConfig,
        *,
        action_scale: Tensor | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.preset == "raw_mse":
            scale = torch.ones(23, dtype=torch.float32)
        elif action_scale is None:
            raise ValueError(f"loss preset {config.preset!r} requires action_scale.")
        else:
            scale = torch.as_tensor(action_scale, dtype=torch.float32)
        if scale.ndim != 1 or scale.shape[0] != 23:
            raise ValueError(f"action_scale must have shape [23], got {tuple(scale.shape)}.")
        scale = scale.clamp_min(float(config.min_action_scale))
        self.register_buffer("action_scale", scale, persistent=True)
        for name, indices in ACTION_GROUPS.items():
            self.register_buffer(f"{name}_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

    def forward(self, pred_delta: Tensor, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        target_delta = batch["target_delta"]
        if self.config.preset == "raw_mse":
            loss = F.mse_loss(pred_delta, target_delta)
            return loss, {"total": loss.detach(), "residual": loss.detach()}

        normalized_error = (pred_delta - target_delta) / self.action_scale.to(
            device=pred_delta.device,
            dtype=pred_delta.dtype,
        )
        group_losses: dict[str, Tensor] = {}
        weighted_losses: list[Tensor] = []
        for name in ACTION_GROUPS:
            indices = getattr(self, f"{name}_indices").to(device=pred_delta.device)
            group_error = normalized_error.index_select(dim=-1, index=indices)
            group_loss = self._elementwise_loss(group_error)
            weight = self._group_weight(name)
            group_losses[name] = group_loss
            weighted_losses.append(group_loss * weight)

        residual_loss = torch.stack(weighted_losses).sum()
        metrics = {
            "total": residual_loss.detach(),
            "residual": residual_loss.detach(),
            "raw_residual_mse": F.mse_loss(pred_delta, target_delta).detach(),
        }
        for name, value in group_losses.items():
            metrics[f"group/{name}"] = value.detach()
        return residual_loss, metrics

    def _elementwise_loss(self, normalized_error: Tensor) -> Tensor:
        if self.config.preset == "norm_mse":
            return normalized_error.square().mean()
        return F.huber_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            delta=float(self.config.huber_delta),
            reduction="mean",
        )

    def _group_weight(self, name: str) -> float:
        if self.config.preset == "norm_huber_gripw" and "gripper" in name:
            return float(self.config.gripper_weight)
        return 1.0
