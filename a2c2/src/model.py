"""A2C2 correction head architecture for online BEHAVIOR/OpenPI inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import torch
from torch import Tensor, nn


REQUIRED_A2C2_FEATURE_FLAGS = ("use_rgb", "use_depth", "use_language")
REQUIRED_A2C2_FEATURE_LABELS = {
    "use_rgb": "RGB",
    "use_depth": "depth",
    "use_language": "task language",
}


@dataclass(frozen=True)
class A2C2CorrectionHeadConfig:
    state_dim: int = 256
    action_dim: int = 23
    action_horizon: int = 32
    base_policy_z_dim: int = 2048
    use_base_policy_z: bool = True
    time_dim: int = 2

    use_rgb: bool = True
    use_depth: bool = True
    use_language: bool = True
    use_cam_rel_poses: bool = False
    use_task_info: bool = False
    use_policy_infer_ms: bool = False
    num_rgb_views: int = 3
    num_depth_views: int = 3
    image_channels: int = 3
    rgb_backbone: str = "resnet18"
    depth_backbone: str = "resnet18"
    depth_preprocess: str = "hha"
    depth_max_m: float = 10.0
    pretrained_rgb: bool = True
    pretrained_depth: bool = True
    freeze_rgb: bool = True
    freeze_depth: bool = True
    language_vocab_size: int = 4096
    language_token_dim: int = 128
    language_hidden_dim: int = 256
    language_max_length: int = 32
    language_dim: int = 512
    cam_rel_pose_dim: int = 21
    task_info_dim: int = 82
    policy_timing_dim: int = 1

    dim_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    mlp_hidden_dim: int = 1024


def validate_required_a2c2_features(
    cfg: A2C2CorrectionHeadConfig,
    *,
    context: str = "A2C2 config",
) -> None:
    disabled = [flag for flag in REQUIRED_A2C2_FEATURE_FLAGS if not getattr(cfg, flag)]
    if disabled:
        labels = ", ".join(REQUIRED_A2C2_FEATURE_LABELS[flag] for flag in disabled)
        raise ValueError(
            f"{context} must enable RGB, depth, and task-language inputs. "
            f"Disabled required feature(s): {labels}. "
            "Pre-RGBD/task-language A2C2 artifacts are unsupported."
        )


def config_from_checkpoint_payload(
    payload: dict[str, Any],
    *,
    context: str = "A2C2 checkpoint",
) -> A2C2CorrectionHeadConfig:
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError(
            f"{context} is missing a serialized A2C2 config. "
            "Pre-RGBD/task-language A2C2 artifacts are unsupported."
        )

    missing = [flag for flag in REQUIRED_A2C2_FEATURE_FLAGS if flag not in raw]
    if missing:
        labels = ", ".join(REQUIRED_A2C2_FEATURE_LABELS[flag] for flag in missing)
        raise ValueError(
            f"{context} config is missing required feature flag(s): {labels}. "
            "Pre-RGBD/task-language A2C2 artifacts are unsupported."
        )

    valid_keys = {field.name for field in fields(A2C2CorrectionHeadConfig)}
    filtered = {key: value for key, value in raw.items() if key in valid_keys}
    cfg = A2C2CorrectionHeadConfig(**filtered)
    validate_required_a2c2_features(cfg, context=f"{context} config")
    return cfg


def _sinusoidal_positions(length: int, dim: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("dim must be even for sinusoidal positional encoding.")

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SmallImageEncoder(nn.Module):
    """Compact per-frame encoder used when no pretrained visual trunk is requested."""

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


class ResNet18ImageEncoder(nn.Module):
    """Online ImageNet ResNet-18 trunk for RGB frames."""

    def __init__(self, out_dim: int, *, pretrained: bool, freeze: bool) -> None:
        super().__init__()
        self.freeze_trunk = freeze
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError("ResNet-18 visual backbone requires torchvision.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = resnet18(weights=weights)
        except Exception as exc:
            hint = "Pass --no-pretrained-rgb or --rgb-backbone small-cnn to avoid loading pretrained weights."
            raise RuntimeError(f"Could not initialize torchvision ResNet-18. {hint}") from exc

        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.proj = nn.Identity() if out_dim == 512 else nn.Linear(512, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet18ImageEncoder":
        super().train(mode)
        if not any(param.requires_grad for param in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        values = (images + 1.0) * 0.5
        values = (values - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)
        if self.freeze_trunk:
            with torch.no_grad():
                features = self.backbone(values).flatten(1)
        else:
            features = self.backbone(values).flatten(1)
        return self.proj(features)


class SwinTImageEncoder(nn.Module):
    """Frozen ImageNet Swin-T trunk exposed as per-view visual features."""

    def __init__(self, out_dim: int, *, pretrained: bool, freeze: bool) -> None:
        super().__init__()
        self.freeze_trunk = freeze
        try:
            from torchvision.models import Swin_T_Weights, swin_t
        except Exception as exc:
            raise RuntimeError("Swin-T visual backbone requires torchvision.") from exc

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        try:
            model = swin_t(weights=weights)
        except Exception as exc:
            hint = "Pass --no-pretrained-rgb/--no-pretrained-depth or choose --*-backbone resnet18 to avoid Swin weights."
            raise RuntimeError(f"Could not initialize torchvision Swin-T. {hint}") from exc

        model.head = nn.Identity()
        self.backbone = model
        self.proj = nn.Identity() if out_dim == 768 else nn.Linear(768, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "SwinTImageEncoder":
        super().train(mode)
        if self.freeze_trunk:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        values = (images + 1.0) * 0.5
        values = (values - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)
        if self.freeze_trunk:
            with torch.no_grad():
                features = self.backbone(values)
        else:
            features = self.backbone(values)
        return self.proj(features)


class TextInstructionEncoder(nn.Module):
    """Small online text encoder over hashed instruction token ids."""

    def __init__(
        self,
        *,
        vocab_size: int,
        token_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.encoder = nn.GRU(token_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        tokens = tokens.to(dtype=torch.long)
        token_mask = token_mask.to(dtype=torch.bool)
        embedded = self.embedding(tokens)
        encoded, _ = self.encoder(embedded)
        mask = token_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        encoded = encoded * mask
        lengths = mask.sum(dim=1).clamp_min(1.0)
        pooled = encoded.sum(dim=1) / lengths
        return self.proj(pooled)


def build_image_encoder(backbone: str, *, out_dim: int, pretrained: bool, freeze: bool) -> nn.Module:
    if backbone == "small-cnn":
        encoder = SmallImageEncoder(3, out_dim)
    elif backbone == "resnet18":
        encoder = ResNet18ImageEncoder(out_dim, pretrained=pretrained, freeze=freeze)
    elif backbone == "swin_t":
        encoder = SwinTImageEncoder(out_dim, pretrained=pretrained, freeze=freeze)
    else:
        raise ValueError(f"Unknown visual backbone: {backbone!r}")

    if freeze and backbone == "small-cnn":
        for param in encoder.parameters():
            param.requires_grad_(False)
        encoder.eval()
    return encoder


class A2C2CorrectionHead(nn.Module):
    """Transformer + MLP residual head with online language and RGB(D) encoders."""

    TYPE_CLS = 0
    TYPE_STATE = 1
    TYPE_Z = 2
    TYPE_TIME = 3
    TYPE_SELECTED_ACTION = 4
    TYPE_CHUNK = 5
    TYPE_RGB = 6
    TYPE_DEPTH = 7
    TYPE_LANGUAGE = 8
    TYPE_CAM = 9
    TYPE_TASK = 10
    TYPE_POLICY_TIMING = 11

    def __init__(self, config: A2C2CorrectionHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or A2C2CorrectionHeadConfig()
        cfg = self.config
        validate_required_a2c2_features(cfg)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim_model))
        self.type_embedding = nn.Parameter(torch.zeros(self._type_count(cfg), cfg.dim_model))

        self.state_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        if cfg.use_base_policy_z:
            self.z_proj = nn.Linear(cfg.base_policy_z_dim, cfg.dim_model)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.dim_model)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.dim_model)

        if cfg.use_rgb:
            self.rgb_encoder = build_image_encoder(
                cfg.rgb_backbone,
                out_dim=cfg.dim_model,
                pretrained=cfg.pretrained_rgb,
                freeze=cfg.freeze_rgb,
            )
            self.rgb_view_embedding = nn.Parameter(torch.zeros(cfg.num_rgb_views, cfg.dim_model))
        if cfg.use_depth:
            self.depth_encoder = build_image_encoder(
                cfg.depth_backbone,
                out_dim=cfg.dim_model,
                pretrained=cfg.pretrained_depth,
                freeze=cfg.freeze_depth,
            )
            self.depth_view_embedding = nn.Parameter(torch.zeros(cfg.num_depth_views, cfg.dim_model))
        if cfg.use_language:
            self.language_encoder = TextInstructionEncoder(
                vocab_size=cfg.language_vocab_size,
                token_dim=cfg.language_token_dim,
                hidden_dim=cfg.language_hidden_dim,
                out_dim=cfg.dim_model,
                dropout=cfg.dropout,
            )
        if cfg.use_cam_rel_poses:
            self.cam_proj = nn.Linear(cfg.cam_rel_pose_dim, cfg.dim_model)
        if cfg.use_task_info:
            self.task_info_proj = nn.Linear(cfg.task_info_dim, cfg.dim_model)
        if cfg.use_policy_infer_ms:
            self.policy_timing_proj = nn.Linear(cfg.policy_timing_dim, cfg.dim_model)

        chunk_pos = _sinusoidal_positions(cfg.action_horizon, cfg.dim_model)
        self.register_buffer("chunk_pos_embedding", chunk_pos, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_encoder_layers)
        self.encoder_norm = nn.LayerNorm(cfg.dim_model)

        self.head_token_names = self._head_token_names(cfg)
        head_input_dim = cfg.dim_model * len(self.head_token_names) + cfg.action_dim
        self.residual_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_dim, cfg.action_dim),
        )

        self._reset_parameters()

    @staticmethod
    def _type_count(cfg: A2C2CorrectionHeadConfig) -> int:
        has_extra = any(
            (
                cfg.use_rgb,
                cfg.use_depth,
                cfg.use_language,
                cfg.use_cam_rel_poses,
                cfg.use_task_info,
                cfg.use_policy_infer_ms,
            )
        )
        return 12 if has_extra else 6

    @staticmethod
    def _head_token_names(cfg: A2C2CorrectionHeadConfig) -> list[str]:
        names = ["cls", "state"]
        if cfg.use_base_policy_z:
            names.append("base_policy_z")
        if cfg.use_language:
            names.append("language")
        if cfg.use_cam_rel_poses:
            names.append("cam_rel_poses")
        if cfg.use_task_info:
            names.append("task_info")
        if cfg.use_policy_infer_ms:
            names.append("policy_infer_ms")
        if cfg.use_rgb:
            names.append("rgb")
        if cfg.use_depth:
            names.append("depth")
        names.extend(["time", "selected_action"])
        return names

    @staticmethod
    def make_time_feature(chunk_index: Tensor, horizon: int) -> Tensor:
        """Create [sin, cos] phase features from chunk indices."""

        idx = chunk_index.to(dtype=torch.float32)
        denom = max(horizon - 1, 1)
        phase = 2.0 * math.pi * idx / denom
        return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)

    def forward(
        self,
        observation_state: Tensor,
        selected_base_action: Tensor,
        base_action_chunk: Tensor,
        base_policy_z: Tensor,
        time_feature: Tensor,
        valid_action_mask: Tensor | None = None,
        *,
        rgb_images: Tensor | None = None,
        depth_images: Tensor | None = None,
        language_tokens: Tensor | None = None,
        language_token_mask: Tensor | None = None,
        cam_rel_poses: Tensor | None = None,
        task_info: Tensor | None = None,
        policy_infer_ms: Tensor | None = None,
    ) -> Tensor:
        """Predict residual action delta.

        Args:
            observation_state: latest state at target frame, shape [B, state_dim].
            selected_base_action: base chunk action being corrected, shape [B, action_dim].
            base_action_chunk: source base action chunk, shape [B, H, action_dim].
            base_policy_z: source-frame base-policy prefix latent, shape [B, z_dim].
            time_feature: sinusoidal chunk-index feature, shape [B, 2].
            valid_action_mask: optional bool tensor [B, H], True for valid chunk entries.
            rgb_images: latest RGB views, shape [B, V, 3, H, W], values in [-1, 1].
            depth_images: latest depth views, shape [B, V, 3, H, W], values in [-1, 1].
            language_tokens: hashed instruction token ids, shape [B, L].
            language_token_mask: bool non-padding mask, shape [B, L].
            cam_rel_poses: camera relative poses, shape [B, cam_rel_pose_dim].
            task_info: BEHAVIOR task-info vector, shape [B, task_info_dim].
            policy_infer_ms: source base-policy inference timing, shape [B, 1].

        Returns:
            Tensor [B, action_dim], the predicted residual delta.
        """

        cfg = self.config
        batch_size = observation_state.shape[0]
        device = observation_state.device
        dtype = observation_state.dtype

        self._validate_inputs(
            observation_state=observation_state,
            selected_base_action=selected_base_action,
            base_action_chunk=base_action_chunk,
            base_policy_z=base_policy_z,
            time_feature=time_feature,
            rgb_images=rgb_images,
            depth_images=depth_images,
            language_tokens=language_tokens,
            language_token_mask=language_token_mask,
            cam_rel_poses=cam_rel_poses,
            task_info=task_info,
            policy_infer_ms=policy_infer_ms,
        )

        tokens: list[Tensor] = []
        spans: dict[str, tuple[int, int]] = {}

        def type_embedding(type_idx: int) -> Tensor:
            return self.type_embedding[type_idx].to(device=device, dtype=dtype)

        def add_tokens(name: str, values: Tensor, type_idx: int) -> None:
            start = sum(token.shape[1] for token in tokens)
            values = values + type_embedding(type_idx)
            tokens.append(values)
            spans[name] = (start, start + values.shape[1])

        cls = self.cls_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        add_tokens("cls", cls, self.TYPE_CLS)

        state_token = self.state_proj(observation_state).unsqueeze(1)
        add_tokens("state", state_token, self.TYPE_STATE)

        if cfg.use_base_policy_z:
            z_token = self.z_proj(base_policy_z).unsqueeze(1)
            add_tokens("base_policy_z", z_token, self.TYPE_Z)

        if cfg.use_language:
            assert language_tokens is not None
            assert language_token_mask is not None
            language_token = self.language_encoder(language_tokens, language_token_mask).unsqueeze(1)
            add_tokens("language", language_token, self.TYPE_LANGUAGE)

        if cfg.use_cam_rel_poses:
            assert cam_rel_poses is not None
            cam_token = self.cam_proj(cam_rel_poses).unsqueeze(1)
            add_tokens("cam_rel_poses", cam_token, self.TYPE_CAM)

        if cfg.use_task_info:
            assert task_info is not None
            task_token = self.task_info_proj(task_info).unsqueeze(1)
            add_tokens("task_info", task_token, self.TYPE_TASK)

        if cfg.use_policy_infer_ms:
            assert policy_infer_ms is not None
            policy_token = self.policy_timing_proj(policy_infer_ms).unsqueeze(1)
            add_tokens("policy_infer_ms", policy_token, self.TYPE_POLICY_TIMING)

        if cfg.use_rgb:
            assert rgb_images is not None
            add_tokens("rgb", self._encode_image_views(rgb_images, self.rgb_encoder, self.rgb_view_embedding), self.TYPE_RGB)

        if cfg.use_depth:
            assert depth_images is not None
            depth_tokens = self._encode_image_views(depth_images, self.depth_encoder, self.depth_view_embedding)
            add_tokens("depth", depth_tokens, self.TYPE_DEPTH)

        time_token = self.time_proj(time_feature).unsqueeze(1)
        add_tokens("time", time_token, self.TYPE_TIME)

        selected_action_token = self.action_proj(selected_base_action).unsqueeze(1)
        add_tokens("selected_action", selected_action_token, self.TYPE_SELECTED_ACTION)

        prefix_len = sum(token.shape[1] for token in tokens)
        chunk_tokens = self.action_proj(base_action_chunk)
        chunk_pos = self.chunk_pos_embedding[: base_action_chunk.shape[1]].to(device=device, dtype=dtype)
        chunk_tokens = chunk_tokens + chunk_pos.unsqueeze(0)
        add_tokens("chunk", chunk_tokens, self.TYPE_CHUNK)

        all_tokens = torch.cat(tokens, dim=1)
        padding_mask = None
        if valid_action_mask is not None:
            valid_action_mask = valid_action_mask.to(device=device, dtype=torch.bool)
            prefix_mask = torch.zeros(batch_size, prefix_len, device=device, dtype=torch.bool)
            padding_mask = torch.cat([prefix_mask, ~valid_action_mask], dim=1)

        encoded = self.encoder(all_tokens, src_key_padding_mask=padding_mask)
        encoded = self.encoder_norm(encoded)

        head_states = [self._pool_span(encoded, spans[name]) for name in self.head_token_names]
        head_input = torch.cat([*head_states, selected_base_action], dim=-1)
        return self.residual_head(head_input)

    def _encode_image_views(self, images: Tensor, encoder: nn.Module, view_embedding: Tensor) -> Tensor:
        cfg = self.config
        batch_size, num_views = images.shape[:2]
        flat = images.reshape(batch_size * num_views, cfg.image_channels, images.shape[-2], images.shape[-1])
        encoded = encoder(flat).reshape(batch_size, num_views, cfg.dim_model)
        view = view_embedding[:num_views].to(device=images.device, dtype=images.dtype).unsqueeze(0)
        return encoded + view

    @staticmethod
    def _pool_span(encoded: Tensor, span: tuple[int, int]) -> Tensor:
        start, end = span
        values = encoded[:, start:end]
        if values.shape[1] == 1:
            return values[:, 0]
        return values.mean(dim=1)

    def _validate_inputs(
        self,
        *,
        observation_state: Tensor,
        selected_base_action: Tensor,
        base_action_chunk: Tensor,
        base_policy_z: Tensor,
        time_feature: Tensor,
        rgb_images: Tensor | None,
        depth_images: Tensor | None,
        language_tokens: Tensor | None,
        language_token_mask: Tensor | None,
        cam_rel_poses: Tensor | None,
        task_info: Tensor | None,
        policy_infer_ms: Tensor | None,
    ) -> None:
        cfg = self.config
        if observation_state.ndim != 2 or observation_state.shape[-1] != cfg.state_dim:
            raise ValueError(f"observation_state must have shape [B, {cfg.state_dim}].")
        if selected_base_action.ndim != 2 or selected_base_action.shape[-1] != cfg.action_dim:
            raise ValueError(f"selected_base_action must have shape [B, {cfg.action_dim}].")
        if base_action_chunk.ndim != 3 or base_action_chunk.shape[-1] != cfg.action_dim:
            raise ValueError(f"base_action_chunk must have shape [B, H, {cfg.action_dim}].")
        if base_action_chunk.shape[1] > cfg.action_horizon:
            raise ValueError(f"base_action_chunk horizon cannot exceed {cfg.action_horizon}.")
        if cfg.use_base_policy_z and (base_policy_z.ndim != 2 or base_policy_z.shape[-1] != cfg.base_policy_z_dim):
            raise ValueError(f"base_policy_z must have shape [B, {cfg.base_policy_z_dim}].")
        if time_feature.ndim != 2 or time_feature.shape[-1] != cfg.time_dim:
            raise ValueError(f"time_feature must have shape [B, {cfg.time_dim}].")
        if cfg.use_rgb:
            self._validate_images("rgb_images", rgb_images, cfg.num_rgb_views)
        if cfg.use_depth:
            self._validate_images("depth_images", depth_images, cfg.num_depth_views)
        if cfg.use_language:
            self._validate_language(language_tokens, language_token_mask)
        if cfg.use_cam_rel_poses and (
            cam_rel_poses is None or cam_rel_poses.ndim != 2 or cam_rel_poses.shape[-1] != cfg.cam_rel_pose_dim
        ):
            raise ValueError(f"cam_rel_poses must have shape [B, {cfg.cam_rel_pose_dim}].")
        if cfg.use_task_info and (task_info is None or task_info.ndim != 2 or task_info.shape[-1] != cfg.task_info_dim):
            raise ValueError(f"task_info must have shape [B, {cfg.task_info_dim}].")
        if cfg.use_policy_infer_ms and (
            policy_infer_ms is None
            or policy_infer_ms.ndim != 2
            or policy_infer_ms.shape[-1] != cfg.policy_timing_dim
        ):
            raise ValueError(f"policy_infer_ms must have shape [B, {cfg.policy_timing_dim}].")

    def _validate_images(self, name: str, images: Tensor | None, expected_views: int) -> None:
        cfg = self.config
        if images is None:
            raise ValueError(f"{name} is required by the model config.")
        if images.ndim != 5:
            raise ValueError(f"{name} must have shape [B, V, C, H, W].")
        if images.shape[1] != expected_views or images.shape[2] != cfg.image_channels:
            raise ValueError(f"{name} must have shape [B, {expected_views}, {cfg.image_channels}, H, W].")

    def _validate_language(self, tokens: Tensor | None, token_mask: Tensor | None) -> None:
        cfg = self.config
        if tokens is None or token_mask is None:
            raise ValueError("language_tokens and language_token_mask are required by the model config.")
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("language_tokens and language_token_mask must have shape [B, L].")
        if tokens.shape != token_mask.shape:
            raise ValueError("language_tokens and language_token_mask must have identical shapes.")
        if tokens.shape[1] != cfg.language_max_length:
            raise ValueError(f"language token length must be {cfg.language_max_length}.")

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)
        for name in ("rgb_view_embedding", "depth_view_embedding"):
            if hasattr(self, name):
                nn.init.trunc_normal_(getattr(self, name), std=0.02)
        visual_module_ids: set[int] = set()
        for name in ("rgb_encoder", "depth_encoder"):
            if hasattr(self, name):
                visual_module_ids.update(id(module) for module in getattr(self, name).modules())
        for module in self.modules():
            if id(module) in visual_module_ids:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
