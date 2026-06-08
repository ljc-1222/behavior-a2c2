"""Online A2C2 wrapper for BEHAVIOR-1K/OpenPI websocket evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import DEPTH_VIDEO_COLUMNS, preprocess_video_frame, tokenize_language_instruction
from model import A2C2CorrectionHead, A2C2CorrectionHeadConfig, config_from_checkpoint_payload
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


logger = logging.getLogger(__name__)


DEPTH_CAMERA_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::depth_linear",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear",
)
TASK_INFO_KEYS = (
    "observation.task_info",
    "task::low_dim",
    "task_info",
    "robot_r1::task_info",
)


@dataclass
class _ActiveChunk:
    base_action_chunk: np.ndarray
    valid_action_mask: np.ndarray
    base_policy_z: np.ndarray
    policy_infer_ms: np.ndarray
    execute_len: int
    offset: int = 0


def config_from_checkpoint(payload: dict[str, Any]) -> A2C2CorrectionHeadConfig:
    return config_from_checkpoint_payload(payload, context="A2C2 online checkpoint")


def image_size_from_checkpoint(payload: dict[str, Any], explicit_image_size: int | None) -> int:
    if explicit_image_size is not None:
        return explicit_image_size
    checkpoint_args = payload.get("args", {})
    if "image_size" in checkpoint_args:
        return int(checkpoint_args["image_size"])
    return 224


def pick_torch_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path.expanduser(), map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"A2C2 checkpoint is missing model_state_dict: {path}")
    return payload


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fit_vector(value: Any, dim: int, *, name: str) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if arr.shape[0] == dim:
        return arr.copy()
    fitted = np.zeros((dim,), dtype=np.float32)
    count = min(dim, arr.shape[0])
    fitted[:count] = arr[:count]
    logger.warning("%s had dim %s; fitted to %s.", name, arr.shape[0], dim)
    return fitted


def _as_action_chunk(value: Any, *, horizon: int, action_dim: int) -> tuple[np.ndarray, np.ndarray, int]:
    actions = _to_numpy(value).astype(np.float32, copy=False)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"Base policy actions must have shape [H, A], got {actions.shape}.")
    if actions.shape[1] < action_dim:
        raise ValueError(f"Base policy action dim {actions.shape[1]} is smaller than A2C2 action dim {action_dim}.")

    returned_len = min(actions.shape[0], horizon)
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    chunk[:returned_len] = actions[:returned_len, :action_dim]
    valid = np.zeros((horizon,), dtype=np.bool_)
    valid[:returned_len] = True
    return chunk, valid, returned_len


class A2C2B1KPolicyWrapper(B1KPolicyWrapper):
    """Wrap a B1K OpenPI policy and add an online A2C2 residual per control step."""

    def __init__(
        self,
        policy,
        *,
        a2c2_checkpoint: str | Path,
        a2c2_device: str = "auto",
        a2c2_image_size: int | None = None,
        a2c2_language_instruction: str | None = None,
        a2c2_correction_scale: float = 1.0,
        task_name: str = "turning_on_radio",
        control_mode: str = "receeding_horizon",
        max_len: int = 32,
        action_horizon: int = 5,
        temporal_ensemble_max: int = 3,
        fine_grained_level: int = 0,
    ) -> None:
        if control_mode == "receding_horizon":
            control_mode = "receeding_horizon"
        if control_mode != "receeding_horizon":
            raise ValueError(
                "Online A2C2 currently supports only control_mode='receeding_horizon', "
                "which matches the training alignment to one source action chunk."
            )
        if not math.isfinite(a2c2_correction_scale):
            raise ValueError("a2c2_correction_scale must be finite.")

        super().__init__(
            policy,
            task_name=task_name,
            control_mode=control_mode,
            max_len=max_len,
            action_horizon=action_horizon,
            temporal_ensemble_max=temporal_ensemble_max,
            fine_grained_level=fine_grained_level,
        )

        self.a2c2_checkpoint = Path(a2c2_checkpoint).expanduser()
        self.a2c2_device = pick_torch_device(a2c2_device)
        payload = _load_torch_checkpoint(self.a2c2_checkpoint)
        self.a2c2_config = config_from_checkpoint(payload)
        self.a2c2_image_size = image_size_from_checkpoint(payload, a2c2_image_size)
        self.a2c2_correction_scale = float(a2c2_correction_scale)

        if self.a2c2_config.action_dim != 23:
            raise ValueError(f"B1K online evaluation expects action_dim=23, got {self.a2c2_config.action_dim}.")
        if self.a2c2_config.state_dim != 256:
            raise ValueError(f"B1K online evaluation expects state_dim=256, got {self.a2c2_config.state_dim}.")

        self.a2c2_model = A2C2CorrectionHead(self.a2c2_config).to(self.a2c2_device)
        self.a2c2_model.load_state_dict(payload["model_state_dict"])
        self.a2c2_model.eval()

        self.a2c2_language_instruction = a2c2_language_instruction or self.task_prompt
        tokens, token_mask = tokenize_language_instruction(
            self.a2c2_language_instruction,
            max_length=self.a2c2_config.language_max_length,
            vocab_size=self.a2c2_config.language_vocab_size,
        )
        self._language_tokens = tokens
        self._language_token_mask = token_mask

        self._active_chunk: _ActiveChunk | None = None
        self._last_policy_prompt = self.task_prompt
        logger.info(
            "Loaded A2C2 checkpoint %s on %s with image_size=%s, use_base_policy_z=%s.",
            self.a2c2_checkpoint,
            self.a2c2_device,
            self.a2c2_image_size,
            self.a2c2_config.use_base_policy_z,
        )

    def reset(self) -> None:
        super().reset()
        self._active_chunk = None

    def act(self, input_obs: dict) -> torch.Tensor:
        processed_obs = self.process_obs(input_obs)
        if self._active_chunk is None or self._active_chunk.offset >= self._active_chunk.execute_len:
            self._active_chunk = self._start_new_chunk(processed_obs)

        assert self._active_chunk is not None
        chunk = self._active_chunk
        offset = chunk.offset
        selected_base_action = chunk.base_action_chunk[offset].copy()
        corrected_action = self._correct_action(
            raw_obs=input_obs,
            processed_obs=processed_obs,
            active_chunk=chunk,
            chunk_offset=offset,
        )
        chunk.offset += 1

        if not np.all(np.isfinite(corrected_action)):
            raise ValueError("A2C2 produced a non-finite corrected action.")
        self.step_counter += 1
        logger.debug(
            "A2C2 step=%s offset=%s residual_norm=%.6f.",
            self.step_counter,
            offset,
            float(np.linalg.norm(corrected_action - selected_base_action)),
        )
        return torch.from_numpy(corrected_action.astype(np.float32, copy=False)[None])

    def _start_new_chunk(self, processed_obs: dict[str, np.ndarray]) -> _ActiveChunk:
        batch = self._make_openpi_batch(processed_obs)
        if self.a2c2_config.use_base_policy_z:
            if not hasattr(self.policy, "infer_with_prefix_z"):
                raise RuntimeError(
                    "A2C2 checkpoint requires base_policy_z, but the base policy does not expose "
                    "infer_with_prefix_z. Serve with the patched OpenPI policy."
                )
            action = self.policy.infer_with_prefix_z(batch)
            if "prefix_z" not in action:
                raise KeyError("Base policy infer_with_prefix_z did not return prefix_z.")
            base_policy_z = _fit_vector(
                action["prefix_z"],
                self.a2c2_config.base_policy_z_dim,
                name="prefix_z",
            )
        else:
            action = self.policy.infer(batch)
            base_policy_z = np.zeros((self.a2c2_config.base_policy_z_dim,), dtype=np.float32)

        base_action_chunk, valid_action_mask, returned_len = _as_action_chunk(
            action["actions"],
            horizon=self.a2c2_config.action_horizon,
            action_dim=self.a2c2_config.action_dim,
        )
        execute_len = min(int(self.max_len), returned_len, self.a2c2_config.action_horizon)
        if execute_len <= 0:
            raise ValueError("Base policy returned an empty action chunk.")

        infer_ms = float(action.get("policy_timing", {}).get("infer_ms", 0.0))
        self.last_action = action
        return _ActiveChunk(
            base_action_chunk=base_action_chunk,
            valid_action_mask=valid_action_mask,
            base_policy_z=base_policy_z,
            policy_infer_ms=np.array([np.log1p(max(infer_ms, 0.0))], dtype=np.float32),
            execute_len=execute_len,
        )

    def _make_openpi_batch(self, processed_obs: dict[str, np.ndarray]) -> dict[str, Any]:
        nbatch = copy.deepcopy(processed_obs)
        if nbatch["observation"].shape[-1] != 3:
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        joint_positions = nbatch["proprio"][0]
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": self.task_prompt,
        }

        if "observation/egocentric_depth" in nbatch:
            batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

        if self.fine_grained_level > 0:
            reasoner_response = self.reasoner.generate_subtask(
                high_level_task=self.task_prompt,
                multi_modals=[batch["observation/egocentric_camera"]],
            )
            logger.info("* %s", reasoner_response)
            batch["prompt"] = reasoner_response

        self._last_policy_prompt = str(batch["prompt"])
        return batch

    @torch.no_grad()
    def _correct_action(
        self,
        *,
        raw_obs: dict,
        processed_obs: dict[str, np.ndarray],
        active_chunk: _ActiveChunk,
        chunk_offset: int,
    ) -> np.ndarray:
        cfg = self.a2c2_config
        selected_base_action = active_chunk.base_action_chunk[chunk_offset]
        time_feature = A2C2CorrectionHead.make_time_feature(
            torch.tensor([chunk_offset], device=self.a2c2_device),
            cfg.action_horizon,
        )
        observation_state = _fit_vector(processed_obs["proprio"][0], cfg.state_dim, name="proprio")
        batch = {
            "observation_state": self._tensor(observation_state[None]),
            "selected_base_action": self._tensor(selected_base_action[None]),
            "base_action_chunk": self._tensor(active_chunk.base_action_chunk[None]),
            "base_policy_z": self._tensor(active_chunk.base_policy_z[None]),
            "time_feature": time_feature,
            "valid_action_mask": self._tensor(active_chunk.valid_action_mask[None], dtype=torch.bool),
        }

        if cfg.use_rgb:
            batch["rgb_images"] = self._tensor(self._rgb_images(processed_obs))
        if cfg.use_depth:
            batch["depth_images"] = self._tensor(self._depth_images(raw_obs))
        if cfg.use_language:
            batch["language_tokens"] = self._tensor(self._language_tokens[None], dtype=torch.long)
            batch["language_token_mask"] = self._tensor(self._language_token_mask[None], dtype=torch.bool)
        if cfg.use_cam_rel_poses:
            batch["cam_rel_poses"] = self._tensor(self._cam_rel_poses(raw_obs)[None])
        if cfg.use_task_info:
            batch["task_info"] = self._tensor(self._task_info(raw_obs)[None])
        if cfg.use_policy_infer_ms:
            batch["policy_infer_ms"] = self._tensor(active_chunk.policy_infer_ms[None])

        delta = self.a2c2_model(
            batch["observation_state"],
            batch["selected_base_action"],
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
        delta_np = delta.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        return selected_base_action + self.a2c2_correction_scale * delta_np

    def _tensor(self, value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.from_numpy(np.asarray(value))
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        elif tensor.dtype in (torch.float64, torch.float16, torch.bfloat16):
            tensor = tensor.to(dtype=torch.float32)
        return tensor.to(self.a2c2_device, non_blocking=True)

    def _rgb_images(self, processed_obs: dict[str, np.ndarray]) -> np.ndarray:
        images = np.asarray(processed_obs["observation"][0])
        if images.ndim != 4:
            raise ValueError(f"Processed RGB observation must have shape [V, H, W, C], got {images.shape}.")
        if images.shape[-1] != 3 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        if images.shape[0] != self.a2c2_config.num_rgb_views:
            return self._missing_array(
                "rgb_images",
                (1, self.a2c2_config.num_rgb_views, 3, self.a2c2_image_size, self.a2c2_image_size),
            )

        import cv2

        out = np.empty(
            (1, self.a2c2_config.num_rgb_views, 3, self.a2c2_image_size, self.a2c2_image_size),
            dtype=np.float32,
        )
        for idx, image in enumerate(images):
            image = np.asarray(image)
            if image.shape[:2] != (self.a2c2_image_size, self.a2c2_image_size):
                image = cv2.resize(image, (self.a2c2_image_size, self.a2c2_image_size), interpolation=cv2.INTER_AREA)
            if np.issubdtype(image.dtype, np.floating):
                values = image.astype(np.float32, copy=False)
                if values.max(initial=0.0) > 2.0:
                    values = values / 127.5 - 1.0
                else:
                    values = values * 2.0 - 1.0
            else:
                values = image.astype(np.float32) / 127.5 - 1.0
            out[0, idx] = np.transpose(values[..., :3], (2, 0, 1))
        return out

    def _depth_images(self, raw_obs: dict) -> np.ndarray:
        import cv2

        frames = []
        for key, column in zip(DEPTH_CAMERA_KEYS, DEPTH_VIDEO_COLUMNS, strict=True):
            if key not in raw_obs:
                return self._missing_array(
                    "depth_images",
                    (1, self.a2c2_config.num_depth_views, 3, self.a2c2_image_size, self.a2c2_image_size),
                )
            frames.append(
                preprocess_video_frame(
                    cv2,
                    _to_numpy(raw_obs[key]),
                    image_size=self.a2c2_image_size,
                    is_depth=True,
                    column=column,
                    depth_preprocess=self.a2c2_config.depth_preprocess,
                    depth_max_m=self.a2c2_config.depth_max_m,
                )
            )
        return np.stack(frames, axis=0)[None].astype(np.float32, copy=False)

    def _cam_rel_poses(self, raw_obs: dict) -> np.ndarray:
        if "robot_r1::cam_rel_poses" not in raw_obs:
            return self._missing_array("cam_rel_poses", (self.a2c2_config.cam_rel_pose_dim,))
        return _fit_vector(raw_obs["robot_r1::cam_rel_poses"], self.a2c2_config.cam_rel_pose_dim, name="cam_rel_poses")

    def _task_info(self, raw_obs: dict) -> np.ndarray:
        for key in TASK_INFO_KEYS:
            if key in raw_obs:
                return _fit_vector(raw_obs[key], self.a2c2_config.task_info_dim, name=key)
        return self._missing_array("task_info", (self.a2c2_config.task_info_dim,))

    def _missing_array(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        raise KeyError(
            f"A2C2 checkpoint requires {name}, but the current online observation does not provide it. "
            "Use an online BEHAVIOR wrapper/checkpoint that supplies every enabled RGB/depth/task-language feature."
        )
