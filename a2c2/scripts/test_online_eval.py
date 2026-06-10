#!/usr/bin/env python3
"""Fast online A2C2 smoke test with a fake BEHAVIOR environment."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


A2C2_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = A2C2_ROOT.parent
OPENPI_ROOT = WORKSPACE_ROOT / "openpi-comet"
sys.path.insert(0, str(A2C2_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
client_src = OPENPI_ROOT / "packages" / "openpi-client" / "src"
if client_src.is_dir():
    sys.path.insert(0, str(client_src))

from model import A2C2CorrectionHead, A2C2CorrectionHeadConfig, config_from_checkpoint_payload  # noqa: E402
from online import A2C2B1KPolicyWrapper, R1PRO_ACTION_HIGH, R1PRO_ACTION_LOW  # noqa: E402


metrics_spec = importlib.util.spec_from_file_location(
    "a2c2_online_metrics",
    WORKSPACE_ROOT / "BEHAVIOR-1K" / "OmniGibson" / "omnigibson" / "learning" / "a2c2_online_metrics.py",
)
assert metrics_spec is not None and metrics_spec.loader is not None
metrics_module = importlib.util.module_from_spec(metrics_spec)
metrics_spec.loader.exec_module(metrics_module)
A2C2OnlineMetricAccumulator = metrics_module.A2C2OnlineMetricAccumulator


ACTION_DIM = 23
STATE_DIM = 256
Z_DIM = 2048
HORIZON = 4
IMAGE_SIZE = 8
LANGUAGE_MAX_LENGTH = 8
TASK_INFO_DIM = 82


def check_openpi_patch_surface() -> None:
    from openpi.models.pi0 import Pi0
    from openpi.policies.policy import Policy, PolicyRecorder

    assert hasattr(Policy, "infer_with_prefix_z"), "Policy.infer_with_prefix_z is missing"
    assert hasattr(PolicyRecorder, "infer_with_prefix_z"), "PolicyRecorder.infer_with_prefix_z is missing"
    sample_signature = inspect.signature(Pi0.sample_actions)
    assert "return_prefix_z" in sample_signature.parameters, "Pi0.sample_actions is missing return_prefix_z"


def check_legacy_checkpoint_rejected() -> None:
    legacy_payload = {
        "model_state_dict": {},
        "config": {
            "action_horizon": HORIZON,
            "use_base_policy_z": True,
        },
    }
    try:
        config_from_checkpoint_payload(legacy_payload, context="legacy smoke checkpoint")
    except ValueError as exc:
        assert "Pre-RGB/task-language" in str(exc)
    else:
        raise AssertionError("Legacy A2C2 checkpoint config was accepted.")

    A2C2CorrectionHead(
        A2C2CorrectionHeadConfig(
            action_horizon=HORIZON,
            use_rgb=True,
            use_depth=False,
            use_language=True,
            rgb_backbone="small-cnn",
            depth_backbone="small-cnn",
            pretrained_rgb=False,
            pretrained_depth=False,
        )
    )


class FakeBasePolicy:
    metadata = {"fake": True}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer_with_prefix_z(self, batch: dict) -> dict:
        call_index = len(self.calls)
        self.calls.append(
            {
                "call_index": call_index,
                "state0": float(batch["observation/state"][0]),
                "state_shape": tuple(batch["observation/state"].shape),
                "image_shapes": (
                    tuple(batch["observation/egocentric_camera"].shape),
                    tuple(batch["observation/wrist_image_left"].shape),
                    tuple(batch["observation/wrist_image_right"].shape),
                ),
                "prompt": batch["prompt"],
            }
        )
        actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        for offset in range(HORIZON):
            actions[offset] = expected_base(call_index, offset)
        return {
            "actions": actions,
            "prefix_z": np.full((Z_DIM,), call_index + 0.5, dtype=np.float32),
            "policy_timing": {"infer_ms": 10.0 + call_index},
        }


class RecordingCorrectionHead:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        observation_state: torch.Tensor,
        selected_base_action: torch.Tensor,
        base_action_chunk: torch.Tensor,
        base_policy_z: torch.Tensor,
        time_feature: torch.Tensor,
        valid_action_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        step = float(observation_state[0, 0].detach().cpu())
        residual = torch.as_tensor(
            expected_residual(step),
            dtype=selected_base_action.dtype,
            device=selected_base_action.device,
        )[None]
        optional = {key: value for key, value in kwargs.items() if value is not None}
        self.calls.append(
            {
                "observation_state_shape": tuple(observation_state.shape),
                "selected_base_action_shape": tuple(selected_base_action.shape),
                "base_action_chunk_shape": tuple(base_action_chunk.shape),
                "base_policy_z_shape": tuple(base_policy_z.shape),
                "time_feature_shape": tuple(time_feature.shape),
                "valid_action_mask_shape": tuple(valid_action_mask.shape) if valid_action_mask is not None else None,
                "optional_keys": sorted(optional),
                "optional_shapes": {key: tuple(value.shape) for key, value in optional.items()},
                "step": step,
                "selected_base_action": selected_base_action.detach().cpu().numpy()[0].copy(),
                "base_action_chunk": base_action_chunk.detach().cpu().numpy()[0].copy(),
                "base_policy_z0": float(base_policy_z.detach().cpu().numpy()[0, 0]),
                "time_feature": time_feature.detach().cpu().numpy()[0].copy(),
                "valid_action_mask": valid_action_mask.detach().cpu().numpy()[0].copy(),
            }
        )
        return residual


class FakeBehaviorEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.actions: list[np.ndarray] = []

    def observe(self) -> dict:
        proprio = np.zeros((STATE_DIM,), dtype=np.float32)
        proprio[0] = float(self.step_index)
        image_value = np.uint8(self.step_index)
        depth_value = np.float32(0.25 + 0.01 * self.step_index)
        return {
            "robot_r1::proprio": proprio,
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.full((IMAGE_SIZE, IMAGE_SIZE, 3), image_value, dtype=np.uint8),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                image_value,
                dtype=np.uint8,
            ),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                image_value,
                dtype=np.uint8,
            ),
            "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.full(
                (IMAGE_SIZE, IMAGE_SIZE),
                depth_value,
                dtype=np.float32,
            ),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear": np.full(
                (IMAGE_SIZE, IMAGE_SIZE),
                depth_value,
                dtype=np.float32,
            ),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear": np.full(
                (IMAGE_SIZE, IMAGE_SIZE),
                depth_value,
                dtype=np.float32,
            ),
            "robot_r1::cam_rel_poses": np.full((21,), float(self.step_index), dtype=np.float32),
            "task::low_dim": np.full((TASK_INFO_DIM + 12,), float(self.step_index), dtype=np.float32),
            "task_id": np.array([18], dtype=np.int64),
        }

    def step(self, action: torch.Tensor) -> None:
        action_np = action.detach().cpu().numpy()
        assert action_np.shape == (1, ACTION_DIM), action_np.shape
        self.actions.append(action_np[0].copy())
        self.step_index += 1


def make_tiny_checkpoint(path: Path) -> None:
    cfg = A2C2CorrectionHeadConfig(
        action_horizon=HORIZON,
        use_base_policy_z=True,
        dim_model=32,
        n_heads=4,
        n_encoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        mlp_hidden_dim=64,
        use_rgb=True,
        rgb_input_kind="resnet18-features",
        rgb_feature_dim=512,
        use_depth=True,
        use_language=True,
        use_cam_rel_poses=True,
        use_task_info=True,
        use_policy_infer_ms=True,
        rgb_backbone="small-cnn",
        depth_backbone="small-cnn",
        pretrained_rgb=False,
        pretrained_depth=False,
        freeze_rgb=False,
        freeze_depth=False,
        language_max_length=LANGUAGE_MAX_LENGTH,
        task_info_dim=TASK_INFO_DIM,
    )
    model = A2C2CorrectionHead(cfg)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "args": {"image_size": IMAGE_SIZE},
        },
        path,
    )


def expected_base(call_index: int, offset: int) -> np.ndarray:
    center = (R1PRO_ACTION_LOW + R1PRO_ACTION_HIGH) * 0.5
    half_span = (R1PRO_ACTION_HIGH - R1PRO_ACTION_LOW) * 0.5
    pattern = ((np.arange(ACTION_DIM, dtype=np.float32) % 5.0) - 2.0) * 0.08
    pattern += float(call_index) * 0.02 + float(offset) * 0.03
    return (center + half_span * pattern).astype(np.float32, copy=False)


def expected_residual(step: float) -> np.ndarray:
    residual = np.full((ACTION_DIM,), float(step), dtype=np.float32)
    residual[0] = -2.0 - float(step)
    residual[14] = 2.0 + float(step)
    residual[22] = -2.0 - float(step)
    return residual


def expected_time_feature(offset: int) -> np.ndarray:
    phase = 2.0 * np.pi * float(offset) / float(HORIZON - 1)
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def copy_a2c2_info(info: dict) -> dict:
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in info.items()}


def fake_proprio(step: int, *, left_open: bool, right_open: bool) -> np.ndarray:
    proprio = np.zeros((STATE_DIM,), dtype=np.float32)
    proprio[186:189] = np.array([0.01 * step, 0.02 * step, 0.03 * step], dtype=np.float32)
    proprio[225:228] = np.array([0.015 * step, 0.01 * step, 0.02 * step], dtype=np.float32)
    angle = 0.01 * step
    quat = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)], dtype=np.float32)
    proprio[189:193] = quat
    proprio[228:232] = quat
    proprio[193:195] = 0.05 if left_open else 0.0
    proprio[232:234] = 0.05 if right_open else 0.0
    return proprio


def run_smoke() -> None:
    check_openpi_patch_surface()
    check_legacy_checkpoint_rejected()
    os.chdir(OPENPI_ROOT)
    with tempfile.TemporaryDirectory(prefix="a2c2_online_smoke_") as tmpdir:
        checkpoint = Path(tmpdir) / "tiny_a2c2.pt"
        make_tiny_checkpoint(checkpoint)

        base_policy = FakeBasePolicy()
        policy = A2C2B1KPolicyWrapper(
            base_policy,
            a2c2_checkpoint=checkpoint,
            a2c2_device="cpu",
            task_name="tidying_bedroom",
            control_mode="receeding_horizon",
            max_len=3,
        )
        recorder = RecordingCorrectionHead()
        policy.a2c2_model = recorder

        env = FakeBehaviorEnv()
        a2c2_infos = []
        for _ in range(5):
            action = policy.act(env.observe())
            assert policy.last_a2c2_info is not None
            a2c2_infos.append(copy_a2c2_info(policy.last_a2c2_info))
            env.step(action)

    assert [call["state0"] for call in base_policy.calls] == [0.0, 3.0]
    assert [call["state_shape"] for call in base_policy.calls] == [(STATE_DIM,), (STATE_DIM,)]
    assert all(call["image_shapes"] == ((224, 224, 3), (224, 224, 3), (224, 224, 3)) for call in base_policy.calls)
    assert all("tidying" in call["prompt"].lower() or "bedroom" in call["prompt"].lower() for call in base_policy.calls)

    expected_offsets = [0, 1, 2, 0, 1]
    expected_chunk_calls = [0, 0, 0, 1, 1]
    for step, record in enumerate(recorder.calls):
        offset = expected_offsets[step]
        chunk_call = expected_chunk_calls[step]
        assert record["observation_state_shape"] == (1, STATE_DIM)
        assert record["selected_base_action_shape"] == (1, ACTION_DIM)
        assert record["base_action_chunk_shape"] == (1, HORIZON, ACTION_DIM)
        assert record["base_policy_z_shape"] == (1, Z_DIM)
        assert record["time_feature_shape"] == (1, 2)
        assert record["valid_action_mask_shape"] == (1, HORIZON)
        assert record["optional_keys"] == [
            "cam_rel_poses",
            "depth_images",
            "language_token_mask",
            "language_tokens",
            "policy_infer_ms",
            "rgb_features",
            "task_info",
        ]
        assert record["optional_shapes"] == {
            "cam_rel_poses": (1, 21),
            "depth_images": (1, 3, 3, IMAGE_SIZE, IMAGE_SIZE),
            "rgb_features": (1, 3, 512),
            "language_token_mask": (1, LANGUAGE_MAX_LENGTH),
            "language_tokens": (1, LANGUAGE_MAX_LENGTH),
            "policy_infer_ms": (1, 1),
            "task_info": (1, TASK_INFO_DIM),
        }
        assert record["step"] == float(step)
        np.testing.assert_allclose(record["selected_base_action"], expected_base(chunk_call, offset), atol=1e-6)
        np.testing.assert_allclose(record["base_action_chunk"][offset], expected_base(chunk_call, offset), atol=1e-6)
        np.testing.assert_allclose(record["time_feature"], expected_time_feature(offset), atol=1e-6)
        np.testing.assert_array_equal(record["valid_action_mask"], np.ones((HORIZON,), dtype=np.bool_))
        assert record["base_policy_z0"] == chunk_call + 0.5

        unclipped_action = expected_base(chunk_call, offset) + expected_residual(float(step))
        expected_action = np.clip(unclipped_action, R1PRO_ACTION_LOW, R1PRO_ACTION_HIGH)
        np.testing.assert_allclose(env.actions[step], expected_action, atol=1e-6)
        np.testing.assert_allclose(env.actions[step][0], R1PRO_ACTION_LOW[0], atol=1e-6)
        np.testing.assert_allclose(env.actions[step][14], R1PRO_ACTION_HIGH[14], atol=1e-6)
        np.testing.assert_allclose(env.actions[step][22], R1PRO_ACTION_LOW[22], atol=1e-6)
        unclipped_dims = np.isclose(expected_action, unclipped_action, atol=1e-6)
        if np.any(unclipped_dims):
            np.testing.assert_allclose(env.actions[step][unclipped_dims], unclipped_action[unclipped_dims], atol=1e-6)

        info = a2c2_infos[step]
        np.testing.assert_allclose(info["base_action"], expected_base(chunk_call, offset), atol=1e-6)
        np.testing.assert_allclose(info["residual_action"], expected_residual(float(step)), atol=1e-6)
        np.testing.assert_allclose(info["corrected_action_unclipped"], unclipped_action, atol=1e-6)
        np.testing.assert_allclose(info["corrected_action"], expected_action, atol=1e-6)
        np.testing.assert_array_equal(info["clip_mask"], np.not_equal(expected_action, unclipped_action))
        assert info["chunk_offset"] == offset
        assert info["execute_len"] == 3
        assert info["step_counter"] == step

    accumulator = A2C2OnlineMetricAccumulator()
    for step, (info, action) in enumerate(zip(a2c2_infos, env.actions, strict=True)):
        pre_obs = {
            "robot_r1::proprio": fake_proprio(
                step,
                left_open=bool(step % 2),
                right_open=not bool(step % 2),
            )
        }
        post_obs = {
            "robot_r1::proprio": fake_proprio(
                step + 1,
                left_open=bool(action[14] > 0.0),
                right_open=bool(action[22] > 0.0),
            )
        }
        accumulator.step_callback(
            pre_obs=pre_obs,
            post_obs=post_obs,
            action=action,
            policy_info={"a2c2": info},
        )
    metrics = accumulator.gather_results(
        success=False,
        q_score_final=0.0,
        steps=len(env.actions),
        n_trials=1,
        n_success_trials=0,
    )
    json.dumps(metrics)
    online_metrics = metrics["a2c2_online"]
    assert online_metrics["rollout"]["q_score_final"] == 0.0
    assert online_metrics["actions"]["out_of_range_action_ratio"] > 0.0
    assert online_metrics["actions"]["per_group_residual_error"]["left_gripper"]["rmse_mean"] is not None
    assert online_metrics["smoothness"]["action_jerk_norm"]["count"] > 0
    assert online_metrics["end_effector_proxy"]["left"]["position_delta"]["mean"] is not None

    print("online A2C2 smoke test passed")
    print("OpenPI prefix latent API surface checked")
    print("legacy no-RGB/task-language checkpoint rejection checked")
    print("task::low_dim task-info source and truncation checked")
    print("R1Pro action-range clipping checked")
    print("A2C2 online metrics payload and accumulator checked")
    print(f"base policy calls at env steps: {[call['state0'] for call in base_policy.calls]}")
    print(f"correction offsets: {expected_offsets}")
    print(f"env actions checked: {len(env.actions)}")


if __name__ == "__main__":
    run_smoke()
