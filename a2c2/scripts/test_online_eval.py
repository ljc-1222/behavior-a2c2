#!/usr/bin/env python3
"""Fast online A2C2 smoke test with a fake BEHAVIOR environment."""

from __future__ import annotations

from dataclasses import asdict
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

from model import A2C2CorrectionHead, A2C2CorrectionHeadConfig  # noqa: E402
from online import A2C2B1KPolicyWrapper  # noqa: E402


ACTION_DIM = 23
STATE_DIM = 256
Z_DIM = 2048
HORIZON = 4


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
            actions[offset] = call_index * 100.0 + offset * 10.0 + np.arange(ACTION_DIM, dtype=np.float32) * 0.001
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
        residual = torch.full_like(selected_base_action, step)
        self.calls.append(
            {
                "observation_state_shape": tuple(observation_state.shape),
                "selected_base_action_shape": tuple(selected_base_action.shape),
                "base_action_chunk_shape": tuple(base_action_chunk.shape),
                "base_policy_z_shape": tuple(base_policy_z.shape),
                "time_feature_shape": tuple(time_feature.shape),
                "valid_action_mask_shape": tuple(valid_action_mask.shape) if valid_action_mask is not None else None,
                "optional_keys": sorted(key for key, value in kwargs.items() if value is not None),
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
        return {
            "robot_r1::proprio": proprio,
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.full((8, 8, 3), image_value, dtype=np.uint8),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.full((8, 8, 3), image_value, dtype=np.uint8),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.full((8, 8, 3), image_value, dtype=np.uint8),
            "robot_r1::cam_rel_poses": np.full((21,), float(self.step_index), dtype=np.float32),
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
        use_rgb=False,
        use_depth=False,
        use_language=False,
        use_cam_rel_poses=False,
        use_task_info=False,
        use_policy_infer_ms=False,
    )
    model = A2C2CorrectionHead(cfg)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "args": {"image_size": 8},
        },
        path,
    )


def expected_base(call_index: int, offset: int) -> np.ndarray:
    return call_index * 100.0 + offset * 10.0 + np.arange(ACTION_DIM, dtype=np.float32) * 0.001


def expected_time_feature(offset: int) -> np.ndarray:
    phase = 2.0 * np.pi * float(offset) / float(HORIZON - 1)
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def run_smoke() -> None:
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
        for _ in range(5):
            action = policy.act(env.observe())
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
        assert record["optional_keys"] == []
        assert record["step"] == float(step)
        np.testing.assert_allclose(record["selected_base_action"], expected_base(chunk_call, offset), atol=1e-6)
        np.testing.assert_allclose(record["base_action_chunk"][offset], expected_base(chunk_call, offset), atol=1e-6)
        np.testing.assert_allclose(record["time_feature"], expected_time_feature(offset), atol=1e-6)
        np.testing.assert_array_equal(record["valid_action_mask"], np.ones((HORIZON,), dtype=np.bool_))
        assert record["base_policy_z0"] == chunk_call + 0.5

        expected_action = expected_base(chunk_call, offset) + float(step)
        np.testing.assert_allclose(env.actions[step], expected_action, atol=1e-6)

    print("online A2C2 smoke test passed")
    print(f"base policy calls at env steps: {[call['state0'] for call in base_policy.calls]}")
    print(f"correction offsets: {expected_offsets}")
    print(f"env actions checked: {len(env.actions)}")


if __name__ == "__main__":
    run_smoke()
