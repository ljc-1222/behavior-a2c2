#!/usr/bin/env python3
"""Serve OpenPI-COMET for BEHAVIOR-1K with online A2C2 residual correction."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import socket
import sys

import tyro


A2C2_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = A2C2_ROOT.parent
sys.path.insert(0, str(A2C2_ROOT / "src"))
openpi_src = WORKSPACE_ROOT / "openpi-comet" / "src"
if openpi_src.is_dir():
    sys.path.insert(0, str(openpi_src))

from online import A2C2B1KPolicyWrapper  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared.b1k_network_utils import WebsocketPolicyServer  # noqa: E402
from openpi.training import config as _config  # noqa: E402


@dataclasses.dataclass
class Checkpoint:
    """Load a base OpenPI policy from a trained checkpoint."""

    config: str = "pi05_b1k-base"
    dir: str = "./checkpoints/pi05-b1kpt50-cs32"


@dataclasses.dataclass
class Default:
    """Use the workspace default task18 OpenPI-COMET checkpoint."""


@dataclasses.dataclass
class Args:
    """Arguments for the online A2C2 BEHAVIOR websocket server."""

    # BEHAVIOR task name used for task prompt lookup.
    task_name: str = "tidying_bedroom"

    # Fallback prompt for the OpenPI policy when the transformed input has no prompt.
    default_prompt: str | None = None

    # Port to serve the websocket policy on.
    port: int = 8000

    # Record the base OpenPI policy's transformed inputs and outputs for debugging.
    record: bool = False

    # Base OpenPI policy checkpoint.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # B1K control mode. Online A2C2 currently requires receding horizon.
    control_mode: str = "receeding_horizon"
    max_len: int = 32
    action_horizon: int = 5
    temporal_ensemble_max: int = 3
    fine_grained_level: int = 0

    # A2C2 correction-head checkpoint.
    a2c2_checkpoint: Path = Path("../a2c2/ckpt/model_latent.pt")
    a2c2_device: str = "auto"
    a2c2_image_size: int | None = None
    a2c2_language_instruction: str | None = None
    a2c2_correction_scale: float = 1.0
    allow_missing_online_features: bool = False


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            checkpoint = args.policy
        case Default():
            checkpoint = Checkpoint()
        case _:
            raise ValueError(f"Unsupported policy spec: {args.policy!r}")

    return _policy_config.create_trained_policy(
        _config.get_config(checkpoint.config),
        checkpoint.dir,
        default_prompt=args.default_prompt,
    )


def main(args: Args) -> None:
    logging.info("Using task_name: %s", args.task_name)
    base_policy = create_policy(args)
    policy_metadata = dict(base_policy.metadata)
    policy_metadata["a2c2"] = {
        "checkpoint": str(args.a2c2_checkpoint),
        "device": args.a2c2_device,
        "correction_scale": args.a2c2_correction_scale,
    }

    if args.record:
        base_policy = _policy.PolicyRecorder(base_policy, "policy_records")

    policy = A2C2B1KPolicyWrapper(
        base_policy,
        task_name=args.task_name,
        control_mode=args.control_mode,
        max_len=args.max_len,
        action_horizon=args.action_horizon,
        temporal_ensemble_max=args.temporal_ensemble_max,
        fine_grained_level=args.fine_grained_level,
        a2c2_checkpoint=args.a2c2_checkpoint,
        a2c2_device=args.a2c2_device,
        a2c2_image_size=args.a2c2_image_size,
        a2c2_language_instruction=args.a2c2_language_instruction,
        a2c2_correction_scale=args.a2c2_correction_scale,
        allow_missing_online_features=args.allow_missing_online_features,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating A2C2 server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
