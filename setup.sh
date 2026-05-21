#!/usr/bin/env bash
set -Eeuo pipefail

# Default to the directory where setup is launched.
ROOT_DIR="${B1K_ROOT:-$(pwd -P)}"
OPENPI_DIR="$ROOT_DIR/openpi-comet"
BEHAVIOR_DIR="$ROOT_DIR/BEHAVIOR-1K"
CONDA_DIR="$ROOT_DIR/miniconda3"
CONDA_ENV="behavior"

OPENPI_COMMIT="4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
BEHAVIOR_TAG="v3.7.2"
BEHAVIOR_COMMIT="88454bd04f75dc57c00ab1f1a00bcde1ff505950"
CHECKPOINT_NAME="pi05-b1kpt12-cs32"
CHECKPOINT_REPO="https://huggingface.co/sunshk/openpi_comet"
CHALLENGE_DEMOS_REPO_ID="behavior-1k/2025-challenge-demos"
CHALLENGE_DEMOS_TASK="task-0001"

export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.uv-python"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/pip"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"
export TMPDIR="$ROOT_DIR/tmp"
export OMNIGIBSON_DATA_PATH="$BEHAVIOR_DIR/OmniGibson/datasets"
export OMNIGIBSON_APPDATA_PATH="$ROOT_DIR/og-appdata"
export PATH="$HOME/.local/bin:$CONDA_DIR/bin:$PATH"

SETUP_LOG="${SETUP_LOG:-$ROOT_DIR/setup_run.log}"
CHALLENGE_DEMOS_DIR="$OMNIGIBSON_DATA_PATH/2025-challenge-demos"

setup_logging() {
  mkdir -p "$ROOT_DIR"
  exec > >(tee "$SETUP_LOG") 2>&1
  printf "Setup log: %s\n" "$SETUP_LOG"
}

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if [ "$n" -ge "$attempts" ]; then
      return 1
    fi
    log "Command failed, retrying ($n/$attempts): $*"
    n=$((n + 1))
    sleep 10
  done
}

require_root_for_apt() {
  if [ "$(id -u)" -ne 0 ]; then
    die "This setup installs apt packages. Please run as root, or preinstall the apt requirements."
  fi
}

prepare_dirs() {
  log "Preparing directories under $ROOT_DIR"
  mkdir -p \
    "$ROOT_DIR" \
    "$UV_CACHE_DIR" \
    "$UV_PYTHON_INSTALL_DIR" \
    "$PIP_CACHE_DIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TMPDIR" \
    "$OMNIGIBSON_APPDATA_PATH"
}

install_apt_packages() {
  require_root_for_apt
  log "Installing system packages"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git git-lfs curl wget ca-certificates pkg-config python3-dev \
    xvfb xauth ffmpeg \
    libxt6 libglu1-mesa libsm6 libxext6 libxrender1 libxi6 \
    libxrandr2 libxcursor1 libxinerama1 libxfixes3 \
    libxkbcommon-x11-0 libegl1 libgl1 libglvnd0 \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
  git lfs install
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "Using existing uv: $(uv --version)"
    return
  fi

  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv install did not put uv on PATH"
  log "Installed $(uv --version)"
}

install_miniconda() {
  if [ -x "$CONDA_DIR/bin/conda" ]; then
    log "Using existing conda: $("$CONDA_DIR/bin/conda" --version)"
    return
  fi

  log "Installing Miniconda to $CONDA_DIR"
  local installer="$ROOT_DIR/Miniconda3-latest-Linux-x86_64.sh"
  retry 3 wget -O "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_DIR"
  rm -f "$installer"
  "$CONDA_DIR/bin/conda" --version
}

accept_conda_tos() {
  log "Accepting Anaconda ToS when supported by this conda version"
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
}

clone_openpi() {
  if [ ! -d "$OPENPI_DIR/.git" ]; then
    log "Cloning openpi-comet"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mli0603/openpi-comet.git "$OPENPI_DIR"
  fi

  log "Checking out openpi-comet commit $OPENPI_COMMIT"
  git -C "$OPENPI_DIR" fetch --all --tags
  git -C "$OPENPI_DIR" checkout "$OPENPI_COMMIT"
  git -C "$OPENPI_DIR" rev-parse HEAD
}

clone_behavior() {
  if [ -e "$BEHAVIOR_DIR" ] && [ ! -d "$BEHAVIOR_DIR/.git" ]; then
    log "Removing non-git BEHAVIOR directory left from an interrupted setup: $BEHAVIOR_DIR"
    rm -rf "$BEHAVIOR_DIR"
  fi

  if [ ! -d "$BEHAVIOR_DIR/.git" ]; then
    log "Cloning BEHAVIOR-1K $BEHAVIOR_TAG with sparse checkout"
    GIT_LFS_SKIP_SMUDGE=1 git clone \
      --depth 1 --filter=blob:none --sparse \
      --branch "$BEHAVIOR_TAG" \
      https://github.com/StanfordVL/BEHAVIOR-1K.git "$BEHAVIOR_DIR"
  fi

  log "Configuring BEHAVIOR sparse checkout"
  git -C "$BEHAVIOR_DIR" sparse-checkout set --no-cone \
    setup.sh README.md OmniGibson bddl3 joylo asset_pipeline eval-jobqueue knowledgebase docs/assets

  local actual_commit
  actual_commit="$(git -C "$BEHAVIOR_DIR" rev-parse HEAD)"
  if [ "$actual_commit" != "$BEHAVIOR_COMMIT" ]; then
    die "Unexpected BEHAVIOR commit: $actual_commit (expected $BEHAVIOR_COMMIT)"
  fi
  log "BEHAVIOR commit: $actual_commit"
}

setup_openpi_env() {
  log "Creating openpi-comet uv environment"
  cd "$OPENPI_DIR"
  GIT_LFS_SKIP_SMUDGE=1 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_CONCURRENT_BUILDS=1 \
    uv sync --no-dev

  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  uv pip install pytest
  uv pip check
}

create_behavior_conda_env() {
  log "Creating conda env: $CONDA_ENV"
  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -n "$CONDA_ENV" python=3.10 -c conda-forge -y
  fi

  conda activate "$CONDA_ENV"
  conda install -c conda-forge git-lfs pip -y
  git lfs install

  python -m pip install "numpy<2" "setuptools<=79"
  python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
}

setup_behavior() {
  log "Running BEHAVIOR / OmniGibson setup without dataset"
  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"

  cd "$BEHAVIOR_DIR"
  mkdir -p "$OMNIGIBSON_DATA_PATH" "$OMNIGIBSON_APPDATA_PATH"

  ./setup.sh --omnigibson --bddl --joylo --eval \
    --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
}

fix_behavior_numpy_stack() {
  log "Pinning BEHAVIOR NumPy / SciPy stack"
  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"

  python -m pip install opencv-contrib-python==4.11.0.86 --no-deps

  local site="$CONDA_DIR/envs/$CONDA_ENV/lib/python3.10/site-packages"
  rm -rf \
    "$site"/numpy \
    "$site"/numpy.libs \
    "$site"/numpy-*.dist-info \
    "$site"/scipy \
    "$site"/scipy.libs \
    "$site"/scipy-*.dist-info

  conda install -c conda-forge numpy=1.26.4 scipy=1.14.1 --force-reinstall -y
  python -m pip check
}

download_behavior_dataset() {
  log "Downloading BEHAVIOR simulator assets and challenge task instances"
  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"

  cd "$BEHAVIOR_DIR"
  mkdir -p "$OMNIGIBSON_DATA_PATH" "$OMNIGIBSON_APPDATA_PATH"
  ./setup.sh --dataset --accept-dataset-tos --confirm-no-conda
}

ensure_huggingface_hub() {
  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"

  if python - <<'PY' >/dev/null 2>&1
import inspect
from huggingface_hub import snapshot_download

params = set(inspect.signature(snapshot_download).parameters)
required = {"repo_id", "repo_type", "local_dir", "allow_patterns", "max_workers"}
raise SystemExit(0 if required.issubset(params) else 1)
PY
  then
    python - <<'PY'
import huggingface_hub
print("huggingface_hub", huggingface_hub.__version__)
PY
    return
  fi

  log "Installing huggingface_hub for selective demo download"
  python -m pip install "huggingface_hub>=0.24.0"
}

verify_challenge_demos_subset() {
  log "Checking 2025 challenge demos subset"

  local ann_dir="$CHALLENGE_DEMOS_DIR/annotations/$CHALLENGE_DEMOS_TASK"
  local data_dir="$CHALLENGE_DEMOS_DIR/data/$CHALLENGE_DEMOS_TASK"
  local meta_dir="$CHALLENGE_DEMOS_DIR/meta/episodes/$CHALLENGE_DEMOS_TASK"
  local meta_root="$CHALLENGE_DEMOS_DIR/meta"
  local videos_dir="$CHALLENGE_DEMOS_DIR/videos/$CHALLENGE_DEMOS_TASK"

  test -f "$CHALLENGE_DEMOS_DIR/.gitattributes" || die "Missing challenge demos .gitattributes"
  test -f "$CHALLENGE_DEMOS_DIR/README.md" || die "Missing challenge demos README.md"
  test -d "$ann_dir" || die "Missing challenge demos annotations: $ann_dir"
  test -d "$data_dir" || die "Missing challenge demos data: $data_dir"
  test -f "$meta_root/info.json" || die "Missing challenge demos meta file: $meta_root/info.json"
  test -f "$meta_root/tasks.jsonl" || die "Missing challenge demos meta file: $meta_root/tasks.jsonl"
  test -f "$meta_root/episodes.jsonl" || die "Missing challenge demos meta file: $meta_root/episodes.jsonl"
  test -f "$meta_root/episodes_stats.jsonl" || die "Missing challenge demos meta file: $meta_root/episodes_stats.jsonl"
  test -d "$meta_dir" || die "Missing challenge demos meta episodes: $meta_dir"
  test -d "$videos_dir" || die "Missing challenge demos videos: $videos_dir"

  local ann_sample data_sample meta_sample video_sample
  ann_sample="$(find "$ann_dir" -type f -name '*.json' -print -quit)"
  data_sample="$(find "$data_dir" -type f -name '*.parquet' -print -quit)"
  meta_sample="$(find "$meta_dir" -type f -name '*.json' -print -quit)"
  video_sample="$(find "$videos_dir" -type f -name '*.mp4' -print -quit)"

  test -n "$ann_sample" || die "No annotation json files found under $ann_dir"
  test -n "$data_sample" || die "No parquet files found under $data_dir"
  test -n "$meta_sample" || die "No meta json files found under $meta_dir"
  test -n "$video_sample" || die "No mp4 files found under $videos_dir"

  if head -c 128 "$data_sample" | grep -q "git-lfs.github.com/spec/v1"; then
    die "Challenge demos data file is still a Git LFS pointer: $data_sample"
  fi
  if head -c 128 "$video_sample" | grep -q "git-lfs.github.com/spec/v1"; then
    die "Challenge demos video file is still a Git LFS pointer: $video_sample"
  fi
  if head -c 128 "$meta_root/episodes_stats.jsonl" | grep -q "git-lfs.github.com/spec/v1"; then
    die "Challenge demos episodes_stats.jsonl is still a Git LFS pointer"
  fi

  du -sh "$CHALLENGE_DEMOS_DIR"
}

download_challenge_demos_snapshot() {
  CHALLENGE_DEMOS_REPO_ID="$CHALLENGE_DEMOS_REPO_ID" \
  CHALLENGE_DEMOS_TASK="$CHALLENGE_DEMOS_TASK" \
  CHALLENGE_DEMOS_DIR="$CHALLENGE_DEMOS_DIR" \
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["CHALLENGE_DEMOS_REPO_ID"]
task = os.environ["CHALLENGE_DEMOS_TASK"]
local_dir = os.environ["CHALLENGE_DEMOS_DIR"]

# The HF dataset stores per-task metadata under meta/episodes/<task>;
# keep the top-level dataset metadata files that loaders expect as well.
allow_patterns = [
    f"annotations/{task}/**",
    f"data/{task}/**",
    f"meta/episodes/{task}/**",
    f"videos/{task}/**",
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    ".gitattributes",
    "README.md",
]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=allow_patterns,
    max_workers=8,
)
PY
}

download_challenge_demos() {
  log "Downloading 2025 challenge demos subset: $CHALLENGE_DEMOS_TASK"
  ensure_huggingface_hub
  mkdir -p "$CHALLENGE_DEMOS_DIR"

  retry 3 download_challenge_demos_snapshot

  verify_challenge_demos_subset
}

install_behavior_deps_into_openpi() {
  log "Installing BEHAVIOR packages into openpi-comet uv env"
  cd "$OPENPI_DIR"
  uv pip install -e "$BEHAVIOR_DIR/bddl3"
  uv pip install -e "$BEHAVIOR_DIR/OmniGibson[eval]"
  uv pip check
}

download_checkpoint() {
  local ckpt_dir="$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME"
  if [ -d "$ckpt_dir" ] && [ -n "$(find "$ckpt_dir" -type f -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
    log "Checkpoint already exists at $ckpt_dir"
    du -sh "$ckpt_dir"
    return
  fi

  log "Downloading checkpoint with Hugging Face git-lfs sparse checkout"
  rm -rf "$ROOT_DIR/hf-openpi-comet-fullgit"
  cd "$ROOT_DIR"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse "$CHECKPOINT_REPO" "$ROOT_DIR/hf-openpi-comet-fullgit"

  cd "$ROOT_DIR/hf-openpi-comet-fullgit"
  git sparse-checkout set "$CHECKPOINT_NAME"
  git lfs install --local
  git config lfs.concurrenttransfers 16
  retry 3 git lfs pull --include="$CHECKPOINT_NAME/**" --exclude=""

  mkdir -p "$OPENPI_DIR/checkpoints"
  rm -rf "$ckpt_dir"
  mv "$ROOT_DIR/hf-openpi-comet-fullgit/$CHECKPOINT_NAME" "$ckpt_dir"
  rm -rf "$ROOT_DIR/hf-openpi-comet-fullgit"
  du -sh "$ckpt_dir"
}

apply_openpi_behavior_patch() {
  log "Copying openpi-comet BEHAVIOR learning patch into OmniGibson"
  cp -rv "$OPENPI_DIR/src/behavior/learning/." "$BEHAVIOR_DIR/OmniGibson/omnigibson/learning/"
}

patch_openpi_server_import() {
  log "Patching openpi-comet server to avoid importing full OmniGibson in the policy process"

  cat > "$OPENPI_DIR/src/openpi/shared/b1k_network_utils.py" <<'PY'
import asyncio
from copy import deepcopy
import functools
import http
import logging
import msgpack
import numpy as np
import time
import traceback
from typing import Any, Optional

import websockets
import websockets.asyncio.server as _server

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WebsocketPolicyServer:
    def __init__(self, policy: Any, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info("Starting websocket server on %s:%s...", self._host, self._port)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                infer_start = time.monotonic()
                action = self._policy.act(deepcopy(result))
                infer_time = time.monotonic() - infer_start

                response = {"action": action.cpu().numpy(), "server_timing": {"infer_ms": infer_time * 1000}}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.error("Error in connection from %s:\n%s", websocket.remote_address, traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
PY

  OPENPI_DIR="$OPENPI_DIR" python - <<'PY'
import os
from pathlib import Path

openpi_dir = Path(os.environ["OPENPI_DIR"])
serve_path = openpi_dir / "scripts/serve_b1k.py"
text = serve_path.read_text()
old = "from omnigibson.learning.utils.network_utils import WebsocketPolicyServer\n"
new = "from openpi.shared.b1k_network_utils import WebsocketPolicyServer\n"
if old not in text and new not in text:
    raise SystemExit("Could not find serve_b1k WebsocketPolicyServer import")
serve_path.write_text(text.replace(old, new))

policy_path = openpi_dir / "src/openpi/policies/b1k_policy.py"
text = policy_path.read_text()
prefix = "import numpy as np\n"
suffix = "\nfrom openpi import transforms\n"
start = text.find(prefix)
end = text.find(suffix, start + len(prefix))
if start == -1 or end == -1:
    raise SystemExit("Could not find b1k_policy import block anchors")

new = '''try:
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
except ModuleNotFoundError:
    from collections import OrderedDict

    PROPRIOCEPTION_INDICES = {
        "R1Pro": OrderedDict(
            {
                "arm_left_qpos": np.s_[158:165],
                "gripper_left_qpos": np.s_[193:195],
                "arm_right_qpos": np.s_[197:204],
                "gripper_right_qpos": np.s_[232:234],
                "trunk_qpos": np.s_[236:240],
                "base_qvel": np.s_[253:256],
            }
        )
    }
'''
policy_path.write_text(text[: start + len(prefix)] + new + text[end:])
PY
}

verify_install() {
  log "Verifying installation"

  cd "$OPENPI_DIR"
  uv run --no-sync python --version
  uv pip check --python "$OPENPI_DIR/.venv/bin/python"
  uv run --no-sync python - <<'PY'
import importlib.util
from pathlib import Path

from openpi.shared.b1k_network_utils import WebsocketPolicyServer

spec = importlib.util.spec_from_file_location("serve_b1k_check", Path("scripts/serve_b1k.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.WebsocketPolicyServer is WebsocketPolicyServer
print("openpi server imports ok")
PY

  # shellcheck source=/dev/null
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  export OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH"
  export OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH"

  python - <<'PY'
import numpy, scipy, torch, omnigibson
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("omnigibson", omnigibson.__version__)
PY
  python -m pip check

  test -d "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" || die "Missing BEHAVIOR challenge instances"
  verify_challenge_demos_subset
  test -d "$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME" || die "Missing checkpoint"
  du -sh "$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME" "$OMNIGIBSON_DATA_PATH"
}

write_evaluation_readme() {
  local out="$ROOT_DIR/evaluation_commands.md"
  log "Writing evaluation commands to $out"
  cat > "$out" <<EOF
# Official-style evaluation commands for Runpod

These follow the openpi-comet README evaluation flow:

1. Start the websocket policy server.
2. Run BEHAVIOR-1K eval.py against that server.

Run them in two terminals after setup finishes.

## Terminal 1: start the openpi-comet websocket policy server

    cd "$OPENPI_DIR"
    export PATH="\$HOME/.local/bin:$CONDA_DIR/bin:\$PATH"
    export UV_CACHE_DIR="$UV_CACHE_DIR"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
    export JAX_COMPILATION_CACHE_DIR="$ROOT_DIR/.cache/jax"

    uv run --no-sync scripts/serve_b1k.py \
      --task_name=picking_up_trash \
      --control_mode=receeding_horizon \
      --max_len=32 \
      policy:checkpoint \
      --policy.config=pi05_b1k-base \
      --policy.dir=./checkpoints/$CHECKPOINT_NAME

## Terminal 2: run BEHAVIOR evaluation

    cd "$BEHAVIOR_DIR"
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"

    RUN_LOG="$BEHAVIOR_DIR/output/picking_up_trash_\$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "\$RUN_LOG"

    export HYDRA_FULL_ERROR=1
    export OMNI_KIT_ACCEPT_EULA=YES
    export OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH"
    export OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH"
    export TMPDIR="$TMPDIR"

    xvfb-run -a -s "-screen 0 1280x720x24" python OmniGibson/omnigibson/learning/eval.py \
      policy=websocket \
      task.name=picking_up_trash \
      log_path="\$RUN_LOG" \
      model.host=127.0.0.1 \
      env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
      eval_instance_ids="[0]" \
      write_video=true

Output videos are written under:

    \$RUN_LOG/videos/

Optional quick listing:

    find "\$RUN_LOG" -name "*.mp4" -print
EOF
}

print_next_steps() {
  cat <<EOF

============================================================
Setup finished.
============================================================

Evaluation commands were written to:
  $ROOT_DIR/evaluation_commands.md

The file follows the official openpi-comet two-step eval flow, with only the
Runpod/headless and video-output options kept.
EOF
}

main() {
  setup_logging
  prepare_dirs
  install_apt_packages
  install_uv
  install_miniconda
  accept_conda_tos
  clone_openpi
  clone_behavior
  setup_openpi_env
  create_behavior_conda_env
  setup_behavior
  fix_behavior_numpy_stack
  download_behavior_dataset
  download_challenge_demos
  install_behavior_deps_into_openpi
  download_checkpoint
  apply_openpi_behavior_patch
  patch_openpi_server_import
  verify_install
  write_evaluation_readme
  print_next_steps
}

main "$@"
