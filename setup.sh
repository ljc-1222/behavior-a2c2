#!/usr/bin/env bash
set -Eeuo pipefail

# Default to the directory where setup is launched.
ROOT_DIR="${B1K_ROOT:-$(pwd -P)}"
OPENPI_DIR="$ROOT_DIR/openpi-comet"
BEHAVIOR_DIR="$ROOT_DIR/BEHAVIOR-1K"
PROJECT_CONDA_DIR="$ROOT_DIR/miniconda3"
DEFAULT_CONDA_DIR="${B1K_DEFAULT_CONDA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/behavior-a2c2/miniforge3}"
CONDA_DIR="${B1K_CONDA_DIR:-}"
CONDA_EXE="${B1K_CONDA_EXE:-}"
CONDA_ENV="${B1K_CONDA_ENV:-behavior}"

OPENPI_SUBMODULE_COMMIT="0a08b229505da406f1041e15cf01c77ebc8953cf"
BEHAVIOR_SUBMODULE_COMMIT="398ff024db4c5b5e8be0fd38e632bc00579eb470"
TASK_NAME="${B1K_TASK_NAME:-tidying_bedroom}"
TASK_DIR="${B1K_TASK_DIR:-task-0018}"
CHECKPOINT_NAME="${B1K_CHECKPOINT_NAME:-pi05-b1kpt50-cs32}"
CHECKPOINT_REPO="https://huggingface.co/sunshk/openpi_comet"
CHALLENGE_DEMOS_REPO_ID="behavior-1k/2025-challenge-demos"
CHALLENGE_DEMOS_TASK="${B1K_CHALLENGE_DEMOS_TASK:-$TASK_DIR}"
CONDA_INSTALLER_URL="${B1K_CONDA_INSTALLER_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"

SYSTEM_PACKAGES="${B1K_SYSTEM_PACKAGES:-auto}"
INSTALL_CONDA="${B1K_INSTALL_CONDA:-auto}"
DOWNLOAD_BEHAVIOR_DATASET="${B1K_DOWNLOAD_BEHAVIOR_DATASET:-1}"
DOWNLOAD_CHALLENGE_DEMOS="${B1K_DOWNLOAD_CHALLENGE_DEMOS:-0}"
DOWNLOAD_CHECKPOINT="${B1K_DOWNLOAD_CHECKPOINT:-1}"
UPDATE_SUBMODULES="${B1K_UPDATE_SUBMODULES:-1}"
BEHAVIOR_SPARSE_CHECKOUT="${B1K_BEHAVIOR_SPARSE_CHECKOUT:-1}"

APT_PACKAGES=(
  git git-lfs curl wget ca-certificates
  xvfb xauth ffmpeg
  libxt6 libglu1-mesa libsm6 libxext6 libxrender1 libxi6
  libxrandr2 libxcursor1 libxinerama1 libxfixes3
  libxkbcommon-x11-0 libegl1 libgl1 libglvnd0
)

export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.uv-python"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/pip"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"
export TMPDIR="$ROOT_DIR/tmp"
export OMNIGIBSON_DATA_PATH="$BEHAVIOR_DIR/OmniGibson/datasets"
export OMNIGIBSON_APPDATA_PATH="$ROOT_DIR/og-appdata"
export PATH="$HOME/.local/bin:$PATH"

SETUP_LOG="${SETUP_LOG:-$ROOT_DIR/setup_run.log}"
CHALLENGE_DEMOS_DIR="$OMNIGIBSON_DATA_PATH/2025-challenge-demos"

usage() {
  cat <<EOF
Usage: bash setup.sh [OPTIONS]

Sets up the behavior-a2c2 workspace using openpi-comet + BEHAVIOR-1K submodules.

Options:
  -h, --help                         Show this help message
  --system-packages MODE             auto, install, or skip (default: $SYSTEM_PACKAGES)
  --skip-system-packages             Alias for --system-packages skip
  --install-system-packages          Alias for --system-packages install
  --conda-dir DIR                    Use or install conda at DIR
  --conda-env NAME                   Conda env name (default: $CONDA_ENV)
  --no-install-conda                 Require an existing conda installation
  --download-conda                   Download Miniforge without prompting if no conda is found
  --skip-behavior-dataset            Do not download BEHAVIOR runtime assets/task instances
  --download-challenge-demos         Download the optional 2025 challenge demos subset
  --skip-challenge-demos             Do not download the optional challenge demos subset
  --skip-checkpoint                  Do not download the policy checkpoint
  --checkpoint-name NAME             Checkpoint directory under sunshk/openpi_comet
  --skip-submodule-update            Require existing submodule checkouts; do not run git submodule update
  --no-behavior-sparse-checkout      Keep the BEHAVIOR-1K submodule working tree as-is

Environment overrides:
  B1K_ROOT, B1K_CONDA_DIR, B1K_CONDA_EXE, B1K_CONDA_ENV,
  B1K_SYSTEM_PACKAGES, B1K_INSTALL_CONDA,
  B1K_DOWNLOAD_BEHAVIOR_DATASET, B1K_DOWNLOAD_CHALLENGE_DEMOS,
  B1K_DOWNLOAD_CHECKPOINT, B1K_CHECKPOINT_NAME,
  B1K_UPDATE_SUBMODULES, B1K_BEHAVIOR_SPARSE_CHECKOUT,
  B1K_TASK_NAME, B1K_TASK_DIR
EOF
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --system-packages)
        [ "$#" -ge 2 ] || die "--system-packages requires auto, install, or skip"
        SYSTEM_PACKAGES="$2"
        shift 2
        ;;
      --skip-system-packages)
        SYSTEM_PACKAGES="skip"
        shift
        ;;
      --install-system-packages)
        SYSTEM_PACKAGES="install"
        shift
        ;;
      --conda-dir)
        [ "$#" -ge 2 ] || die "--conda-dir requires a path"
        CONDA_DIR="$2"
        shift 2
        ;;
      --conda-env)
        [ "$#" -ge 2 ] || die "--conda-env requires a name"
        CONDA_ENV="$2"
        shift 2
        ;;
      --no-install-conda)
        INSTALL_CONDA="never"
        shift
        ;;
      --download-conda)
        INSTALL_CONDA="download"
        shift
        ;;
      --skip-behavior-dataset)
        DOWNLOAD_BEHAVIOR_DATASET=0
        shift
        ;;
      --download-challenge-demos)
        DOWNLOAD_CHALLENGE_DEMOS=1
        shift
        ;;
      --skip-challenge-demos)
        DOWNLOAD_CHALLENGE_DEMOS=0
        shift
        ;;
      --skip-checkpoint)
        DOWNLOAD_CHECKPOINT=0
        shift
        ;;
      --checkpoint-name)
        [ "$#" -ge 2 ] || die "--checkpoint-name requires a name"
        CHECKPOINT_NAME="$2"
        shift 2
        ;;
      --skip-submodule-update)
        UPDATE_SUBMODULES=0
        shift
        ;;
      --no-behavior-sparse-checkout)
        BEHAVIOR_SPARSE_CHECKOUT=0
        shift
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

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

apt_get() {
  if [ "$(id -u)" -eq 0 ]; then
    DEBIAN_FRONTEND=noninteractive apt-get "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo env DEBIAN_FRONTEND=noninteractive apt-get "$@"
  else
    return 127
  fi
}

find_missing_apt_packages() {
  MISSING_APT_PACKAGES=()

  if ! command -v dpkg-query >/dev/null 2>&1; then
    MISSING_APT_PACKAGES=("${APT_PACKAGES[@]}")
    return
  fi

  local pkg
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      MISSING_APT_PACKAGES+=("$pkg")
    fi
  done
}

install_apt_packages() {
  case "$SYSTEM_PACKAGES" in
    auto|install|skip) ;;
    *) die "Invalid --system-packages mode: $SYSTEM_PACKAGES" ;;
  esac

  if [ "$SYSTEM_PACKAGES" = "skip" ]; then
    log "Skipping apt packages by request"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    [ "$SYSTEM_PACKAGES" = "auto" ] && { log "apt-get not found; assuming system packages are managed externally"; return; }
    die "apt-get not found; cannot install system packages"
  fi

  find_missing_apt_packages
  if [ "${#MISSING_APT_PACKAGES[@]}" -eq 0 ]; then
    log "System apt packages already present"
    git lfs install
    return
  fi

  log "Installing missing system packages: ${MISSING_APT_PACKAGES[*]}"
  if ! apt_get update; then
    die "Could not run apt-get update. Install these packages manually or rerun with --skip-system-packages: ${MISSING_APT_PACKAGES[*]}"
  fi
  if ! apt_get install -y --no-install-recommends "${MISSING_APT_PACKAGES[@]}"; then
    die "Could not install apt packages. Install them manually or rerun with --skip-system-packages: ${MISSING_APT_PACKAGES[*]}"
  fi
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

install_managed_conda() {
  [ "$INSTALL_CONDA" != "never" ] || die "No conda found and automatic conda installation is disabled"

  log "Installing Miniforge to $CONDA_DIR"
  mkdir -p "$(dirname "$CONDA_DIR")"
  local installer="$TMPDIR/Miniforge3-Linux-x86_64.sh"
  retry 3 wget -O "$installer" "$CONDA_INSTALLER_URL"
  bash "$installer" -b -p "$CONDA_DIR"
  rm -f "$installer"
  "$CONDA_DIR/bin/conda" --version
}

is_conda_root() {
  local candidate="${1%/}"
  [ -n "$candidate" ] && [ -x "$candidate/bin/conda" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]
}

canonical_dir() {
  local candidate="${1%/}"
  (cd "$candidate" && pwd -P)
}

expand_user_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

find_outer_conda_root() {
  local dir parent candidate name
  dir="$ROOT_DIR"

  while :; do
    parent="$(dirname "$dir")"
    [ "$parent" != "$dir" ] || break

    for name in miniforge3 miniconda3 anaconda3 mambaforge conda .conda opt/conda; do
      candidate="$parent/$name"
      if is_conda_root "$candidate"; then
        canonical_dir "$candidate"
        return 0
      fi
    done

    for candidate in "$parent"/conda/* "$parent"/.conda/*; do
      [ -e "$candidate" ] || continue
      if is_conda_root "$candidate"; then
        canonical_dir "$candidate"
        return 0
      fi
    done

    dir="$parent"
  done

  return 1
}

prompt_for_conda() {
  local response candidate
  [ "$INSTALL_CONDA" != "never" ] || die "No conda found. Rerun with --conda-dir DIR or remove --no-install-conda."

  if [ "$INSTALL_CONDA" = "download" ]; then
    CONDA_DIR="$DEFAULT_CONDA_DIR"
    install_managed_conda
    return
  fi

  if [ ! -t 0 ]; then
    die "No conda root found outside $ROOT_DIR. Rerun with --conda-dir DIR, set B1K_CONDA_DIR, or use --download-conda."
  fi

  while :; do
    printf '\nNo conda root was found outside: %s\n' "$ROOT_DIR"
    printf 'Choose [s] specify existing conda root, [d] download Miniforge to %s, [q] quit: ' "$DEFAULT_CONDA_DIR"
    read -r response

    case "$response" in
      s|S)
        printf 'Conda root path: '
        read -r candidate
        candidate="$(expand_user_path "$candidate")"
        if is_conda_root "$candidate"; then
          CONDA_DIR="$(canonical_dir "$candidate")"
          return
        fi
        printf 'No conda executable found at %s/bin/conda. Download Miniforge there? [y/N]: ' "$candidate"
        read -r response
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
          CONDA_DIR="$candidate"
          install_managed_conda
          return
        fi
        ;;
      d|D)
        CONDA_DIR="$DEFAULT_CONDA_DIR"
        install_managed_conda
        return
        ;;
      q|Q)
        die "Conda setup cancelled"
        ;;
      *)
        printf 'Please answer s, d, or q.\n'
        ;;
    esac
  done
}

resolve_conda() {
  local found_conda_dir

  if [ -n "$CONDA_EXE" ]; then
    [ -x "$CONDA_EXE" ] || die "B1K_CONDA_EXE is not executable: $CONDA_EXE"
    CONDA_DIR="$("$CONDA_EXE" info --base)"
  elif [ -n "$CONDA_DIR" ]; then
    CONDA_DIR="$(expand_user_path "$CONDA_DIR")"
    if ! is_conda_root "$CONDA_DIR"; then
      install_managed_conda
    fi
  elif found_conda_dir="$(find_outer_conda_root)"; then
    CONDA_DIR="$found_conda_dir"
    log "Found conda root outside project: $CONDA_DIR"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
    found_conda_dir="$("$CONDA_EXE" info --base)"
    if [ "${found_conda_dir%/}" = "${PROJECT_CONDA_DIR%/}" ]; then
      log "Ignoring project-local conda unless it is explicitly selected: $PROJECT_CONDA_DIR"
      prompt_for_conda
    else
      CONDA_DIR="$found_conda_dir"
      log "Found conda on PATH: $CONDA_DIR"
    fi
  else
    prompt_for_conda
  fi

  CONDA_EXE="$CONDA_DIR/bin/conda"
  [ -x "$CONDA_EXE" ] || die "conda executable not found at $CONDA_EXE"
  export PATH="$CONDA_DIR/bin:$PATH"
  log "Using conda: $("$CONDA_EXE" --version) at $CONDA_DIR"
}

source_conda() {
  local conda_sh="$CONDA_DIR/etc/profile.d/conda.sh"
  [ -f "$conda_sh" ] || die "conda.sh not found: $conda_sh"
  # shellcheck source=/dev/null
  source "$conda_sh"
}

accept_conda_tos() {
  if "$CONDA_EXE" tos --help >/dev/null 2>&1; then
    log "Accepting Anaconda ToS for conda distributions that require it"
    "$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    "$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  else
    log "Conda distribution does not require the Anaconda ToS plugin step"
  fi
}

is_git_checkout() {
  local dir="$1"
  [ -e "$dir/.git" ] && git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

initialize_submodules() {
  truthy "$UPDATE_SUBMODULES" || die "Submodules are missing. Rerun without --skip-submodule-update."
  git -C "$ROOT_DIR" rev-parse --show-toplevel >/dev/null 2>&1 || \
    die "This setup expects a git clone of behavior-a2c2. Clone with submodules instead of using a source archive."
  [ -f "$ROOT_DIR/.gitmodules" ] || die "Missing .gitmodules; cannot initialize project submodules."

  log "Initializing project submodules"
  git -C "$ROOT_DIR" submodule sync --recursive
  GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT_DIR" submodule update \
    --init --recursive --depth 1 --filter=blob:none \
    openpi-comet BEHAVIOR-1K
}

configure_behavior_sparse_checkout() {
  if ! truthy "$BEHAVIOR_SPARSE_CHECKOUT"; then
    log "Keeping BEHAVIOR-1K working tree as-is"
    return
  fi

  log "Configuring BEHAVIOR-1K sparse checkout"
  git -C "$BEHAVIOR_DIR" sparse-checkout set --no-cone \
    setup.sh README.md OmniGibson bddl3 joylo eval-jobqueue
}

verify_submodule_commit() {
  local name="$1"
  local dir="$2"
  local expected="$3"
  local actual

  is_git_checkout "$dir" || die "$name is not checked out at $dir"
  actual="$(git -C "$dir" rev-parse HEAD)"
  if [ "$actual" != "$expected" ]; then
    die "$name submodule is at $actual, expected $expected. Run git submodule update --init --recursive, or update setup.sh and the gitlink together."
  fi
  log "$name submodule commit: $actual"
}

verify_submodule_patches() {
  log "Verifying fork-specific submodule files"
  grep -q "openpi.shared.b1k_network_utils" "$OPENPI_DIR/scripts/serve_b1k.py" || \
    die "openpi-comet submodule is missing the self-contained B1K websocket server patch"
  test -f "$OPENPI_DIR/src/openpi/shared/b1k_network_utils.py" || \
    die "openpi-comet submodule is missing src/openpi/shared/b1k_network_utils.py"
  grep -q "def infer_with_prefix_z" "$OPENPI_DIR/src/openpi/policies/policy.py" || \
    die "openpi-comet submodule is missing Policy.infer_with_prefix_z required by latent A2C2 online eval"
  grep -q "return_prefix_z" "$OPENPI_DIR/src/openpi/models/pi0.py" || \
    die "openpi-comet submodule is missing Pi0 return_prefix_z required by latent A2C2 online eval"
  test -f "$BEHAVIOR_DIR/OmniGibson/omnigibson/learning/eval_custom.py" || \
    die "BEHAVIOR-1K submodule is missing OmniGibson learning eval_custom.py"
  test -f "$BEHAVIOR_DIR/OmniGibson/omnigibson/learning/wrappers/rgb_wrapper.py" || \
    die "BEHAVIOR-1K submodule is missing RGBWrapper"
}

ensure_submodules() {
  if ! is_git_checkout "$OPENPI_DIR" || ! is_git_checkout "$BEHAVIOR_DIR"; then
    initialize_submodules
  elif truthy "$UPDATE_SUBMODULES"; then
    log "Synchronizing project submodules to the pinned commits"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT_DIR" submodule update \
      --init --recursive --depth 1 --filter=blob:none \
      openpi-comet BEHAVIOR-1K
  else
    log "Using existing submodule checkouts"
  fi

  configure_behavior_sparse_checkout
  verify_submodule_commit "openpi-comet" "$OPENPI_DIR" "$OPENPI_SUBMODULE_COMMIT"
  verify_submodule_commit "BEHAVIOR-1K" "$BEHAVIOR_DIR" "$BEHAVIOR_SUBMODULE_COMMIT"
  verify_submodule_patches
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
  source_conda

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
  source_conda
  conda activate "$CONDA_ENV"

  cd "$BEHAVIOR_DIR"
  mkdir -p "$OMNIGIBSON_DATA_PATH" "$OMNIGIBSON_APPDATA_PATH"

  ./setup.sh --omnigibson --bddl --joylo --eval \
    --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
}

fix_behavior_numpy_stack() {
  log "Pinning BEHAVIOR NumPy / SciPy stack"
  source_conda
  conda activate "$CONDA_ENV"

  python -m pip install opencv-contrib-python==4.11.0.86 --no-deps

  local site="$CONDA_PREFIX/lib/python3.10/site-packages"
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
  source_conda
  conda activate "$CONDA_ENV"

  cd "$BEHAVIOR_DIR"
  mkdir -p "$OMNIGIBSON_DATA_PATH" "$OMNIGIBSON_APPDATA_PATH"
  ./setup.sh --dataset --accept-dataset-tos --confirm-no-conda
}

ensure_huggingface_hub() {
  source_conda
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

refresh_behavior_editables() {
  log "Refreshing BEHAVIOR editable installs in the conda env"
  source_conda
  conda activate "$CONDA_ENV"
  python -m pip install -e "$BEHAVIOR_DIR/bddl3"
  python -m pip install -e "$BEHAVIOR_DIR/OmniGibson[eval]"
  python -m pip check
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

  source_conda
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

  if truthy "$DOWNLOAD_BEHAVIOR_DATASET"; then
    test -d "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" || die "Missing BEHAVIOR challenge instances"
  elif [ -d "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" ]; then
    log "BEHAVIOR challenge instances already present"
  else
    log "Skipping BEHAVIOR dataset presence check because --skip-behavior-dataset was used"
  fi

  if truthy "$DOWNLOAD_CHALLENGE_DEMOS"; then
    verify_challenge_demos_subset
  elif [ -d "$CHALLENGE_DEMOS_DIR" ]; then
    log "Challenge demos are present but optional; skipping strict verification"
  else
    log "Skipping optional challenge demos verification"
  fi

  if truthy "$DOWNLOAD_CHECKPOINT"; then
    test -d "$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME" || die "Missing checkpoint"
  elif [ -d "$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME" ]; then
    log "Checkpoint already present: $OPENPI_DIR/checkpoints/$CHECKPOINT_NAME"
  else
    log "Skipping checkpoint presence check because --skip-checkpoint was used"
  fi

  du -sh "$OPENPI_DIR/checkpoints/$CHECKPOINT_NAME" "$OMNIGIBSON_DATA_PATH" 2>/dev/null || true
}

print_next_steps() {
  cat <<EOF

============================================================
Setup finished.
============================================================

Baseline and online A2C2 evaluation commands are documented in:
  $ROOT_DIR/README.md
EOF
}

log_config() {
  log "Setup configuration"
  printf 'ROOT_DIR=%s\n' "$ROOT_DIR"
  printf 'SYSTEM_PACKAGES=%s\n' "$SYSTEM_PACKAGES"
  printf 'OPENPI_SUBMODULE_COMMIT=%s\n' "$OPENPI_SUBMODULE_COMMIT"
  printf 'BEHAVIOR_SUBMODULE_COMMIT=%s\n' "$BEHAVIOR_SUBMODULE_COMMIT"
  printf 'UPDATE_SUBMODULES=%s\n' "$UPDATE_SUBMODULES"
  printf 'BEHAVIOR_SPARSE_CHECKOUT=%s\n' "$BEHAVIOR_SPARSE_CHECKOUT"
  printf 'TASK_NAME=%s\n' "$TASK_NAME"
  printf 'TASK_DIR=%s\n' "$TASK_DIR"
  printf 'CHALLENGE_DEMOS_TASK=%s\n' "$CHALLENGE_DEMOS_TASK"
  printf 'CONDA_DIR=%s\n' "${CONDA_DIR:-<auto>}"
  printf 'CONDA_ENV=%s\n' "$CONDA_ENV"
  printf 'DOWNLOAD_BEHAVIOR_DATASET=%s\n' "$DOWNLOAD_BEHAVIOR_DATASET"
  printf 'DOWNLOAD_CHALLENGE_DEMOS=%s\n' "$DOWNLOAD_CHALLENGE_DEMOS"
  printf 'DOWNLOAD_CHECKPOINT=%s\n' "$DOWNLOAD_CHECKPOINT"
  printf 'CHECKPOINT_NAME=%s\n' "$CHECKPOINT_NAME"
}

main() {
  parse_args "$@"
  setup_logging
  log_config
  prepare_dirs
  install_apt_packages
  install_uv
  resolve_conda
  accept_conda_tos
  ensure_submodules
  setup_openpi_env
  create_behavior_conda_env
  setup_behavior
  fix_behavior_numpy_stack

  if truthy "$DOWNLOAD_BEHAVIOR_DATASET"; then
    download_behavior_dataset
  else
    log "Skipping BEHAVIOR dataset download by request"
  fi

  if truthy "$DOWNLOAD_CHALLENGE_DEMOS"; then
    download_challenge_demos
  else
    log "Skipping optional challenge demos download"
  fi

  refresh_behavior_editables

  if truthy "$DOWNLOAD_CHECKPOINT"; then
    download_checkpoint
  else
    log "Skipping checkpoint download by request"
  fi

  verify_install
  print_next_steps
}

if [ "${B1K_SOURCE_ONLY:-0}" != "1" ]; then
  main "$@"
fi
