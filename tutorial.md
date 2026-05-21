# Runpod 重現 openpi-comet + BEHAVIOR-1K `picking_up_trash`

這份文件是交接 runbook。真正的一鍵安裝流程以本目錄的 `setup.sh` 為準；本文件只保留接手者需要先知道的硬體軟體要求、HF token 設定、空間估算、evaluation 入口，以及 `setup.sh` 可能無法自動處理的常見錯誤與修法。

本文用 `B1K_ROOT` 表示你選定的安裝根目錄。`setup.sh` 預設使用執行時的目前目錄；`openpi-comet/`、`BEHAVIOR-1K/`、`miniconda3/`、cache 和 output 都會直接放在這個目錄底下，不會再多建立一層 `b1k/` 外包裝。新開 terminal 要照本文指令操作時，先進入安裝根目錄並執行 `export B1K_ROOT="$(pwd -P)"`。

實測日期：2026-05-19。

## 硬體 / 軟體要求

硬體建議：

- GPU：NVIDIA GPU，至少 24GB VRAM。RTX 4090 24GB 可跑通，但 server + OmniGibson eval 會很貼邊。
- RAM：建議 64GB 以上。
- CPU：建議 16 vCPU 以上。
- 磁碟：最低抓 120GB，建議 150GB 以上；如果會反覆跑 eval 或保留影片，建議 200GB 到 500GB。
- 網路：需要能連 GitHub、Hugging Face、NVIDIA / PyTorch wheel、Ubuntu apt mirror。

軟體假設：

- OS：Ubuntu 22.04 LTS。不要優先用 Ubuntu 24.04，Isaac Sim / Kit 比較容易遇到 native crash。
- NVIDIA driver / CUDA：實測 driver `550.127.05`，`nvidia-smi` 顯示 CUDA `12.4`。
- `setup.sh` 會自動安裝 apt 套件、Miniconda、`uv`、Python env、BEHAVIOR assets、checkpoint。

固定版本：

- openpi-comet commit：`4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`
- BEHAVIOR-1K tag：`v3.7.2`
- BEHAVIOR-1K commit：`88454bd04f75dc57c00ab1f1a00bcde1ff505950`
- OmniGibson：`3.7.2`
- Torch：`2.6.0+cu124`
- NumPy：`1.26.4`
- SciPy：`1.14.1`
- Checkpoint：`sunshk/openpi_comet/pi05-b1kpt12-cs32`

## HF_TOKEN 設定 (選用，能加速環境安裝)

Hugging Face token 不是每次都必要，因為目前使用的 checkpoint / assets 多數可以匿名下載。但匿名下載可能遇到 rate limit、速度慢，或 Hugging Face 回 `401` / `403` / `429`。建議正式交接時先設定 `HF_TOKEN`。

官方重點：`huggingface_hub` 會讀取 `HF_TOKEN` 作為 Hub 認證 token；`HF_HOME` 和 `HF_HUB_CACHE` 可指定 token / cache 的位置。參考 Hugging Face 官方文件：https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

建立 token：

1. 登入 Hugging Face。
2. 打開 https://huggingface.co/settings/tokens
3. 建立一個 token。只下載公開模型 / dataset 時，通常 `Read` 權限就夠。
4. 複製 token。token 只會完整顯示一次，且不要貼到 git、Slack、issue、log。

在 Runpod shell 設定 token。建議用 `read -rsp`，避免 token 進 shell history：

```bash
mkdir -p ~/.config/huggingface
chmod 700 ~/.config/huggingface

read -rsp "Paste HF_TOKEN: " HF_TOKEN
echo

printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" > ~/.config/huggingface/token.env
chmod 600 ~/.config/huggingface/token.env

source ~/.config/huggingface/token.env
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is set"
```

驗證 token 是否有效：

```bash
curl -sS \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami-v2
```

如果 token 有效，會看到包含使用者資訊的 JSON。如果看到 `Invalid user token`、`401` 或空 token，重新建立 token 並重新 `source ~/.config/huggingface/token.env`。

執行 setup 前，先在同一個 shell 載入 token：

```bash
source ~/.config/huggingface/token.env
```

如果 Hugging Face `git lfs pull` 仍然遇到授權或 rate limit，可讓 git / curl 也使用 token：

```bash
cat > ~/.netrc <<EOF
machine huggingface.co
  login __token__
  password $HF_TOKEN
EOF
chmod 600 ~/.netrc
```

注意：`~/.netrc` 會把 token 寫到磁碟。只在自己的 pod 使用；如果 token 曾外洩，立刻到 Hugging Face tokens 頁面 revoke / rotate。

## 一鍵安裝

在乾淨 pod 上，先進入你要作為安裝根目錄的資料夾，並把 `setup.sh` 和 `tutorial.md` 放在這個資料夾。若已經在該資料夾內，直接跑：

```bash
bash setup.sh
```

成功時會看到類似：

```text
openpi server imports ok
numpy 1.26.4
scipy 1.14.1
torch 2.6.0+cu124 cuda True
omnigibson 3.7.2
No broken requirements found.
12G $B1K_ROOT/openpi-comet/checkpoints/pi05-b1kpt12-cs32
36G $B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets
Setup finished.
```

`setup.sh` 會自動使用目前目錄作為安裝根目錄，並寫出安裝 log 和完整 evaluation 指令：

```text
$B1K_ROOT/setup_run.log
$B1K_ROOT/evaluation_commands.txt
```

## 空間估算

實測主要大小：

```text
$B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets            36G
$B1K_ROOT/openpi-comet/checkpoints/pi05-b1kpt12-cs32 12G
$B1K_ROOT/miniconda3                                  約 20G
$B1K_ROOT/openpi-comet/.venv                          約 9G
$B1K_ROOT/.uv-cache                                   約 9G
$B1K_ROOT/og-appdata                                  約 10G
```

建議：

- 只安裝並跑一次：至少 120GB。
- 需要保留 logs / videos / 多次 output：150GB 以上。
- 想反覆 debug、避免清 cache：200GB 到 500GB。

可用這些指令查空間：

```bash
df -h "$B1K_ROOT"
du -sh "$B1K_ROOT"/* "$B1K_ROOT"/.cache "$B1K_ROOT"/.uv-cache 2>/dev/null | sort -h
```

## 跑 evaluation

安裝完成後直接看：

```bash
sed -n '1,220p' "$B1K_ROOT/evaluation_commands.txt"
```

需要兩個 terminal：

- Terminal 1：啟動 openpi-comet websocket policy server。
- Terminal 2：跑 BEHAVIOR evaluation。

注意：若 `uv` 是由 `setup.sh` 裝到 root 的 local bin，新開的 terminal 不一定已經有 `$HOME/.local/bin`。請以 `$B1K_ROOT/evaluation_commands.txt` 的完整指令為準；Terminal 1 需先把 `$HOME/.local/bin` 加回 `PATH`，否則會出現 `uv: command not found`。

這份指令刻意接近 openpi-comet 官方 README 的 evaluation 方式：先 `serve_b1k.py`，再 `eval.py policy=websocket task.name=... log_path=...`。多出來的少數參數是 Runpod/headless 與本任務需要的設定：

- `xvfb-run`：Runpod 沒有實體螢幕，OmniGibson / Isaac Sim 需要虛擬 display。
- `env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper`：Comet 官方也建議 evaluation 使用 RGBWrapper。
- `eval_instance_ids='[0]'`：只跑一個 instance，方便確認 `picking_up_trash` 能完整推論一次。
- `write_video=true`：保留輸出影片，符合本專案目標。

影片輸出位置會類似：

```text
$B1K_ROOT/BEHAVIOR-1K/output/picking_up_trash_*/videos/*.mp4
```

不需要用 `ffprobe` 才算完成；那只是交接時可選的 sanity check。想快速找影片時用：

```bash
find "$RUN_LOG" -name '*.mp4' -print
```

## 常見錯誤與修法

### 1. Hugging Face 下載慢、`401` / `403` / `429`

症狀：

```text
Warning: You are sending unauthenticated requests to the HF Hub
401 Client Error
403 Client Error
429 Too Many Requests
Repository Not Found
git lfs pull failed
```

修法：

```bash
source ~/.config/huggingface/token.env
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is set"

curl -sS \
  -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami-v2
```

如果 `curl` 驗證失敗，重新建立 token。若 `git lfs pull` 還是不吃 token，加入 `~/.netrc`：

```bash
cat > ~/.netrc <<EOF
machine huggingface.co
  login __token__
  password $HF_TOKEN
EOF
chmod 600 ~/.netrc
```

然後重跑：

```bash
cd "$B1K_ROOT"
bash setup.sh
```

### 2. 磁碟不足

症狀：

```text
No space left on device
OSError: [Errno 28]
git-lfs: smudge filter lfs failed
```

先查：

```bash
df -h "$B1K_ROOT"
du -sh "$B1K_ROOT"/* "$B1K_ROOT"/.cache "$B1K_ROOT"/.uv-cache 2>/dev/null | sort -h
```

如果只是要清暫存：

```bash
rm -rf "$B1K_ROOT"/tmp/*
rm -rf "$B1K_ROOT"/hf-openpi-comet-fullgit
```

如果可以刪 cache，會省很多空間，但下次會重新下載：

```bash
rm -rf "$B1K_ROOT"/.cache "$B1K_ROOT"/.uv-cache
```

如果可以刪舊 evaluation output：

```bash
rm -rf "$B1K_ROOT"/BEHAVIOR-1K/output/picking_up_trash_*
```

如果要回到最乾淨狀態，只留文件和腳本：

```bash
find "$B1K_ROOT" -mindepth 1 -maxdepth 1 \
  ! -name setup.sh \
  ! -name tutorial.md \
  -exec rm -rf -- {} +
```

### 3. NumPy / SciPy ABI 或 `numpy.ufunc` 錯誤

症狀：

```text
ValueError: All ufuncs must have type numpy.ufunc
numpy.dtype size changed
module compiled against ABI version
isaacsim-core requires numpy<2.0.0, but you have numpy 2.x
```

原因：JoyLo / mediapipe 可能把 NumPy 升到 2.x，但 OmniGibson / Isaac Sim 這組需要 NumPy 1.x。`setup.sh` 已經自動修一次。

手動修：

```bash
source "$B1K_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate behavior

python -m pip install opencv-contrib-python==4.11.0.86 --no-deps

site="$B1K_ROOT/miniconda3/envs/behavior/lib/python3.10/site-packages"
rm -rf \
  "$site"/numpy \
  "$site"/numpy.libs \
  "$site"/numpy-*.dist-info \
  "$site"/scipy \
  "$site"/scipy.libs \
  "$site"/scipy-*.dist-info

conda install -c conda-forge numpy=1.26.4 scipy=1.14.1 --force-reinstall -y
python -m pip check
```

確認：

```bash
python - <<'PY'
import numpy, scipy, torch, omnigibson
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("omnigibson", omnigibson.__version__)
PY
```

### 4. `Data path ... does not exist`

症狀：

```text
AssertionError: Data path $B1K_ROOT/BEHAVIOR-1K/datasets does not exist!
```

原因：沒有設 `OMNIGIBSON_DATA_PATH`，或設到舊路徑。

修法：

```bash
export OMNIGIBSON_DATA_PATH="$B1K_ROOT/BEHAVIOR-1K/OmniGibson/datasets"
export OMNIGIBSON_APPDATA_PATH="$B1K_ROOT/og-appdata"
test -d "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances"
```

跑 evaluation 時直接使用 `$B1K_ROOT/evaluation_commands.txt` 裡的完整 env vars。

### 5. openpi server import 失敗

症狀：

```text
ModuleNotFoundError: No module named 'omnigibson'
ModuleNotFoundError: No module named 'pytest'
```

原因：openpi policy server 用 Python 3.11 / uv env，BEHAVIOR simulator 用 Python 3.10 / conda env。兩邊不能簡單混在一起。`setup.sh` 會：

- 在 openpi env 補 `pytest`
- 寫入 `openpi.shared.b1k_network_utils`
- patch `scripts/serve_b1k.py`，避免 policy server import 完整 OmniGibson
- patch `openpi/policies/b1k_policy.py`，讓 proprioception indices 有 fallback

修法：

```bash
cd "$B1K_ROOT"
bash setup.sh
```

確認 server import：

```bash
cd "$B1K_ROOT"/openpi-comet
UV_CACHE_DIR="$B1K_ROOT/.uv-cache" uv run --no-sync python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("serve_b1k_check", Path("scripts/serve_b1k.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("serve_b1k import ok")
PY
```

### 6. Evaluation 連不上 server

症狀：

```text
Health check failed, waiting for server
Connection refused
Websocket connection failed
```

修法：

1. 確認 Terminal 1 還在跑 server，且沒有 crash。
2. 在 Terminal 2 檢查 health check：

```bash
curl -v http://127.0.0.1:8000/healthz
```

3. 如果 port 被占用：

```bash
lsof -i :8000 || true
```

4. 如果要換 port，Terminal 1 加 `--port=8001`，Terminal 2 evaluation 加 `model.port=8001`。

### 7. CUDA OOM

症狀：

```text
CUDA out of memory
RESOURCE_EXHAUSTED
XlaRuntimeError
```

修法：

```bash
nvidia-smi
```

先停掉舊 server / 舊 eval，只留一組 Terminal 1 + Terminal 2。若仍 OOM，把 Terminal 1 的 JAX memory fraction 調低：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
```

然後重啟 policy server。若想先短測，可以臨時在 Terminal 2 的 eval 指令尾端加 `max_steps=2 write_video=false`；正式交接指令不把 smoke test 列為必要步驟。

### 8. Isaac Sim / OmniGibson headless 問題

症狀：

```text
Failed to open display
GLX / EGL error
OmniKit crash
native segfault
```

修法：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
xvfb-run -a -s "-screen 0 1280x720x24" python OmniGibson/omnigibson/learning/eval.py ...
```

不要直接裸跑 `python eval.py`。如果在 Ubuntu 24.04 出現 native crash，改用 Ubuntu 22.04 pod。

### 9. Conda Terms of Service

症狀：

```text
CondaToSNonInteractiveError
Terms of Service have not been accepted
```

修法：

```bash
"$B1K_ROOT/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$B1K_ROOT/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

`setup.sh` 已包含這一步，但若手動操作 conda 時遇到，可以直接跑以上指令。

### 10. setup 被中斷

通常直接重跑即可：

```bash
cd "$B1K_ROOT"
bash setup.sh
```

如果 repo / env 半殘，最乾淨修法是回到只留 `setup.sh` 和 `tutorial.md`：

```bash
find "$B1K_ROOT" -mindepth 1 -maxdepth 1 \
  ! -name setup.sh \
  ! -name tutorial.md \
  -exec rm -rf -- {} +

cd "$B1K_ROOT"
bash setup.sh
```

## 注意事項

- `setup.sh` 是 source of truth。不要把 tutorial 當手動安裝逐步教學。
- 不要隨便 `git pull` 或換 commit；openpi-comet、BEHAVIOR、OmniGibson、Isaac Sim、NumPy 版本彼此很敏感。
- 不要手動升級 NumPy 到 2.x。
- 不要把 `HF_TOKEN` commit、貼到 log、貼到 issue。
- `xvfb-run` 是 headless Runpod 的顯示需求，不是影片驗證；在本環境建議保留。
- full evaluation 不保證任務成功率為 1；本流程目標是可重現推論與輸出影片，不是保證 policy 完成任務。
