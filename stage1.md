# Stage 1: Range-Normalized Group Residual Loss

這一階段只修改訓練目標，不改 A2C2 model architecture，也不改 online inference 行為。
目的很單純：先確認 action scale imbalance 和 outlier residual 是否正在傷害訓練。

## 已修改檔案

- `a2c2/src/loss.py`
  - 新增 `A2C2ResidualLoss`。
  - 支援 `raw_mse`、`norm_mse`、`norm_huber`、`norm_huber_gripw` 四種 preset。
  - 內建 23-D BEHAVIOR/R1Pro action group：
    - `base`: `[0:3]`
    - `torso`: `[3:7]`
    - `left_arm`: `[7:14]`
    - `left_gripper`: `[14]`
    - `right_arm`: `[15:22]`
    - `right_gripper`: `[22]`

- `a2c2/scripts/train.py`
  - 新增 loss ablation args。
  - 預設 `--rgb-cache-kind resnet18-features`，training 直接讀 cached RGB feature parquet，不再預設重新 decode RGB mp4。
  - 非 raw loss 會從 train split 的 parquet `action` 欄位計算 action statistics。
  - 會把 `action_stats.json` 寫到 run output dir。
  - checkpoint 會保存 `loss_config` 和 `action_stats`。
  - training / validation 會 log stage1 loss components 和 group loss。

- `a2c2/scripts/eval.py`
  - 讀 checkpoint 裡的 `loss_config` 和 `action_stats`。
  - 額外輸出 stage1 loss preset、total loss、raw residual MSE、各 group normalized loss。

## 新增訓練參數

```bash
--loss-preset raw_mse|norm_mse|norm_huber|norm_huber_gripw
--action-stats-path PATH
--action-stat-q-low 0.01
--action-stat-q-high 0.99
--min-action-scale 1e-4
--huber-delta 1.0
--gripper-weight 2.0
```

## Preset 說明

- `raw_mse`
  - 原本 baseline。
  - 不需要 action stats。

- `norm_mse`
  - 對 residual error 做 dimension-wise range normalization：
    `error_i = (delta_pred_i - delta_target_i) / scale_i`
  - 用 MSE。

- `norm_huber`
  - 同樣做 range normalization。
  - 把 MSE 換成 Huber loss。
  - 預設 `--huber-delta 1.0`。

- `norm_huber_gripw`
  - 在 `norm_huber` 基礎上，左右 gripper group 加權。
  - 預設 `--gripper-weight 2.0`。

## 建議最小 ablation

機器有限時先跑 4 組就好：

```bash
--loss-preset raw_mse
--loss-preset norm_mse
--loss-preset norm_huber
--loss-preset norm_huber_gripw --gripper-weight 2.0
```

我會優先比較：

- validation `corrected_action_mse` 是否低於 `base_action_mse`
- `loss/group/left_gripper` 和 `loss/group/right_gripper` 是否改善
- rollout 裡 gripper timing 是否比 raw MSE 更穩
- correction magnitude 是否變得過大

## 範例命令

```bash
python a2c2/scripts/train.py \
  --loss-preset norm_huber_gripw \
  --gripper-weight 2.0 \
  --huber-delta 1.0 \
  --output-dir a2c2/runs/task18_stage1_norm_huber_gripw
```

現在 `train.py` 預設使用：

```bash
--rgb-cache-kind resnet18-features
```

所以 RGB 會直接從：

```text
<dataset-root>/rgb_features_resnet18/<task-dir>/
```

讀 cached feature parquet，不會再 decode RGB mp4。

如果要完全避免讀任何 video，還要關掉 depth branch：

```bash
--no-use-depth
```

第一次跑非 raw preset 時會自動計算：

```text
a2c2/runs/<run_name>/action_stats.json
```

之後如果想固定同一組 statistics 做公平比較，可以傳：

```bash
--action-stats-path a2c2/runs/<run_name>/action_stats.json
```

## 這階段刻意不做的事

- 不加 gripper event classification head。
- 不改 `model.py` 的 forward output schema。
- 不加 action clamp 到 online inference。
- 不加 temporal smoothness。

原因是這些都會擴大修改面。Stage 1 先回答一個問題：

```text
只換成 range-normalized / robust / group-aware residual regression，
是否已經比 raw residual MSE 更穩？
```

如果 Stage 1 的 rollout 還是主要敗在 gripper timing，再進 Stage 2 加 gripper event head。
