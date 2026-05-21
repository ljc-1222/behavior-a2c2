- 模型公式：

    $$a_{t+k}^{exec} = a_{t+k}^{base} + \Delta a_{t+k}$$
    $$\Delta a_{t+k} = \pi_{a2c2}(o_{t+k}, a_{t+k}^{base}, \tau_k, z_{t+k}, l)$$

- 模型架構：

    1. Base model (OpenPi-COMET)
    2. Correction head model
        - 建議採用 Transformer Encoder + MLP 的混合架構
        - 參考 A2C2 在 LIBERO 任務的設計

- 模型輸入：

    1. 基礎策略最新表示 $z_t$：
        - 從 OpenPi-COMET 的 Transformer 中提取最後一層的 Hidden States。
        - 代表了對當前 BEHAVIOR 任務（如「清潔地板」）的高階語義與環境摘要。

    2. 即時觀測值 $o_{t+k}$：
        - OmniGibson 提供的實時 RGB-D 影像（頂部與手腕視角），通過輕量級 encoder（建議選擇 ResNet-18）編碼後的向量
        - 本體感受 (State)：包含機器人關節角度、末端執行器座標與夾爪狀態。

    3. 基礎動作與時間特徵 ($a_{t+k}^{base}, \tau_k$)：
        - 來自 OpenPi-COMET 在 t 時間產出的 Chunk 中對應第 $k$ 步的 3D 動作指令。
        - $\tau_k = (\sin(2\pi \frac{k}{H}), \cos(2\pi \frac{k}{H}))$ ,  where H = Horizon Length
    
    4. 任務文字輸入

- 訓練模型：

    1. 收集數據：
        - 將 BEHAVIOR 演示數據集中的影像與指令輸入模型。記錄模型產出的預測動作塊 $\hat{A}_t$ 以及隱藏特徵 $z_t$ 。
        - 將上述資料與該時刻的「專家真實動作 ($a^{expert}_t$)」、「即時觀測值 $o_{t}$」配對。
        - 計算專家真實動作 $a_{expert}$ 與模型預測動作 $a_{base}$ 之間的差值：$\Delta a = a^{expert} - a^{base}$ 

    3. 訓練：
        - 損失函數：建議使用均方誤差（MSE Loss）最小化 A2C2 預測值與 $a_{expert}$ 的差距 $\Delta a_{target}$。  
        - 訓練細節：建議使用極小的學習率（如 $1 \times 10^{-5}$），確保修正頭穩定收斂。