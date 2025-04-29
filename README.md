# BFL-Optimistic-Poison-Defend

一個基於區塊鏈技術的去中心化聯邦學習系統，提供安全、可驗證且具抗毒性的模型訓練功能。

## 項目概述

本項目實現了一個以測試驅動開發 (TDD) 方法為基礎的區塊鏈聯邦學習系統，整合了 Flower 框架用於分散式機器學習與 Arbitrum 的 Layer-2 捲軸技術。系統使用 IPFS 進行去中心化模型存儲，並實現 Krum 防禦機制抵禦投毒攻擊。

## 架構

系統採用多層架構設計：

1. **聯邦學習層**：使用 Flower 框架實現客戶端-服務器通信
2. **存儲層**：IPFS 用於去中心化模型和更新存儲
3. **執行層**：Arbitrum 捲軸提供高效、可擴展的計算能力
4. **安全層**：Krum 聚合實現拜占庭容錯能力

## 系統流程

1. 請求者通過 Flower 服務器發起聯邦學習任務
2. 初始模型上傳至 IPFS 並在鏈上註冊
3. 選定的客戶端下載模型、本地訓練並提交更新
4. 更新通過 Krum 防禦機制進行篩選
5. 聚合結果通過 Arbitrum 發佈與確認
6. 最終模型用於下一輪訓練或任務完成

## 測試驅動開發方法

本項目採用測試驅動開發 (TDD) 方法，確保每個功能模塊都有完整的測試覆蓋：

1. 先撰寫測試：為每個功能模塊定義預期行為
2. 實現功能：根據測試需求開發功能
3. 通過測試確認：確保實現符合預期行為
4. 重構優化：在保持測試通過的前提下改進代碼

## 主要組件

### 智能合約 (單一合約架構)

智能合約實現了以下核心功能：

- 任務管理：創建、監控和完成聯邦學習任務
- 輪次管理：初始化、追蹤和完成訓練輪次
- 客戶端管理：註冊、選擇和貢獻評估
- Krum 防禦：抵禦惡意更新的整合式防禦機制
- 獎勵系統：基於貢獻度的獎勵計算與分發

### IPFS 連接器

IPFS 連接器負責模型和更新的去中心化存儲：

- 模型上傳與下載：高效處理機器學習模型
- 更新管理：存儲和檢索模型更新
- 雜湊驗證：確保模型完整性和真實性
- 批次操作：高效處理多個更新
- 錯誤處理：實現重試機制以提高穩定性

### 區塊鏈連接器

區塊鏈連接器連接聯邦學習系統與 Arbitrum 網絡：

- Arbitrum 整合：適配 Layer-2 特定功能
- 交易管理：提交、監控和重試機制
- 事件處理：事件監聽與響應
- Gas 優化：減少交易成本

### 聯邦學習核心

基於 Flower 框架的聯邦學習實現：

**服務器端**：

- 任務初始化與配置
- 客戶端選擇策略
- 全局模型管理與聚合
- 模型評估與進度追蹤

**客戶端**：

- 本地訓練邏輯
- 模型更新生成
- 區塊鏈更新提交
- 安全通信機制

### 防禦機制

整合 Krum 算法抵禦投毒攻擊：

- 更新距離計算
- 鄰居選擇邏輯
- 分數機制
- 最佳更新選擇

### 攻擊模擬

實現多種攻擊模型用於測試防禦效果：

- 標籤翻轉攻擊
- 模型替換攻擊
- 拜占庭行為模擬

## 設置與安裝

### 環境要求

- Python 3.9+
- Foundry (用於智能合約開發)
- IPFS 節點
- 以太坊錢包

```bash
# 克隆儲存庫
git clone https://github.com/yourusername/BFL-optimistic-poison-defend.git
cd BFL-optimistic-poison-defend

# 安裝 Python 依賴
pip install -r requirements.txt

# 安裝 Foundry (如果尚未安裝)
curl -L https://foundry.paradigm.xyz | bash
foundryup

# 編譯智能合約
forge build
```

### 配置

1. 創建 `.env` 文件並設置以下變數：

```
PRIVATE_KEY=your_private_key_here
ARBITRUM_SEPOLIA_RPC_URL=https://sepolia-rollup.arbitrum.io/rpc
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
ARBISCAN_API_KEY=your_arbiscan_api_key
```

2. 確保 IPFS 節點正在運行：

```bash
# 啟動本地 IPFS 節點 (如果尚未運行)
ipfs daemon
```

## 智能合約部署

將合約部署到 Arbitrum 測試網：

```bash
# 部署到 Arbitrum Sepolia 測試網
forge script script/Deploy.s.sol --rpc-url arbitrum_sepolia --private-key $PRIVATE_KEY --broadcast --verify

# 運行合約測試
forge test
```

## 使用方法

```bash
# 啟動 Flower 服務器 (請求者)
python fl/server/server.py --task_params "config/fl_params.json" --initial_model "model.h5"

# 運行 Flower 客戶端
python fl/client/client.py --client_id "client1" --data_path "data/"

# 運行攻擊模擬
python attack/label_flipping.py --client_id "malicious1" --intensity 0.3

# 運行評估
python evaluation/defense_effectiveness.py --results_dir "results/"
```

## 檔案結構

```
BFL-Optimistic-Poison-Defend/
├── .gitignore                    # Git ignore file
├── foundry.toml                  # Foundry configuration
├── README.md                     # Project documentation
├── TASKS.md                      # Task tracking list
├── requirements.txt              # Python dependencies
│
├── fl/                           # Federated learning Python code
│   └── blockchain_connector.py   # Blockchain connector for Flower integration
│
├── ipfs_connector.py             # IPFS model storage connector
│
├── src/                          # Smart contract source code
│   ├── FederatedLearning.sol     # Main federated learning contract
│   └── libraries/                # Supporting libraries
│       └── KrumDefense.sol       # Krum defense algorithm implementation
│
├── test/                         # Contract tests
│   ├── FederatedLearning.t.sol   # Main contract tests
│   ├── KrumDefense.t.sol         # Krum defense tests
│   └── Setup.t.sol               # Environment setup tests
│
├── script/                       # Deployment scripts
│   └── Deploy.s.sol              # Contract deployment script
│
├── rules/                        # Development guidelines
│   └── task-list.mdc             # Task list management rules
│
└── docs/                         # Documentation
    └── sequenceDiagram.mmd       # System sequence diagram
```

## 性能與安全特性

- **抗投毒防禦**：Krum 算法過濾惡意更新
- **效率優化**：基於 Arbitrum Layer-2 的高效交易
- **可驗證性**：所有操作都在區塊鏈上可追踪
- **激勵機制**：獎勵貢獻度高的客戶端
- **可擴展性**：適應大規模分布式學習場景

## 開發路線圖

本項目分為四個主要階段開發：

### 階段 1 (2 週)

- 完成智能合約基礎設施和任務管理
- 完成 IPFS 模型儲存功能
- 完成 Arbitrum 連接設定
- 完成基本測試環境設置

### 階段 2 (2 週)

- 完成輪次管理和客戶端管理
- 完成 Krum 防禦機制合約部分
- 完成區塊鏈交易和事件處理
- 開始 Flower 伺服器和客戶端實作

### 階段 3 (2 週)

- 完成 Flower 伺服器和客戶端
- 完成 Krum 防禦策略整合
- 實作並測試惡意客戶端攻擊
- 開始防禦效果評估

### 階段 4 (2 週)

- 完成攻擊模擬和防禦效果評估
- 完成 Arbitrum 部署和 Gas 分析
- 完成性能評估和視覺化
- 整合所有組件並進行最終測試
