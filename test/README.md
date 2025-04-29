# BFL-Optimistic-Poison-Defend 測試指南

本目錄包含 BFL-Optimistic-Poison-Defend 專案的單元測試。這些測試使用 Python 的 `unittest` 框架實現，旨在驗證系統各個組件的正確性與安全性。

## 測試結構

測試套件包含以下模組的單元測試：

1. **區塊鏈連接器** (`test_blockchain_connector.py`)：測試與 Arbitrum 區塊鏈的交互
2. **IPFS 連接器** (`test_ipfs_connector.py`)：測試模型存儲與檢索
3. **Flower 服務器** (`test_flower_server_unit.py`)：測試聯邦學習服務器功能
4. **Flower 客戶端** (`test_flower_client_unit.py`)：測試聯邦學習客戶端功能
5. **攻擊模擬器** (`test_attack_simulator.py`)：測試各種攻擊策略實現
6. **防禦評估器** (`test_defense_effectiveness.py`)：測試防禦機制的評估功能

## 運行測試

### 運行所有測試

要運行所有測試，請在專案根目錄中執行以下命令：

```bash
python -m test.run_all_tests
```

這將依次運行所有測試，並在結束時提供摘要報告。

### 運行單個測試模組

要運行特定的測試模組，可以直接使用 Python 執行對應的測試檔案：

```bash
python -m test.test_blockchain_connector
python -m test.test_ipfs_connector
python -m test.test_flower_server_unit
python -m test.test_flower_client_unit
python -m test.test_attack_simulator
python -m test.test_defense_effectiveness
```

### 運行特定測試案例

要運行特定的測試案例，可以使用以下格式：

```bash
python -m unittest test.test_blockchain_connector.TestBlockchainConnector.test_initialization
```

## 測試依賴

這些測試依賴以下軟體包：

- Python 3.9+
- unittest (Python 標準庫)
- unittest.mock (Python 標準庫)
- numpy
- torch
- matplotlib (用於防禦評估視覺化)

## 測試覆蓋範圍

目前的測試套件覆蓋了以下功能：

### 區塊鏈連接器

- 初始化與連接
- 交易建立與發送
- 模型雜湊計算
- 系統狀態查詢
- 客戶端註冊
- 模型更新提交
- Krum 防禦應用
- 錯誤處理

### IPFS 連接器

- 初始化與連接
- 模型上傳 (PyTorch 模型和 NumPy 陣列)
- 模型下載與反序列化
- 模型差異計算
- 聯邦平均
- 批量操作
- 錯誤處理

### Flower 服務器

- 初始化與配置
- 任務創建
- 輪次管理
- 客戶端選擇
- 模型聚合 (使用 Krum 防禦)
- Krum 防禦機制
- 獎勵分發

### Flower 客戶端

- 初始化與配置
- 區塊鏈註冊
- 模型下載與載入
- 本地訓練
- 模型評估
- 更新提交
- 輪次參與

### 攻擊模擬器

- 標籤翻轉攻擊
- 模型替換攻擊
- 拜占庭攻擊
- 目標型模型中毒攻擊
- 攻擊執行流程

### 防禦評估器

- 任務歷史獲取
- 模型下載與評估
- 模型比較
- Krum 防禦評估
- 防禦有無比較
- 攻擊影響評估
- 報告生成
- 視覺化圖表

## 注意事項

1. 這些測試使用模擬 (mock) 對象來模擬外部依賴，如區塊鏈節點和 IPFS 節點，因此不需要實際連接到這些服務即可運行測試。

2. 部分測試會在臨時目錄中創建檔案，如生成的報告和圖表，這些檔案會在測試完成後自動清理。

3. 攻擊模擬器測試包含隨機元素，偶爾可能會因為隨機性而失敗。如遇此情況，請重新運行測試。

4. 如果您修改了代碼，請確保運行測試以驗證修改沒有破壞現有功能。

## 擴展測試

如需為新功能新增測試，請按照以下步驟：

1. 確定測試應該屬於哪個現有模組，或者是否需要創建新的測試模組。

2. 對於新的測試模組，請遵循現有測試的結構：

   - 導入必要的模組
   - 定義一個繼承 `unittest.TestCase` 的測試類
   - 實現 `setUp` 和 `tearDown` 方法
   - 為每個需要測試的功能添加以 `test_` 開頭的測試方法

3. 更新 `run_all_tests.py` 以包含新的測試模組。

4. 確保新的測試不會干擾其他測試，尤其是清理測試創建的任何資源。
