"""
blockchain_connector.py

區塊鏈連接器模組，負責 Flower 與以太坊智能合約之間的交互，
針對 Arbitrum 網絡和 Foundry 部署進行了優化
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import logging

from web3 import Web3
import numpy as np

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blockchain_connector")

class BlockchainConnector:
    """連接聯邦學習系統與區塊鏈的工具"""

    def __init__(
        self, 
        contract_address: str,
        client_id: Optional[int] = None,
        private_key: Optional[str] = None,
        node_url: str = "http://127.0.0.1:8545",
        contract_abi_path: Optional[str] = None,
    ):
        """
        初始化區塊鏈連接器
        
        參數:
            contract_address: 已部署的 FederatedLearning 合約地址
            client_id: 客戶端 ID (用於客戶端)
            private_key: 私鑰 (用於客戶端)
            node_url: 以太坊節點 URL
            contract_abi_path: 合約 ABI 檔案路徑
        """
        self.client_id = client_id
        self.contract_address = contract_address
        self.currentTaskId = None  # 當前任務 ID
        
        # 連接到以太坊節點
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        
        # 添加 PoA 中間件（針對 Arbitrum 可能需要）
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception as e:
            logger.warning(f"無法添加 PoA 中間件: {str(e)}")
        
        # 檢查連接
        if not self.w3.is_connected():
            raise ConnectionError(f"無法連接到區塊鏈節點: {node_url}")
        
        # 設定帳戶
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            logger.info(f"使用帳戶: {self.account.address}")
        else:
            try:
                self.account = self.w3.eth.accounts[0]
                logger.info(f"使用默認帳戶: {self.account}")
            except:
                self.account = None
                logger.warning("未提供私鑰且無法訪問默認帳戶，僅支持讀取操作")
        
        # 載入合約 ABI
        if contract_abi_path is None:
            # 嘗試從預設位置載入 ABI
            contract_abi = self._load_contract_abi()
        else:
            # 使用指定的 ABI
            with open(contract_abi_path, "r") as file:
                contract_abi = json.load(file)
        
        # 初始化合約接口
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=contract_abi)
        logger.info(f"已連接到合約: {contract_address}")
        
        # 獲取系統狀態
        try:
            system_status = self.get_system_status()
            self.currentTaskId = system_status.get("currentTaskId", 0)
            logger.info(f"當前任務ID: {self.currentTaskId}")
        except Exception as e:
            logger.warning(f"無法獲取系統狀態: {str(e)}")
    
    def _load_contract_abi(self) -> List[Dict[str, Any]]:
        """嘗試從多個位置載入合約 ABI"""
        # 首先嘗試從 Foundry 產生的 json 文件中加載
        try:
            # 嘗試從 out 目錄載入
            artifact_path = Path("out/FederatedLearning.sol/FederatedLearning.json")
            if artifact_path.exists():
                with open(artifact_path, "r") as file:
                    artifact = json.load(file)
                    return artifact["abi"]
            
            # 嘗試從 artifacts 目錄直接載入
            artifacts_path = Path("artifacts/contracts/FederatedLearning.sol/FederatedLearning.json")
            if artifacts_path.exists():
                with open(artifacts_path, "r") as file:
                    artifact = json.load(file)
                    return artifact["abi"]
                
            # 嘗試從相對路徑 (針對模塊導入情況)
            mod_path = Path(__file__).parent.parent / "out/FederatedLearning.sol/FederatedLearning.json"
            if mod_path.exists():
                with open(mod_path, "r") as file:
                    artifact = json.load(file)
                    return artifact["abi"]
        except Exception as e:
            logger.warning(f"從 Foundry 生成的文件載入 ABI 失敗: {str(e)}")
        
        # 如果上述方法都失敗，使用硬編碼的 ABI
        logger.warning("使用硬編碼的合約 ABI")
        return [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "inputs": [],
                "name": "initialize",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "string", "name": "initialModelHash", "type": "string"}, 
                          {"internalType": "uint256", "name": "totalRounds", "type": "uint256"}],
                "name": "createTask",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "taskId", "type": "uint256"}],
                "name": "startRound",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "clientId", "type": "uint256"},
                    {"internalType": "uint256", "name": "roundId", "type": "uint256"},
                    {"internalType": "string", "name": "modelUpdateHash", "type": "string"}
                ],
                "name": "submitModelUpdate",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "roundId", "type": "uint256"},
                    {"internalType": "string", "name": "globalModelHash", "type": "string"}
                ],
                "name": "updateGlobalModel",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "roundId", "type": "uint256"}],
                "name": "completeRound",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "registerClient",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "roundId", "type": "uint256"}],
                "name": "applyKrumDefense",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getSystemStatus",
                "outputs": [
                    {"internalType": "uint256", "name": "totalClients", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalRounds", "type": "uint256"},
                    {"internalType": "uint256", "name": "currentRound", "type": "uint256"},
                    {"internalType": "uint8", "name": "currentRoundStatus", "type": "uint8"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "taskId", "type": "uint256"}],
                "name": "getTaskInfo",
                "outputs": [
                    {"internalType": "uint256", "name": "", "type": "uint256"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "uint256", "name": "startTime", "type": "uint256"},
                    {"internalType": "uint256", "name": "completedRounds", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalRounds", "type": "uint256"},
                    {"internalType": "string", "name": "initialModelHash", "type": "string"},
                    {"internalType": "string", "name": "currentModelHash", "type": "string"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "roundId", "type": "uint256"}],
                "name": "getRoundInfo",
                "outputs": [
                    {"internalType": "uint256", "name": "", "type": "uint256"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "uint256", "name": "startTime", "type": "uint256"},
                    {"internalType": "uint256", "name": "endTime", "type": "uint256"},
                    {"internalType": "uint256", "name": "participantCount", "type": "uint256"},
                    {"internalType": "uint256", "name": "completedUpdates", "type": "uint256"},
                    {"internalType": "string", "name": "globalModelHash", "type": "string"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "clientId", "type": "uint256"}],
                "name": "getClientInfo",
                "outputs": [
                    {"internalType": "address", "name": "clientAddress", "type": "address"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "uint256", "name": "contributionScore", "type": "uint256"},
                    {"internalType": "uint256", "name": "lastUpdateTimestamp", "type": "uint256"},
                    {"internalType": "bool", "name": "selectedForRound", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _get_transaction_params(self) -> Dict:
        """獲取交易參數"""
        if not self.account:
            raise ValueError("未設置帳戶，無法發送交易")
            
        return {
            "from": self.account.address if hasattr(self.account, "address") else self.account,
            "gasPrice": self.w3.eth.gas_price,
            "nonce": self.w3.eth.get_transaction_count(
                self.account.address if hasattr(self.account, "address") else self.account
            ),
        }
    
    def _sign_and_send_transaction(self, transaction, max_retries: int = 3, retry_delay: int = 5) -> str:
        """簽署並發送交易，帶有重試機制"""
        # 估算 gas
        try:
            gas_estimate = self.w3.eth.estimate_gas(transaction)
            transaction["gas"] = int(gas_estimate * 1.2)  # 增加 20% 以防止 gas 不足
        except Exception as e:
            logger.error(f"估算 gas 失敗: {str(e)}")
            raise
        
        # 簽署交易 (如果有私鑰)
        if hasattr(self.account, "key"):
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            
            # 嘗試發送交易，帶有重試
            for attempt in range(max_retries):
                try:
                    tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:  # 最後一次嘗試
                        logger.error(f"發送交易失敗: {str(e)}")
                        raise
                    else:
                        logger.warning(f"發送交易嘗試 {attempt+1} 失敗: {str(e)}，將在 {retry_delay} 秒後重試")
                        time.sleep(retry_delay)
                        # 更新 nonce
                        transaction["nonce"] = self.w3.eth.get_transaction_count(self.account.address)
                        signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
        else:
            # 使用解鎖的帳戶
            tx_hash = self.w3.eth.send_transaction(transaction)
        
        # 等待交易確認
        logger.info(f"交易已提交，等待確認: {tx_hash.hex()}")
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # 檢查交易狀態
        if receipt.status == 0:
            logger.error(f"交易失敗: {receipt}")
            raise Exception(f"交易失敗，可能是合約執行出錯: {receipt.transactionHash.hex()}")
            
        return receipt["transactionHash"].hex()
    
    def hash_model_parameters(self, parameters: List[np.ndarray]) -> str:
        """將模型參數轉換為雜湊值"""
        # 將參數序列化為二進制格式
        serialized = b""
        for param in parameters:
            serialized += param.tobytes()
        
        # 計算 SHA-256 雜湊值
        hash_object = hashlib.sha256(serialized)
        return hash_object.hexdigest()
    
    def createTask(self, initial_model_hash: str, total_rounds: int) -> int:
        """創建一個新的聯邦學習任務"""
        # 建立交易
        transaction = self.contract.functions.createTask(initial_model_hash, total_rounds).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"創建任務成功，交易雜湊: {tx_hash}")
            
            # 獲取任務 ID
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            # 解析事件找出任務 ID
            task_id_event = self.contract.events.TaskCreated().process_receipt(receipt)
            if task_id_event:
                task_id = task_id_event[0]['args']['taskId']
                self.currentTaskId = task_id
                return task_id
            else:
                # 嘗試從系統狀態獲取
                self.get_system_status()
                return self.currentTaskId
        except Exception as e:
            logger.error(f"創建任務失敗: {str(e)}")
            raise
    
    def register_client(self) -> bool:
        """註冊客戶端"""
        if self.client_id is not None:
            logger.info(f"使用預設客戶端 ID: {self.client_id}")
            return True
            
        # 建立交易
        transaction = self.contract.functions.registerClient().build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            
            # 從事件中獲取客戶端 ID
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            client_reg_event = self.contract.events.ClientRegistered().process_receipt(receipt)
            if client_reg_event:
                self.client_id = client_reg_event[0]['args']['clientId']
                logger.info(f"客戶端註冊成功，ID: {self.client_id}，交易雜湊: {tx_hash}")
            else:
                logger.warning(f"客戶端註冊成功，但無法從事件獲取 ID，交易雜湊: {tx_hash}")
                
            return True
        except Exception as e:
            logger.error(f"註冊客戶端失敗: {str(e)}")
            return False
    
    def start_round(self, task_id: int) -> int:
        """開始新的訓練輪次"""
        # 建立交易
        transaction = self.contract.functions.startRound(task_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            
            # 從事件中獲取輪次 ID
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            round_event = self.contract.events.RoundStarted().process_receipt(receipt)
            if round_event:
                round_id = round_event[0]['args']['roundId']
                logger.info(f"輪次 {round_id} 開始成功，交易雜湊: {tx_hash}")
                return round_id
            else:
                # 嘗試從系統狀態獲取當前輪次
                status = self.get_system_status()
                logger.info(f"輪次開始成功，當前輪次 ID: {status['currentRound']}，交易雜湊: {tx_hash}")
                return status['currentRound']
        except Exception as e:
            logger.error(f"開始輪次失敗: {str(e)}")
            raise
    
    def submit_model_update(self, client_id: int, round_id: int, parameters: List[np.ndarray]) -> str:
        """提交模型更新"""
        if self.client_id is not None and client_id != self.client_id:
            logger.warning(f"提交模型更新時使用的客戶端 ID ({client_id}) 與初始化時不符 ({self.client_id})")
        
        # 計算模型參數的雜湊值
        model_hash = self.hash_model_parameters(parameters)
        
        # 建立交易
        transaction = self.contract.functions.submitModelUpdate(
            client_id, round_id, model_hash
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"客戶端 {client_id} 提交模型更新成功，輪次: {round_id}，交易雜湊: {tx_hash}")
            return model_hash
        except Exception as e:
            logger.error(f"提交模型更新失敗: {str(e)}")
            raise
    
    def updateGlobalModel(self, round_id: int, global_model_hash: str) -> bool:
        """更新全局模型"""
        # 建立交易
        transaction = self.contract.functions.updateGlobalModel(
            round_id, global_model_hash
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"更新輪次 {round_id} 的全局模型成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            logger.error(f"更新全局模型失敗: {str(e)}")
            raise
    
    def complete_round(self, round_id: int) -> bool:
        """完成訓練輪次"""
        # 建立交易
        transaction = self.contract.functions.completeRound(round_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"輪次 {round_id} 完成成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            logger.error(f"完成輪次失敗: {str(e)}")
            raise
    
    def completeTask(self, task_id: int, final_model_hash: str) -> bool:
        """完成聯邦學習任務"""
        # 建立交易
        transaction = self.contract.functions.completeTask(task_id, final_model_hash).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"任務 {task_id} 完成成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            logger.error(f"完成任務失敗: {str(e)}")
            raise
    
    def selectClients(self, round_id: int, client_ids: List[int]) -> bool:
        """選擇客戶端參與當前輪次"""
        # 建立交易
        transaction = self.contract.functions.selectClients(round_id, client_ids).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"輪次 {round_id} 選擇 {len(client_ids)} 個客戶端成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            logger.error(f"選擇客戶端失敗: {str(e)}")
            raise
    
    def applyKrumDefense(self, round_id: int) -> int:
        """應用 Krum 防禦機制"""
        # 建立交易
        transaction = self.contract.functions.applyKrumDefense(round_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            
            # 從交易收據中獲取選中的客戶端 ID
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            # 解析事件
            accept_event = self.contract.events.ModelUpdateAccepted().process_receipt(receipt)
            if accept_event:
                selected_client_id = accept_event[0]['args']['clientId']
                logger.info(f"Krum 防禦選中客戶端 {selected_client_id}，交易雜湊: {tx_hash}")
                return selected_client_id
            else:
                logger.warning(f"Krum 防禦應用成功但無法從事件獲取選中的客戶端 ID，交易雜湊: {tx_hash}")
                return 0
        except Exception as e:
            logger.error(f"應用 Krum 防禦失敗: {str(e)}")
            raise
    
    def distributeRewards(self, client_ids: List[int], round_id: int) -> bool:
        """分發獎勵給客戶端"""
        # 建立交易
        transaction = self.contract.functions.distributeRewards(client_ids, round_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            logger.info(f"輪次 {round_id} 為 {len(client_ids)} 個客戶端分發獎勵成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            logger.error(f"分發獎勵失敗: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict:
        """獲取系統狀態"""
        try:
            status = self.contract.functions.getSystemStatus().call()
            
            result = {
                "totalClients": status[0],
                "totalRounds": status[1],
                "currentRound": status[2],
                "currentRoundStatus": status[3]
            }
            
            # 嘗試獲取當前任務 ID
            try:
                current_task_id = self.contract.functions.currentTaskId().call()
                result["currentTaskId"] = current_task_id
                self.currentTaskId = current_task_id
            except:
                pass
                
            return result
        except Exception as e:
            logger.error(f"獲取系統狀態失敗: {str(e)}")
            raise
    
    def getTaskInfo(self, task_id: int) -> Dict:
        """獲取任務資訊"""
        try:
            task_info = self.contract.functions.getTaskInfo(task_id).call()
            
            return {
                "taskId": task_info[0],
                "status": task_info[1],
                "startTime": task_info[2],
                "completedRounds": task_info[3],
                "totalRounds": task_info[4],
                "initialModelHash": task_info[5],
                "currentModelHash": task_info[6]
            }
        except Exception as e:
            logger.error(f"獲取任務資訊失敗: {str(e)}")
            raise
    
    def get_round_info(self, round_id: int) -> Dict:
        """獲取輪次資訊"""
        try:
            round_info = self.contract.functions.getRoundInfo(round_id).call()
            
            return {
                "roundId": round_info[0],
                "status": round_info[1],
                "startTime": round_info[2],
                "endTime": round_info[3],
                "participantCount": round_info[4],
                "completedUpdates": round_info[5],
                "globalModelHash": round_info[6]
            }
        except Exception as e:
            logger.error(f"獲取輪次資訊失敗: {str(e)}")
            raise
    
    def get_client_info(self, client_id: int) -> Dict:
        """獲取客戶端資訊"""
        try:
            client_info = self.contract.functions.getClientInfo(client_id).call()
            
            return {
                "clientAddress": client_info[0],
                "status": client_info[1],
                "contributionScore": client_info[2],
                "lastUpdateTimestamp": client_info[3],
                "selectedForRound": client_info[4]
            }
        except Exception as e:
            logger.error(f"獲取客戶端資訊失敗: {str(e)}")
            raise
    
    def get_registered_clients(self) -> List[Dict]:
        """獲取所有已註冊的客戶端"""
        try:
            system_status = self.get_system_status()
            total_clients = system_status["totalClients"]
            
            clients = []
            for client_id in range(1, total_clients + 1):
                try:
                    client_info = self.get_client_info(client_id)
                    client_info["clientId"] = client_id
                    clients.append(client_info)
                except Exception as e:
                    logger.warning(f"獲取客戶端 {client_id} 資訊失敗: {str(e)}")
            
            return clients
        except Exception as e:
            logger.error(f"獲取已註冊客戶端失敗: {str(e)}")
            raise