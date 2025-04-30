"""Integration tests for Blockchain Federated Learning system.

This module tests the integration between the Federated Learning components
and the blockchain infrastructure.
"""

import unittest
import os
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_fl_blockchain")


class TestFLBlockchainIntegration(unittest.TestCase):
    """Test suite for FL-blockchain integration."""

    def setUp(self):
        """Set up test environment."""
        # Create a simple model for testing
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy dataset
        self.x_train = np.random.random((20, 5))
        self.y_train = np.random.randint(0, 2, 20)
        self.x_val = np.random.random((10, 5))
        self.y_val = np.random.randint(0, 2, 10)
        
        # Create temp dir for model storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch environment variables for blockchain private key
        self.env_patcher = patch.dict('os.environ', {'PRIVATE_KEY': '0x' + 'a' * 64})
        self.env_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        # Remove temp dir
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Stop patchers
        self.env_patcher.stop()

    @patch('fl.blockchain_connector.BlockchainConnector')
    @patch('fl.ipfs_connector.ModelIPFSConnector')
    def test_client_server_interaction(self, mock_ipfs_class, mock_blockchain_class):
        """Test basic interaction between client and server components."""
        from fl.client import BlockchainFlowerClient
        from fl.server import BlockchainFlowerServer
        
        # Mock blockchain connector
        mock_blockchain = MagicMock()
        mock_blockchain.currentTaskId = 1
        mock_blockchain.get_system_status.return_value = {
            "totalClients": 1,
            "totalRounds": 0,
            "currentRound": 0,
            "currentRoundStatus": 0  # Inactive
        }
        mock_blockchain.create_task.return_value = 1
        mock_blockchain.start_round.return_value = 1
        mock_blockchain.submit_model_update.return_value = "QmHashOfModelUpdate"
        mock_blockchain_class.return_value = mock_blockchain
        
        # Mock IPFS connector
        mock_ipfs = MagicMock()
        mock_ipfs.upload_model.return_value = {"Hash": "QmTestHash"}
        mock_ipfs.download_model.return_value = self.model.get_weights()
        mock_ipfs_class.return_value = mock_ipfs
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract",
            ipfs_url="http://localhost:5001"
        )
        
        # Create client
        client = BlockchainFlowerClient(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            client_id=1,
            contract_address="0xTestContract"
        )
        
        # Simulate FL workflow
        
        # 1. Server creates task
        task_id = server.create_task(total_rounds=3)
        self.assertEqual(task_id, 1)
        mock_blockchain.createTask.assert_called_once()
        mock_ipfs.upload_model.assert_called_once()
        
        # 2. Server starts round
        round_id = server.start_round(task_id)
        self.assertEqual(round_id, 1)
        mock_blockchain.start_round.assert_called_once_with(1)
        
        # 3. Client gets parameters from server
        parameters = client.get_parameters({})
        
        # 4. Client performs local training
        config = {"current_round": 1, "local_epochs": 1}
        updated_parameters, num_examples, metrics = client.fit(parameters, config)
        
        # Check that client submitted update to blockchain
        mock_blockchain.submit_model_update.assert_called_once()
        self.assertIsInstance(metrics, dict)
        self.assertIn("blockchain_hash", metrics)
        
        # 5. Server applies defense
        server.use_krum = True
        selected_client_id = server.apply_defense(round_id)
        mock_blockchain.applyKrumDefense.assert_called_once_with(1)
        
        # 6. Server completes round
        mock_ipfs.upload_model.reset_mock()
        server.complete_round(round_id, "QmNewGlobalModelHash")
        mock_blockchain.updateGlobalModel.assert_called_once_with(1, "QmNewGlobalModelHash")
        mock_blockchain.complete_round.assert_called_once_with(1)
        
        # 7. Client evaluates model
        loss, num_examples, eval_metrics = client.evaluate(updated_parameters, {})
        self.assertIsInstance(loss, float)
        self.assertEqual(num_examples, len(self.x_val))
        self.assertIn("accuracy", eval_metrics)
        
        # 8. Server completes task
        mock_ipfs.upload_model.reset_mock()
        server.complete_task(task_id, "QmFinalModelHash")
        mock_blockchain.completeTask.assert_called_once_with(1, "QmFinalModelHash")

    @patch('fl.blockchain_connector.BlockchainConnector')
    @patch('fl.ipfs_connector.ModelIPFSConnector')
    def test_krum_defense_integration(self, mock_ipfs_class, mock_blockchain_class):
        """Test integration of Krum defense with FL system."""
        from fl.client import BlockchainFlowerClient, ByzantineClient
        from fl.server import BlockchainFlowerServer, KrumFedAvg
        from fl.defense import krum_select
        
        # Mock blockchain connector
        mock_blockchain = MagicMock()
        mock_blockchain.currentTaskId = 1
        mock_blockchain.get_system_status.return_value = {
            "totalClients": 5,
            "totalRounds": 0,
            "currentRound": 0,
            "currentRoundStatus": 0  # Inactive
        }
        mock_blockchain.applyKrumDefense.return_value = 1  # Selected client ID
        mock_blockchain_class.return_value = mock_blockchain
        
        # Mock IPFS connector
        mock_ipfs = MagicMock()
        mock_ipfs.upload_model.return_value = {"Hash": "QmTestHash"}
        mock_ipfs.download_model.return_value = self.model.get_weights()
        mock_ipfs_class.return_value = mock_ipfs
        
        # Create honest clients
        honest_clients = []
        for i in range(3):
            client = BlockchainFlowerClient(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train,
                x_val=self.x_val,
                y_val=self.y_val,
                client_id=i+1,
                contract_address="0xTestContract"
            )
            honest_clients.append(client)
        
        # Create byzantine clients
        byzantine_clients = []
        for i in range(2):
            client = ByzantineClient(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train,
                x_val=self.x_val,
                y_val=self.y_val,
                client_id=i+4,
                contract_address="0xTestContract",
                attack_type="model_replacement",
                attack_params={"scale_factor": 10.0}
            )
            byzantine_clients.append(client)
        
        all_clients = honest_clients + byzantine_clients
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract",
            use_krum=True,
            byzantine_threshold=1
        )
        
        # Test Krum strategy directly
        from flwr.common import Parameters
        from flwr.server.client_proxy import ClientProxy
        
        # Create strategy
        strategy = KrumFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            byzantine_threshold=1
        )
        
        # Create mock client proxies and fit results
        proxies = []
        fit_results = []
        
        # Get initial weights
        initial_weights = self.model.get_weights()
        
        # Get client updates
        client_updates = []
        for client in all_clients:
            updated_parameters, _, _ = client.fit(initial_weights, {"current_round": 1})
            client_updates.append(updated_parameters)
        
        # Verify that Krum can identify malicious updates
        client_ids = [i+1 for i in range(5)]
        selected_idx, selected_update = krum_select(client_updates, client_ids, 1)
        
        # Selected update should be from an honest client (index 0-2)
        self.assertIn(selected_idx, [0, 1, 2])
        
        # Test blockchain integration of Krum
        selected_client_id = server.apply_defense(1)
        mock_blockchain.applyKrumDefense.assert_called_once_with(1)
        self.assertEqual(selected_client_id, 1)  # The mock returns 1


if __name__ == "__main__":
    unittest.main()