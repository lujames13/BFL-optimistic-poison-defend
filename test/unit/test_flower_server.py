"""Test suite for the blockchain-integrated Flower server."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import tensorflow as tf
from typing import Dict, List, Tuple

# We'll need to patch these imports in the tests
# from fl.server import BlockchainFlowerServer, get_strategy
# from fl.blockchain_connector import BlockchainConnector
# from fl.ipfs_connector import ModelIPFSConnector


class TestBlockchainFlowerServer(unittest.TestCase):
    """Test suite for the blockchain-integrated Flower server."""

    def setUp(self):
        """Set up test data and mocks."""
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
        
        # Create dummy dataset for server-side evaluation
        self.x_test = np.random.random((30, 5))
        self.y_test = np.random.randint(0, 2, 30)
        
        # Mock blockchain connector
        self.mock_blockchain = MagicMock()
        self.mock_blockchain.currentTaskId = 1
        self.mock_blockchain.get_system_status.return_value = {
            "totalClients": 3,
            "totalRounds": 0,
            "currentRound": 0,
            "currentRoundStatus": 0  # Inactive
        }
        self.mock_blockchain.start_round.return_value = 1
        self.mock_blockchain.applyKrumDefense.return_value = 1  # Selected client ID
        
        # Mock IPFS connector
        self.mock_ipfs = MagicMock()
        self.mock_ipfs.upload_model.return_value = {"Hash": "QmTestHash"}
        self.mock_ipfs.download_model.return_value = self.model.get_weights()

    @patch('fl.server.BlockchainConnector')
    @patch('fl.server.ModelIPFSConnector')
    def test_server_initialization(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the server initializes correctly."""
        from fl.server import BlockchainFlowerServer
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract",
            ipfs_url="http://localhost:5001",
            x_test=self.x_test,
            y_test=self.y_test
        )
        
        # Check that server is initialized with correct values
        self.assertIsNotNone(server.model)
        self.assertIsNotNone(server.blockchain)
        self.assertIsNotNone(server.ipfs)
        
        # Check that blockchain connector was initialized with correct contract address
        mock_blockchain_class.assert_called_once()
        args, kwargs = mock_blockchain_class.call_args
        self.assertEqual(kwargs.get("contract_address"), "0xTestContract")
        
        # Check that IPFS connector was initialized with correct URL
        mock_ipfs_class.assert_called_once()
        args, kwargs = mock_ipfs_class.call_args
        self.assertEqual(kwargs.get("api_url"), "http://localhost:5001")

    @patch('fl.server.BlockchainConnector')
    @patch('fl.server.ModelIPFSConnector')
    def test_create_task(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the server correctly creates a new federated learning task."""
        from fl.server import BlockchainFlowerServer
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
        # Mock the blockchain createTask method
        self.mock_blockchain.createTask.return_value = 1  # Task ID
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract"
        )
        
        # Create a new task
        task_id = server.create_task(total_rounds=5)
        
        # Check that a task was created
        self.assertEqual(task_id, 1)
        
        # Check that the blockchain connector was called
        self.mock_blockchain.createTask.assert_called_once()
        
        # Check that model was uploaded to IPFS
        self.mock_ipfs.upload_model.assert_called_once()

    @patch('fl.server.BlockchainConnector')
    @patch('fl.server.ModelIPFSConnector')
    def test_start_round(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the server correctly starts a new training round."""
        from fl.server import BlockchainFlowerServer
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract"
        )
        
        # Start a new round
        round_id = server.start_round(task_id=1)
        
        # Check that a round was started
        self.assertEqual(round_id, 1)
        
        # Check that the blockchain connector was called
        self.mock_blockchain.start_round.assert_called_once_with(1)

    @patch('fl.server.BlockchainConnector')
    @patch('fl.server.ModelIPFSConnector')
    def test_apply_krum_defense(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the server correctly applies Krum defense."""
        from fl.server import BlockchainFlowerServer
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
        # Create server
        server = BlockchainFlowerServer(
            initial_model=self.model,
            contract_address="0xTestContract",
            use_krum=True,
            byzantine_threshold=1
        )
        
        # Apply Krum defense
        selected_client_id = server.apply_defense(round_id=1)
        
        # Check that Krum defense was applied
        self.assertEqual(selected_client_id, 1)
        
        # Check that the blockchain connector was called
        self.mock_blockchain.applyKrumDefense.assert_called_once_with(1)

    @patch('fl.server.BlockchainConnector')
    @patch('fl.server.ModelIPFSConnector')
    def test_get_strategy(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the get_strategy function returns the correct strategy."""
        from fl.server import get_strategy
        
        # Create evaluation function
        def evaluate_fn(weights):
            model = self.model
            model.set_weights(weights)
            loss, accuracy = model.evaluate(self.x_test, self.y_test)
            return loss, {"accuracy": accuracy}
        
        # Get strategy with Krum defense
        strategy = get_strategy(
            evaluate_fn=evaluate_fn,
            fraction_fit=1.0,
            min_fit_clients=3,
            min_available_clients=3,
            use_krum=True,
            byzantine_threshold=1
        )
        
        # Check that strategy is correctly configured
        self.assertEqual(strategy.fraction_fit, 1.0)
        self.assertEqual(strategy.min_fit_clients, 3)
        self.assertEqual(strategy.min_available_clients, 3)
        
        # Get strategy without Krum defense
        strategy = get_strategy(
            evaluate_fn=evaluate_fn,
            fraction_fit=0.5,
            min_fit_clients=2,
            min_available_clients=3,
            use_krum=False
        )
        
        # Check that strategy is correctly configured
        self.assertEqual(strategy.fraction_fit, 0.5)
        self.assertEqual(strategy.min_fit_clients, 2)
        self.assertEqual(strategy.min_available_clients, 3)


if __name__ == "__main__":
    unittest.main()