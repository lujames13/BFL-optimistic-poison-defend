"""
Test suite for the Flower server implementation with blockchain integration.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# The modules we'll be testing (these will be implemented next)
from fl.server.server import FlowerServer
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestFlowerServer(unittest.TestCase):
    """Test cases for the Flower server implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the blockchain connector
        self.blockchain_mock = MagicMock(spec=BlockchainConnector)
        self.blockchain_mock.get_system_status.return_value = {
            "totalClients": 5,
            "totalRounds": 0,
            "currentRound": 0,
            "currentRoundStatus": 0  # Inactive
        }
        
        # Mock successful task creation
        self.blockchain_mock.createTask.return_value = 1  # Task ID
        
        # Mock the IPFS connector
        self.ipfs_mock = MagicMock(spec=ModelIPFSConnector)
        self.ipfs_mock.upload_model.return_value = {"Hash": "QmInitialModelHash"}
        
        # Create a simple test model
        self.test_model = self._create_test_model()
        
        # Config for the server
        self.config = {
            "rounds": 5,
            "min_clients": 3,
            "sample_fraction": 0.8,
            "min_fit_clients": 3,
            "min_eval_clients": 3,
            "accept_failures": False
        }
        
        # Create the server with mocked dependencies
        self.server = FlowerServer(
            blockchain_connector=self.blockchain_mock,
            ipfs_connector=self.ipfs_mock,
            model=self.test_model,
            config=self.config
        )

    def _create_test_model(self):
        """Create a simple PyTorch model for testing."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                return self.fc(x)
        
        return SimpleModel()

    def test_initialization(self):
        """Test server initialization."""
        self.assertIsNotNone(self.server)
        self.assertEqual(self.server.config, self.config)
        self.assertEqual(self.server.total_rounds, 5)

    def test_initialize_task(self):
        """Test task initialization."""
        task_id = self.server.initialize_task()
        
        # Assert blockchain connector was called to create a task
        self.blockchain_mock.createTask.assert_called_once()
        
        # Assert IPFS connector was called to upload the initial model
        self.ipfs_mock.upload_model.assert_called_once()
        
        # Assert task ID was returned correctly
        self.assertEqual(task_id, 1)

    def test_start_round(self):
        """Test starting a new training round."""
        # Setup
        self.server.task_id = 1
        self.blockchain_mock.start_round.return_value = 1  # Round ID
        
        # Run
        round_id = self.server.start_round()
        
        # Assert
        self.blockchain_mock.start_round.assert_called_once_with(1)
        self.assertEqual(round_id, 1)

    def test_select_clients(self):
        """Test client selection for a round."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Mock registered clients
        clients = [{"clientId": i, "status": 1} for i in range(1, 6)]  # 5 clients
        self.blockchain_mock.get_registered_clients.return_value = clients
        
        # Run
        selected_clients = self.server.select_clients(num_clients=3)
        
        # Assert
        self.assertEqual(len(selected_clients), 3)
        self.blockchain_mock.selectClients.assert_called_once()

    def test_aggregate_updates_with_krum(self):
        """Test aggregating updates using Krum defense."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Mock client updates
        updates = [
            {"clientId": 1, "modelHash": "QmUpdate1", "parameters": [np.ones((10, 2)), np.ones(2)]},
            {"clientId": 2, "modelHash": "QmUpdate2", "parameters": [np.ones((10, 2)) * 1.1, np.ones(2) * 1.1]},
            {"clientId": 3, "modelHash": "QmUpdate3", "parameters": [np.ones((10, 2)) * 0.9, np.ones(2) * 0.9]},
            {"clientId": 4, "modelHash": "QmMalicious", "parameters": [np.ones((10, 2)) * 100, np.ones(2) * 100]}
        ]
        
        # Mock blockchain Krum defense call
        self.blockchain_mock.applyKrumDefense.return_value = 2  # Client 2 selected
        
        # Mock IPFS upload for the aggregated model
        self.ipfs_mock.upload_model.return_value = {"Hash": "QmAggregatedModel"}
        
        # Run
        result = self.server.aggregate_updates(updates)
        
        # Assert
        self.blockchain_mock.applyKrumDefense.assert_called_once()
        self.ipfs_mock.upload_model.assert_called()
        self.assertEqual(result["selectedClientId"], 2)
        self.assertEqual(result["aggregatedModelHash"], "QmAggregatedModel")

    def test_complete_round(self):
        """Test completing a training round."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Run
        self.server.complete_round()
        
        # Assert
        self.blockchain_mock.complete_round.assert_called_once_with(1)

    def test_complete_task(self):
        """Test completing a federated learning task."""
        # Setup
        self.server.task_id = 1
        self.ipfs_mock.upload_model.return_value = {"Hash": "QmFinalModel"}
        
        # Run
        self.server.complete_task()
        
        # Assert
        self.blockchain_mock.completeTask.assert_called_once_with(
            1, "QmFinalModel"
        )

    def test_distribute_rewards(self):
        """Test reward distribution to participants."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        client_ids = [1, 2, 3]
        
        # Run
        self.server.distribute_rewards(client_ids)
        
        # Assert
        self.blockchain_mock.distributeRewards.assert_called_once_with(
            client_ids, 1
        )

    def test_handle_round_failure(self):
        """Test handling of round failures."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Run with not enough clients
        result = self.server.handle_round_failure("Not enough clients")
        
        # Assert
        self.assertFalse(result["success"])
        self.assertIn("Not enough clients", result["message"])


if __name__ == "__main__":
    unittest.main()