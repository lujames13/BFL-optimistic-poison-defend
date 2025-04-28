"""
Test suite for the Flower client implementation with blockchain integration.
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
from fl.client.client import FlowerClient
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestFlowerClient(unittest.TestCase):
    """Test cases for the Flower client implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the blockchain connector
        self.blockchain_mock = MagicMock(spec=BlockchainConnector)
        self.blockchain_mock.get_system_status.return_value = {
            "totalClients": 5,
            "totalRounds": 1,
            "currentRound": 1,
            "currentRoundStatus": 1  # Active
        }
        
        # Mock client registration
        self.blockchain_mock.register_client.return_value = True
        
        # Mock the IPFS connector
        self.ipfs_mock = MagicMock(spec=ModelIPFSConnector)
        
        # Mock downloading model from IPFS
        self.model_params = [np.ones((10, 2)), np.ones(2)]
        self.ipfs_mock.download_model.return_value = self.model_params
        
        # Mock uploading model to IPFS
        self.ipfs_mock.upload_model.return_value = {"Hash": "QmUpdatedModelHash"}
        
        # Create a simple test model class
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                return self.fc(x)
        
        # Create test data
        self.x_train = torch.randn(20, 10)
        self.y_train = torch.randint(0, 2, (20,))
        
        # Client configuration
        self.config = {
            "client_id": 1,
            "local_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.01
        }
        
        # Create the client with mocked dependencies
        self.client = FlowerClient(
            blockchain_connector=self.blockchain_mock,
            ipfs_connector=self.ipfs_mock,
            model_class=SimpleModel,
            config=self.config,
            train_data=(self.x_train, self.y_train),
            test_data=(self.x_train, self.y_train)  # Using same data for simplicity
        )

    def test_initialization(self):
        """Test client initialization."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.client_id, 1)
        self.assertEqual(self.client.config["local_epochs"], 3)

    def test_register_client(self):
        """Test client registration with blockchain."""
        # Run
        result = self.client.register_client()
        
        # Assert
        self.blockchain_mock.register_client.assert_called_once()
        self.assertTrue(result)

    def test_download_model(self):
        """Test downloading model from IPFS."""
        # Setup
        model_hash = "QmModelHash"
        
        # Run
        params = self.client.download_model(model_hash)
        
        # Assert
        self.ipfs_mock.download_model.assert_called_once_with(model_hash)
        self.assertEqual(params, self.model_params)

    def test_load_parameters(self):
        """Test loading parameters into model."""
        # Setup
        params = [np.ones((10, 2)), np.ones(2)]
        
        # Run
        self.client.load_parameters(params)
        
        # Get model parameters after loading
        model_params = []
        for param in self.client.model.parameters():
            model_params.append(param.detach().numpy())
        
        # Assert parameters were loaded correctly (approximately)
        for i in range(len(model_params)):
            np.testing.assert_allclose(model_params[i], params[i], rtol=1e-5)

    def test_train_locally(self):
        """Test local training."""
        # Setup - load parameters first
        self.client.load_parameters(self.model_params)
        
        # Store original parameters for comparison
        original_params = []
        for param in self.client.model.parameters():
            original_params.append(param.detach().clone().numpy())
        
        # Run training
        new_params = self.client.train_locally()
        
        # Assert parameters changed after training
        for i in range(len(original_params)):
            with self.subTest(param_idx=i):
                self.assertFalse(np.array_equal(original_params[i], new_params[i]),
                                "Parameters should change after training")

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Setup - load parameters
        self.client.load_parameters(self.model_params)
        
        # Run evaluation
        metrics = self.client.evaluate_model()
        
        # Assert
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)

    def test_submit_update(self):
        """Test submitting model update to blockchain."""
        # Setup
        round_id = 1
        parameters = [np.ones((10, 2)), np.ones(2)]
        
        # Mock blockchain submitModelUpdate
        self.blockchain_mock.submit_model_update.return_value = "QmUpdateHash"
        
        # Run
        result = self.client.submit_update(round_id, parameters)
        
        # Assert
        self.ipfs_mock.upload_model.assert_called_once()
        self.blockchain_mock.submit_model_update.assert_called_once_with(
            round_id, parameters
        )
        self.assertEqual(result, "QmUpdateHash")

    def test_participate_in_round(self):
        """Test full round participation flow."""
        # Setup
        round_id = 1
        model_hash = "QmGlobalModelHash"
        
        # Mock blockchain getGlobalModelHash
        self.blockchain_mock.get_round_info.return_value = {
            "globalModelHash": model_hash
        }
        
        # Mock client selection status
        self.blockchain_mock.get_client_info.return_value = {
            "selectedForRound": True
        }
        
        # Mock registration and submission
        self.blockchain_mock.register_client.return_value = True
        self.blockchain_mock.submit_model_update.return_value = "QmUpdateHash"
        
        # Run
        result = self.client.participate_in_round(round_id)
        
        # Assert
        self.ipfs_mock.download_model.assert_called_once()
        self.assertTrue(result["success"])
        self.assertEqual(result["updateHash"], "QmUpdateHash")

    def test_handle_not_selected(self):
        """Test handling when client is not selected for round."""
        # Setup
        round_id = 1
        
        # Mock client selection status - not selected
        self.blockchain_mock.get_client_info.return_value = {
            "selectedForRound": False
        }
        
        # Run
        result = self.client.participate_in_round(round_id)
        
        # Assert
        self.assertFalse(result["success"])
        self.assertIn("not selected", result["message"])
        self.ipfs_mock.download_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()