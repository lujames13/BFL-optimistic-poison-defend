"""
Unit tests for the Flower client module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import numpy as np
import torch
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.client.client import FlowerClient, FlowerClientFl
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestFlowerClient(unittest.TestCase):
    """Test cases for FlowerClient class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock blockchain connector
        self.blockchain_connector_patcher = patch('fl.blockchain_connector.BlockchainConnector')
        self.mock_blockchain_connector_class = self.blockchain_connector_patcher.start()
        self.mock_blockchain_connector = MagicMock(spec=BlockchainConnector)
        self.mock_blockchain_connector_class.return_value = self.mock_blockchain_connector
        
        # Mock IPFS connector
        self.ipfs_connector_patcher = patch('fl.ipfs_connector.ModelIPFSConnector')
        self.mock_ipfs_connector_class = self.ipfs_connector_patcher.start()
        self.mock_ipfs_connector = MagicMock(spec=ModelIPFSConnector)
        self.mock_ipfs_connector_class.return_value = self.mock_ipfs_connector
        
        # Setup mock responses
        self.mock_blockchain_connector.get_system_status.return_value = {
            "totalClients": 5,
            "totalRounds": 1,
            "currentRound": 1,
            "currentRoundStatus": 1  # Active
        }
        
        self.mock_blockchain_connector.register_client.return_value = True
        
        # Set up model parameters for testing
        self.model_params = [
            np.ones((10, 20), dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones((20, 10), dtype=np.float32),
            np.ones(10, dtype=np.float32)
        ]
        
        self.mock_ipfs_connector.download_model.return_value = self.model_params
        self.mock_ipfs_connector.upload_model.return_value = {"Hash": "QmUpdatedModelHash"}
        
        # Create a simple test model class
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(20, 10)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        self.model_class = SimpleModel
        
        # Create test data
        self.x_train = torch.randn(20, 10)
        self.y_train = torch.randint(0, 10, (20,))
        self.x_test = torch.randn(10, 10)
        self.y_test = torch.randint(0, 10, (10,))
        
        # Client configuration
        self.client_config = {
            "client_id": 1,
            "local_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.01
        }
        
        # Create the client
        self.client = FlowerClient(
            blockchain_connector=self.mock_blockchain_connector,
            ipfs_connector=self.mock_ipfs_connector,
            model_class=self.model_class,
            config=self.client_config,
            train_data=(self.x_train, self.y_train),
            test_data=(self.x_test, self.y_test)
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.blockchain_connector_patcher.stop()
        self.ipfs_connector_patcher.stop()
    
    def test_initialization(self):
        """Test client initialization."""
        # Check config was properly set
        self.assertEqual(self.client.client_id, 1)
        self.assertEqual(self.client.config["local_epochs"], 3)
        self.assertEqual(self.client.config["batch_size"], 4)
        
        # Check that model was instantiated
        self.assertIsInstance(self.client.model, self.model_class)
        
        # Check that data loaders were created
        self.assertIsInstance(self.client.train_loader, DataLoader)
        self.assertIsInstance(self.client.test_loader, DataLoader)
    
    def test_register_client(self):
        """Test client registration with blockchain."""
        # Test registration
        result = self.client.register_client()
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.register_client.assert_called_once()
        
        # Verify result
        self.assertTrue(result)
    
    def test_download_model(self):
        """Test downloading model from IPFS."""
        # Test download
        model_hash = "QmModelHash"
        parameters = self.client.download_model(model_hash)
        
        # Verify IPFS interaction
        self.mock_ipfs_connector.download_model.assert_called_once_with(model_hash)
        
        # Verify parameters
        self.assertEqual(parameters, self.model_params)
    
    def test_load_parameters(self):
        """Test loading parameters into model."""
        # Before loading, get initial model parameters
        initial_params = []
        for param in self.client.model.parameters():
            initial_params.append(param.detach().cpu().numpy())
        
        # Load parameters
        self.client.load_parameters(self.model_params)
        
        # Get model parameters after loading
        loaded_params = []
        for param in self.client.model.parameters():
            loaded_params.append(param.detach().cpu().numpy())
        
        # Verify parameters were loaded
        for i in range(len(loaded_params)):
            np.testing.assert_array_almost_equal(loaded_params[i], self.model_params[i])
            
        # Verify parameters changed
        for i in range(len(loaded_params)):
            self.assertFalse(np.array_equal(initial_params[i], loaded_params[i]))
    
    def test_train_locally(self):
        """Test local training."""
        # First load parameters to ensure consistent starting point
        self.client.load_parameters(self.model_params)
        
        # Store original parameters for comparison
        original_params = []
        for param in self.client.model.parameters():
            original_params.append(param.detach().clone().numpy())
        
        # Train locally
        new_params = self.client.train_locally()
        
        # Verify parameters have changed
        for i in range(len(new_params)):
            with self.subTest(param_idx=i):
                self.assertFalse(np.array_equal(original_params[i], new_params[i]),
                               "Parameters should change after training")
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Load parameters
        self.client.load_parameters(self.model_params)
        
        # Evaluate model
        metrics = self.client.evaluate_model()
        
        # Verify metrics
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)
    
    def test_submit_update(self):
        """Test submitting model update to blockchain."""
        # Setup
        round_id = 1
        
        # Mock blockchain submitModelUpdate
        self.mock_blockchain_connector.submit_model_update.return_value = "QmUpdateHash"
        
        # Test submit update
        result = self.client.submit_update(round_id, self.model_params)
        
        # Verify IPFS interaction
        self.mock_ipfs_connector.upload_model.assert_called_once()
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.submit_model_update.assert_called_once()
        
        # Verify result
        self.assertEqual(result, "QmUpdateHash")
    
    def test_participate_in_round_not_selected(self):
        """Test participation when client is not selected."""
        # Setup
        round_id = 1
        
        # Mock client not selected
        self.mock_blockchain_connector.get_client_info.return_value = {
            "selectedForRound": False
        }
        
        # Test participation
        result = self.client.participate_in_round(round_id)
        
        # Verify client info was checked
        self.mock_blockchain_connector.get_client_info.assert_called_once_with(1)
        
        # Verify no model download attempted
        self.mock_ipfs_connector.download_model.assert_not_called()
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("not selected", result["message"])
    
    def test_participate_in_round_selected(self):
        """Test participation when client is selected."""
        # Setup
        round_id = 1
        model_hash = "QmGlobalModelHash"
        
        # Mock client selected
        self.mock_blockchain_connector.get_client_info.return_value = {
            "selectedForRound": True
        }
        
        # Mock round info
        self.mock_blockchain_connector.get_round_info.return_value = {
            "globalModelHash": model_hash
        }
        
        # Mock model training and submission
        self.mock_blockchain_connector.submit_model_update.return_value = "QmUpdateHash"
        
        # Test participation
        result = self.client.participate_in_round(round_id)
        
        # Verify client info was checked
        self.mock_blockchain_connector.get_client_info.assert_called_once_with(1)
        
        # Verify round info was fetched
        self.mock_blockchain_connector.get_round_info.assert_called_once_with(round_id)
        
        # Verify model was downloaded
        self.mock_ipfs_connector.download_model.assert_called_once_with(model_hash)
        
        # Verify model update was submitted
        self.mock_blockchain_connector.submit_model_update.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["updateHash"], "QmUpdateHash")
        self.assertIn("metrics", result)


class TestFlowerClientFl(unittest.TestCase):
    """Test cases for FlowerClientFl (Flower NumPyClient) class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock blockchain connector
        self.mock_blockchain_connector = MagicMock(spec=BlockchainConnector)
        
        # Mock IPFS connector
        self.mock_ipfs_connector = MagicMock(spec=ModelIPFSConnector)
        
        # Create a simple test model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
        
        # Create test data loaders
        x_train = torch.randn(20, 10)
        y_train = torch.randint(0, 10, (20,))
        x_test = torch.randn(10, 10)
        y_test = torch.randint(0, 10, (10,))
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=4)
        self.test_loader = DataLoader(test_dataset, batch_size=4)
        
        # Client configuration
        self.client_config = {
            "local_epochs": 3,
            "learning_rate": 0.01
        }
        
        # Create the client
        self.fl_client = FlowerClientFl(
            client_id=1,
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            blockchain_connector=self.mock_blockchain_connector,
            ipfs_connector=self.mock_ipfs_connector,
            config=self.client_config
        )
    
    def test_get_parameters(self):
        """Test getting model parameters."""
        # Get parameters
        params = self.fl_client.get_parameters({})
        
        # Verify parameters
        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 4)  # 2 layers, each with weights and bias
        self.assertIsInstance(params[0], np.ndarray)
        self.assertEqual(params[0].shape, (20, 10))  # First layer weights, transposed in numpy
    
    def test_set_parameters(self):
        """Test setting model parameters."""
        # Create test parameters
        params = [
            np.ones((20, 10), dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones((10, 20), dtype=np.float32),
            np.ones(10, dtype=np.float32)
        ]
        
        # Set parameters
        self.fl_client.set_parameters(params)
        
        # Get parameters to verify they were set
        new_params = self.fl_client.get_parameters({})
        
        # Verify parameters
        for i in range(len(params)):
            np.testing.assert_array_almost_equal(params[i], new_params[i])
    
    def test_fit(self):
        """Test model training."""
        # Create test parameters
        params = [
            np.ones((20, 10), dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones((10, 20), dtype=np.float32),
            np.ones(10, dtype=np.float32)
        ]
        
        # Mock blockchain submitModelUpdate
        self.mock_blockchain_connector.submit_model_update.return_value = "QmUpdateHash"
        
        # Fit model
        config = {"round_id": 1, "epochs": 1}
        updated_params, num_examples, metrics = self.fl_client.fit(params, config)
        
        # Verify updated parameters
        self.assertIsInstance(updated_params, list)
        self.assertEqual(len(updated_params), 4)
        
        # Verify number of examples
        self.assertEqual(num_examples, 20)
        
        # Verify metrics
        self.assertIn("model_hash", metrics)
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.submit_model_update.assert_called_once()
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Create test parameters
        params = [
            np.ones((20, 10), dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones((10, 20), dtype=np.float32),
            np.ones(10, dtype=np.float32)
        ]
        
        # Evaluate model
        loss, num_examples, metrics = self.fl_client.evaluate(params, {})
        
        # Verify loss
        self.assertIsInstance(loss, float)
        
        # Verify number of examples
        self.assertEqual(num_examples, 10)
        
        # Verify metrics
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["accuracy"], float)


if __name__ == '__main__':
    unittest.main()