"""Test suite for the Flower client with blockchain integration."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import tensorflow as tf
from typing import Dict, List, Tuple

# We'll need to patch these imports in the tests
# from fl.client import BlockchainFlowerClient
# from fl.blockchain_connector import BlockchainConnector
# from fl.ipfs_connector import ModelIPFSConnector


class TestBlockchainFlowerClient(unittest.TestCase):
    """Test suite for the blockchain-integrated Flower client."""

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
        
        # Create dummy dataset
        self.x_train = np.random.random((20, 5))
        self.y_train = np.random.randint(0, 2, 20)
        self.x_val = np.random.random((10, 5))
        self.y_val = np.random.randint(0, 2, 10)
        
        # Mock blockchain connector
        self.mock_blockchain = MagicMock()
        self.mock_blockchain.client_id = 1
        self.mock_blockchain.submit_model_update.return_value = "QmHashOfModelUpdate"
        
        # Mock IPFS connector
        self.mock_ipfs = MagicMock()
        self.mock_ipfs.upload_model.return_value = {"Hash": "QmTestHash"}
        self.mock_ipfs.download_model.return_value = self.model.get_weights()

    @patch('fl.client.BlockchainConnector')
    @patch('fl.client.ModelIPFSConnector')
    def test_client_initialization(self, mock_ipfs_class, mock_blockchain_class):
        """Test that the client initializes correctly."""
        from fl.client import BlockchainFlowerClient
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
        # Create client
        client = BlockchainFlowerClient(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            client_id=1,
            contract_address="0xTestContract",
            ipfs_url="http://localhost:5001"
        )
        
        # Check that client is initialized with correct values
        self.assertEqual(client.client_id, 1)
        self.assertIsNotNone(client.model)
        self.assertIsNotNone(client.blockchain)
        self.assertIsNotNone(client.ipfs)
        
        # Check that blockchain connector was initialized with correct contract address
        mock_blockchain_class.assert_called_once()
        args, kwargs = mock_blockchain_class.call_args
        self.assertEqual(kwargs.get("contract_address"), "0xTestContract")
        
        # Check that IPFS connector was initialized with correct URL
        mock_ipfs_class.assert_called_once()
        args, kwargs = mock_ipfs_class.call_args
        self.assertEqual(kwargs.get("api_url"), "http://localhost:5001")

    @patch('fl.client.BlockchainConnector')
    @patch('fl.client.ModelIPFSConnector')
    def test_get_parameters(self, mock_ipfs_class, mock_blockchain_class):
        """Test that client correctly returns model parameters."""
        from fl.client import BlockchainFlowerClient
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
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
        
        # Get parameters
        parameters = client.get_parameters({})
        
        # Check that parameters match the model weights
        model_weights = self.model.get_weights()
        self.assertEqual(len(parameters), len(model_weights))
        for p1, p2 in zip(parameters, model_weights):
            np.testing.assert_allclose(p1, p2)

    @patch('fl.client.BlockchainConnector')
    @patch('fl.client.ModelIPFSConnector')
    def test_fit(self, mock_ipfs_class, mock_blockchain_class):
        """Test that client correctly performs local training and submits update."""
        from fl.client import BlockchainFlowerClient
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
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
        
        # Create parameters
        parameters = client.get_parameters({})
        
        # Create config with current round
        config = {"current_round": 1, "local_epochs": 1}
        
        # Call fit
        updated_parameters, num_examples, metrics = client.fit(parameters, config)
        
        # Check that fit returns the right type of values
        self.assertIsInstance(updated_parameters, list)
        self.assertEqual(len(updated_parameters), len(parameters))
        self.assertEqual(num_examples, len(self.x_train))
        self.assertIsInstance(metrics, dict)
        
        # Check that blockchain connector was called to submit update
        self.mock_blockchain.submit_model_update.assert_called_once()
        args, kwargs = self.mock_blockchain.submit_model_update.call_args
        self.assertEqual(args[0], 1)  # client_id
        self.assertEqual(args[1], 1)  # round_id

        # Check that IPFS was used to upload the model
        self.mock_ipfs.upload_model.assert_called_once()

    @patch('fl.client.BlockchainConnector')
    @patch('fl.client.ModelIPFSConnector')
    def test_evaluate(self, mock_ipfs_class, mock_blockchain_class):
        """Test that client correctly evaluates the model."""
        from fl.client import BlockchainFlowerClient
        
        # Setup mocks
        mock_blockchain_class.return_value = self.mock_blockchain
        mock_ipfs_class.return_value = self.mock_ipfs
        
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
        
        # Get parameters
        parameters = client.get_parameters({})
        
        # Call evaluate
        loss, num_examples, metrics = client.evaluate(parameters, {})
        
        # Check that evaluate returns the right type of values
        self.assertIsInstance(loss, float)
        self.assertEqual(num_examples, len(self.x_val))
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIsInstance(metrics['accuracy'], float)


if __name__ == "__main__":
    unittest.main()