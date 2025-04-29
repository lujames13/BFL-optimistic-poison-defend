"""
Unit tests for the Flower server module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import numpy as np
import torch
import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.server.server import FlowerServer, KrumAggregationStrategy
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestFlowerServer(unittest.TestCase):
    """Test cases for FlowerServer class."""

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
            "totalRounds": 0,
            "currentRound": 0,
            "currentRoundStatus": 0  # Inactive
        }
        
        self.mock_blockchain_connector.createTask.return_value = 1  # Task ID
        
        self.mock_ipfs_connector.upload_model.return_value = {
            "Hash": "QmInitialModelHash",
            "metadata": {"model_id": "initial_model"}
        }
        
        # Create a simple test model
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        )
        
        # Server configuration
        self.server_config = {
            "rounds": 5,
            "min_clients": 3,
            "sample_fraction": 0.8,
            "min_fit_clients": 3,
            "min_eval_clients": 3,
            "accept_failures": False,
            "byzantine_tolerance": 1
        }
        
        # Create the server
        self.server = FlowerServer(
            blockchain_connector=self.mock_blockchain_connector,
            ipfs_connector=self.mock_ipfs_connector,
            model=self.test_model,
            config=self.server_config
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.blockchain_connector_patcher.stop()
        self.ipfs_connector_patcher.stop()
    
    def test_initialization(self):
        """Test server initialization."""
        # Check that configuration was properly set
        self.assertEqual(self.server.config, self.server_config)
        self.assertEqual(self.server.total_rounds, 5)
        self.assertEqual(self.server.min_clients, 3)
        self.assertEqual(self.server.byzantine_tolerance, 1)
        
        # Check that connectors were properly set
        self.assertEqual(self.server.blockchain_connector, self.mock_blockchain_connector)
        self.assertEqual(self.server.ipfs_connector, self.mock_ipfs_connector)
        
        # Check that model was properly set
        self.assertEqual(type(self.server.model), type(self.test_model))
    
    def test_initialize_task(self):
        """Test task initialization."""
        # Test initialize task
        task_id = self.server.initialize_task()
        
        # Check model was uploaded to IPFS
        self.mock_ipfs_connector.upload_model.assert_called_once()
        
        # Check that task was created on blockchain
        self.mock_blockchain_connector.createTask.assert_called_once()
        
        # Check that task ID was returned correctly
        self.assertEqual(task_id, 1)
        self.assertEqual(self.server.task_id, 1)
    
    def test_start_round(self):
        """Test starting a new training round."""
        # Setup
        self.server.task_id = 1
        self.mock_blockchain_connector.start_round.return_value = 1  # Round ID
        
        # Test start round
        round_id = self.server.start_round()
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.start_round.assert_called_once_with(1)
        
        # Verify returned round ID
        self.assertEqual(round_id, 1)
        self.assertEqual(self.server.current_round, 1)
    
    def test_select_clients(self):
        """Test client selection for a round."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Mock registered clients
        mock_clients = [
            {"clientId": 1, "status": 1, "contributionScore": 10},
            {"clientId": 2, "status": 1, "contributionScore": 20},
            {"clientId": 3, "status": 1, "contributionScore": 30},
            {"clientId": 4, "status": 1, "contributionScore": 40},
            {"clientId": 5, "status": 1, "contributionScore": 50}
        ]
        
        self.mock_blockchain_connector.get_registered_clients.return_value = mock_clients
        
        # Test select clients
        selected_clients = self.server.select_clients(num_clients=3)
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.get_registered_clients.assert_called_once()
        self.mock_blockchain_connector.selectClients.assert_called_once()
        
        # Verify number of selected clients
        self.assertEqual(len(selected_clients), 3)
        
        # Verify all selected clients are from the registered list
        for client_id in selected_clients:
            self.assertIn(client_id, [c["clientId"] for c in mock_clients])
    
    def test_aggregate_updates(self):
        """Test aggregating updates with Krum defense."""
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
        self.mock_blockchain_connector.applyKrumDefense.return_value = 2  # Client 2 selected
        
        # Mock IPFS upload for the aggregated model
        self.mock_ipfs_connector.upload_model.return_value = {"Hash": "QmAggregatedModel"}
        
        # Test aggregate updates
        result = self.server.aggregate_updates(updates)
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.applyKrumDefense.assert_called_once_with(1)
        
        # Verify IPFS interaction for uploading the selected model
        self.mock_ipfs_connector.upload_model.assert_called()
        
        # Verify blockchain interaction for updating global model
        self.mock_blockchain_connector.updateGlobalModel.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["selectedClientId"], 2)
        self.assertEqual(result["aggregatedModelHash"], "QmAggregatedModel")
    
    def test_complete_round(self):
        """Test completing a training round."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Test complete round
        result = self.server.complete_round()
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.complete_round.assert_called_once_with(1)
    
    def test_complete_task(self):
        """Test completing a federated learning task."""
        # Setup
        self.server.task_id = 1
        
        # Mock IPFS upload for final model
        self.mock_ipfs_connector.upload_model.return_value = {"Hash": "QmFinalModel"}
        
        # Test complete task
        result = self.server.complete_task()
        
        # Verify IPFS interaction
        self.mock_ipfs_connector.upload_model.assert_called_once()
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.completeTask.assert_called_once_with(1, "QmFinalModel")
    
    def test_distribute_rewards(self):
        """Test reward distribution to participants."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        client_ids = [1, 2, 3]
        
        # Test distribute rewards
        result = self.server.distribute_rewards(client_ids)
        
        # Verify blockchain interaction
        self.mock_blockchain_connector.distributeRewards.assert_called_once_with(client_ids, 1)
    
    def test_handle_round_failure(self):
        """Test handling of round failures."""
        # Setup
        self.server.task_id = 1
        self.server.current_round = 1
        
        # Test handle round failure
        result = self.server.handle_round_failure("Not enough clients")
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("Not enough clients", result["message"])
    
    def test_start_training(self):
        """Test full training workflow."""
        # Setup mock responses for initialization
        self.mock_blockchain_connector.createTask.return_value = 1
        self.mock_blockchain_connector.start_round.return_value = 1
        
        # Mock client selection
        mock_clients = [{"clientId": i, "status": 1} for i in range(1, 6)]
        self.mock_blockchain_connector.get_registered_clients.return_value = mock_clients
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            # Test start training
            result = self.server.start_training()
        
        # Verify task was initialized
        self.mock_blockchain_connector.createTask.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["task_id"], 1)


class TestKrumAggregationStrategy(unittest.TestCase):
    """Test cases for KrumAggregationStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock blockchain connector
        self.mock_blockchain_connector = MagicMock(spec=BlockchainConnector)
        
        # Mock IPFS connector
        self.mock_ipfs_connector = MagicMock(spec=ModelIPFSConnector)
        
        # Create strategy instance
        self.strategy = KrumAggregationStrategy(
            blockchain_connector=self.mock_blockchain_connector,
            ipfs_connector=self.mock_ipfs_connector,
            task_id=1,
            round_id=1,
            byzantine_tolerance=1,
            min_clients=3,
            fraction_fit=0.8,
            min_fit_clients=3
        )
        
        # Create mock clients and parameters
        self.client1 = MagicMock()
        self.client1.properties = {"client_id": "1"}
        
        self.client2 = MagicMock()
        self.client2.properties = {"client_id": "2"}
        
        self.client3 = MagicMock()
        self.client3.properties = {"client_id": "3"}
        
        # Create mock fit results with parameters
        self.fit_res1 = MagicMock()
        self.fit_res1.parameters = b"params1"
        
        self.fit_res2 = MagicMock()
        self.fit_res2.parameters = b"params2"
        
        self.fit_res3 = MagicMock()
        self.fit_res3.parameters = b"params3"
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.task_id, 1)
        self.assertEqual(self.strategy.round_id, 1)
        self.assertEqual(self.strategy.byzantine_tolerance, 1)
        self.assertEqual(self.strategy.min_clients, 3)
    
    @patch('fl.server.server.parameters_to_ndarrays')
    def test_aggregate_fit_not_enough_clients(self, mock_parameters_to_ndarrays):
        """Test aggregation with not enough clients for Krum."""
        # Setup mock parameters
        mock_parameters_to_ndarrays.side_effect = lambda p: [np.ones((5, 5))]
        
        # Only two clients, not enough for Krum with f=1
        results = [(self.client1, self.fit_res1), (self.client2, self.fit_res2)]
        failures = []
        
        # Mock fallback strategy's aggregate_fit method
        with patch.object(self.strategy.fallback_strategy, 'aggregate_fit') as mock_fallback:
            mock_fallback.return_value = (b"fallback_params", {})
            
            # Call aggregate_fit
            params, metrics = self.strategy.aggregate_fit(1, results, failures)
            
            # Verify fallback was used
            mock_fallback.assert_called_once()
            self.assertEqual(params, b"fallback_params")
    
    @patch('fl.server.server.parameters_to_ndarrays')
    @patch('fl.server.server.ndarrays_to_parameters')
    def test_aggregate_fit_with_krum(self, mock_ndarrays_to_parameters, mock_parameters_to_ndarrays):
        """Test aggregation with Krum defense."""
        # Setup mock parameters conversion
        mock_params1 = [np.ones((5, 5))]
        mock_params2 = [np.ones((5, 5)) * 1.1]
        mock_params3 = [np.ones((5, 5)) * 0.9]
        
        mock_parameters_to_ndarrays.side_effect = [mock_params1, mock_params2, mock_params3]
        mock_ndarrays_to_parameters.return_value = b"selected_params"
        
        # Three clients, enough for Krum with f=1
        results = [
            (self.client1, self.fit_res1),
            (self.client2, self.fit_res2),
            (self.client3, self.fit_res3)
        ]
        failures = []
        
        # Mock IPFS uploads
        self.mock_ipfs_connector.upload_model.side_effect = [
            {"Hash": "QmHash1"}, {"Hash": "QmHash2"}, {"Hash": "QmHash3"}, 
            {"Hash": "QmGlobalHash"}
        ]
        
        # Mock blockchain Krum defense
        self.mock_blockchain_connector.applyKrumDefense.return_value = 2  # Client 2 selected
        
        # Mock blockchain submit model update
        self.mock_blockchain_connector.submit_model_update.return_value = True
        
        # Call aggregate_fit
        params, metrics = self.strategy.aggregate_fit(1, results, failures)
        
        # Verify model uploads to IPFS
        self.assertEqual(self.mock_ipfs_connector.upload_model.call_count, 4)
        
        # Verify Krum defense call
        self.mock_blockchain_connector.applyKrumDefense.assert_called_once_with(1)
        
        # Verify global model update
        self.mock_blockchain_connector.updateGlobalModel.assert_called_once()
        
        # Verify returned parameters
        self.assertEqual(params, b"selected_params")
        self.assertEqual(metrics["aggregation"], "krum")
        self.assertEqual(metrics["selected_client"], 2)


if __name__ == '__main__':
    unittest.main()