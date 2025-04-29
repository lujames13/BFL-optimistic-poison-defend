"""
Unit tests for the blockchain connector module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.blockchain_connector import BlockchainConnector


class TestBlockchainConnector(unittest.TestCase):
    """Test cases for BlockchainConnector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Web3 and contract interactions
        self.web3_patcher = patch('fl.blockchain_connector.Web3')
        self.mock_web3 = self.web3_patcher.start()
        
        # Setup mock contract
        self.mock_contract = MagicMock()
        self.mock_web3.eth.contract.return_value = self.mock_contract
        
        # Setup mock connection
        self.mock_web3.is_connected.return_value = True
        
        # Setup mock accounts
        self.mock_web3.eth.accounts = [
            '0x1234567890123456789012345678901234567890'
        ]
        
        # Setup test parameters
        self.contract_address = '0x1234567890123456789012345678901234567890'
        self.client_id = 1
        self.node_url = 'http://localhost:8545'
        
        # Create connector instance
        self.connector = BlockchainConnector(
            contract_address=self.contract_address,
            client_id=self.client_id,
            node_url=self.node_url
        )
        
        # Setup mock system status for testing
        self.mock_system_status = (5, 2, 1, 1)  # totalClients, totalRounds, currentRound, status
        self.mock_contract.functions.getSystemStatus().call.return_value = self.mock_system_status
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.web3_patcher.stop()
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertEqual(self.connector.client_id, self.client_id)
        self.assertEqual(self.connector.contract_address, self.contract_address)
        self.mock_web3.HTTPProvider.assert_called_once_with(self.node_url)
        self.mock_web3.eth.contract.assert_called_once()
    
    def test_hash_model_parameters(self):
        """Test hashing model parameters."""
        # Create test parameters
        params = [np.ones((2, 2)), np.zeros((3, 3))]
        
        # Test hash calculation
        hash_value = self.connector.hash_model_parameters(params)
        
        # Assert hash is a non-empty string
        self.assertIsInstance(hash_value, str)
        self.assertTrue(len(hash_value) > 0)
        
        # Test same parameters produce same hash
        hash_value2 = self.connector.hash_model_parameters(params)
        self.assertEqual(hash_value, hash_value2)
        
        # Test different parameters produce different hash
        params2 = [np.ones((2, 2)), np.ones((3, 3))]
        hash_value3 = self.connector.hash_model_parameters(params2)
        self.assertNotEqual(hash_value, hash_value3)
    
    def test_get_system_status(self):
        """Test getting system status."""
        # Call method
        status = self.connector.get_system_status()
        
        # Verify the contract call
        self.mock_contract.functions.getSystemStatus.assert_called_once()
        
        # Verify output format
        self.assertIsInstance(status, dict)
        self.assertEqual(status['totalClients'], 5)
        self.assertEqual(status['totalRounds'], 2)
        self.assertEqual(status['currentRound'], 1)
        self.assertEqual(status['currentRoundStatus'], 1)
    
    def test_createTask(self):
        """Test creating a new task."""
        # Setup mock for task creation
        self.mock_contract.functions.createTask().build_transaction.return_value = {
            'from': '0x1234',
            'gasPrice': 2000000000,
            'nonce': 1
        }
        
        # Mock transaction signing and sending
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = '0xabcd'
        self.mock_web3.eth.send_transaction.return_value = mock_tx_hash
        
        # Mock transaction receipt with TaskCreated event
        mock_receipt = {'transactionHash': mock_tx_hash}
        self.mock_web3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        
        # Mock event processing to simulate task ID from event
        mock_event = [{'args': {'taskId': 42}}]
        self.mock_contract.events.TaskCreated().process_receipt.return_value = mock_event
        
        # Test create task
        task_id = self.connector.createTask("QmInitialModelHash", 5)
        
        # Verify contract interaction
        self.mock_contract.functions.createTask.assert_called_with("QmInitialModelHash", 5)
        
        # Verify transaction processing
        self.mock_web3.eth.send_transaction.assert_called_once()
        self.mock_web3.eth.wait_for_transaction_receipt.assert_called_once_with(mock_tx_hash)
        
        # Verify result
        self.assertEqual(task_id, 42)
        self.assertEqual(self.connector.currentTaskId, 42)
    
    def test_register_client(self):
        """Test client registration."""
        # Setup mock for client registration
        self.mock_contract.functions.registerClient().build_transaction.return_value = {
            'from': '0x1234',
            'gasPrice': 2000000000,
            'nonce': 1
        }
        
        # Mock transaction signing and sending
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = '0xabcd'
        self.mock_web3.eth.send_transaction.return_value = mock_tx_hash
        
        # Mock transaction receipt
        mock_receipt = {'transactionHash': mock_tx_hash}
        self.mock_web3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        
        # Mock event processing to simulate client ID from event
        mock_event = [{'args': {'clientId': 5}}]
        self.mock_contract.events.ClientRegistered().process_receipt.return_value = mock_event
        
        # Test register client
        result = self.connector.register_client()
        
        # Verify contract interaction
        self.mock_contract.functions.registerClient.assert_called_once()
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(self.connector.client_id, 5)
    
    def test_submit_model_update(self):
        """Test submitting model update."""
        # Setup test parameters
        round_id = 1
        client_id = 1
        parameters = [np.ones((2, 2)), np.zeros((3, 3))]
        
        # Setup mock for submitModelUpdate
        self.mock_contract.functions.submitModelUpdate().build_transaction.return_value = {
            'from': '0x1234',
            'gasPrice': 2000000000,
            'nonce': 1
        }
        
        # Mock transaction signing and sending
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = '0xabcd'
        self.mock_web3.eth.send_transaction.return_value = mock_tx_hash
        
        # Mock transaction receipt
        mock_receipt = {'transactionHash': mock_tx_hash}
        self.mock_web3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        
        # Test submit model update
        model_hash = self.connector.submit_model_update(client_id, round_id, parameters)
        
        # Verify hash was computed
        self.assertIsInstance(model_hash, str)
        self.assertTrue(len(model_hash) > 0)
        
        # Verify contract interaction
        self.mock_contract.functions.submitModelUpdate.assert_called_once()
        
        # Verify transaction was processed
        self.mock_web3.eth.send_transaction.assert_called_once()
    
    def test_applyKrumDefense(self):
        """Test applying Krum defense."""
        # Setup test parameters
        round_id = 1
        
        # Setup mock for applyKrumDefense
        self.mock_contract.functions.applyKrumDefense().build_transaction.return_value = {
            'from': '0x1234',
            'gasPrice': 2000000000,
            'nonce': 1
        }
        
        # Mock transaction signing and sending
        mock_tx_hash = MagicMock()
        mock_tx_hash.hex.return_value = '0xabcd'
        self.mock_web3.eth.send_transaction.return_value = mock_tx_hash
        
        # Mock transaction receipt
        mock_receipt = {'transactionHash': mock_tx_hash}
        self.mock_web3.eth.wait_for_transaction_receipt.return_value = mock_receipt
        
        # Mock event processing to simulate selected client ID
        mock_event = [{'args': {'clientId': 3}}]
        self.mock_contract.events.ModelUpdateAccepted().process_receipt.return_value = mock_event
        
        # Test apply Krum defense
        selected_client_id = self.connector.applyKrumDefense(round_id)
        
        # Verify contract interaction
        self.mock_contract.functions.applyKrumDefense.assert_called_with(round_id)
        
        # Verify result
        self.assertEqual(selected_client_id, 3)
    
    def test_get_registered_clients(self):
        """Test getting registered clients."""
        # Setup mock for system status
        self.mock_contract.functions.getSystemStatus().call.return_value = (3, 2, 1, 1)
        
        # Setup mock for client info
        client1_info = ('0x1111', 1, 10, 12345, True)
        client2_info = ('0x2222', 1, 20, 12346, False)
        client3_info = ('0x3333', 1, 30, 12347, True)
        
        self.mock_contract.functions.getClientInfo().call.side_effect = [
            client1_info, client2_info, client3_info
        ]
        
        # Test get registered clients
        clients = self.connector.get_registered_clients()
        
        # Verify contract interactions
        self.mock_contract.functions.getSystemStatus.assert_called_once()
        self.assertEqual(self.mock_contract.functions.getClientInfo().call.call_count, 3)
        
        # Verify result
        self.assertEqual(len(clients), 3)
        
        # Verify first client info
        self.assertEqual(clients[0]['clientId'], 1)
        self.assertEqual(clients[0]['clientAddress'], '0x1111')
        self.assertEqual(clients[0]['contributionScore'], 10)
        self.assertEqual(clients[0]['selectedForRound'], True)
    
    def test_error_handling(self):
        """Test error handling in connector."""
        # Mock a failed contract call
        self.mock_contract.functions.getSystemStatus().call.side_effect = Exception("Test error")
        
        # Test that error is propagated
        with self.assertRaises(Exception) as context:
            self.connector.get_system_status()
        
        self.assertIn("Test error", str(context.exception))


if __name__ == '__main__':
    unittest.main()