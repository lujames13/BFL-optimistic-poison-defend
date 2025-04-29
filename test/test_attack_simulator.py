"""
Unit tests for the attack simulator module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.attack.attack_simulator import (
    AttackSimulator, LabelFlippingAttack, ModelReplacementAttack, 
    ByzantineClientAttack, TargetedModelPoisoningAttack
)
from fl.client.client import FlowerClient
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestAttackSimulator(unittest.TestCase):
    """Test cases for attack simulator classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock client
        self.mock_client = MagicMock(spec=FlowerClient)
        
        # Mock blockchain connector
        self.mock_blockchain_connector = MagicMock(spec=BlockchainConnector)
        self.mock_client.blockchain_connector = self.mock_blockchain_connector
        
        # Mock IPFS connector
        self.mock_ipfs_connector = MagicMock(spec=ModelIPFSConnector)
        self.mock_client.ipfs_connector = self.mock_ipfs_connector
        
        # Set client ID
        self.mock_client.client_id = 42
        
        # Create test parameters
        self.test_params = [
            np.ones((10, 10), dtype=np.float32),
            np.ones(10, dtype=np.float32),
            np.ones((10, 5), dtype=np.float32),
            np.ones(5, dtype=np.float32)
        ]
        
        # Mock client methods
        self.mock_client.train_locally.return_value = self.test_params
        self.mock_client.download_model.return_value = self.test_params
        self.mock_client.submit_update.return_value = "QmUpdateHash"
        self.mock_client.evaluate_model.return_value = {"loss": 0.5, "accuracy": 0.8}
        
        # Mock blockchain responses
        self.mock_blockchain_connector.get_client_info.return_value = {
            "selectedForRound": True
        }
        
        self.mock_blockchain_connector.get_round_info.return_value = {
            "globalModelHash": "QmGlobalHash"
        }
        
        # Create base attack config
        self.attack_config = {
            "type": "generic",
            "intensity": 0.5
        }
        
        # Create base attack simulator
        self.simulator = AttackSimulator(self.mock_client, self.attack_config)
    
    def test_base_attack_initialization(self):
        """Test base attack simulator initialization."""
        self.assertEqual(self.simulator.client, self.mock_client)
        self.assertEqual(self.simulator.attack_config, self.attack_config)
        self.assertEqual(self.simulator.attack_type, "generic")
        self.assertEqual(self.simulator.intensity, 0.5)
    
    def test_base_attack_no_modification(self):
        """Test that base attack doesn't modify parameters."""
        # Base implementation should return parameters unchanged
        result = self.simulator.attack(self.test_params)
        
        # Verify result
        for i in range(len(result)):
            np.testing.assert_array_equal(result[i], self.test_params[i])
    
    def test_execute_attack_not_selected(self):
        """Test execute_attack when client is not selected."""
        # Mock client not selected
        self.mock_blockchain_connector.get_client_info.return_value = {
            "selectedForRound": False
        }
        
        # Execute attack
        result = self.simulator.execute_attack(1)
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("not selected", result["message"])
        
        # Verify no model download attempted
        self.mock_client.download_model.assert_not_called()
    
    def test_execute_attack_selected(self):
        """Test execute_attack when client is selected."""
        # Execute attack
        result = self.simulator.execute_attack(1)
        
        # Verify client info was checked
        self.mock_blockchain_connector.get_client_info.assert_called_with(42)
        
        # Verify round info was fetched
        self.mock_blockchain_connector.get_round_info.assert_called_with(1)
        
        # Verify model was downloaded
        self.mock_client.download_model.assert_called_with("QmGlobalHash")
        
        # Verify attack was performed (but no modification in base class)
        
        # Verify model update was submitted
        self.mock_client.submit_update.assert_called_with(1, self.test_params)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["updateHash"], "QmUpdateHash")
        self.assertIn("honestMetrics", result)
        self.assertIn("attackedMetrics", result)
    
    def test_label_flipping_attack(self):
        """Test label flipping attack."""
        # Create label flipping attack
        attack_config = {
            "type": "label_flipping",
            "intensity": 0.8
        }
        
        simulator = LabelFlippingAttack(self.mock_client, attack_config)
        
        # Attack parameters
        result = simulator.attack(self.test_params)
        
        # Verify parameters were modified
        for i in range(len(result)):
            if i % 2 == 0:  # Weights, not biases
                # The attack should flip the sign of weights
                expected = self.test_params[i] * -0.8
                np.testing.assert_array_almost_equal(result[i], expected)
    
    def test_model_replacement_attack(self):
        """Test model replacement attack."""
        # Create model replacement attack
        attack_config = {
            "type": "model_replacement",
            "intensity": 0.5
        }
        
        simulator = ModelReplacementAttack(self.mock_client, attack_config)
        
        # Attack parameters
        result = simulator.attack(self.test_params)
        
        # Verify parameters were modified
        for i in range(len(result)):
            # The attack should scale parameters by a large factor
            expected = self.test_params[i] * 50.0  # 100.0 * 0.5
            np.testing.assert_array_almost_equal(result[i], expected)
    
    def test_byzantine_client_attack(self):
        """Test Byzantine client attack."""
        # Create Byzantine attack
        attack_config = {
            "type": "byzantine",
            "intensity": 0.5
        }
        
        simulator = ByzantineClientAttack(self.mock_client, attack_config)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Attack parameters
        result = simulator.attack(self.test_params)
        
        # Verify parameters were modified (different from original)
        modified = False
        for i in range(len(result)):
            if not np.array_equal(result[i], self.test_params[i]):
                modified = True
                break
        
        self.assertTrue(modified, "Byzantine attack should modify parameters")
    
    def test_targeted_model_poisoning_attack(self):
        """Test targeted model poisoning attack."""
        # Create targeted attack
        attack_config = {
            "type": "targeted",
            "intensity": 0.6,
            "target_class": 2
        }
        
        simulator = TargetedModelPoisoningAttack(self.mock_client, attack_config)
        
        # Attack parameters
        result = simulator.attack(self.test_params)
        
        # Verify parameters were modified
        output_weights = result[2]  # Assuming 3rd parameter is output layer weights
        output_bias = result[3]  # Assuming 4th parameter is output layer bias
        
        # For the target class, weights should be reduced
        if len(output_weights.shape) > 1:  # For 2D weights
            # Check that only the target class weights were modified
            for i in range(output_weights.shape[1]):
                if i == 2:  # Target class
                    expected = self.test_params[2][:, i] * (1 - 0.6)
                    np.testing.assert_array_almost_equal(output_weights[:, i], expected)
                else:
                    np.testing.assert_array_equal(output_weights[:, i], self.test_params[2][:, i])
        
        # For the target class bias, it should be reduced
        if len(output_bias.shape) > 0:  # For 1D bias
            expected = self.test_params[3][2] * (1 - 0.6)
            self.assertAlmostEqual(output_bias[2], expected)


class TestAttackStrategies(unittest.TestCase):
    """Test different attack strategies more thoroughly."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test parameters mimicking a real model
        # First layer: 784 (input) -> 128 (hidden)
        # Second layer: 128 (hidden) -> 10 (output, classes)
        self.test_params = [
            np.random.randn(784, 128).astype(np.float32),  # Layer 1 weights
            np.random.randn(128).astype(np.float32),       # Layer 1 bias
            np.random.randn(128, 10).astype(np.float32),   # Layer 2 weights
            np.random.randn(10).astype(np.float32)         # Layer 2 bias
        ]
        
        # Mock client
        self.mock_client = MagicMock(spec=FlowerClient)
    
    def test_label_flipping_effect(self):
        """Test the effect of label flipping attack on model predictions."""
        # Create attack with different intensities
        intensities = [0.0, 0.5, 1.0]
        
        for intensity in intensities:
            attack_config = {
                "type": "label_flipping",
                "intensity": intensity
            }
            
            simulator = LabelFlippingAttack(self.mock_client, attack_config)
            
            # Attack parameters
            result = simulator.attack(self.test_params.copy())
            
            # Check output layer weights - should be flipped in sign by intensity
            output_weights = result[2]
            output_bias = result[3]
            
            # Calculate the average change in weights
            weight_change = np.mean(np.abs(output_weights - self.test_params[2]))
            
            # The change should be proportional to intensity
            if intensity == 0.0:
                self.assertAlmostEqual(weight_change, 0.0, places=5)
            else:
                self.assertGreater(weight_change, 0.0)
                # Higher intensity should cause bigger change
                self.assertAlmostEqual(
                    weight_change, 
                    np.mean(np.abs(self.test_params[2] * intensity)),
                    places=5
                )
    
    def test_model_replacement_effect(self):
        """Test the effect of model replacement attack."""
        # Create attack with different intensities
        intensities = [0.0, 0.25, 1.0]
        
        for intensity in intensities:
            attack_config = {
                "type": "model_replacement",
                "intensity": intensity
            }
            
            simulator = ModelReplacementAttack(self.mock_client, attack_config)
            
            # Attack parameters
            result = simulator.attack(self.test_params.copy())
            
            # Calculate how much the parameters have been scaled
            scaling_factor = np.mean(result[0] / self.test_params[0])
            
            # The scaling factor should match the expected value
            expected_scale = 100.0 * intensity
            self.assertAlmostEqual(scaling_factor, expected_scale, places=5)
    
    def test_byzantine_randomness(self):
        """Test that Byzantine attacks produce different results each time."""
        attack_config = {
            "type": "byzantine",
            "intensity": 0.8
        }
        
        simulator = ByzantineClientAttack(self.mock_client, attack_config)
        
        # Attack parameters multiple times
        result1 = simulator.attack(self.test_params.copy())
        result2 = simulator.attack(self.test_params.copy())
        
        # Results should be different, showing randomness
        different = False
        for i in range(len(result1)):
            if not np.array_equal(result1[i], result2[i]):
                different = True
                break
        
        self.assertTrue(different, "Byzantine attack should produce different results each time")
    
    def test_targeted_attack_specificity(self):
        """Test that targeted attack only affects the target class."""
        attack_config = {
            "type": "targeted",
            "intensity": 0.7,
            "target_class": 5  # Target class 5
        }
        
        simulator = TargetedModelPoisoningAttack(self.mock_client, attack_config)
        
        # Attack parameters
        result = simulator.attack(self.test_params.copy())
        
        # Check that only the target class was affected
        output_weights = result[2]
        output_bias = result[3]
        
        # For each class output
        for i in range(10):
            if i == 5:  # Target class
                # Should be reduced by intensity
                expected_weights = self.test_params[2][:, i] * (1 - 0.7)
                np.testing.assert_array_almost_equal(output_weights[:, i], expected_weights)
                expected_bias = self.test_params[3][i] * (1 - 0.7)
                self.assertAlmostEqual(output_bias[i], expected_bias)
            else:
                # Other classes should be unchanged
                np.testing.assert_array_equal(output_weights[:, i], self.test_params[2][:, i])
                self.assertEqual(output_bias[i], self.test_params[3][i])


if __name__ == '__main__':
    unittest.main()