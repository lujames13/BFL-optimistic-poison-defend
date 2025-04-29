"""
Unit tests for the defense effectiveness evaluation module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock, mock_open
import json
import numpy as np
import torch
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.evaluation.defense_effectiveness import DefenseEvaluator
from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector


class TestDefenseEvaluator(unittest.TestCase):
    """Test cases for DefenseEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for test output
        self.temp_dir = tempfile.mkdtemp()
        
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
        # Task info
        self.mock_blockchain_connector.getTaskInfo.return_value = {
            "taskId": 1,
            "status": 2,  # Completed
            "startTime": 1000000,
            "completedRounds": 3,
            "totalRounds": 5,
            "initialModelHash": "QmInitialHash",
            "currentModelHash": "QmFinalHash"
        }
        
        # Round info
        self.mock_round_info = {
            "roundId": 1,
            "status": 2,  # Completed
            "startTime": 1000000,
            "endTime": 1001000,
            "participantCount": 5,
            "completedUpdates": 5,
            "globalModelHash": "QmGlobalHash"
        }
        self.mock_blockchain_connector.get_round_info.side_effect = mock_get_round_updates
        
        # Models for different rounds and clients
        global_models = [self.model_class() for _ in range(3)]
        honest_models = [self.model_class() for _ in range(3 * 3)]  # 3 honest clients, 3 rounds
        attacker_models = [self.model_class() for _ in range(3 * 2)]  # 2 attackers, 3 rounds
        
        # Make attacker models significantly different
        for model in attacker_models:
            for param in model.parameters():
                with torch.no_grad():
                    param.mul_(10.0)
        
        # Function to get model based on client ID and round
        models_mapping = {}
        
        # Add globals
        for i in range(3):
            models_mapping[f"Global_{i+1}"] = [p.detach().numpy() for p in global_models[i].parameters()]
        
        # Add honest
        honest_idx = 0
        for r in range(3):
            for c in honest_ids:
                models_mapping[f"Honest_{c}_{r+1}"] = [p.detach().numpy() for p in honest_models[honest_idx].parameters()]
                honest_idx += 1
        
        # Add attackers
        attacker_idx = 0
        for r in range(3):
            for c in attacker_ids:
                models_mapping[f"Attacker_{c}_{r+1}"] = [p.detach().numpy() for p in attacker_models[attacker_idx].parameters()]
                attacker_idx += 1
        
        # Mock download model
        def mock_download_model(model_hash):
            if "Global" in model_hash:
                round_num = int(model_hash[-1])
                return models_mapping[f"Global_{round_num}"]
            elif any(str(id) in model_hash for id in honest_ids):
                for hid in honest_ids:
                    if str(hid) in model_hash:
                        for r in range(3):
                            if str(r+1) in model_hash:
                                return models_mapping[f"Honest_{hid}_{r+1}"]
            elif any(str(id) in model_hash for id in attacker_ids):
                for aid in attacker_ids:
                    if str(aid) in model_hash:
                        for r in range(3):
                            if str(r+1) in model_hash:
                                return models_mapping[f"Attacker_{aid}_{r+1}"]
            
            # Default case
            return models_mapping["Global_1"]
        
        self.mock_ipfs_connector.download_model.side_effect = mock_download_model
        
        # Mock model update hashes
        def mock_get_model_update_hash(client_id, round_id):
            if client_id in honest_ids:
                return f"QmUpdate_Honest_{client_id}_{round_id}"
            else:
                return f"QmUpdate_Attacker_{client_id}_{round_id}"
        
        self.mock_blockchain_connector.getModelUpdateHash.side_effect = mock_get_model_update_hash
        
        # Evaluate attack impact
        with patch('torch.device') as mock_device:
            # Mock device to use CPU
            mock_device.return_value = torch.device('cpu')
            
            result = self.evaluator.evaluate_attack_impact(
                task_id=task_id,
                attacker_ids=attacker_ids,
                honest_ids=honest_ids,
                model_class=self.model_class,
                test_data=self.test_data
            )
        
        # Verify blockchain interactions
        self.mock_blockchain_connector.getTaskInfo.assert_called_once_with(task_id)
        self.assertEqual(self.mock_blockchain_connector.get_round_info.call_count, 3)
        
        # Verify result structure
        self.assertIn("task_id", result)
        self.assertIn("total_rounds", result)
        self.assertIn("attacker_ids", result)
        self.assertIn("honest_ids", result)
        self.assertIn("selected_clients", result)
        self.assertIn("global_accuracies", result)
        self.assertIn("attacker_accuracies", result)
        self.assertIn("honest_accuracies", result)
        self.assertIn("attack_success_rate", result)
        self.assertIn("defense_effectiveness", result)
        
        # Verify defense effectiveness calculation
        # 1 attacker success out of 3 rounds = 1/3 attack success rate = 2/3 defense effectiveness
        self.assertAlmostEqual(result["attack_success_rate"], 1/3)
        self.assertAlmostEqual(result["defense_effectiveness"], 2/3)
    
    @patch('fl.evaluation.defense_effectiveness.plt')
    def test_plot_accuracies_over_rounds(self, mock_plt):
        """Test plotting accuracies over rounds."""
        # Sample results
        results = {
            "task_id": 1,
            "total_rounds": 3,
            "attacker_ids": [4, 5],
            "honest_ids": [1, 2, 3],
            "selected_clients": [1, 4, 2],  # Round 1: honest, Round 2: attacker, Round 3: honest
            "global_accuracies": [0.82, 0.70, 0.85],  # Drop in round 2 when attacker was selected
            "attacker_accuracies": [0.30, 0.30, 0.30],
            "honest_accuracies": [0.85, 0.85, 0.85],
            "attack_success_rate": 1/3,
            "defense_effectiveness": 2/3
        }
        
        # Generate plot with specific filename
        filename = "test_accuracy_over_rounds.png"
        filepath = self.evaluator.plot_accuracies_over_rounds(results, filename)
        
        # Verify mock plt interactions
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.plot.call_count, 3)  # Global, attacker, honest lines
        mock_plt.title.assert_called_once()
        mock_plt.xlabel.assert_called_once()
        mock_plt.ylabel.assert_called_once()
        mock_plt.legend.assert_called_once()
        mock_plt.savefig.assert_called_once()
        
        # Verify file path
        expected_path = os.path.join(self.temp_dir, filename)
        self.assertEqual(filepath, expected_path)


if __name__ == '__main__':
    unittest.main()
return_value = self.mock_round_info
        
        # System status
        self.mock_blockchain_connector.get_system_status.return_value = {
            "totalClients": 5,
            "totalRounds": 3,
            "currentRound": 3,
            "currentRoundStatus": 2  # Completed
        }
        
        # Client participation
        self.mock_blockchain_connector.didClientParticipate.return_value = True
        
        # Model update hash
        self.mock_blockchain_connector.getModelUpdateHash.return_value = "QmUpdateHash"
        
        # IPFS model download
        model_params = [
            np.ones((10, 20), dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones((20, 10), dtype=np.float32),
            np.ones(10, dtype=np.float32)
        ]
        self.mock_ipfs_connector.download_model.return_value = model_params
        
        # Create the evaluator
        self.config = {"results_dir": self.temp_dir}
        self.evaluator = DefenseEvaluator(
            blockchain_connector=self.mock_blockchain_connector,
            ipfs_connector=self.mock_ipfs_connector,
            config=self.config
        )
        
        # Define a simple test model class for evaluation
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
        self.x_test = torch.randn(50, 10)
        self.y_test = torch.randint(0, 10, (50,))
        self.test_data = (self.x_test, self.y_test)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.blockchain_connector_patcher.stop()
        self.ipfs_connector_patcher.stop()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.results_dir, self.temp_dir)
        self.assertEqual(self.evaluator.blockchain_connector, self.mock_blockchain_connector)
        self.assertEqual(self.evaluator.ipfs_connector, self.mock_ipfs_connector)
    
    def test_get_task_history(self):
        """Test getting task history."""
        # Mock round infos
        round_infos = [
            {"roundId": 1, "status": 2, "globalModelHash": "QmHash1"},
            {"roundId": 2, "status": 2, "globalModelHash": "QmHash2"},
            {"roundId": 3, "status": 2, "globalModelHash": "QmHash3"}
        ]
        
        self.mock_blockchain_connector.get_round_info.side_effect = round_infos
        
        # Get task history
        history = self.evaluator.get_task_history(1)
        
        # Verify blockchain interactions
        self.mock_blockchain_connector.getTaskInfo.assert_called_once_with(1)
        self.assertEqual(self.mock_blockchain_connector.get_round_info.call_count, 3)
        
        # Verify history structure
        self.assertIn("task", history)
        self.assertIn("rounds", history)
        self.assertEqual(len(history["rounds"]), 3)
        self.assertEqual(history["task"]["taskId"], 1)
    
    def test_download_model_for_evaluation(self):
        """Test downloading model for evaluation."""
        # Download model
        model = self.evaluator.download_model_for_evaluation("QmModelHash", self.model_class)
        
        # Verify IPFS interaction
        self.mock_ipfs_connector.download_model.assert_called_once_with("QmModelHash")
        
        # Verify model
        self.assertIsInstance(model, self.model_class)
        
        # Verify parameters were loaded
        params = list(model.parameters())
        self.assertEqual(len(params), 4)
        
        # Check first layer weights
        np.testing.assert_array_equal(
            params[0].detach().numpy(),
            np.ones((20, 10), dtype=np.float32)  # PyTorch stores transposed
        )
    
    def test_compare_models(self):
        """Test comparing multiple models on test data."""
        # Create models
        model1 = self.model_class()
        model2 = self.model_class()
        
        # Modify model2 to be different
        for param in model2.parameters():
            with torch.no_grad():
                param.mul_(2.0)
        
        # Compare models
        with patch('torch.device') as mock_device:
            # Mock device to use CPU
            mock_device.return_value = torch.device('cpu')
            
            results = self.evaluator.compare_models(
                [("Model1", model1), ("Model2", model2)],
                self.test_data
            )
        
        # Verify results
        self.assertIn("Model1", results)
        self.assertIn("Model2", results)
        self.assertIn("loss", results["Model1"])
        self.assertIn("accuracy", results["Model1"])
        self.assertIn("loss", results["Model2"])
        self.assertIn("accuracy", results["Model2"])
    
    @patch('fl.evaluation.defense_effectiveness.plt')
    def test_evaluate_krum_defense(self, mock_plt):
        """Test evaluating Krum defense."""
        # Setup mock updates
        client_ids = [1, 2, 3, 4]
        
        # Mock client participation
        self.mock_blockchain_connector.didClientParticipate.side_effect = lambda cid, rid: cid in client_ids
        
        # Mock model update hashes
        self.mock_blockchain_connector.getModelUpdateHash.side_effect = lambda cid, rid: f"QmUpdate{cid}"
        
        # Evaluate Krum defense
        result = self.evaluator.evaluate_krum_defense(
            task_id=1, 
            round_id=1, 
            model_class=self.model_class, 
            test_data=self.test_data
        )
        
        # Verify blockchain interactions
        self.mock_blockchain_connector.get_round_info.assert_called_with(1)
        self.assertEqual(self.mock_blockchain_connector.didClientParticipate.call_count, 5)
        
        # Verify IPFS interactions
        self.assertEqual(self.mock_ipfs_connector.download_model.call_count, 5)  # 4 clients + global
        
        # Verify result structure
        self.assertIn("task_id", result)
        self.assertIn("round_id", result)
        self.assertIn("global_model_hash", result)
        self.assertIn("client_models", result)
        self.assertEqual(result["task_id"], 1)
        self.assertEqual(result["round_id"], 1)
        self.assertEqual(result["global_model_hash"], "QmGlobalHash")
    
    def test_generate_report(self):
        """Test generating a report from evaluation results."""
        # Sample results
        results = {
            "task_id": 1,
            "round_id": 1,
            "global_model_hash": "QmGlobalHash",
            "selected_client_id": 2,
            "client_models": {
                "Client 1": {"loss": 0.5, "accuracy": 0.8},
                "Client 2": {"loss": 0.4, "accuracy": 0.85},
                "Global": {"loss": 0.45, "accuracy": 0.82}
            }
        }
        
        # Generate report with specific filename
        filename = "test_report.json"
        filepath = self.evaluator.generate_report(results, filename)
        
        # Verify file was created
        expected_path = os.path.join(self.temp_dir, filename)
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(filepath))
        
        # Verify file contents
        with open(filepath, 'r') as f:
            saved_results = json.load(f)
        
        self.assertEqual(saved_results, results)
    
    @patch('fl.evaluation.defense_effectiveness.plt')
    def test_plot_accuracy_comparison(self, mock_plt):
        """Test plotting accuracy comparison."""
        # Sample results
        results = {
            "task_id": 1,
            "round_id": 1,
            "global_model_hash": "QmGlobalHash",
            "selected_client_id": 2,
            "client_models": {
                "Client 1": {"loss": 0.5, "accuracy": 0.8},
                "Client 2": {"loss": 0.4, "accuracy": 0.85},
                "Global (Krum selected)": {"loss": 0.45, "accuracy": 0.82}
            }
        }
        
        # Generate plot with specific filename
        filename = "test_plot.png"
        filepath = self.evaluator.plot_accuracy_comparison(results, filename)
        
        # Verify mock plt interactions
        mock_plt.figure.assert_called_once()
        mock_plt.bar.assert_called_once()
        mock_plt.savefig.assert_called_once()
        
        # Verify file path
        expected_path = os.path.join(self.temp_dir, filename)
        self.assertEqual(filepath, expected_path)
    
    def test_compare_with_without_defense(self):
        """Test comparing performance with and without defense."""
        # Setup mock responses for two tasks
        self.mock_blockchain_connector.get_round_info.side_effect = [
            {"globalModelHash": "QmWithDefenseHash"},  # With defense
            {"globalModelHash": "QmWithoutDefenseHash"}  # Without defense
        ]
        
        # Different models for with/without defense
        model_with_defense = self.model_class()
        model_without_defense = self.model_class()
        
        # Make without-defense model worse
        for param in model_without_defense.parameters():
            with torch.no_grad():
                param.mul_(10.0)  # Make predictions worse
        
        # Mock download model
        self.mock_ipfs_connector.download_model.side_effect = [
            [p.detach().numpy() for p in model_with_defense.parameters()],
            [p.detach().numpy() for p in model_without_defense.parameters()]
        ]
        
        # Compare with/without defense
        with patch('torch.device') as mock_device:
            # Mock device to use CPU
            mock_device.return_value = torch.device('cpu')
            
            result = self.evaluator.compare_with_without_defense(
                task_with_defense=1,
                task_without_defense=2,
                round_id=1,
                model_class=self.model_class,
                test_data=self.test_data
            )
        
        # Verify blockchain interactions
        self.assertEqual(self.mock_blockchain_connector.get_round_info.call_count, 2)
        
        # Verify IPFS interactions
        self.assertEqual(self.mock_ipfs_connector.download_model.call_count, 2)
        
        # Verify result structure
        self.assertIn("task_with_defense", result)
        self.assertIn("task_without_defense", result)
        self.assertIn("with_defense_hash", result)
        self.assertIn("without_defense_hash", result)
        self.assertIn("comparison", result)
        self.assertIn("improvement_percentage", result)
        
        # Verify correct task IDs
        self.assertEqual(result["task_with_defense"], 1)
        self.assertEqual(result["task_without_defense"], 2)
    
    @patch('fl.evaluation.defense_effectiveness.plt')
    def test_evaluate_attack_impact(self, mock_plt):
        """Test evaluating attack impact."""
        # Setup
        task_id = 1
        attacker_ids = [4, 5]
        honest_ids = [1, 2, 3]
        
        # Mock task info
        self.mock_blockchain_connector.getTaskInfo.return_value = {
            "taskId": 1,
            "completedRounds": 3
        }
        
        # Mock round infos
        round_infos = [
            {"roundId": 1, "globalModelHash": "QmGlobalHash1"},
            {"roundId": 2, "globalModelHash": "QmGlobalHash2"},
            {"roundId": 3, "globalModelHash": "QmGlobalHash3"}
        ]
        self.mock_blockchain_connector.get_round_info.side_effect = round_infos
        
        # Mock client participation - all clients participate
        self.mock_blockchain_connector.didClientParticipate.return_value = True
        
        # Mock check if update was accepted
        round_updates = {
            # Round 1 - honest client selected
            "1": {"1": {"accepted": True}, "2": {"accepted": False}, "3": {"accepted": False}, 
                 "4": {"accepted": False}, "5": {"accepted": False}},
            # Round 2 - attacker managed to get selected
            "2": {"1": {"accepted": False}, "2": {"accepted": False}, "3": {"accepted": False}, 
                 "4": {"accepted": True}, "5": {"accepted": False}},
            # Round 3 - honest client selected
            "3": {"1": {"accepted": False}, "2": {"accepted": True}, "3": {"accepted": False}, 
                 "4": {"accepted": False}, "5": {"accepted": False}}
        }
        
        def mock_get_round_updates(round_id):
            info = round_infos[round_id - 1].copy()
            info["updates"] = round_updates[str(round_id)]
            return info
        
        self.mock_blockchain_connector.get_round_info.