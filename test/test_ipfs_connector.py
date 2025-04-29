"""
Unit tests for the IPFS connector module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import numpy as np
import torch
import os
import sys
import tempfile
from pathlib import Path
import io

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from fl.ipfs_connector import ModelIPFSConnector


class TestModelIPFSConnector(unittest.TestCase):
    """Test cases for ModelIPFSConnector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch requests
        self.requests_patcher = patch('fl.ipfs_connector.requests')
        self.mock_requests = self.requests_patcher.start()
        
        # Setup mock responses
        self.mock_id_response = MagicMock()
        self.mock_id_response.status_code = 200
        self.mock_id_response.json.return_value = {
            "ID": "QmTest",
            "Addresses": ["address1", "address2"]
        }
        
        self.mock_add_response = MagicMock()
        self.mock_add_response.status_code = 200
        self.mock_add_response.json.return_value = {
            "Hash": "QmUploadedModel",
            "Name": "test.pt",
            "Size": "1024"
        }
        
        self.mock_cat_response = MagicMock()
        self.mock_cat_response.status_code = 200
        
        self.mock_pin_response = MagicMock()
        self.mock_pin_response.status_code = 200
        self.mock_pin_response.json.return_value = {
            "Pins": ["QmUploadedModel"]
        }
        
        # Setup mock request sequences
        self.mock_requests.post.side_effect = [
            self.mock_id_response,  # Initial connection check
        ]
        
        # Create connector instance with mock API
        self.api_url = 'http://localhost:5001/api/v0'
        self.connector = ModelIPFSConnector(api_url=self.api_url)
        
        # Reset requests mock after connector initialization
        self.mock_requests.post.side_effect = None
        self.mock_requests.post.reset_mock()
        
        # Create test model parameters
        self.test_params = [
            np.random.randn(5, 10).astype(np.float32),
            np.random.randn(10, 2).astype(np.float32)
        ]
        
        # Create test PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        self.test_model = SimpleModel()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.requests_patcher.stop()
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertEqual(self.connector.api_url, self.api_url)
        self.assertTrue(self.connector.connected)
        self.mock_requests.post.assert_called_once_with(f"{self.api_url}/id", timeout=30)
    
    def test_upload_model_numpy_arrays(self):
        """Test uploading model as NumPy arrays."""
        # Setup mock responses for this test
        self.mock_requests.post.return_value = self.mock_add_response
        
        # Test model upload
        result = self.connector.upload_model(self.test_params, model_id="test_model")
        
        # Verify request was made correctly
        self.mock_requests.post.assert_called_with(
            f"{self.api_url}/add", 
            files=unittest.mock.ANY,
            timeout=unittest.mock.ANY
        )
        
        # Verify result
        self.assertEqual(result["Hash"], "QmUploadedModel")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_id"], "test_model")
    
    def test_upload_model_pytorch(self):
        """Test uploading PyTorch model."""
        # Setup mock responses for this test
        self.mock_requests.post.return_value = self.mock_add_response
        
        # Test model upload
        result = self.connector.upload_model(
            self.test_model, 
            model_id="pytorch_model",
            metadata={"framework": "pytorch"}
        )
        
        # Verify request was made correctly
        self.mock_requests.post.assert_called_with(
            f"{self.api_url}/add", 
            files=unittest.mock.ANY,
            timeout=unittest.mock.ANY
        )
        
        # Verify result
        self.assertEqual(result["Hash"], "QmUploadedModel")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_id"], "pytorch_model")
        self.assertEqual(result["metadata"]["framework"], "pytorch")
    
    def test_download_model(self):
        """Test downloading model."""
        # Create a simple state dict to serialize
        state_dict = {
            "__metadata__": {"model_id": "test_model"},
            "__weights__": [p.tolist() for p in self.test_params],
            "__shapes__": [p.shape for p in self.test_params],
            "__dtypes__": [str(p.dtype) for p in self.test_params]
        }
        
        # Convert to bytes for mock response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            json.dump(state_dict, temp_file)
            temp_path = temp_file.name
        
        with open(temp_path, 'rb') as f:
            mock_content = f.read()
        
        # Set mock response
        self.mock_cat_response.content = mock_content
        self.mock_requests.post.return_value = self.mock_cat_response
        
        # Test model download
        try:
            params = self.connector.download_model("QmModelHash", as_numpy=True)
            
            # Verify request was made correctly
            self.mock_requests.post.assert_called_with(
                f"{self.api_url}/cat?arg=QmModelHash", 
                timeout=unittest.mock.ANY
            )
            
            # Verify result type
            self.assertIsInstance(params, list)
            self.assertTrue(all(isinstance(p, np.ndarray) for p in params))
            
            # Verify shapes match
            self.assertEqual(len(params), len(self.test_params))
            for i in range(len(params)):
                self.assertEqual(params[i].shape, self.test_params[i].shape)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_pin_model(self):
        """Test pinning model."""
        # Setup mock response
        self.mock_requests.post.return_value = self.mock_pin_response
        
        # Test pin model
        result = self.connector.pin_model("QmTestModel")
        
        # Verify request was made correctly
        self.mock_requests.post.assert_called_once_with(
            f"{self.api_url}/pin/add?arg=QmTestModel",
            timeout=unittest.mock.ANY
        )
        
        # Verify result
        self.assertIn("Pins", result)
        self.assertIn("QmUploadedModel", result["Pins"])
    
    def test_unpin_model(self):
        """Test unpinning model."""
        # Setup mock response
        self.mock_requests.post.return_value = self.mock_pin_response
        
        # Test unpin model
        result = self.connector.unpin_model("QmTestModel")
        
        # Verify request was made correctly
        self.mock_requests.post.assert_called_once_with(
            f"{self.api_url}/pin/rm?arg=QmTestModel",
            timeout=unittest.mock.ANY
        )
    
    def test_batch_upload_models(self):
        """Test batch upload of models."""
        # Setup mock response for multiple uploads
        self.mock_requests.post.return_value = self.mock_add_response
        
        # Prepare multiple models
        models = [
            {"weights": self.test_params, "model_id": "model1", "metadata": {"type": "test"}},
            {"weights": self.test_params, "model_id": "model2", "metadata": {"type": "test"}}
        ]
        
        # Test batch upload
        result = self.connector.batch_upload_models(models)
        
        # Verify calls made
        self.assertEqual(self.mock_requests.post.call_count, 2)
        
        # Verify result
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(len(result["results"]), 2)
    
    def test_calculate_model_diff(self):
        """Test calculating model difference."""
        # Create two similar but different models
        model1 = [np.ones((3, 3)), np.zeros((2, 2))]
        model2 = [np.ones((3, 3)) * 2, np.ones((2, 2))]
        
        # Calculate difference
        diff, distance = self.connector.calculate_model_diff(model1, model2)
        
        # Verify difference calculation
        self.assertIsInstance(diff, list)
        self.assertEqual(len(diff), 2)
        self.assertEqual(diff[0].shape, (3, 3))
        self.assertEqual(diff[1].shape, (2, 2))
        
        # Verify first parameter difference is -1 everywhere
        np.testing.assert_array_equal(diff[0], np.ones((3, 3)) * -1)
        
        # Verify second parameter difference is -1 everywhere
        np.testing.assert_array_equal(diff[1], np.ones((2, 2)) * -1)
        
        # Verify distance calculation (should be sqrt(9 + 4) = sqrt(13))
        expected_distance = np.sqrt(9 + 4)  # 3x3 diff of 1 plus 2x2 diff of 1
        self.assertAlmostEqual(distance, expected_distance)
    
    def test_federated_average(self):
        """Test federated averaging."""
        # Create multiple models
        model1 = [np.ones((2, 2)), np.zeros((3, 3))]
        model2 = [np.zeros((2, 2)), np.ones((3, 3))]
        model3 = [np.ones((2, 2)) * 2, np.ones((3, 3)) * 2]
        
        models = [model1, model2, model3]
        
        # Test equal weights
        avg_model = self.connector.federated_average(models)
        
        # Verify result dimensions
        self.assertEqual(len(avg_model), 2)
        self.assertEqual(avg_model[0].shape, (2, 2))
        self.assertEqual(avg_model[1].shape, (3, 3))
        
        # Verify average calculation (each model contributes 1/3)
        expected_avg1 = np.ones((2, 2)) * (1 + 0 + 2) / 3  # = 1
        expected_avg2 = np.ones((3, 3)) * (0 + 1 + 2) / 3  # = 1
        
        np.testing.assert_array_almost_equal(avg_model[0], expected_avg1)
        np.testing.assert_array_almost_equal(avg_model[1], expected_avg2)
        
        # Test with explicit weights
        weights = [0.5, 0.25, 0.25]
        avg_model_weighted = self.connector.federated_average(models, weights)
        
        # Verify weighted average
        expected_avg1_weighted = np.ones((2, 2)) * (1*0.5 + 0*0.25 + 2*0.25)  # = 1
        expected_avg2_weighted = np.ones((3, 3)) * (0*0.5 + 1*0.25 + 2*0.25)  # = 0.75
        
        np.testing.assert_array_almost_equal(avg_model_weighted[0], expected_avg1_weighted)
        np.testing.assert_array_almost_equal(avg_model_weighted[1], expected_avg2_weighted)
    
    def test_error_handling(self):
        """Test error handling."""
        # Mock a failed request
        error_response = MagicMock()
        error_response.status_code = 404
        error_response.text = "Not found"
        self.mock_requests.post.return_value = error_response
        
        # Test uploading with error
        with self.assertRaises(Exception) as context:
            self.connector.upload_model(self.test_params)
        
        self.assertIn("IPFS add request failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()