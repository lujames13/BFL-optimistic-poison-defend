"""
IPFS connector specialized for machine learning model parameter storage and retrieval.
Optimized for federated learning with support for various model formats and batch operations.
"""

import os
import json
import time
import tempfile
import logging
import io
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import requests
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ipfs_connector")

class ModelIPFSConnector:
    """
    IPFS connector specialized for machine learning model parameter storage and retrieval.
    Provides robust handling of model parameters for federated learning.
    """
    
    def __init__(self, api_url: str = "http://localhost:5001/api/v0", timeout: int = 30):
        """
        Initialize IPFS connector.
        
        Args:
            api_url: IPFS API endpoint, defaults to local node
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        self.connected = False
        
        # Test connection
        try:
            response = requests.post(f"{self.api_url}/id", timeout=self.timeout)
            if response.status_code == 200:
                node_id = response.json()["ID"]
                logger.info(f"Successfully connected to IPFS node: {node_id}")
                self.connected = True
                # Log node addresses for possible multiaddr connections
                addresses = response.json().get("Addresses", [])
                if addresses:
                    logger.info(f"Node addresses: {', '.join(addresses[:3])}")
                    if len(addresses) > 3:
                        logger.info(f"...and {len(addresses) - 3} more")
            else:
                logger.error(f"Could not connect to IPFS API: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to IPFS: {str(e)}")
    
    def upload_model(self, model_weights: Union[List[np.ndarray], Dict, torch.nn.Module], 
                    model_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Serialize model weights and upload to IPFS.
        
        Args:
            model_weights: Model weights as list of NumPy arrays, state dict, or PyTorch model
            model_id: Optional identifier for the model
            metadata: Additional metadata to store with the model
            
        Returns:
            Dictionary containing CID and metadata
        """
        if not self.connected:
            logger.warning("IPFS connection is not established. Attempting to connect...")
            try:
                response = requests.post(f"{self.api_url}/id", timeout=self.timeout)
                if response.status_code == 200:
                    self.connected = True
                    logger.info("Successfully connected to IPFS node")
                else:
                    logger.error(f"Could not connect to IPFS API: {response.status_code}")
                    raise ConnectionError("Failed to connect to IPFS")
            except Exception as e:
                logger.error(f"Error connecting to IPFS: {str(e)}")
                raise ConnectionError(f"Failed to connect to IPFS: {str(e)}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            temp_path = temp_file.name
        
        try:
            # Prepare metadata
            meta = metadata or {}
            if model_id:
                meta['model_id'] = model_id
            meta['timestamp'] = time.time()
            meta['date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Serialize model weights based on type
            if isinstance(model_weights, torch.nn.Module):
                # If full PyTorch model is provided
                logger.info(f"Serializing full PyTorch model to: {temp_path}")
                state_dict = model_weights.state_dict()
                state_dict['__metadata__'] = meta
                torch.save(state_dict, temp_path)
            elif isinstance(model_weights, dict):
                # If state dict is provided
                logger.info(f"Serializing PyTorch state dict to: {temp_path}")
                state_dict = model_weights.copy()
                state_dict['__metadata__'] = meta
                torch.save(state_dict, temp_path)
            else:
                # If list of NumPy arrays is provided (Flower format)
                logger.info(f"Serializing NumPy arrays to: {temp_path}")
                state_dict = {}
                state_dict['__metadata__'] = meta
                state_dict['__weights__'] = [arr.tolist() for arr in model_weights]
                state_dict['__shapes__'] = [arr.shape for arr in model_weights]
                state_dict['__dtypes__'] = [str(arr.dtype) for arr in model_weights]
                
                with open(temp_path, 'w') as f:
                    json.dump(state_dict, f)
            
            # Upload to IPFS
            logger.info(f"Uploading model to IPFS")
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f)}
                response = requests.post(
                    f"{self.api_url}/add", 
                    files=files,
                    timeout=self.timeout
                )
                
            if response.status_code != 200:
                raise Exception(f"IPFS add request failed: {response.text}")
                
            result = response.json()
            logger.info(f"Model uploaded to IPFS with CID: {result['Hash']}")
            
            # Add metadata to result
            result['metadata'] = meta
            result['size_bytes'] = os.path.getsize(temp_path)
            
            # Pin the file to ensure it remains in IPFS storage
            try:
                self.pin_model(result['Hash'])
            except Exception as e:
                logger.warning(f"Could not pin model {result['Hash']}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error uploading model to IPFS: {str(e)}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def download_model(self, cid: str, output_path: Optional[str] = None, 
                      as_numpy: bool = True, timeout: Optional[int] = None) -> Union[List[np.ndarray], Dict]:
        """
        Download model weights from IPFS using CID.
        
        Args:
            cid: Content identifier for the model
            output_path: Optional path to save the model
            as_numpy: If True, returns list of NumPy arrays (Flower format), 
                     otherwise returns state dict
            timeout: Optional timeout override for large models
            
        Returns:
            Model weights
        """
        logger.info(f"Downloading model from IPFS with CID: {cid}")
        
        # Use provided timeout or default
        req_timeout = timeout or self.timeout
        
        try:
            response = requests.post(
                f"{self.api_url}/cat?arg={cid}", 
                timeout=req_timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"IPFS cat request failed: {response.text}")
                
            content = response.content
            
            # Create temporary file for downloaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # Save to output path if specified
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    logger.info(f"Saved model to: {output_path}")
                
                # Deserialize model weights
                return self._deserialize_model(temp_path, as_numpy)
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Error downloading model from IPFS: {str(e)}")
            # Try alternative method for large files
            if "timeout" in str(e).lower():
                logger.info("Trying alternative method for large file download")
                return self._download_large_model(cid, output_path, as_numpy)
            raise
    
    def _deserialize_model(self, file_path: str, as_numpy: bool) -> Union[List[np.ndarray], Dict]:
        """
        Deserialize model from file path.
        
        Args:
            file_path: Path to model file
            as_numpy: If True, returns list of NumPy arrays
            
        Returns:
            Model weights
        """
        try:
            # Try loading as PyTorch model
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            
            # Extract metadata if present
            metadata = state_dict.pop('__metadata__', {}) if isinstance(state_dict, dict) else {}
            
            if as_numpy:
                # Convert to NumPy arrays (Flower format)
                if isinstance(state_dict, dict) and '__weights__' in state_dict:
                    # Handle JSON format with weights list
                    weights_list = state_dict['__weights__']
                    shapes = state_dict.get('__shapes__', None)
                    dtypes = state_dict.get('__dtypes__', None)
                    
                    if shapes and dtypes:
                        # Use shape and dtype information if available
                        return [np.array(arr, dtype=np.dtype(dt)).reshape(shape) 
                                for arr, shape, dt in zip(weights_list, shapes, dtypes)]
                    else:
                        return [np.array(arr) for arr in weights_list]
                else:
                    # Handle PyTorch state dict
                    # Skip metadata and other special keys
                    return [val.cpu().numpy() for key, val in state_dict.items() 
                            if not key.startswith('__') and isinstance(val, torch.Tensor)]
            else:
                return state_dict
                
        except Exception as e:
            logger.warning(f"Error loading as PyTorch model: {str(e)}")
            
            # Try loading as JSON
            try:
                with open(file_path, 'r') as f:
                    state_dict = json.load(f)
                
                if '__weights__' in state_dict:
                    weights_list = state_dict['__weights__']
                    shapes = state_dict.get('__shapes__', None)
                    dtypes = state_dict.get('__dtypes__', None)
                    
                    if as_numpy:
                        if shapes and dtypes:
                            # Use shape and dtype information if available
                            return [np.array(arr, dtype=np.dtype(dt)).reshape(shape) 
                                    for arr, shape, dt in zip(weights_list, shapes, dtypes)]
                        else:
                            return [np.array(arr) for arr in weights_list]
                    else:
                        return state_dict
                else:
                    raise ValueError("Invalid model format: missing weights")
            except Exception as json_error:
                logger.error(f"Error loading JSON: {str(json_error)}")
                raise ValueError(f"Failed to deserialize model: {str(e)}, then failed JSON parsing: {str(json_error)}")
    
    def _download_large_model(self, cid: str, output_path: Optional[str] = None, 
                             as_numpy: bool = True) -> Union[List[np.ndarray], Dict]:
        """
        Alternative method to download large models using IPFS get.
        
        Args:
            cid: Content identifier for the model
            output_path: Optional path to save the model
            as_numpy: If True, returns list of NumPy arrays
            
        Returns:
            Model weights
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Use IPFS get instead of cat for large files
            logger.info(f"Downloading large model with CID: {cid} to {temp_dir}")
            response = requests.post(
                f"{self.api_url}/get?arg={cid}&output={temp_dir}",
                timeout=self.timeout * 2  # Double timeout for large files
            )
            
            if response.status_code != 200:
                raise Exception(f"IPFS get request failed: {response.text}")
            
            # Find downloaded file
            downloaded_file = None
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if cid in file or file.endswith('.pt') or file.endswith('.json'):
                        downloaded_file = os.path.join(root, file)
                        break
                if downloaded_file:
                    break
            
            if not downloaded_file:
                raise FileNotFoundError(f"Could not find downloaded file for CID: {cid}")
            
            logger.info(f"Found downloaded file: {downloaded_file}")
            
            # Save to output path if specified
            if output_path:
                import shutil
                shutil.copy2(downloaded_file, output_path)
                logger.info(f"Saved model to: {output_path}")
            
            # Deserialize the model
            return self._deserialize_model(downloaded_file, as_numpy)
            
        except Exception as e:
            logger.error(f"Error in large model download: {str(e)}")
            raise
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def pin_model(self, cid: str) -> Dict[str, Any]:
        """
        Pin a model to ensure it remains in IPFS storage.
        
        Args:
            cid: Content identifier to pin
            
        Returns:
            API response
        """
        logger.info(f"Pinning CID: {cid}")
        response = requests.post(
            f"{self.api_url}/pin/add?arg={cid}",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin add request failed: {response.text}")
            
        result = response.json()
        logger.info(f"CID pinned: {result}")
        return result
    
    def unpin_model(self, cid: str) -> Dict[str, Any]:
        """
        Unpin a model to allow it to be garbage collected.
        
        Args:
            cid: Content identifier to unpin
            
        Returns:
            API response
        """
        logger.info(f"Removing pin for CID: {cid}")
        response = requests.post(
            f"{self.api_url}/pin/rm?arg={cid}",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin rm request failed: {response.text}")
            
        result = response.json()
        logger.info(f"Pin removed: {result}")
        return result
    
    def list_pinned_models(self) -> Dict[str, Any]:
        """
        List all pinned models.
        
        Returns:
            Dictionary of pinned CIDs
        """
        response = requests.post(
            f"{self.api_url}/pin/ls",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin ls request failed: {response.text}")
            
        result = response.json()
        return result

    def get_node_info(self) -> Dict[str, Any]:
        """
        Get IPFS node information.
        
        Returns:
            Node information dictionary
        """
        response = requests.post(
            f"{self.api_url}/id",
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get node info: {response.text}")
            
        return response.json()
    
    def batch_upload_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Upload multiple models in batch.
        
        Args:
            models: List of dictionaries with keys:
                   - 'weights': Model weights
                   - 'model_id': Optional model identifier
                   - 'metadata': Optional additional metadata
            
        Returns:
            List of upload results
        """
        results = []
        errors = []
        
        for i, model_info in enumerate(models):
            try:
                weights = model_info.get('weights')
                model_id = model_info.get('model_id', f"batch_model_{i}")
                metadata = model_info.get('metadata', {})
                
                if not weights:
                    raise ValueError(f"No weights provided for model {i}")
                
                result = self.upload_model(weights, model_id, metadata)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error uploading model {i}: {str(e)}")
                errors.append({"index": i, "error": str(e)})
        
        return {
            "results": results,
            "errors": errors,
            "success_count": len(results),
            "error_count": len(errors),
            "total": len(models)
        }
    
    def batch_download_models(self, cids: List[str], as_numpy: bool = True) -> List[Dict[str, Any]]:
        """
        Download multiple models in batch.
        
        Args:
            cids: List of content identifiers
            as_numpy: If True, returns models as lists of NumPy arrays
            
        Returns:
            List of download results
        """
        results = []
        errors = []
        
        for i, cid in enumerate(cids):
            try:
                model = self.download_model(cid, as_numpy=as_numpy)
                results.append({
                    "cid": cid,
                    "model": model,
                    "index": i
                })
                
            except Exception as e:
                logger.error(f"Error downloading model {cid}: {str(e)}")
                errors.append({"cid": cid, "index": i, "error": str(e)})
        
        return {
            "models": results,
            "errors": errors,
            "success_count": len(results),
            "error_count": len(errors),
            "total": len(cids)
        }
    
    def calculate_model_diff(self, model1: List[np.ndarray], model2: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Calculate difference between two models (for Krum defense).
        
        Args:
            model1: First model parameters
            model2: Second model parameters
            
        Returns:
            Tuple of (parameter differences, overall distance)
        """
        if len(model1) != len(model2):
            raise ValueError(f"Models have different number of parameters: {len(model1)} vs {len(model2)}")
        
        # Calculate differences and distance
        diff_params = []
        squared_distance = 0
        
        for p1, p2 in zip(model1, model2):
            if p1.shape != p2.shape:
                raise ValueError(f"Parameter shapes don't match: {p1.shape} vs {p2.shape}")
            
            diff = p1 - p2
            diff_params.append(diff)
            
            # Frobenius norm squared
            squared_distance += np.sum(diff**2)
        
        distance = np.sqrt(squared_distance)
        
        return diff_params, distance
    
    def federated_average(self, models: List[List[np.ndarray]], weights: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Perform federated averaging of models.
        
        Args:
            models: List of model parameters (each as list of NumPy arrays)
            weights: Optional weights for averaging (e.g., based on data sizes)
            
        Returns:
            Averaged model parameters
        """
        if not models:
            raise ValueError("No models provided for averaging")
        
        # Use equal weights if none provided
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        if len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of models ({len(models)})")
        
        # Initialize with zeros
        avg_model = [np.zeros_like(param) for param in models[0]]
        
        # Weighted average
        for model_params, weight in zip(models, weights):
            for i, param in enumerate(model_params):
                avg_model[i] += param * weight
        
        return avg_model