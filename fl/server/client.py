"""
Federated learning client implementation with blockchain integration.

This module implements a Flower client with integrated blockchain and IPFS
capabilities, supporting secure model updates and blockchain verification.
"""

import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

# Import our custom blockchain and IPFS connectors
from ..blockchain_connector import BlockchainConnector
from ..ipfs_connector import ModelIPFSConnector


class FlowerClientFl(fl.client.NumPyClient):
    """Flower client implementation with blockchain integration."""
    
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        blockchain_connector: BlockchainConnector,
        ipfs_connector: ModelIPFSConnector,
        config: Dict[str, Any]
    ):
        """Initialize the Flower client.
        
        Args:
            client_id: Client ID
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            blockchain_connector: Connector to the blockchain network
            ipfs_connector: Connector to IPFS
            config: Client configuration
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.blockchain_connector = blockchain_connector
        self.ipfs_connector = ipfs_connector
        self.config = config
        
        # Extract configuration
        self.local_epochs = config.get("local_epochs", 3)
        self.learning_rate = config.get("learning_rate", 0.01)
        
        # Setup optimizer and loss function
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Initialized client {client_id} on device: {self.device}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get model parameters.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of model parameters as NumPy arrays
        """
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray], config: Dict[str, Scalar] = None) -> None:
        """Set model parameters.
        
        Args:
            parameters: List of model parameters as NumPy arrays
            config: Configuration from server
        """
        params_dict = zip(self.model.parameters(), parameters)
        with torch.no_grad():
            for model_param, param in params_dict:
                model_param.copy_(torch.tensor(param))
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the model.
        
        Args:
            parameters: Current model parameters
            config: Configuration from server
            
        Returns:
            Tuple of (updated parameters, number of training examples, training metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get round information from config
        round_id = config.get("round_id", 0)
        
        # Train the model
        num_examples = self._train(config)
        
        # Get updated parameters
        parameters_updated = self.get_parameters(config)
        
        # Submit update to blockchain
        model_hash = self.blockchain_connector.submit_model_update(
            self.client_id, round_id, parameters_updated
        )
        
        # Store the model hash for reference
        metrics = {"model_hash": model_hash}
        
        return parameters_updated, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model.
        
        Args:
            parameters: Current model parameters
            config: Configuration from server
            
        Returns:
            Tuple of (loss, number of test examples, evaluation metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        loss, accuracy, num_examples = self._evaluate()
        
        return float(loss), num_examples, {"accuracy": float(accuracy)}
    
    def _train(self, config: Dict[str, Scalar]) -> int:
        """Train the model for a specified number of epochs.
        
        Args:
            config: Training configuration
            
        Returns:
            Number of training examples
        """
        # Get training epochs from config or use default
        epochs = int(config.get("epochs", self.local_epochs))
        
        # Set model to training mode
        self.model.train()
        
        # Keep track of number of examples
        num_examples = 0
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Increment example count
                num_examples += len(inputs)
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100.0 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        return num_examples
    
    def _evaluate(self) -> Tuple[float, float, int]:
        """Evaluate the model on the test dataset.
        
        Returns:
            Tuple of (loss, accuracy, number of test examples)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Keep track of metrics
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate overall metrics
        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, total


class FlowerClient:
    """High-level client interface with blockchain and IPFS integration."""
    
    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        ipfs_connector: ModelIPFSConnector,
        model_class: Type[torch.nn.Module],
        config: Dict[str, Any],
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
    ):
        """Initialize the client.
        
        Args:
            blockchain_connector: Connector to the blockchain network
            ipfs_connector: Connector to IPFS
            model_class: PyTorch model class to use
            config: Client configuration
            train_data: Training data as (inputs, targets) tuple
            test_data: Test data as (inputs, targets) tuple
        """
        self.blockchain_connector = blockchain_connector
        self.ipfs_connector = ipfs_connector
        self.model_class = model_class
        self.config = config
        
        # Extract client ID from config
        self.client_id = config.get("client_id")
        if self.client_id is None:
            raise ValueError("Client ID must be specified in config")
        
        # Create data loaders
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        batch_size = config.get("batch_size", 32)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = model_class()
        
        print(f"Initialized client {self.client_id}")
    
    def register_client(self) -> bool:
        """Register client with the blockchain.
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            print(f"Registering client {self.client_id} with blockchain")
            result = self.blockchain_connector.register_client()
            print(f"Registration result: {result}")
            return result
        except Exception as e:
            print(f"Error registering client: {str(e)}")
            return False
    
    def download_model(self, model_hash: str) -> List[np.ndarray]:
        """Download model from IPFS.
        
        Args:
            model_hash: IPFS hash of the model
            
        Returns:
            List of model parameters as NumPy arrays
        """
        try:
            print(f"Downloading model from IPFS with hash: {model_hash}")
            parameters = self.ipfs_connector.download_model(model_hash)
            print("Model downloaded successfully")
            return parameters
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return None
    
    def load_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load parameters into the model.
        
        Args:
            parameters: List of model parameters as NumPy arrays
        """
        params_dict = zip(self.model.parameters(), parameters)
        with torch.no_grad():
            for model_param, param in params_dict:
                model_param.copy_(torch.tensor(param))
    
    def train_locally(self) -> List[np.ndarray]:
        """Train the model locally.
        
        Returns:
            Updated model parameters
        """
        print("Training model locally")
        
        # Setup optimizer and loss function
        learning_rate = self.config.get("learning_rate", 0.01)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Train for a specified number of epochs
        epochs = self.config.get("local_epochs", 3)
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Set model to training mode
            self.model.train()
            
            for inputs, targets in self.train_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100.0 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # Return updated parameters
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model")
        
        # Setup loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Keep track of metrics
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Track metrics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate overall metrics
        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / total
        
        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        }
        
        print(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def submit_update(self, round_id: int, parameters: List[np.ndarray]) -> str:
        """Submit model update to blockchain.
        
        Args:
            round_id: Current round ID
            parameters: Updated model parameters
            
        Returns:
            IPFS hash of the submitted model update
        """
        try:
            print(f"Uploading model update to IPFS")
            upload_result = self.ipfs_connector.upload_model(
                parameters, f"client_{self.client_id}_round_{round_id}"
            )
            model_hash = upload_result["Hash"]
            
            print(f"Model uploaded to IPFS with hash: {model_hash}")
            
            print(f"Submitting model update to blockchain")
            self.blockchain_connector.submit_model_update(
                self.client_id, round_id, model_hash
            )
            
            return model_hash
        except Exception as e:
            print(f"Error submitting update: {str(e)}")
            return None
    
    def participate_in_round(self, round_id: int) -> Dict[str, Any]:
        """Participate in a training round.
        
        Args:
            round_id: Current round ID
            
        Returns:
            Dictionary with participation results
        """
        try:
            # Check if client is selected for this round
            client_info = self.blockchain_connector.get_client_info(self.client_id)
            if not client_info.get("selectedForRound", False):
                print(f"Client {self.client_id} not selected for round {round_id}")
                return {
                    "success": False,
                    "message": f"Client not selected for round {round_id}"
                }
            
            # Get round information
            round_info = self.blockchain_connector.get_round_info(round_id)
            global_model_hash = round_info.get("globalModelHash")
            
            if not global_model_hash:
                return {
                    "success": False,
                    "message": "Global model hash not available"
                }
            
            # Download global model
            global_parameters = self.download_model(global_model_hash)
            if not global_parameters:
                return {
                    "success": False,
                    "message": "Failed to download global model"
                }
            
            # Load parameters into model
            self.load_parameters(global_parameters)
            
            # Train locally
            updated_parameters = self.train_locally()
            
            # Submit update
            update_hash = self.submit_update(round_id, updated_parameters)
            if not update_hash:
                return {
                    "success": False,
                    "message": "Failed to submit update"
                }
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            return {
                "success": True,
                "message": "Successfully participated in round",
                "roundId": round_id,
                "updateHash": update_hash,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error participating in round: {str(e)}")
            return {
                "success": False,
                "message": f"Error participating in round: {str(e)}"
            }
    
    def start_fl_client(self) -> None:
        """Start the Flower client.
        
        This method is used when integrating with the Flower framework.
        """
        # Create Flower client
        fl_client = FlowerClientFl(
            client_id=self.client_id,
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            blockchain_connector=self.blockchain_connector,
            ipfs_connector=self.ipfs_connector,
            config=self.config
        )
        
        # Add client ID to properties
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=fl_client,
            grpc_max_message_length=1024 * 1024 * 1024,
            root_certificates=None
        )


def main():
    """Main function to run the client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client with Blockchain")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--contract_address", type=str, required=True, help="Blockchain contract address")
    parser.add_argument("--node_url", type=str, default="http://127.0.0.1:8545", help="Blockchain node URL")
    parser.add_argument("--ipfs_url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--config_path", type=str, help="Path to client configuration JSON")
    
    args = parser.parse_args()
    
    # Load client configuration
    if args.config_path:
        with open(args.config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "client_id": args.client_id,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01
        }
    
    # Ensure client_id is set
    config["client_id"] = args.client_id
    
    # Initialize blockchain connector
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        client_id=args.client_id,
        node_url=args.node_url
    )
    
    # Initialize IPFS connector
    ipfs_connector = ModelIPFSConnector(api_url=args.ipfs_url)
    
    # Create a simple model class
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(20, 10)
            self.softmax = torch.nn.Softmax(dim=1)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x
    
    # In a real implementation, load data from args.data_path
    # For simplicity, create random data
    print(f"Loading data from {args.data_path}")
    try:
        # Try to load actual data (simplified example)
        # This would be replaced with proper data loading code
        data = torch.load(args.data_path)
        x_train, y_train = data["train"]
        x_test, y_test = data["test"]
    except:
        print("Could not load data, using random data for demonstration")
        # Create random data for demonstration
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 10, (100,))
        x_test = torch.randn(20, 10)
        y_test = torch.randint(0, 10, (20,))
    
    # Initialize client
    client = FlowerClient(
        blockchain_connector=blockchain_connector,
        ipfs_connector=ipfs_connector,
        model_class=SimpleModel,
        config=config,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test)
    )
    
    # Register client with blockchain
    registered = client.register_client()
    if not registered:
        print("Failed to register client, exiting")
        return
    
    # Start the client
    print(f"Client {args.client_id} registered successfully")
    print("Starting Federated Learning client")
    
    # In a standalone mode, we would poll the blockchain for active rounds
    try:
        # Check system status
        system_status = blockchain_connector.get_system_status()
        current_round = system_status.get("currentRound", 0)
        
        if current_round > 0:
            print(f"Active round detected: {current_round}")
            result = client.participate_in_round(current_round)
            print(f"Participation result: {json.dumps(result, indent=2)}")
        else:
            print("No active round detected, waiting for server to start a round")
            # In a real implementation, we would set up a loop to check periodically
            # For demonstration, we'll start the Flower client
            client.start_fl_client()
    
    except Exception as e:
        print(f"Error during client execution: {str(e)}")


if __name__ == "__main__":
    main()