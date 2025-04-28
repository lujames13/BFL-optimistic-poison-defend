"""
Attack simulator for testing Byzantine robustness of federated learning.

This module implements different attack models to test the Krum defense
mechanism in a blockchain-based federated learning system.
"""

import os
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type

from ..client.client import FlowerClient
from ..blockchain_connector import BlockchainConnector
from ..ipfs_connector import ModelIPFSConnector


class AttackSimulator:
    """Base class for attack simulation."""
    
    def __init__(
        self,
        client: FlowerClient,
        attack_config: Dict[str, Any]
    ):
        """Initialize the attack simulator.
        
        Args:
            client: The FlowerClient to perform attacks with
            attack_config: Attack configuration
        """
        self.client = client
        self.attack_config = attack_config
        self.attack_type = attack_config.get("type", "generic")
        self.intensity = attack_config.get("intensity", 0.5)
        
        print(f"Initialized {self.attack_type} attack simulator with intensity {self.intensity}")
    
    def attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Perform attack on model parameters.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Attacked model parameters
        """
        # Base implementation does nothing
        print("Base attack class does not modify parameters")
        return parameters
    
    def execute_attack(self, round_id: int) -> Dict[str, Any]:
        """Execute attack in a federated learning round.
        
        Args:
            round_id: Current round ID
            
        Returns:
            Dictionary with attack results
        """
        try:
            # Check if client is selected for this round
            client_info = self.client.blockchain_connector.get_client_info(self.client.client_id)
            if not client_info.get("selectedForRound", False):
                print(f"Client {self.client.client_id} not selected for round {round_id}")
                return {
                    "success": False,
                    "message": f"Client not selected for round {round_id}"
                }
            
            # Get round information
            round_info = self.client.blockchain_connector.get_round_info(round_id)
            global_model_hash = round_info.get("globalModelHash")
            
            if not global_model_hash:
                return {
                    "success": False,
                    "message": "Global model hash not available"
                }
            
            # Download global model
            global_parameters = self.client.download_model(global_model_hash)
            if not global_parameters:
                return {
                    "success": False,
                    "message": "Failed to download global model"
                }
            
            # Load parameters into model
            self.client.load_parameters(global_parameters)
            
            # Train locally (this will be honest training)
            honest_parameters = self.client.train_locally()
            
            # Apply attack to parameters
            print(f"Applying {self.attack_type} attack")
            attacked_parameters = self.attack(honest_parameters)
            
            # Submit malicious update
            update_hash = self.client.submit_update(round_id, attacked_parameters)
            if not update_hash:
                return {
                    "success": False,
                    "message": "Failed to submit update"
                }
            
            # For reference, calculate honest evaluation and attacked evaluation
            self.client.load_parameters(honest_parameters)
            honest_metrics = self.client.evaluate_model()
            
            self.client.load_parameters(attacked_parameters)
            attacked_metrics = self.client.evaluate_model()
            
            return {
                "success": True,
                "message": f"Successfully executed {self.attack_type} attack",
                "roundId": round_id,
                "updateHash": update_hash,
                "honestMetrics": honest_metrics,
                "attackedMetrics": attacked_metrics,
                "attackType": self.attack_type,
                "intensity": self.intensity
            }
            
        except Exception as e:
            print(f"Error executing attack: {str(e)}")
            return {
                "success": False,
                "message": f"Error executing attack: {str(e)}"
            }


class LabelFlippingAttack(AttackSimulator):
    """Label flipping attack implementation."""
    
    def attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Perform label flipping attack.
        
        This attack targets the last layer weights to flip the predictions.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Attacked model parameters
        """
        # Clone parameters to avoid modifying original
        attacked_params = [np.copy(param) for param in parameters]
        
        # In label flipping attack, we typically target the last layer
        # We'll modify the last layer's weights (usually the layer before softmax)
        last_layer_weights = attacked_params[-2]  # Assuming last layer is just before bias
        last_layer_bias = attacked_params[-1]  # Assuming last element is bias
        
        # Flip the sign of the last layer weights and scale by intensity
        last_layer_weights *= -1.0 * self.intensity
        last_layer_bias *= -1.0 * self.intensity
        
        print(f"Label flipping attack applied with intensity {self.intensity}")
        return attacked_params


class ModelReplacementAttack(AttackSimulator):
    """Model replacement attack implementation."""
    
    def attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Perform model replacement attack.
        
        This attack replaces the model with a malicious model that maximizes
        the distance from the honest model.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Attacked model parameters
        """
        # Clone parameters to avoid modifying original
        attacked_params = [np.copy(param) for param in parameters]
        
        # Calculate scale factor based on intensity
        scale_factor = 100.0 * self.intensity
        
        # Scale all parameters to create a model far from the original
        for i in range(len(attacked_params)):
            attacked_params[i] *= scale_factor
        
        print(f"Model replacement attack applied with scale factor {scale_factor}")
        return attacked_params


class ByzantineClientAttack(AttackSimulator):
    """Byzantine client attack with random behavior."""
    
    def attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Perform Byzantine attack with random behavior.
        
        This attack randomly modifies model parameters in unpredictable ways.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Attacked model parameters
        """
        # Clone parameters to avoid modifying original
        attacked_params = [np.copy(param) for param in parameters]
        
        # Choose a random attack strategy
        attack_strategy = random.choice(["random", "scale", "noise", "flip"])
        
        if attack_strategy == "random":
            # Replace with random values
            for i in range(len(attacked_params)):
                shape = attacked_params[i].shape
                attacked_params[i] = np.random.randn(*shape) * self.intensity
                
        elif attack_strategy == "scale":
            # Scale parameters by a large factor
            scale_factor = 10.0 * self.intensity
            for i in range(len(attacked_params)):
                attacked_params[i] *= scale_factor
                
        elif attack_strategy == "noise":
            # Add noise to parameters
            for i in range(len(attacked_params)):
                shape = attacked_params[i].shape
                noise = np.random.randn(*shape) * self.intensity
                attacked_params[i] += noise
                
        elif attack_strategy == "flip":
            # Negate parameters
            for i in range(len(attacked_params)):
                attacked_params[i] *= -1.0 * self.intensity
        
        print(f"Byzantine attack applied using strategy: {attack_strategy}")
        return attacked_params


class TargetedModelPoisoningAttack(AttackSimulator):
    """Targeted model poisoning attack implementation."""
    
    def attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Perform targeted model poisoning attack.
        
        This attack subtly modifies model parameters to perform well on most data
        but fail specifically on targeted samples.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Attacked model parameters
        """
        # Clone parameters to avoid modifying original
        attacked_params = [np.copy(param) for param in parameters]
        
        # Target class (this would be configured in a real attack)
        target_class = self.attack_config.get("target_class", 0)
        
        # Find the output layer weights for the target class
        # Assuming the last two parameters are the weights and bias of the output layer
        output_weights = attacked_params[-2]
        output_bias = attacked_params[-1]
        
        # Subtly modify the weights for the target class
        # This makes the model perform poorly only on the target class
        if len(output_weights.shape) > 1:  # For fully connected layer
            output_weights[:, target_class] *= (1 - self.intensity)
        else:  # For 1D case (unusual but possible)
            output_weights[target_class] *= (1 - self.intensity)
        
        if len(output_bias.shape) > 0:  # If bias is a vector
            output_bias[target_class] *= (1 - self.intensity)
        
        print(f"Targeted model poisoning attack applied on class {target_class}")
        return attacked_params


def main():
    """Main function to run the attack simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Attack Simulator")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--contract_address", type=str, required=True, help="Blockchain contract address")
    parser.add_argument("--node_url", type=str, default="http://127.0.0.1:8545", help="Blockchain node URL")
    parser.add_argument("--ipfs_url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--attack_type", type=str, required=True, 
                      choices=["label_flipping", "model_replacement", "byzantine", "targeted"],
                      help="Type of attack to simulate")
    parser.add_argument("--intensity", type=float, default=0.5, help="Attack intensity (0.0-1.0)")
    parser.add_argument("--target_class", type=int, default=0, help="Target class for targeted attack")
    
    args = parser.parse_args()
    
    # Create client configuration
    client_config = {
        "client_id": args.client_id,
        "local_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.01
    }
    
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
        config=client_config,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test)
    )
    
    # Register client with blockchain
    registered = client.register_client()
    if not registered:
        print("Failed to register client, exiting")
        return
    
    # Create attack configuration
    attack_config = {
        "type": args.attack_type,
        "intensity": args.intensity,
        "target_class": args.target_class
    }
    
    # Initialize attack simulator based on attack type
    if args.attack_type == "label_flipping":
        simulator = LabelFlippingAttack(client, attack_config)
    elif args.attack_type == "model_replacement":
        simulator = ModelReplacementAttack(client, attack_config)
    elif args.attack_type == "byzantine":
        simulator = ByzantineClientAttack(client, attack_config)
    elif args.attack_type == "targeted":
        simulator = TargetedModelPoisoningAttack(client, attack_config)
    else:
        print(f"Unknown attack type: {args.attack_type}")
        return
    
    # Check system status
    system_status = blockchain_connector.get_system_status()
    current_round = system_status.get("currentRound", 0)
    
    if current_round > 0:
        print(f"Active round detected: {current_round}")
        print(f"Executing {args.attack_type} attack in round {current_round}")
        result = simulator.execute_attack(current_round)
        print(f"Attack result: {json.dumps(result, indent=2)}")
    else:
        print("No active round detected, waiting for server to start a round")
        # In a real implementation, we would set up a loop to check periodically


if __name__ == "__main__":
    main()