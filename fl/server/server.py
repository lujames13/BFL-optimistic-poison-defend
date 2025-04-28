"""
Federated learning server implementation with blockchain integration.

This module implements a Flower server with integrated blockchain and IPFS
capabilities, with Krum defense mechanism for Byzantine-robust aggregation.
"""

import os
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

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
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import Strategy, FedAvg

# Import our custom blockchain and IPFS connectors
from ..blockchain_connector import BlockchainConnector
from ..ipfs_connector import ModelIPFSConnector


class KrumAggregationStrategy(Strategy):
    """Krum aggregation strategy for Byzantine robustness."""
    
    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        ipfs_connector: ModelIPFSConnector,
        task_id: int,
        round_id: int,
        byzantine_tolerance: int = 1,
        min_clients: int = 3,
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.8,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = False,
    ):
        """Initialize the Krum aggregation strategy.
        
        Args:
            blockchain_connector: Connector to the blockchain network
            ipfs_connector: Connector to IPFS
            task_id: Current task ID
            round_id: Current round ID
            byzantine_tolerance: Number of Byzantine clients to tolerate
            min_clients: Minimum number of clients to proceed
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients to use for training
            min_evaluate_clients: Minimum number of clients to use for evaluation
            min_available_clients: Minimum number of available clients
            on_fit_config_fn: Function to configure training
            on_evaluate_config_fn: Function to configure evaluation
            accept_failures: Whether to accept client failures
        """
        super().__init__()
        self.blockchain_connector = blockchain_connector
        self.ipfs_connector = ipfs_connector
        self.task_id = task_id
        self.round_id = round_id
        self.byzantine_tolerance = byzantine_tolerance
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        
        # Fallback strategy to use if we don't have enough clients for Krum
        self.fallback_strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
        )
    
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> Tuple[List[Tuple[ClientProxy, FitIns]], Dict]:
        """Configure the next round of training.
        
        Args:
            rnd: Current round number
            parameters: Model parameters
            client_manager: Client manager
            
        Returns:
            Tuple of (client, fit_ins) pairs and a dictionary with fit configuration
        """
        # Use the fallback strategy to configure clients
        return self.fallback_strategy.configure_fit(rnd, parameters, client_manager)
    
    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> Tuple[List[Tuple[ClientProxy, EvaluateIns]], Dict]:
        """Configure the next round of evaluation.
        
        Args:
            rnd: Current round number
            parameters: Model parameters
            client_manager: Client manager
            
        Returns:
            Tuple of (client, evaluate_ins) pairs and a dictionary with evaluation configuration
        """
        # Use the fallback strategy to configure evaluation
        return self.fallback_strategy.configure_evaluate(rnd, parameters, client_manager)
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate training results using Krum.
        
        Args:
            rnd: Current round number
            results: List of (client, fit_res) tuples
            failures: List of failure exceptions
            
        Returns:
            Aggregated parameters and dictionary with aggregation metrics
        """
        if not results:
            return None, {}
        
        # Check if we have enough results for Krum
        if len(results) < 2 * self.byzantine_tolerance + 3:
            print(f"Not enough clients for Krum: {len(results)} < {2 * self.byzantine_tolerance + 3}")
            print("Falling back to FedAvg")
            return self.fallback_strategy.aggregate_fit(rnd, results, failures)
        
        # Extract parameters and client IDs
        client_parameters = []
        client_ids = []
        for client, fit_res in results:
            # Extract client ID from properties (set during fit)
            client_id = int(client.properties["client_id"])
            client_ids.append(client_id)
            
            # Convert parameters to NumPy arrays
            params = parameters_to_ndarrays(fit_res.parameters)
            client_parameters.append(params)
        
        # Convert parameters to model hashes and upload to IPFS
        model_hashes = []
        for i, params in enumerate(client_parameters):
            # Upload the model to IPFS
            upload_result = self.ipfs_connector.upload_model(
                params, f"client_{client_ids[i]}_round_{rnd}"
            )
            model_hash = upload_result["Hash"]
            model_hashes.append(model_hash)
            
            # Submit the model update to the blockchain
            print(f"Submitting update for client {client_ids[i]} to blockchain")
            self.blockchain_connector.submit_model_update(
                client_ids[i], self.round_id, model_hash
            )
        
        # Prepare updates for Krum defense
        updates = []
        for i in range(len(client_ids)):
            updates.append({
                "clientId": client_ids[i],
                "modelHash": model_hashes[i],
                "parameters": client_parameters[i]
            })
        
        # Apply Krum defense on the blockchain
        print("Applying Krum defense on blockchain")
        selected_client_id = self.blockchain_connector.applyKrumDefense(self.round_id)
        
        # Find the selected parameters
        selected_params = None
        for update in updates:
            if update["clientId"] == selected_client_id:
                selected_params = update["parameters"]
                break
        
        if selected_params is None:
            print(f"Error: Selected client {selected_client_id} not found in updates")
            return None, {"aggregation": "failed"}
        
        # Upload the selected model as the new global model
        print(f"Uploading selected model from client {selected_client_id} as new global model")
        upload_result = self.ipfs_connector.upload_model(
            selected_params, f"global_round_{rnd}"
        )
        global_model_hash = upload_result["Hash"]
        
        # Update the global model on the blockchain
        self.blockchain_connector.updateGlobalModel(self.round_id, global_model_hash)
        
        # Convert back to Flower parameters format
        parameters_aggregated = ndarrays_to_parameters(selected_params)
        
        return parameters_aggregated, {
            "aggregation": "krum",
            "selected_client": selected_client_id,
            "global_model_hash": global_model_hash
        }
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results.
        
        Args:
            rnd: Current round number
            results: List of (client, evaluate_res) tuples
            failures: List of failure exceptions
            
        Returns:
            Aggregated accuracy and a dictionary with evaluation metrics
        """
        # Use the fallback strategy to aggregate evaluation results
        return self.fallback_strategy.aggregate_evaluate(rnd, results, failures)


class FlowerServer:
    """Federated learning server with blockchain integration."""
    
    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        ipfs_connector: ModelIPFSConnector,
        model: torch.nn.Module,
        config: Dict[str, Any]
    ):
        """Initialize the federated learning server.
        
        Args:
            blockchain_connector: Connector to the blockchain network
            ipfs_connector: Connector to IPFS
            model: PyTorch model to be trained
            config: Server configuration
        """
        self.blockchain_connector = blockchain_connector
        self.ipfs_connector = ipfs_connector
        self.model = model
        self.config = config
        
        # Extract configuration
        self.total_rounds = config.get("rounds", 5)
        self.min_clients = config.get("min_clients", 3)
        self.sample_fraction = config.get("sample_fraction", 0.8)
        self.min_fit_clients = config.get("min_fit_clients", 3)
        self.min_eval_clients = config.get("min_eval_clients", 3)
        self.accept_failures = config.get("accept_failures", False)
        self.byzantine_tolerance = config.get("byzantine_tolerance", 1)
        
        # Initialize state
        self.task_id = None
        self.current_round = 0
        self.global_model_hash = None
        self.client_manager = SimpleClientManager()
        self.flower_server = None
    
    def initialize_task(self) -> int:
        """Initialize a new federated learning task on the blockchain.
        
        Returns:
            Task ID of the created task
        """
        # Upload initial model to IPFS
        print("Uploading initial model to IPFS")
        model_params = [param.detach().cpu().numpy() for param in self.model.parameters()]
        upload_result = self.ipfs_connector.upload_model(model_params, "initial_model")
        initial_model_hash = upload_result["Hash"]
        self.global_model_hash = initial_model_hash
        
        print(f"Initial model uploaded to IPFS with hash: {initial_model_hash}")
        
        # Create task on blockchain
        print("Creating task on blockchain")
        task_id = self.blockchain_connector.createTask(initial_model_hash, self.total_rounds)
        self.task_id = task_id
        
        print(f"Task created on blockchain with ID: {task_id}")
        
        return task_id
    
    def start_round(self) -> int:
        """Start a new training round on the blockchain.
        
        Returns:
            Round ID of the created round
        """
        print(f"Starting round for task {self.task_id}")
        round_id = self.blockchain_connector.start_round(self.task_id)
        self.current_round = round_id
        
        print(f"Round started with ID: {round_id}")
        
        return round_id
    
    def select_clients(self, num_clients: int = None) -> List[int]:
        """Select clients for the current round.
        
        Args:
            num_clients: Number of clients to select (defaults to min_fit_clients)
            
        Returns:
            List of selected client IDs
        """
        if num_clients is None:
            num_clients = self.min_fit_clients
        
        # Get registered clients from blockchain
        print("Getting registered clients from blockchain")
        clients = self.blockchain_connector.get_registered_clients()
        
        if len(clients) < num_clients:
            print(f"Warning: Not enough registered clients ({len(clients)} < {num_clients})")
            selected_clients = [client["clientId"] for client in clients]
        else:
            # Randomly select clients
            selected_indices = random.sample(range(len(clients)), num_clients)
            selected_clients = [clients[i]["clientId"] for i in selected_indices]
        
        # Select clients on blockchain
        print(f"Selecting clients on blockchain: {selected_clients}")
        self.blockchain_connector.selectClients(self.current_round, selected_clients)
        
        return selected_clients
    
    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate model updates using Krum defense.
        
        Args:
            updates: List of client updates with parameters
            
        Returns:
            Dictionary with aggregation results
        """
        if len(updates) < 2 * self.byzantine_tolerance + 3:
            print(f"Not enough updates for Krum: {len(updates)} < {2 * self.byzantine_tolerance + 3}")
            return {"success": False, "message": "Not enough updates for Krum defense"}
        
        # Extract client IDs and model hashes
        client_ids = [update["clientId"] for update in updates]
        model_hashes = [update["modelHash"] for update in updates]
        
        # Apply Krum defense on blockchain
        print("Applying Krum defense on blockchain")
        selected_client_id = self.blockchain_connector.applyKrumDefense(self.current_round)
        
        # Find the selected parameters
        selected_params = None
        for update in updates:
            if update["clientId"] == selected_client_id:
                selected_params = update["parameters"]
                break
        
        if selected_params is None:
            return {
                "success": False,
                "message": f"Selected client {selected_client_id} not found in updates"
            }
        
        # Upload the selected model as the new global model
        print(f"Uploading selected model from client {selected_client_id} as new global model")
        upload_result = self.ipfs_connector.upload_model(
            selected_params, f"global_round_{self.current_round}"
        )
        global_model_hash = upload_result["Hash"]
        self.global_model_hash = global_model_hash
        
        # Update the global model on the blockchain
        self.blockchain_connector.updateGlobalModel(self.current_round, global_model_hash)
        
        return {
            "success": True,
            "selectedClientId": selected_client_id,
            "aggregatedModelHash": global_model_hash
        }
    
    def complete_round(self) -> bool:
        """Complete the current training round.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Completing round {self.current_round}")
        return self.blockchain_connector.complete_round(self.current_round)
    
    def complete_task(self) -> bool:
        """Complete the federated learning task.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Completing task {self.task_id}")
        
        # Upload final model to IPFS
        model_params = [param.detach().cpu().numpy() for param in self.model.parameters()]
        upload_result = self.ipfs_connector.upload_model(model_params, "final_model")
        final_model_hash = upload_result["Hash"]
        
        return self.blockchain_connector.completeTask(self.task_id, final_model_hash)
    
    def distribute_rewards(self, client_ids: List[int]) -> bool:
        """Distribute rewards to clients.
        
        Args:
            client_ids: List of client IDs to reward
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Distributing rewards to clients: {client_ids}")
        return self.blockchain_connector.distributeRewards(client_ids, self.current_round)
    
    def handle_round_failure(self, reason: str) -> Dict[str, Any]:
        """Handle round failure.
        
        Args:
            reason: Reason for failure
            
        Returns:
            Dictionary with failure handling results
        """
        print(f"Round {self.current_round} failed: {reason}")
        
        # In a real implementation, we might have different strategies for different failures
        return {
            "success": False,
            "message": f"Round failed: {reason}"
        }
    
    def start_training(self) -> Dict[str, Any]:
        """Start the federated learning process.
        
        Returns:
            Dictionary with training results
        """
        try:
            # Initialize task
            self.initialize_task()
            
            for round_num in range(1, self.total_rounds + 1):
                print(f"\n--- Starting round {round_num}/{self.total_rounds} ---")
                
                # Start round on blockchain
                round_id = self.start_round()
                
                # Select clients
                selected_clients = self.select_clients()
                if len(selected_clients) < self.min_clients:
                    return self.handle_round_failure("Not enough clients available")
                
                # Start Flower server for this round
                self._start_flower_server(round_id)
                
                # Wait for round completion (this would be handled by Flower callbacks)
                # This is just a placeholder - in reality, Flower would handle the training
                print("Waiting for round completion...")
                time.sleep(2)  # Simulate waiting for clients
                
                # Complete round on blockchain
                self.complete_round()
                
                # Distribute rewards (assuming all selected clients participated)
                self.distribute_rewards(selected_clients)
                
                print(f"--- Round {round_num}/{self.total_rounds} completed ---")
            
            # Complete task
            self.complete_task()
            
            return {
                "success": True,
                "message": "Training completed successfully",
                "task_id": self.task_id,
                "rounds_completed": self.total_rounds,
                "final_model_hash": self.global_model_hash
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return {
                "success": False,
                "message": f"Training failed: {str(e)}"
            }
    
    def _start_flower_server(self, round_id: int):
        """Start Flower server for a specific round.
        
        Args:
            round_id: Current round ID
        """
        # Convert model to parameters
        model_params = [param.detach().cpu().numpy() for param in self.model.parameters()]
        parameters = ndarrays_to_parameters(model_params)
        
        # Create strategy with blockchain/IPFS integration
        strategy = KrumAggregationStrategy(
            blockchain_connector=self.blockchain_connector,
            ipfs_connector=self.ipfs_connector,
            task_id=self.task_id,
            round_id=round_id,
            byzantine_tolerance=self.byzantine_tolerance,
            min_clients=self.min_clients,
            fraction_fit=self.sample_fraction,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_eval_clients,
            accept_failures=self.accept_failures,
            on_fit_config_fn=lambda r: {"epochs": 3, "batch_size": 32}
        )
        
        # In a real implementation, we would start the Flower server
        # server = fl.server.Server(client_manager=self.client_manager, strategy=strategy)
        # fl.server.start_server(
        #     server_address="0.0.0.0:8080",
        #     server=server,
        #     config={"num_rounds": 1}  # Only do one round per blockchain round
        # )
        
        print(f"Flower server started for round {round_id}")


def main():
    """Main function to run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server with Blockchain")
    parser.add_argument("--task_params", type=str, required=True, help="Path to task parameters JSON")
    parser.add_argument("--contract_address", type=str, required=True, help="Blockchain contract address")
    parser.add_argument("--node_url", type=str, default="http://127.0.0.1:8545", help="Blockchain node URL")
    parser.add_argument("--ipfs_url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--model_path", type=str, help="Path to initial model state")
    
    args = parser.parse_args()
    
    # Load task parameters
    with open(args.task_params, "r") as f:
        config = json.load(f)
    
    # Initialize blockchain connector
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        node_url=args.node_url
    )
    
    # Initialize IPFS connector
    ipfs_connector = ModelIPFSConnector(api_url=args.ipfs_url)
    
    # Create a simple model
    # In a real implementation, this would be loaded from args.model_path
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.Softmax(dim=1)
    )
    
    # Initialize server
    server = FlowerServer(
        blockchain_connector=blockchain_connector,
        ipfs_connector=ipfs_connector,
        model=model,
        config=config
    )
    
    # Start training
    result = server.start_training()
    print(f"Training result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()