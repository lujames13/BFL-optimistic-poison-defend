"""Blockchain-integrated Flower server implementation.

This module implements a Flower server that integrates with blockchain
technology for secure model update verification and aggregation with
Byzantine fault tolerance through the Krum algorithm.
"""

import os
import flwr as fl
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes

from fl.blockchain_connector import BlockchainConnector
from fl.ipfs_connector import ModelIPFSConnector
from fl.defense import krum_select


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters to NumPy arrays.
    
    Args:
        parameters: Parameters object from Flower.
        
    Returns:
        List of NumPy arrays.
    """
    return fl.common.parameters_to_ndarrays(parameters)


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy arrays to parameters.
    
    Args:
        ndarrays: List of NumPy arrays.
        
    Returns:
        Parameters object for Flower.
    """
    return fl.common.ndarrays_to_parameters(ndarrays)


def get_strategy(
    evaluate_fn: Optional[Callable[[NDArrays], Tuple[float, Dict[str, float]]]] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    use_krum: bool = False,
    byzantine_threshold: int = 1,
    multi_krum: bool = False
) -> Strategy:
    """Get the appropriate Flower strategy based on configuration.
    
    Args:
        evaluate_fn: Optional function to evaluate the aggregated model parameters.
        fraction_fit: Fraction of clients to sample for training.
        fraction_evaluate: Fraction of clients to sample for evaluation.
        min_fit_clients: Minimum number of clients to sample for training.
        min_evaluate_clients: Minimum number of clients to sample for evaluation.
        min_available_clients: Minimum number of clients that need to be available.
        use_krum: Whether to use Krum for Byzantine-robust aggregation.
        byzantine_threshold: Number of Byzantine clients to tolerate (f).
        multi_krum: Whether to use Multi-Krum (aggregates multiple updates).
    
    Returns:
        A Flower strategy configured based on the provided parameters.
    """
    if use_krum:
        # Use a Byzantine-robust strategy
        return KrumFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            byzantine_threshold=byzantine_threshold,
            multi_krum=multi_krum
        )
    else:
        # Use standard FedAvg
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn
        )


class KrumFedAvg(FedAvg):
    """Federated Averaging with Krum defense for Byzantine robustness."""
    
    def __init__(
        self,
        *args,
        byzantine_threshold: int = 1,
        multi_krum: bool = False,
        m: Optional[int] = None,
        **kwargs
    ):
        """Initialize the KrumFedAvg strategy.
        
        Args:
            *args: Arguments to pass to the parent FedAvg class.
            byzantine_threshold: Number of Byzantine clients to tolerate (f).
            multi_krum: Whether to use Multi-Krum (aggregates multiple updates).
            m: For Multi-Krum, the number of candidates to consider.
            **kwargs: Keyword arguments to pass to the parent FedAvg class.
        """
        super().__init__(*args, **kwargs)
        self.byzantine_threshold = byzantine_threshold
        self.multi_krum = multi_krum
        self.m = m
        
        # Keep track of client selection for research purposes
        self.selected_clients = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using Krum Byzantine-robust aggregation.
        
        Args:
            server_round: Current round of federated learning.
            results: List of tuples of (client, fit_res) for successful clients.
            failures: List of failures that occurred during fitting.
            
        Returns:
            Tuple of (parameters, metrics) where parameters are the new global model
            parameters and metrics is a dict of aggregation metrics.
        """
        if not results:
            return None, {}
        
        # Check if we have enough clients to apply Krum
        if len(results) < 2 * self.byzantine_threshold + 3:
            # If not enough clients, fall back to standard FedAvg
            return super().aggregate_fit(server_round, results, failures)
        
        # Extract client updates
        client_ids = [int(client.cid) for client, _ in results]
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        weights = [w for w, _ in weights_results]
        num_examples = [n for _, n in weights_results]
        
        if self.multi_krum:
            # Apply Multi-Krum to select multiple clients
            selected_indices, selected_weights = krum_select(
                weights, client_ids, self.byzantine_threshold, 
                multi_krum=True, m=self.m or len(weights),
                weights=num_examples
            )
            # Keep track of selected clients for analysis
            self.selected_clients.append([client_ids[i] for i in selected_indices])
            
            # Perform weighted average of selected updates
            selected_num_examples = [num_examples[i] for i in selected_indices]
            aggregated_weights = self.aggregate_ndarrays(selected_weights, selected_num_examples)
        else:
            # Apply standard Krum to select the most representative update
            selected_idx, selected_weight = krum_select(
                weights, client_ids, self.byzantine_threshold,
                weights=num_examples
            )
            # Keep track of selected client for analysis
            self.selected_clients.append(client_ids[selected_idx])
            
            # Use the selected update directly
            aggregated_weights = selected_weight
        
        # Convert back to parameters format
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        
        # Calculate metrics
        metrics = {
            "krum_selected": self.selected_clients[-1],
            "num_examples": sum(num_examples),
            "num_clients": len(results),
        }
        
        return aggregated_parameters, metrics
    
    def aggregate_ndarrays(
        self, weights: List[NDArrays], num_examples: List[int]
    ) -> NDArrays:
        """Aggregate model updates using weighted average.
        
        Args:
            weights: List of model weights to aggregate.
            num_examples: Number of examples used for training by each client.
            
        Returns:
            Aggregated model weights.
        """
        # Weighted average of selected updates
        total_examples = sum(num_examples)
        weighted_updates = []
        
        for i, update in enumerate(weights):
            weight = num_examples[i] / total_examples
            weighted_update = [layer * weight for layer in update]
            weighted_updates.append(weighted_update)
        
        # Sum all weighted updates
        aggregate = [np.zeros_like(layer) for layer in weights[0]]
        for update in weighted_updates:
            for i, layer in enumerate(update):
                aggregate[i] += layer
        
        return aggregate


class BlockchainFlowerServer:
    """Federated Learning server with blockchain integration.
    
    This server integrates with blockchain technology for secure model update
    verification and with IPFS for decentralized model storage.
    """
    
    def __init__(
        self,
        initial_model: tf.keras.Model,
        contract_address: str,
        ipfs_url: str = "http://localhost:5001",
        private_key: Optional[str] = None,
        node_url: str = "http://127.0.0.1:8545",
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        use_krum: bool = True,
        byzantine_threshold: int = 1,
        round_timeout: int = 3600,  # Seconds
    ):
        """Initialize the blockchain-integrated Flower server.
        
        Args:
            initial_model: Initial model to start federated learning.
            contract_address: Address of the deployed FederatedLearning contract.
            ipfs_url: URL of the IPFS node API.
            private_key: Optional Ethereum private key for blockchain transactions.
            node_url: URL of the Ethereum node.
            x_test: Optional test data features for server-side evaluation.
            y_test: Optional test data labels for server-side evaluation.
            use_krum: Whether to use Krum for Byzantine-robust aggregation.
            byzantine_threshold: Number of Byzantine clients to tolerate (f).
            round_timeout: Timeout for each round in seconds.
        """
        self.model = initial_model
        self.x_test = x_test
        self.y_test = y_test
        self.use_krum = use_krum
        self.byzantine_threshold = byzantine_threshold
        self.round_timeout = round_timeout
        
        # Initialize blockchain connector
        self.blockchain = BlockchainConnector(
            contract_address=contract_address,
            private_key=private_key,
            node_url=node_url,
        )
        
        # Initialize IPFS connector for model storage
        self.ipfs = ModelIPFSConnector(api_url=ipfs_url)
        
        # Get current blockchain state
        self.system_status = self.blockchain.get_system_status()
        
        # Store metrics for different rounds
        self.metrics_history = {}
    
    def create_task(self, total_rounds: int) -> int:
        """Create a new federated learning task on the blockchain.
        
        Args:
            total_rounds: Total number of rounds for the task.
            
        Returns:
            The ID of the created task.
        """
        # Upload initial model to IPFS
        ipfs_result = self.ipfs.upload_model(
            self.model.get_weights(),
            model_id="initial_model",
            metadata={"round": 0, "type": "initial"},
        )
        
        initial_model_hash = ipfs_result["Hash"]
        
        # Create task on blockchain
        task_id = self.blockchain.createTask(initial_model_hash, total_rounds)
        
        # Update system status
        self.system_status = self.blockchain.get_system_status()
        
        return task_id
    
    def start_round(self, task_id: int) -> int:
        """Start a new training round on the blockchain.
        
        Args:
            task_id: ID of the task to start a round for.
            
        Returns:
            The ID of the started round.
        """
        # Start round on blockchain
        round_id = self.blockchain.start_round(task_id)
        
        # Update system status
        self.system_status = self.blockchain.get_system_status()
        
        return round_id
    
    def select_clients(self, round_id: int, client_ids: List[int]) -> bool:
        """Select clients to participate in the current round.
        
        Args:
            round_id: ID of the current round.
            client_ids: List of client IDs to select.
            
        Returns:
            True if clients were successfully selected.
        """
        # Select clients on blockchain
        return self.blockchain.selectClients(round_id, client_ids)
    
    def apply_defense(self, round_id: int) -> int:
        """Apply Byzantine-robust defense mechanism.
        
        Args:
            round_id: ID of the current round.
            
        Returns:
            The ID of the selected client.
        """
        if self.use_krum:
            # Apply Krum defense on blockchain
            selected_client_id = self.blockchain.applyKrumDefense(round_id)
            return selected_client_id
        else:
            # No defense, return 0
            return 0
    
    def complete_round(self, round_id: int, global_model_hash: str) -> bool:
        """Complete the current training round on the blockchain.
        
        Args:
            round_id: ID of the current round.
            global_model_hash: IPFS hash of the global model.
            
        Returns:
            True if round was successfully completed.
        """
        # Update global model on blockchain
        self.blockchain.updateGlobalModel(round_id, global_model_hash)
        
        # Complete round on blockchain
        return self.blockchain.complete_round(round_id)
    
    def complete_task(self, task_id: int, final_model_hash: str) -> bool:
        """Complete the federated learning task on the blockchain.
        
        Args:
            task_id: ID of the task to complete.
            final_model_hash: IPFS hash of the final global model.
            
        Returns:
            True if task was successfully completed.
        """
        # Complete task on blockchain
        return self.blockchain.completeTask(task_id, final_model_hash)
    
    def distribute_rewards(self, client_ids: List[int], round_id: int) -> bool:
        """Distribute rewards to clients for their contributions.
        
        Args:
            client_ids: List of client IDs to reward.
            round_id: ID of the round for which to distribute rewards.
            
        Returns:
            True if rewards were successfully distributed.
        """
        # Distribute rewards on blockchain
        return self.blockchain.distributeRewards(client_ids, round_id)
    
    def evaluate_model(self, parameters: NDArrays) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on the server's test dataset.
        
        Args:
            parameters: Model parameters to evaluate.
            
        Returns:
            Tuple of (loss, metrics).
        """
        if self.x_test is None or self.y_test is None:
            # No test data available
            return 0.0, {"accuracy": 0.0}
        
        # Set model parameters
        self.model.set_weights(parameters)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Return evaluation results
        return float(loss), {"accuracy": float(accuracy)}
    
    def run_federated_learning(
        self,
        num_rounds: int,
        num_clients: int,
        min_clients: int,
        client_manager: fl.server.client_manager.ClientManager,
        fraction_fit: float = 1.0,
        server_address: str = "[::]:8080",
    ) -> Dict:
        """Run the federated learning process.
        
        Args:
            num_rounds: Number of federated learning rounds to run.
            num_clients: Number of clients to expect.
            min_clients: Minimum number of clients required.
            client_manager: Flower client manager.
            fraction_fit: Fraction of clients to sample for training.
            server_address: Server address for Flower server.
            
        Returns:
            Dictionary containing the federated learning history.
        """
        # Create a new task on blockchain
        task_id = self.create_task(num_rounds)
        
        # Define the evaluation function for the strategy
        def evaluate_fn(weights):
            return self.evaluate_model(weights)
        
        # Get the appropriate strategy
        strategy = get_strategy(
            evaluate_fn=evaluate_fn,
            fraction_fit=fraction_fit,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            use_krum=self.use_krum,
            byzantine_threshold=self.byzantine_threshold
        )
        
        # Start Flower server
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_manager=client_manager,
        )
        
        # Complete the task on blockchain
        final_model_weights = self.model.get_weights()
        final_model_hash = self.ipfs.upload_model(
            final_model_weights,
            model_id=f"final_model_task_{task_id}",
            metadata={"task_id": task_id, "type": "final"},
        )["Hash"]
        
        self.complete_task(task_id, final_model_hash)
        
        return history