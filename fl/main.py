"""Main entry point for running the Blockchain Federated Learning system.

This module provides the entry point for running a federated learning process
with blockchain integration, Byzantine-robust aggregation, and IPFS storage.
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

from fl.server import BlockchainFlowerServer
from fl.client import BlockchainFlowerClient, ByzantineClient
from fl.dataset_preparation import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl.main")

# Ensure TensorFlow logs are not too verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the Blockchain Federated Learning system.
    
    Args:
        cfg: Hydra configuration.
    """
    # Print the configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Check if we're in client or server mode
    if cfg.mode == "server":
        run_server(cfg)
    elif cfg.mode == "client":
        run_client(cfg)
    elif cfg.mode == "simulation":
        run_simulation(cfg)
    else:
        logger.error(f"Unknown mode: {cfg.mode}")


def run_server(cfg: DictConfig) -> None:
    """Run the server component of the BFL system.
    
    Args:
        cfg: Hydra configuration.
    """
    logger.info("Starting server...")
    
    # Load model architecture
    model = get_model(cfg.model)
    
    # Load test dataset for server-side evaluation (if enabled)
    x_test, y_test = None, None
    if cfg.server.evaluate:
        x_test, y_test, _, _ = load_dataset(
            dataset_name=cfg.dataset.name,
            num_clients=1,  # Only need one partition for the server
            iid=True,  # IID split is fine for the server test set
            test_ratio=cfg.dataset.test_ratio
        )
        # Take the first (and only) partition
        x_test, y_test = x_test[0], y_test[0]
    
    # Create server instance
    server = BlockchainFlowerServer(
        initial_model=model,
        contract_address=cfg.blockchain.contract_address,
        ipfs_url=cfg.ipfs.url,
        private_key=cfg.blockchain.private_key,
        node_url=cfg.blockchain.node_url,
        x_test=x_test,
        y_test=y_test,
        use_krum=cfg.defense.use_krum,
        byzantine_threshold=cfg.defense.byzantine_threshold,
        round_timeout=cfg.server.round_timeout
    )
    
    # Prepare client manager
    client_manager = fl.server.SimpleClientManager()
    
    # Run federated learning
    history = server.run_federated_learning(
        num_rounds=cfg.fl.num_rounds,
        num_clients=cfg.fl.num_clients,
        min_clients=cfg.fl.min_clients,
        client_manager=client_manager,
        fraction_fit=cfg.fl.fraction_fit,
        server_address=cfg.server.address
    )
    
    logger.info(f"Server completed {cfg.fl.num_rounds} rounds of federated learning")
    logger.info(f"Final metrics: {history.metrics_centralized}")


def run_client(cfg: DictConfig) -> None:
    """Run a client component of the BFL system.
    
    Args:
        cfg: Hydra configuration.
    """
    client_id = cfg.client.id
    logger.info(f"Starting client {client_id}...")
    
    # Load model architecture
    model = get_model(cfg.model)
    
    # Load dataset partition for this client
    x_train, y_train, x_val, y_val = load_client_data(
        cfg.dataset.name, 
        cfg.fl.num_clients, 
        client_id,
        cfg.dataset.iid,
        cfg.dataset.val_ratio
    )
    
    # Create client based on type
    if cfg.client.byzantine and cfg.client.attack_type:
        logger.info(f"Creating Byzantine client with attack: {cfg.client.attack_type}")
        client = ByzantineClient(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            client_id=client_id,
            contract_address=cfg.blockchain.contract_address,
            ipfs_url=cfg.ipfs.url,
            private_key=cfg.blockchain.private_key,
            node_url=cfg.blockchain.node_url,
            attack_type=cfg.client.attack_type,
            attack_params=cfg.client.attack_params
        )
    else:
        logger.info("Creating honest client")
        client = BlockchainFlowerClient(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            client_id=client_id,
            contract_address=cfg.blockchain.contract_address,
            ipfs_url=cfg.ipfs.url,
            private_key=cfg.blockchain.private_key,
            node_url=cfg.blockchain.node_url
        )
    
    # Convert NumPyClient to Client
    fl_client = client.to_client()
    
    # Start client
    fl.client.start_client(
        server_address=cfg.server.address,
        client=fl_client
    )
    
    logger.info(f"Client {client_id} completed federated learning")


def run_simulation(cfg: DictConfig) -> None:
    """Run a local simulation of the BFL system.
    
    Args:
        cfg: Hydra configuration.
    """
    logger.info("Starting simulation...")
    
    # Load model architecture
    model = get_model(cfg.model)
    
    # Load dataset
    clients_data = load_dataset(
        dataset_name=cfg.dataset.name,
        num_clients=cfg.fl.num_clients,
        iid=cfg.dataset.iid,
        test_ratio=cfg.dataset.test_ratio,
        val_ratio=cfg.dataset.val_ratio
    )
    
    if cfg.simulation.use_ray:
        # Run simulation with Ray
        run_ray_simulation(cfg, model, clients_data)
    else:
        # Run simulation with standard Flower
        run_standard_simulation(cfg, model, clients_data)
    
    logger.info("Simulation completed")


def run_ray_simulation(cfg: DictConfig, model: tf.keras.Model, clients_data: Tuple) -> None:
    """Run simulation using Flower's Ray integration.
    
    Args:
        cfg: Hydra configuration.
        model: TensorFlow model to train.
        clients_data: Tuple containing client datasets.
    """
    import ray
    from flwr.simulation import start_simulation
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(include_dashboard=cfg.simulation.ray_dashboard)
    
    # Extract data
    x_trains, y_trains, x_vals, y_vals = clients_data
    
    # Create client function
    def client_fn(cid: str) -> fl.client.Client:
        # Convert client ID from string to int
        client_id = int(cid)
        
        # Determine if this client should be Byzantine
        is_byzantine = client_id in cfg.simulation.byzantine_clients
        
        if is_byzantine:
            logger.info(f"Creating Byzantine client {client_id} with attack: {cfg.client.attack_type}")
            client = ByzantineClient(
                model=get_model(cfg.model),  # New model instance per client
                x_train=x_trains[client_id],
                y_train=y_trains[client_id],
                x_val=x_vals[client_id],
                y_val=y_vals[client_id],
                client_id=client_id,
                contract_address=cfg.blockchain.contract_address,
                ipfs_url=cfg.ipfs.url,
                attack_type=cfg.client.attack_type,
                attack_params=cfg.client.attack_params
            )
        else:
            client = BlockchainFlowerClient(
                model=get_model(cfg.model),  # New model instance per client
                x_train=x_trains[client_id],
                y_train=y_trains[client_id],
                x_val=x_vals[client_id],
                y_val=y_vals[client_id],
                client_id=client_id,
                contract_address=cfg.blockchain.contract_address,
                ipfs_url=cfg.ipfs.url
            )
        
        return client.to_client()
    
    # Load test dataset for server-side evaluation
    x_test, y_test = load_test_dataset(cfg.dataset.name)
    
    # Create server
    server = BlockchainFlowerServer(
        initial_model=model,
        contract_address=cfg.blockchain.contract_address,
        ipfs_url=cfg.ipfs.url,
        x_test=x_test,
        y_test=y_test,
        use_krum=cfg.defense.use_krum,
        byzantine_threshold=cfg.defense.byzantine_threshold
    )
    
    # Define the evaluation function for the strategy
    def evaluate_fn(weights):
        return server.evaluate_model(weights)
    
    # Get the appropriate strategy
    from fl.server import get_strategy
    strategy = get_strategy(
        evaluate_fn=evaluate_fn,
        fraction_fit=cfg.fl.fraction_fit,
        min_fit_clients=cfg.fl.min_clients,
        min_available_clients=cfg.fl.min_clients,
        use_krum=cfg.defense.use_krum,
        byzantine_threshold=cfg.defense.byzantine_threshold
    )
    
    # Start simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fl.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        strategy=strategy,
        client_resources=cfg.simulation.client_resources
    )
    
    logger.info(f"Simulation completed {cfg.fl.num_rounds} rounds of federated learning")
    logger.info(f"Final metrics: {history.metrics_centralized}")


def run_standard_simulation(cfg: DictConfig, model: tf.keras.Model, clients_data: Tuple) -> None:
    """Run simulation using standard Flower (without Ray).
    
    Args:
        cfg: Hydra configuration.
        model: TensorFlow model to train.
        clients_data: Tuple containing client datasets.
    """
    # Extract data
    x_trains, y_trains, x_vals, y_vals = clients_data
    
    # Load test dataset for server-side evaluation
    x_test, y_test = load_test_dataset(cfg.dataset.name)
    
    # Create clients
    clients = []
    for client_id in range(cfg.fl.num_clients):
        # Determine if this client should be Byzantine
        is_byzantine = client_id in cfg.simulation.byzantine_clients
        
        if is_byzantine:
            logger.info(f"Creating Byzantine client {client_id} with attack: {cfg.client.attack_type}")
            client = ByzantineClient(
                model=get_model(cfg.model),  # New model instance per client
                x_train=x_trains[client_id],
                y_train=y_trains[client_id],
                x_val=x_vals[client_id],
                y_val=y_vals[client_id],
                client_id=client_id,
                contract_address=cfg.blockchain.contract_address,
                ipfs_url=cfg.ipfs.url,
                attack_type=cfg.client.attack_type,
                attack_params=cfg.client.attack_params
            )
        else:
            client = BlockchainFlowerClient(
                model=get_model(cfg.model),  # New model instance per client
                x_train=x_trains[client_id],
                y_train=y_trains[client_id],
                x_val=x_vals[client_id],
                y_val=y_vals[client_id],
                client_id=client_id,
                contract_address=cfg.blockchain.contract_address,
                ipfs_url=cfg.ipfs.url
            )
        
        clients.append(client)
    
    # Create server
    server = BlockchainFlowerServer(
        initial_model=model,
        contract_address=cfg.blockchain.contract_address,
        ipfs_url=cfg.ipfs.url,
        x_test=x_test,
        y_test=y_test,
        use_krum=cfg.defense.use_krum,
        byzantine_threshold=cfg.defense.byzantine_threshold
    )
    
    # Create a task on blockchain
    task_id = server.create_task(total_rounds=cfg.fl.num_rounds)
    
    # Run federated learning simulation round by round
    for round_num in range(1, cfg.fl.num_rounds + 1):
        logger.info(f"Starting round {round_num}...")
        
        # Start a new round on blockchain
        round_id = server.start_round(task_id)
        
        # Select clients for this round (all clients in simulation)
        client_ids = list(range(cfg.fl.num_clients))
        server.select_clients(round_id, client_ids)
        
        # Get global model parameters
        global_params = server.model.get_weights()
        
        # Train on each client
        updated_params_list = []
        for client in clients:
            # Train
            updated_params, _, _ = client.fit(
                global_params, 
                {"current_round": round_num, "local_epochs": cfg.fl.local_epochs}
            )
            updated_params_list.append(updated_params)
        
        # Apply Krum defense
        if cfg.defense.use_krum:
            selected_client_id = server.apply_defense(round_id)
            logger.info(f"Krum selected client {selected_client_id}")
            
            # Update global model with selected client's model
            if selected_client_id > 0:
                selected_idx = selected_client_id - 1  # Convert to 0-indexed
                server.model.set_weights(updated_params_list[selected_idx])
        else:
            # Simple average aggregation
            avg_params = [np.zeros_like(param) for param in global_params]
            for params in updated_params_list:
                for i, param in enumerate(params):
                    avg_params[i] += param / len(clients)
            
            # Update global model
            server.model.set_weights(avg_params)
        
        # Upload new global model to IPFS
        ipfs_result = server.ipfs.upload_model(
            server.model.get_weights(),
            model_id=f"global_model_round_{round_num}",
            metadata={"round": round_num, "type": "global"}
        )
        
        # Complete the round on blockchain
        server.complete_round(round_id, ipfs_result["Hash"])
        
        # Evaluate global model
        loss, metrics = server.evaluate_model(server.model.get_weights())
        logger.info(f"Round {round_num} completed. Loss: {loss}, Accuracy: {metrics['accuracy']}")
    
    # Complete the task on blockchain
    final_ipfs_result = server.ipfs.upload_model(
        server.model.get_weights(),
        model_id=f"final_model_task_{task_id}",
        metadata={"task_id": task_id, "type": "final"}
    )
    
    server.complete_task(task_id, final_ipfs_result["Hash"])
    logger.info(f"Task {task_id} completed")


def get_model(model_config: DictConfig) -> tf.keras.Model:
    """Create a TensorFlow model based on configuration.
    
    Args:
        model_config: Model configuration.
        
    Returns:
        TensorFlow model instance.
    """
    if model_config.name == "mlp":
        # Build a simple MLP model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(model_config.input_shape,)),
            tf.keras.layers.Dense(model_config.hidden_layers[0], activation="relu"),
            tf.keras.layers.Dense(model_config.hidden_layers[1], activation="relu"),
            tf.keras.layers.Dense(model_config.num_classes, activation="softmax")
        ])
    elif model_config.name == "cnn":
        # Build a simple CNN model for image data
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=model_config.input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(model_config.num_classes, activation="softmax")
        ])
    else:
        raise ValueError(f"Unknown model type: {model_config.name}")
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def load_client_data(dataset_name: str, num_clients: int, client_id: int, iid: bool, val_ratio: float) -> Tuple:
    """Load dataset partition for a specific client.
    
    Args:
        dataset_name: Name of the dataset to load.
        num_clients: Total number of clients.
        client_id: ID of the client to load data for.
        iid: Whether to use IID (independent and identically distributed) splitting.
        val_ratio: Ratio of data to use for validation.
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val).
    """
    # Load the dataset for all clients
    x_trains, y_trains, x_vals, y_vals = load_dataset(
        dataset_name=dataset_name,
        num_clients=num_clients,
        iid=iid,
        val_ratio=val_ratio
    )
    
    # Return the partition for the specified client
    return x_trains[client_id], y_trains[client_id], x_vals[client_id], y_vals[client_id]


def load_test_dataset(dataset_name: str) -> Tuple:
    """Load test dataset for server-side evaluation.
    
    Args:
        dataset_name: Name of the dataset to load.
        
    Returns:
        Tuple of (x_test, y_test).
    """
    if dataset_name == "mnist":
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Normalize pixel values
        x_test = x_test.astype("float32") / 255.0
        # Reshape for MLP
        x_test = x_test.reshape(x_test.shape[0], -1)
    elif dataset_name == "fashion_mnist":
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # Normalize pixel values
        x_test = x_test.astype("float32") / 255.0
        # Reshape for MLP
        x_test = x_test.reshape(x_test.shape[0], -1)
    elif dataset_name == "cifar10":
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # Normalize pixel values
        x_test = x_test.astype("float32") / 255.0
        # Flatten labels
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return x_test, y_test


if __name__ == "__main__":
    main()