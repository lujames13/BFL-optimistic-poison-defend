"""Dataset preparation for federated learning.

This module handles loading and partitioning datasets for federated learning
with support for creating both IID and non-IID partitions.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl.dataset_preparation")


def load_dataset(
    dataset_name: str,
    num_clients: int,
    iid: bool = True,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load and partition a dataset for federated learning.
    
    Args:
        dataset_name: Name of the dataset to load ('mnist', 'fashion_mnist', 'cifar10').
        num_clients: Number of clients to partition the data for.
        iid: Whether to use IID (independent and identically distributed) splitting.
        test_ratio: Ratio of data to use for testing (server-side evaluation).
        val_ratio: Ratio of each client's data to use for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of lists (x_trains, y_trains, x_vals, y_vals) where each list
        contains the data for each client.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Load the appropriate dataset
    x_all, y_all = _load_raw_dataset(dataset_name)
    
    # Split into train and test
    x_train, y_train, x_test, y_test = _split_train_test(x_all, y_all, test_ratio)
    
    # Partition training data for clients
    if iid:
        x_trains, y_trains = _partition_data_iid(x_train, y_train, num_clients)
    else:
        x_trains, y_trains = _partition_data_non_iid(x_train, y_train, num_clients, dataset_name)
    
    # Create validation set for each client
    x_vals, y_vals = [], []
    for i in range(num_clients):
        x_train_i, y_train_i, x_val_i, y_val_i = _split_train_val(
            x_trains[i], y_trains[i], val_ratio
        )
        x_trains[i] = x_train_i
        y_trains[i] = y_train_i
        x_vals.append(x_val_i)
        y_vals.append(y_val_i)
    
    logger.info(f"Loaded {dataset_name} and partitioned for {num_clients} clients (IID: {iid})")
    logger.info(f"Data shapes for client 0: x_train: {x_trains[0].shape}, y_train: {y_trains[0].shape}, "
               f"x_val: {x_vals[0].shape}, y_val: {y_vals[0].shape}")
    
    return x_trains, y_trains, x_vals, y_vals


def _load_raw_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the raw dataset.
    
    Args:
        dataset_name: Name of the dataset to load.
        
    Returns:
        Tuple of (x, y) data.
    """
    if dataset_name == "mnist":
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        # Normalize pixel values
        x_train = x_train.astype("float32") / 255.0
        # Reshape for MLP
        x_train = x_train.reshape(x_train.shape[0], -1)
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        # Normalize pixel values
        x_train = x_train.astype("float32") / 255.0
        # Reshape for MLP
        x_train = x_train.reshape(x_train.shape[0], -1)
    elif dataset_name == "cifar10":
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        # Normalize pixel values
        x_train = x_train.astype("float32") / 255.0
        # Flatten labels
        y_train = y_train.flatten()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return x_train, y_train


def _split_train_test(
    x: np.ndarray, y: np.ndarray, test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and test sets.
    
    Args:
        x: Feature data.
        y: Label data.
        test_ratio: Ratio of data to use for testing.
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    # Calculate split index
    split_idx = int(len(x) * (1 - test_ratio))
    
    # Shuffle data (with the same random seed for x and y)
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    # Split data
    x_train, x_test = x_shuffled[:split_idx], x_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    return x_train, y_train, x_test, y_test


def _split_train_val(
    x: np.ndarray, y: np.ndarray, val_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and validation sets.
    
    Args:
        x: Feature data.
        y: Label data.
        val_ratio: Ratio of data to use for validation.
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val).
    """
    # Calculate split index
    split_idx = int(len(x) * (1 - val_ratio))
    
    # Shuffle data (with the same random seed for x and y)
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    # Split data
    x_train, x_val = x_shuffled[:split_idx], x_shuffled[split_idx:]
    y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    return x_train, y_train, x_val, y_val


def _partition_data_iid(
    x: np.ndarray, y: np.ndarray, num_clients: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Partition data in an IID (independent and identically distributed) manner.
    
    Args:
        x: Feature data.
        y: Label data.
        num_clients: Number of clients to partition the data for.
        
    Returns:
        Tuple of lists (x_partitions, y_partitions) containing the partitioned data.
    """
    # Shuffle data
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    # Calculate number of samples per client
    samples_per_client = len(x) // num_clients
    
    # Create partitions
    x_partitions = []
    y_partitions = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(x)
        
        x_partitions.append(x_shuffled[start_idx:end_idx])
        y_partitions.append(y_shuffled[start_idx:end_idx])
    
    return x_partitions, y_partitions


def _partition_data_non_iid(
    x: np.ndarray, y: np.ndarray, num_clients: int, dataset_name: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Partition data in a non-IID manner (skewed class distribution).
    
    Args:
        x: Feature data.
        y: Label data.
        num_clients: Number of clients to partition the data for.
        dataset_name: Name of the dataset (to determine number of classes).
        
    Returns:
        Tuple of lists (x_partitions, y_partitions) containing the partitioned data.
    """
    # Determine number of classes based on dataset
    if dataset_name in ["mnist", "fashion_mnist", "cifar10"]:
        num_classes = 10
    else:
        # Count unique classes
        num_classes = len(np.unique(y))
    
    # Create class indices
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    # Determine classes per client
    # Each client gets a subset of classes (2 for 10-class datasets)
    classes_per_client = max(2, num_classes // (num_clients // 2))
    
    # Create partitions
    x_partitions = []
    y_partitions = []
    
    # Assign classes to clients
    for i in range(num_clients):
        # Select classes for this client
        client_classes = np.random.choice(
            range(num_classes), 
            size=classes_per_client, 
            replace=False
        )
        
        # Collect indices for the selected classes
        client_indices = np.concatenate([class_indices[c] for c in client_classes])
        
        # Shuffle the indices
        np.random.shuffle(client_indices)
        
        # Limit number of samples per client to balance dataset sizes
        samples_per_client = len(x) // (num_clients * classes_per_client // 2)
        client_indices = client_indices[:samples_per_client]
        
        # Create client partition
        x_partitions.append(x[client_indices])
        y_partitions.append(y[client_indices])
    
    return x_partitions, y_partitions


def create_label_flipping_attack(
    x: np.ndarray, y: np.ndarray, flip_ratio: float = 0.5, num_classes: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a label flipping attack.
    
    Args:
        x: Feature data.
        y: Label data.
        flip_ratio: Ratio of labels to flip.
        num_classes: Number of classes in the dataset.
        
    Returns:
        Tuple of (x, y) with flipped labels.
    """
    # Determine number of examples to flip
    n_samples = len(y)
    n_to_flip = int(n_samples * flip_ratio)
    
    # Randomly select indices to flip
    indices_to_flip = np.random.choice(n_samples, n_to_flip, replace=False)
    
    # Create a flipped version of the labels
    y_flipped = y.copy()
    for idx in indices_to_flip:
        # Flip to a different class
        current_label = y[idx]
        # Choose a different label randomly
        new_label = (current_label + np.random.randint(1, num_classes)) % num_classes
        y_flipped[idx] = new_label
    
    return x, y_flipped


def create_model_replacement_attack(
    x: np.ndarray, y: np.ndarray, target_class: int = 0, scale_factor: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a model replacement attack.
    
    This attack doesn't modify the data but is used to generate malicious model updates
    in the client. This function is a placeholder to document the attack.
    
    Args:
        x: Feature data.
        y: Label data.
        target_class: Target class for the attack.
        scale_factor: Factor to scale the model updates.
        
    Returns:
        The original data (x, y).
    """
    # This attack is implemented in the ByzantineClient class
    # This function is a placeholder to document the attack
    return x, y