"""Model definitions for federated learning.

This module provides model architectures for federated learning tasks,
including MLP and CNN models for different datasets.
"""

import tensorflow as tf
from typing import List, Optional, Tuple, Union


def get_mlp_model(
    input_shape: int,
    hidden_layers: List[int],
    num_classes: int,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Create a simple MLP model.
    
    Args:
        input_shape: Input shape (flattened).
        hidden_layers: List of hidden layer sizes.
        num_classes: Number of output classes.
        learning_rate: Learning rate for the Adam optimizer.
        
    Returns:
        Compiled TensorFlow model.
    """
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
    
    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def get_cnn_model(
    input_shape: Union[int, Tuple[int, int, int]],
    num_classes: int,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Create a simple CNN model for image data.
    
    Args:
        input_shape: Input shape (H, W, C) for images or flattened shape for MLP.
        num_classes: Number of output classes.
        learning_rate: Learning rate for the Adam optimizer.
        
    Returns:
        Compiled TensorFlow model.
    """
    # Determine if we're dealing with flattened input
    if isinstance(input_shape, int):
        # Assume square image for simplicity
        side_length = int(input_shape ** 0.5)
        channels = 1  # Assume grayscale
        input_shape = (side_length, side_length, channels)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def get_model(
    model_type: str,
    input_shape: Union[int, Tuple[int, int, int]],
    hidden_layers: Optional[List[int]] = None,
    num_classes: int = 10,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Get a model based on the specified type.
    
    Args:
        model_type: Type of model to create ("mlp" or "cnn").
        input_shape: Input shape for the model.
        hidden_layers: List of hidden layer sizes (for MLP).
        num_classes: Number of output classes.
        learning_rate: Learning rate for the optimizer.
        
    Returns:
        Compiled TensorFlow model.
    """
    if model_type.lower() == "mlp":
        if hidden_layers is None:
            hidden_layers = [128, 64]
        return get_mlp_model(input_shape, hidden_layers, num_classes, learning_rate)
    elif model_type.lower() == "cnn":
        return get_cnn_model(input_shape, num_classes, learning_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")