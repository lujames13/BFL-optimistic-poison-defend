"""Integration tests for Krum defense effectiveness.

This module tests the effectiveness of the Krum Byzantine-robust aggregation
against different attack scenarios.
"""

import unittest
import numpy as np
import tensorflow as tf
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_krum_defense")


class TestKrumDefenseEffectiveness(unittest.TestCase):
    """Test suite for Krum defense effectiveness against Byzantine attacks."""

    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create a simple model for testing
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy dataset
        self.x_train = np.random.random((100, 5))
        self.y_train = np.random.randint(0, 2, 100)
        self.x_test = np.random.random((50, 5))
        self.y_test = np.random.randint(0, 2, 50)
        
        # Create initial model weights
        self.initial_weights = self.model.get_weights()

    def test_label_flipping_attack(self):
        """Test Krum defense against label flipping attack."""
        from fl.defense import krum_select
        
        # Create honest updates
        honest_updates = []
        for i in range(8):
            # Create a clone of the model
            model_clone = tf.keras.models.clone_model(self.model)
            model_clone.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model_clone.set_weights(self.initial_weights)
            
            # Train on random subset of data
            indices = np.random.choice(len(self.x_train), 50, replace=False)
            x_subset = self.x_train[indices]
            y_subset = self.y_train[indices]
            
            model_clone.fit(x_subset, y_subset, epochs=1, verbose=0)
            
            # Get updated weights
            honest_updates.append(model_clone.get_weights())
        
        # Create label flipping attack updates
        malicious_updates = []
        for i in range(2):
            # Create a clone of the model
            model_clone = tf.keras.models.clone_model(self.model)
            model_clone.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model_clone.set_weights(self.initial_weights)
            
            # Train on random subset of data with flipped labels
            indices = np.random.choice(len(self.x_train), 50, replace=False)
            x_subset = self.x_train[indices]
            y_subset = (self.y_train[indices] + 1) % 2  # Flip labels
            
            model_clone.fit(x_subset, y_subset, epochs=1, verbose=0)
            
            # Get updated weights
            malicious_updates.append(model_clone.get_weights())
        
        # Combine updates
        all_updates = honest_updates + malicious_updates
        client_ids = list(range(1, len(all_updates) + 1))
        
        # Apply Krum defense
        selected_idx, selected_update = krum_select(all_updates, client_ids, byzantine_threshold=2)
        
        # Selected update should be from an honest client (index 0-7)
        self.assertLess(selected_idx, 8)
        
        # Evaluate the selected model
        self.model.set_weights(selected_update)
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # For comparison, evaluate with a simple average aggregation
        avg_weights = []
        for i in range(len(self.initial_weights)):
            # Simple average of all updates including malicious ones
            avg_layer = np.mean([update[i] for update in all_updates], axis=0)
            avg_weights.append(avg_layer)
        
        self.model.set_weights(avg_weights)
        _, avg_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Krum should be at least as accurate as simple average
        logger.info(f"Krum accuracy: {accuracy}, Simple average accuracy: {avg_accuracy}")
        self.assertGreaterEqual(accuracy, avg_accuracy)

    def test_model_replacement_attack(self):
        """Test Krum defense against model replacement attack."""
        from fl.defense import krum_select
        
        # Create honest updates
        honest_updates = []
        for i in range(8):
            # Create a clone of the model
            model_clone = tf.keras.models.clone_model(self.model)
            model_clone.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model_clone.set_weights(self.initial_weights)
            
            # Train on random subset of data
            indices = np.random.choice(len(self.x_train), 50, replace=False)
            x_subset = self.x_train[indices]
            y_subset = self.y_train[indices]
            
            model_clone.fit(x_subset, y_subset, epochs=1, verbose=0)
            
            # Get updated weights
            honest_updates.append(model_clone.get_weights())
        
        # Create model replacement attack updates
        malicious_updates = []
        for i in range(2):
            # Create a malicious update by scaling the initial weights
            scale_factor = 10.0
            malicious_update = []
            for layer in self.initial_weights:
                malicious_update.append(layer * scale_factor)
            
            malicious_updates.append(malicious_update)
        
        # Combine updates
        all_updates = honest_updates + malicious_updates
        client_ids = list(range(1, len(all_updates) + 1))
        
        # Apply Krum defense
        selected_idx, selected_update = krum_select(all_updates, client_ids, byzantine_threshold=2)
        
        # Selected update should be from an honest client (index 0-7)
        self.assertLess(selected_idx, 8)
        
        # Evaluate the selected model
        self.model.set_weights(selected_update)
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # For comparison, evaluate with a simple average aggregation
        avg_weights = []
        for i in range(len(self.initial_weights)):
            # Simple average of all updates including malicious ones
            avg_layer = np.mean([update[i] for update in all_updates], axis=0)
            avg_weights.append(avg_layer)
        
        self.model.set_weights(avg_weights)
        _, avg_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Krum should be at least as accurate as simple average
        logger.info(f"Krum accuracy: {accuracy}, Simple average accuracy: {avg_accuracy}")
        self.assertGreaterEqual(accuracy, avg_accuracy)

    def test_multi_krum(self):
        """Test Multi-Krum defense against Byzantine attacks."""
        from fl.defense import krum_select
        
        # Create honest updates
        honest_updates = []
        for i in range(8):
            # Create a clone of the model
            model_clone = tf.keras.models.clone_model(self.model)
            model_clone.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model_clone.set_weights(self.initial_weights)
            
            # Train on random subset of data
            indices = np.random.choice(len(self.x_train), 50, replace=False)
            x_subset = self.x_train[indices]
            y_subset = self.y_train[indices]
            
            model_clone.fit(x_subset, y_subset, epochs=1, verbose=0)
            
            # Get updated weights
            honest_updates.append(model_clone.get_weights())
        
        # Create malicious updates (random values)
        malicious_updates = []
        for i in range(2):
            # Create a malicious update with random values
            malicious_update = []
            for layer in self.initial_weights:
                malicious_update.append(np.random.normal(0, 10.0, layer.shape))
            
            malicious_updates.append(malicious_update)
        
        # Combine updates
        all_updates = honest_updates + malicious_updates
        client_ids = list(range(1, len(all_updates) + 1))
        
        # Apply Multi-Krum defense (select m-f updates)
        selected_indices, selected_updates = krum_select(
            all_updates, client_ids, byzantine_threshold=2, multi_krum=True, m=6
        )
        
        # All selected updates should be from honest clients (index 0-7)
        self.assertTrue(all(idx < 8 for idx in selected_indices))
        
        # Multi-Krum should select 6-2=4 updates
        self.assertEqual(len(selected_updates), 4)
        
        # Average the selected updates
        avg_weights = []
        for i in range(len(self.initial_weights)):
            # Average of selected updates
            avg_layer = np.mean([update[i] for update in selected_updates], axis=0)
            avg_weights.append(avg_layer)
        
        # Evaluate the Multi-Krum model
        self.model.set_weights(avg_weights)
        _, multi_krum_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # For comparison, evaluate with standard Krum
        _, selected_update = krum_select(all_updates, client_ids, byzantine_threshold=2)
        self.model.set_weights(selected_update)
        _, krum_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # For comparison, evaluate with a simple average aggregation
        simple_avg_weights = []
        for i in range(len(self.initial_weights)):
            # Simple average of all updates including malicious ones
            avg_layer = np.mean([update[i] for update in all_updates], axis=0)
            simple_avg_weights.append(avg_layer)
        
        self.model.set_weights(simple_avg_weights)
        _, avg_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Log results
        logger.info(f"Multi-Krum accuracy: {multi_krum_accuracy}")
        logger.info(f"Standard Krum accuracy: {krum_accuracy}")
        logger.info(f"Simple average accuracy: {avg_accuracy}")
        
        # Both Krum variants should be at least as accurate as simple average
        self.assertGreaterEqual(multi_krum_accuracy, avg_accuracy)
        self.assertGreaterEqual(krum_accuracy, avg_accuracy)


if __name__ == "__main__":
    unittest.main()