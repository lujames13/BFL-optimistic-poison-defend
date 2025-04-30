"""Test suite for Krum Byzantine-robust aggregation mechanism."""

import unittest
import numpy as np
from typing import List, Dict, Tuple

# Import the module we're testing (will implement later)
# from fl.defense import krum_select, calculate_distances, compute_krum_score


class TestKrumDefense(unittest.TestCase):
    """Test suite for Krum Byzantine-robust aggregation algorithm."""

    def setUp(self):
        """Set up test data."""
        # Create 5 simple model updates (as NumPy arrays)
        # 4 honest updates (close to each other) and 1 malicious (far from others)
        self.honest_updates = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1]),
            np.array([0.9, 1.9, 2.9, 3.9]),
            np.array([1.05, 2.05, 3.05, 4.05]),
        ]
        
        self.malicious_update = np.array([10.0, 20.0, 30.0, 40.0])
        
        self.all_updates = self.honest_updates + [self.malicious_update]
        self.all_client_ids = [1, 2, 3, 4, 5]
        
        # Expected distances between the updates
        # We'll verify our distance calculation against these
        self.expected_distance_01 = np.sqrt(0.04)  # Distance between update 0 and 1
        self.expected_distance_04 = np.sqrt(3774.04)  # Distance between update 0 and 4 (malicious)

    def test_calculate_distances(self):
        """Test the distance calculation between model updates."""
        from fl.defense import calculate_distances
        
        # Calculate distances between all pairs of updates
        distances = calculate_distances(self.all_updates)
        
        # Check the distance calculation for a few pairs
        distance_01 = distances[(0, 1)]  # Distance between first and second update
        self.assertAlmostEqual(distance_01, self.expected_distance_01, places=5)
        
        distance_04 = distances[(0, 4)]  # Distance between first and malicious update
        self.assertAlmostEqual(distance_04, self.expected_distance_04, places=5)
        
        # Verify the distance matrix is symmetric
        self.assertEqual(distances[(0, 1)], distances[(1, 0)])
        self.assertEqual(distances[(0, 4)], distances[(4, 0)])
        
        # Verify all pairs are calculated
        self.assertEqual(len(distances), (len(self.all_updates) * (len(self.all_updates) - 1)) // 2)

    def test_compute_krum_score(self):
        """Test the Krum score computation for each update."""
        from fl.defense import compute_krum_score
        
        # With 5 clients and f=1 Byzantine, we consider n-f-2 = 2 nearest neighbors
        f = 1
        scores = compute_krum_score(self.all_updates, f)
        
        # Malicious update should have the highest score (worst)
        self.assertEqual(np.argmax(scores), 4)
        
        # One of the honest updates should have the lowest score (best)
        best_idx = np.argmin(scores)
        self.assertIn(best_idx, [0, 1, 2, 3])
        
        # Verify that scores are non-negative
        self.assertTrue(all(score >= 0 for score in scores))
        
        # Check that we get an error if f is too large
        with self.assertRaises(ValueError):
            compute_krum_score(self.all_updates, 3)  # f too large for n=5

    def test_krum_select(self):
        """Test the Krum selection function that returns the best update."""
        from fl.defense import krum_select
        
        # With 5 clients and f=1 Byzantine, we select the best update
        f = 1
        selected_idx, selected_update = krum_select(self.all_updates, self.all_client_ids, f)
        
        # Selected update should be one of the honest ones
        self.assertIn(selected_idx, [0, 1, 2, 3])
        
        # The selected client ID should match
        self.assertEqual(self.all_client_ids[selected_idx], selected_idx + 1)
        
        # The selected update should be from honest updates
        np.testing.assert_array_equal(selected_update, self.honest_updates[selected_idx])
        
        # Test the multi-krum version (selects m-f updates)
        m = 3  # Select 3-1=2 updates
        selected_indices, selected_updates = krum_select(
            self.all_updates, self.all_client_ids, f, multi_krum=True, m=m
        )
        
        # Should select m-f = 2 updates
        self.assertEqual(len(selected_indices), m - f)
        self.assertEqual(len(selected_updates), m - f)
        
        # All selected updates should be from honest clients
        for idx in selected_indices:
            self.assertIn(idx, [0, 1, 2, 3])

    def test_krum_with_weights(self):
        """Test Krum with client weight information."""
        from fl.defense import krum_select
        
        # Define weights for each client (e.g., based on data size)
        weights = [10, 20, 15, 5, 10]  # Weights for clients 1-5
        
        # With 5 clients and f=1 Byzantine, select the best update considering weights
        f = 1
        selected_idx, selected_update = krum_select(
            self.all_updates, self.all_client_ids, f, weights=weights
        )
        
        # Selected update should still be one of the honest ones
        self.assertIn(selected_idx, [0, 1, 2, 3])
        
        # The selected update should be from honest updates
        np.testing.assert_array_equal(selected_update, self.honest_updates[selected_idx])
        
        # Client with more data (client 2, index 1) should be preferred if scores are close
        # This is a probabilistic test, so it might not always select client 2
        # But we run it to make sure the function handles weights without errors


if __name__ == "__main__":
    unittest.main()