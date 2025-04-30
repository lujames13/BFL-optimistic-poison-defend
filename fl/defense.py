"""Byzantine-robust defense mechanism implementations for Federated Learning.

This module implements the Krum algorithm, a Byzantine-robust aggregation
rule for federated learning that is resilient to poisoning attacks.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional


def calculate_distances(updates: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
    """Calculate pairwise Euclidean distances between model updates.
    
    Args:
        updates: List of model parameter updates as numpy arrays.
        
    Returns:
        Dictionary mapping update pairs (i,j) to their Euclidean distance.
    """
    n = len(updates)
    distances = {}
    
    # Calculate distances between all pairs of updates
    for i in range(n):
        for j in range(i+1, n):
            # Ensure the updates are flattened for distance calculation
            flat_update_i = updates[i].flatten()
            flat_update_j = updates[j].flatten()
            
            # Calculate Euclidean distance
            dist = np.linalg.norm(flat_update_i - flat_update_j)
            
            # Store distance in both directions for convenience
            distances[(i, j)] = dist
            distances[(j, i)] = dist
    
    return distances


def compute_krum_score(updates: List[np.ndarray], f: int) -> np.ndarray:
    """Compute the Krum score for each update.
    
    The Krum score for an update is the sum of distances to its n-f-2 nearest neighbors.
    Lower score is better.
    
    Args:
        updates: List of model parameter updates.
        f: Number of Byzantine clients to tolerate.
        
    Returns:
        Array of Krum scores for each update.
    
    Raises:
        ValueError: If the number of Byzantine clients f is too large relative
                   to the total number of clients.
    """
    n = len(updates)
    
    # Krum requires at least 2f+3 clients to tolerate f Byzantine clients
    if n < 2*f + 3:
        raise ValueError(
            f"Not enough clients ({n}) for Krum with f={f}. Need at least {2*f+3} clients."
        )
    
    # Calculate distances between all pairs of updates
    distances = calculate_distances(updates)
    
    # Calculate Krum score for each update
    scores = np.zeros(n)
    for i in range(n):
        # Get distances from update i to all other updates
        client_distances = [distances[(i, j)] for j in range(n) if j != i]
        
        # Sort distances to find n-f-2 nearest neighbors
        client_distances.sort()
        
        # Sum the distances to the n-f-2 nearest neighbors
        nearest_neighbors_count = n - f - 2
        scores[i] = sum(client_distances[:nearest_neighbors_count])
    
    return scores


def krum_select(
    updates: List[np.ndarray], 
    client_ids: List[int], 
    f: int, 
    multi_krum: bool = False, 
    m: int = None, 
    weights: Optional[List[float]] = None
) -> Union[Tuple[int, np.ndarray], Tuple[List[int], List[np.ndarray]]]:
    """Select the most representative model update using the Krum algorithm.
    
    Args:
        updates: List of model parameter updates.
        client_ids: List of client IDs corresponding to the updates.
        f: Number of Byzantine clients to tolerate.
        multi_krum: If True, use Multi-Krum to select multiple updates.
        m: For Multi-Krum, the number of candidates to consider.
        weights: Optional list of weights for each client (e.g., based on data size).
                If provided, weights are considered when scores are close.
        
    Returns:
        If multi_krum is False:
            Tuple of (selected_index, selected_update)
        If multi_krum is True:
            Tuple of (list of selected indices, list of selected updates)
    
    Raises:
        ValueError: If f is too large relative to the number of clients.
    """
    n = len(updates)
    
    # Compute Krum scores for all updates
    scores = compute_krum_score(updates, f)
    
    # Apply weight adjustments if provided
    if weights is not None:
        # Small adjustment based on weights (scale by max score for stability)
        # This gives slight preference to clients with more data when scores are close
        max_score = max(scores)
        weight_factor = max_score * 0.01  # Small factor to keep weight influence limited
        normalized_weights = np.array(weights) / max(weights)
        score_adjustments = weight_factor * (1 - normalized_weights)
        scores = scores + score_adjustments
    
    if not multi_krum:
        # Standard Krum: select the update with the lowest score
        selected_idx = np.argmin(scores)
        return selected_idx, updates[selected_idx]
    else:
        # Multi-Krum: select m-f updates with the lowest scores
        if m is None:
            m = n  # Default to all clients
        
        if m <= f:
            raise ValueError(f"For Multi-Krum, m ({m}) must be greater than f ({f})")
        
        # Get indices of updates sorted by score (best to worst)
        sorted_indices = np.argsort(scores)
        
        # Select the m-f best updates
        num_to_select = m - f
        selected_indices = sorted_indices[:num_to_select].tolist()
        selected_updates = [updates[i] for i in selected_indices]
        
        return selected_indices, selected_updates