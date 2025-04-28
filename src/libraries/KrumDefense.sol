// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title KrumDefense
 * @dev Library for implementing the Krum Byzantine-robust aggregation algorithm
 * for federated learning model updates
 */
library KrumDefense {
    struct Distance {
        uint256 from;
        uint256 to;
        uint256 value;
    }
    
    struct KrumResult {
        uint256 selectedClientId;
        uint256 score;
    }
    
    /**
     * @dev Calculate distance between two model hashes
     * @param hash1 First model hash
     * @param hash2 Second model hash
     * @return Distance value
     */
    // 在 KrumDefense.sol 中，可以改進 calculateDistance 函數
    function calculateDistance(string memory hash1, string memory hash2) public pure returns (uint256) {
        // 首先檢查是否完全相同
        if (keccak256(abi.encodePacked(hash1)) == keccak256(abi.encodePacked(hash2))) {
            return 0;
        }
        
        // 使用 XOR 計算距離
        bytes32 h1 = keccak256(abi.encodePacked(hash1));
        bytes32 h2 = keccak256(abi.encodePacked(hash2));
        
        uint256 distance = 0;
        for (uint256 i = 0; i < 32; i++) {
            // 計算每個字節的 XOR 距離
            uint8 xor = uint8(h1[i] ^ h2[i]);
            distance += uint256(xor); // 累計距離
        }
        
        return distance;
    }
    
    /**
     * @dev Compute pairwise distances between all model updates
     * @param hashes Array of model update hashes
     * @param clientIds Array of client IDs corresponding to each hash
     * @return Array of pairwise distances
     */
    function computeDistances(
        string[] memory hashes, 
        uint256[] memory clientIds
    ) public pure returns (Distance[] memory) {
        uint256 n = hashes.length;
        require(n == clientIds.length, "Arrays must have same length");
        
        // Number of pairwise distances: n*(n-1)/2
        uint256 distanceCount = (n * (n - 1)) / 2;
        Distance[] memory distances = new Distance[](distanceCount);
        
        uint256 idx = 0;
        for (uint256 i = 0; i < n; i++) {
            for (uint256 j = i + 1; j < n; j++) {
                distances[idx].from = clientIds[i];
                distances[idx].to = clientIds[j];
                distances[idx].value = calculateDistance(hashes[i], hashes[j]);
                idx++;
            }
        }
        
        return distances;
    }
    
    /**
     * @dev Execute Krum algorithm to select the most representative model update
     * @param hashes Array of model update hashes
     * @param clientIds Array of client IDs corresponding to each hash
     * @param f Number of Byzantine clients to tolerate
     * @return Result containing the selected client ID and its score
     */
    function executeKrum(
        string[] memory hashes, 
        uint256[] memory clientIds,
        uint256 f
    ) public pure returns (KrumResult memory) {
        uint256 n = hashes.length;
        require(n == clientIds.length, "Arrays must have same length");
        
        // Krum requires at least 2f+3 clients to tolerate f Byzantine clients
        require(n >= 2 * f + 3, "Not enough clients for Krum with given f");
        
        // Compute distances
        Distance[] memory distances = computeDistances(hashes, clientIds);
        
        // For each client, compute sum of squared distances to n-f-1 nearest neighbors
        uint256[] memory scores = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            // Create array to store distances from client i to all other clients
            uint256[] memory clientDistances = new uint256[](n - 1);
            uint256 distIdx = 0;
            
            // Find all distances involving client i
            for (uint256 j = 0; j < n; j++) {
                if (i == j) continue;
                
                // Look for distance between i and j
                bool found = false;
                for (uint256 k = 0; k < distances.length; k++) {
                    if ((distances[k].from == clientIds[i] && distances[k].to == clientIds[j]) ||
                        (distances[k].from == clientIds[j] && distances[k].to == clientIds[i])) {
                        clientDistances[distIdx] = distances[k].value;
                        distIdx++;
                        found = true;
                        break;
                    }
                }
                
                require(found, "Distance not found");
            }
            
            // Sort distances (simple bubble sort - could be optimized)
            for (uint256 j = 0; j < clientDistances.length; j++) {
                for (uint256 k = 0; k < clientDistances.length - j - 1; k++) {
                    if (clientDistances[k] > clientDistances[k + 1]) {
                        uint256 temp = clientDistances[k];
                        clientDistances[k] = clientDistances[k + 1];
                        clientDistances[k + 1] = temp;
                    }
                }
            }
            
            // Compute score: sum of squared distances to n-f-1 nearest neighbors
            uint256 score = 0;
            uint256 neighborsToConsider = n - f - 1;
            for (uint256 j = 0; j < neighborsToConsider; j++) {
                score += clientDistances[j] * clientDistances[j]; // Square the distance
            }
            
            scores[i] = score;
        }
        
        // Find client with the smallest score (most central)
        uint256 minScore = type(uint256).max;
        uint256 selectedIdx = 0;
        
        for (uint256 i = 0; i < n; i++) {
            if (scores[i] < minScore) {
                minScore = scores[i];
                selectedIdx = i;
            }
        }
        
        KrumResult memory result;
        result.selectedClientId = clientIds[selectedIdx];
        result.score = minScore;
        
        return result;
    }
}