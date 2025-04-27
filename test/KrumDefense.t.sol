
pragma solidity ^0.8.17;

import "forge-std/Test.sol";
import "../src/libraries/KrumDefense.sol";

contract KrumDefenseTest is Test {
    using KrumDefense for *;
    
    function setUp() public {}
    
    function testCalculateDistance() public {
        string memory hash1 = "QmHash1";
        string memory hash2 = "QmHash2";
        string memory hash3 = "QmHash1"; // Same as hash1
        
        uint256 distance1 = KrumDefense.calculateDistance(hash1, hash2);
        uint256 distance2 = KrumDefense.calculateDistance(hash1, hash3);
        uint256 distance3 = KrumDefense.calculateDistance(hash2, hash3);
        
        // Same hash should have zero distance
        assertEq(distance2, 0, "Distance between identical hashes should be 0");
        
        // Different hashes should have non-zero distance
        assertGt(distance1, 0, "Distance between different hashes should be non-zero");
        assertGt(distance3, 0, "Distance between different hashes should be non-zero");
        
        // Distance should be symmetric
        assertEq(
            KrumDefense.calculateDistance(hash1, hash2),
            KrumDefense.calculateDistance(hash2, hash1),
            "Distance calculation should be symmetric"
        );
    }
    
    function testComputeDistances() public {
        string[] memory hashes = new string[](3);
        hashes[0] = "QmHash1";
        hashes[1] = "QmHash2";
        hashes[2] = "QmHash3";
        
        uint256[] memory clientIds = new uint256[](3);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        
        KrumDefense.Distance[] memory distances = KrumDefense.computeDistances(hashes, clientIds);
        
        // Should have 3 distances (for 3 clients)
        assertEq(distances.length, 3, "Should compute 3 distances for 3 clients");
        
        // Verify distances are between correct clients
        assertEq(distances[0].from, 1, "First distance should be from client 1");
        assertEq(distances[0].to, 2, "First distance should be to client 2");
        
        assertEq(distances[1].from, 1, "Second distance should be from client 1");
        assertEq(distances[1].to, 3, "Second distance should be to client 3");
        
        assertEq(distances[2].from, 2, "Third distance should be from client 2");
        assertEq(distances[2].to, 3, "Third distance should be to client 3");
    }
    
    function testExecuteKrum() public {
        string[] memory hashes = new string[](5);
        hashes[0] = "QmModelHash1";   // Normal model
        hashes[1] = "QmModelHash2";   // Normal model (slightly different)
        hashes[2] = "QmModelHash3";   // Normal model (slightly different)
        hashes[3] = "QmMaliciousHash";  // Malicious model (very different)
        hashes[4] = "QmAnotherMaliciousHash"; // Another malicious model
        
        uint256[] memory clientIds = new uint256[](5);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        clientIds[3] = 4;
        clientIds[4] = 5;
        
        // Execute Krum with f=1 (1 Byzantine client)
        KrumDefense.KrumResult memory result = KrumDefense.executeKrum(hashes, clientIds, 1);
        
        // The selected client should be one of the normal models (1, 2, or 3)
        assertTrue(
            result.selectedClientId == 1 || 
            result.selectedClientId == 2 || 
            result.selectedClientId == 3,
            "Krum should select one of the normal clients"
        );
        
        // It shouldn't select the malicious models
        assertTrue(
            result.selectedClientId != 4 && 
            result.selectedClientId != 5,
            "Krum should not select a malicious client"
        );
    }
    
    function testKrumWithNotEnoughClients() public {
        string[] memory hashes = new string[](3);
        hashes[0] = "QmHash1";
        hashes[1] = "QmHash2";
        hashes[2] = "QmHash3";
        
        uint256[] memory clientIds = new uint256[](3);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        
        // Try to execute Krum with f=1, which requires at least 2*f+3=5 clients
        vm.expectRevert("Not enough clients for Krum with given f");
        KrumDefense.executeKrum(hashes, clientIds, 1);
    }
}// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "forge-std/Test.sol";
import "../src/libraries/KrumDefense.sol";

contract KrumDefenseTest is Test {
    using KrumDefense for *;
    
    function setUp() public {}
    
    function testCalculateDistance() public {
        string memory hash1 = "QmHash1";
        string memory hash2 = "QmHash2";
        string memory hash3 = "QmHash1"; // Same as hash1
        
        uint256 distance1 = KrumDefense.calculateDistance(hash1, hash2);
        uint256 distance2 = KrumDefense.calculateDistance(hash1, hash3);
        uint256 distance3 = KrumDefense.calculateDistance(hash2, hash3);
        
        // Same hash should have zero distance
        assertEq(distance2, 0, "Distance between identical hashes should be 0");
        
        // Different hashes should have non-zero distance
        assertGt(distance1, 0, "Distance between different hashes should be non-zero");
        assertGt(distance3, 0, "Distance between different hashes should be non-zero");
        
        // Distance should be symmetric
        assertEq(
            KrumDefense.calculateDistance(hash1, hash2),
            KrumDefense.calculateDistance(hash2, hash1),
            "Distance calculation should be symmetric"
        );
    }
    
    function testComputeDistances() public {
        string[] memory hashes = new string[](3);
        hashes[0] = "QmHash1";
        hashes[1] = "QmHash2";
        hashes[2] = "QmHash3";
        
        uint256[] memory clientIds = new uint256[](3);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        
        KrumDefense.Distance[] memory distances = KrumDefense.computeDistances(hashes, clientIds);
        
        // Should have 3 distances (for 3 clients)
        assertEq(distances.length, 3, "Should compute 3 distances for 3 clients");
        
        // Verify distances are between correct clients
        assertEq(distances[0].from, 1, "First distance should be from client 1");
        assertEq(distances[0].to, 2, "First distance should be to client 2");
        
        assertEq(distances[1].from, 1, "Second distance should be from client 1");
        assertEq(distances[1].to, 3, "Second distance should be to client 3");
        
        assertEq(distances[2].from, 2, "Third distance should be from client 2");
        assertEq(distances[2].to, 3, "Third distance should be to client 3");
    }
    
    function testExecuteKrum() public {
        string[] memory hashes = new string[](5);
        hashes[0] = "QmModelHash1";   // Normal model
        hashes[1] = "QmModelHash2";   // Normal model (slightly different)
        hashes[2] = "QmModelHash3";   // Normal model (slightly different)
        hashes[3] = "QmMaliciousHash";  // Malicious model (very different)
        hashes[4] = "QmAnotherMaliciousHash"; // Another malicious model
        
        uint256[] memory clientIds = new uint256[](5);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        clientIds[3] = 4;
        clientIds[4] = 5;
        
        // Execute Krum with f=1 (1 Byzantine client)
        KrumDefense.KrumResult memory result = KrumDefense.executeKrum(hashes, clientIds, 1);
        
        // The selected client should be one of the normal models (1, 2, or 3)
        assertTrue(
            result.selectedClientId == 1 || 
            result.selectedClientId == 2 || 
            result.selectedClientId == 3,
            "Krum should select one of the normal clients"
        );
        
        // It shouldn't select the malicious models
        assertTrue(
            result.selectedClientId != 4 && 
            result.selectedClientId != 5,
            "Krum should not select a malicious client"
        );
    }
    
    function testKrumWithNotEnoughClients() public {
        string[] memory hashes = new string[](3);
        hashes[0] = "QmHash1";
        hashes[1] = "QmHash2";
        hashes[2] = "QmHash3";
        
        uint256[] memory clientIds = new uint256[](3);
        clientIds[0] = 1;
        clientIds[1] = 2;
        clientIds[2] = 3;
        
        // Try to execute Krum with f=1, which requires at least 2*f+3=5 clients
        vm.expectRevert("Not enough clients for Krum with given f");
        KrumDefense.executeKrum(hashes, clientIds, 1);
    }
}