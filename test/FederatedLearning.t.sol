// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "forge-std/Test.sol";
import "../src/FederatedLearning.sol";

contract FederatedLearningTest is Test {
    FederatedLearning public flContract;
    address public admin;
    address public operator;
    address public client1;
    address public client2;
    
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    function setUp() public {
        admin = address(this);
        operator = address(0x3);
        client1 = address(0x1);
        client2 = address(0x2);
        
        // Deploy the FederatedLearning contract
        flContract = new FederatedLearning();
        
        // Initialize the contract
        flContract.initialize();
        
        // Grant operator role
        flContract.grantRole(OPERATOR_ROLE, operator);
    }

    function testInitialization() public {
        // Test initial state variables
        (uint256 totalClients, uint256 totalRounds, uint256 currentRound, uint8 currentRoundStatus) = flContract.getSystemStatus();
        
        assertEq(totalClients, 0, "Initial total clients should be 0");
        assertEq(totalRounds, 0, "Initial total rounds should be 0");
        assertEq(currentRound, 0, "Initial current round should be 0");
        assertEq(currentRoundStatus, uint8(FederatedLearning.RoundStatus.Inactive), "Initial round status should be Inactive");
        
        // Verify the admin is set correctly
        assertEq(flContract.admin(), admin, "Admin should be set to the deployer");
    }
    
    function testAccessControl() public {
        // Verify admin role
        assertTrue(flContract.hasRole(ADMIN_ROLE, admin), "Admin should have ADMIN_ROLE");
        assertTrue(flContract.hasRole(DEFAULT_ADMIN_ROLE, admin), "Admin should have DEFAULT_ADMIN_ROLE");
        
        // Verify operator role
        assertTrue(flContract.hasRole(OPERATOR_ROLE, operator), "Operator should have OPERATOR_ROLE");
        assertFalse(flContract.hasRole(ADMIN_ROLE, operator), "Operator should not have ADMIN_ROLE");
        
        // Test role revocation
        flContract.revokeRole(OPERATOR_ROLE, operator);
        assertFalse(flContract.hasRole(OPERATOR_ROLE, operator), "Operator should not have OPERATOR_ROLE after revocation");
    }
    
    function testAccessControlRestriction() public {
        // Try to initialize the contract from a non-admin account
        vm.prank(client1);
        vm.expectRevert("AccessControl: account 0x0000000000000000000000000000000000000001 is missing role 0xa49807205ce4d355092ef5a8a18f56e8913cf4a201fbe287825b095693c21775");
        flContract.initialize();
    }
    
    function testTaskCreation() public {
        // Create a federated learning task
        string memory initialModelHash = "QmInitialModelHash123";
        uint256 totalRounds = 10;
        
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.TaskCreated(1, initialModelHash);
        
        uint256 taskId = flContract.createTask(initialModelHash, totalRounds);
        
        // Verify task was created correctly
        assertEq(taskId, 1, "Task ID should be 1");
        assertEq(flContract.taskCount(), 1, "Task count should be 1");
        assertEq(flContract.currentTaskId(), 1, "Current task ID should be 1");
        
        // Get task information
        (
            uint256 storedTaskId,
            uint8 status,
            uint256 startTime,
            uint256 completedRounds,
            uint256 storedTotalRounds,
            string memory storedInitialModelHash,
            string memory currentModelHash
        ) = flContract.getTaskInfo(taskId);
        
        assertEq(storedTaskId, taskId, "Stored task ID should match");
        assertEq(status, uint8(FederatedLearning.TaskStatus.Active), "Task status should be Active");
        assertEq(completedRounds, 0, "Completed rounds should be 0");
        assertEq(storedTotalRounds, totalRounds, "Total rounds should match");
        assertEq(storedInitialModelHash, initialModelHash, "Initial model hash should match");
        assertEq(currentModelHash, initialModelHash, "Current model hash should initially match initial model hash");
    }
    
    function testTaskStatusManagement() public {
        // Create a task
        string memory initialModelHash = "QmInitialModelHash123";
        uint256 totalRounds = 5;
        uint256 taskId = flContract.createTask(initialModelHash, totalRounds);
        
        // Complete the task
        string memory finalModelHash = "QmFinalModelHash456";
        
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.TaskCompleted(taskId, finalModelHash);
        
        flContract.completeTask(taskId, finalModelHash);
        
        // Verify task status
        (
            ,
            uint8 status,
            ,
            ,
            ,
            ,
            string memory currentModelHash
        ) = flContract.getTaskInfo(taskId);
        
        assertEq(status, uint8(FederatedLearning.TaskStatus.Completed), "Task status should be Completed");
        assertEq(currentModelHash, finalModelHash, "Current model hash should be updated to final model hash");
    }
    
    function testTaskStatusRestrictions() public {
        // Create a task
        string memory initialModelHash = "QmInitialModelHash123";
        uint256 totalRounds = 5;
        uint256 taskId = flContract.createTask(initialModelHash, totalRounds);
        
        // Complete the task
        flContract.completeTask(taskId, "QmFinalModelHash456");
        
        // Try to complete an already completed task
        vm.expectRevert("Task is not active");
        flContract.completeTask(taskId, "QmNewFinalModelHash789");
        
        // Try to terminate a non-existent task
        vm.expectRevert("Task does not exist");
        flContract.terminateTask(999, "Termination reason");
        
        // Terminate the completed task
        flContract.terminateTask(taskId, "Testing termination");
        
        // Verify task status
        (
            uint256 _,
            uint8 status,
            uint256 __,
            uint256 ___,
            uint256 ____,
            string memory _____,
            
        ) = flContract.getTaskInfo(taskId);
        
        assertEq(status, uint8(FederatedLearning.TaskStatus.Terminated), "Task status should be Terminated");
    }
    
    function testRoundInitialization() public {
        // Create a task first
        string memory initialModelHash = "QmInitialModelHash123";
        uint256 totalRounds = 5;
        uint256 taskId = flContract.createTask(initialModelHash, totalRounds);
        
        // Start a new round
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.RoundStarted(1, block.timestamp);
        
        flContract.startRound(taskId);
        
        // Verify round was started correctly
        (uint256 totalClients, uint256 totalRounds, uint256 currentRoundId, uint8 currentRoundStatus) = flContract.getSystemStatus();
        
        assertEq(totalRounds, 1, "Total rounds should be 1");
        assertEq(currentRoundId, 1, "Current round ID should be 1");
        assertEq(currentRoundStatus, uint8(FederatedLearning.RoundStatus.Active), "Round status should be Active");
        
        // Get round information
        (
            uint256 roundId,
            uint8 status,
            uint256 startTime,
            uint256 endTime,
            uint256 participantCount,
            uint256 completedUpdates,
            string memory globalModelHash
        ) = flContract.getRoundInfo(1);
        
        assertEq(roundId, 1, "Round ID should be 1");
        assertEq(status, uint8(FederatedLearning.RoundStatus.Active), "Round status should be Active");
        assertGt(endTime, startTime, "End time should be greater than start time");
        assertEq(participantCount, 0, "Initial participant count should be 0");
        assertEq(completedUpdates, 0, "Initial completed updates should be 0");
        assertEq(globalModelHash, initialModelHash, "Global model hash should match initial model hash");
    }
    
    function testRoundCompletion() public {
        // Create a task
        string memory initialModelHash = "QmInitialModelHash123";
        uint256 taskId = flContract.createTask(initialModelHash, 5);
        
        // Start a round
        flContract.startRound(taskId);
        
        // Update global model (simulating aggregation)
        string memory newGlobalModelHash = "QmNewGlobalModelHash456";
        
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.GlobalModelUpdated(1, newGlobalModelHash);
        
        flContract.updateGlobalModel(1, newGlobalModelHash);
        
        // Complete the round
        vm.expectEmit(true, false, false, false);
        emit FederatedLearning.RoundCompleted(1, block.timestamp);
        
        flContract.completeRound(1);
        
        // Verify round status
        (
            uint256 _1,
            uint8 status,
            uint256 _2,
            uint256 _3,
            uint256 _4,
            uint256 _5,
            string memory _6
        ) = flContract.getRoundInfo(1);
        
        assertEq(status, uint8(FederatedLearning.RoundStatus.Completed), "Round status should be Completed");
        
        // Verify task was updated
        (
            uint256 unused1,
            uint8 unused2,
            uint256 unused3,
            uint256 completedRounds,
            uint256 unused4,
            string memory unused5,
            string memory currentModelHash
        ) = flContract.getTaskInfo(taskId);
        
        assertEq(completedRounds, 1, "Task completed rounds should be incremented");
        assertEq(currentModelHash, newGlobalModelHash, "Task current model hash should be updated");
    }
    
    function testRoundStatusTransitions() public {
        // Create a task
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        
        // Start a round
        flContract.startRound(taskId);
        
        // Try to start another round before completing the current one
        vm.expectRevert("Active round exists");
        flContract.startRound(taskId);
        
        // Complete the round (need to update global model first)
        flContract.updateGlobalModel(1, "QmNewGlobalModelHash");
        flContract.completeRound(1);
        
        // Start another round
        flContract.startRound(taskId);
        
        // Verify round ID
        (uint256 unused1, uint256 unused2, uint256 currentRoundId, uint8 unused3) = flContract.getSystemStatus();
        assertEq(currentRoundId, 2, "Current round ID should be 2");
    }
    
    function testClientRegistration() public {
        // Register clients
        vm.startPrank(client1);
        
        vm.expectEmit(true, true, false, false);
        emit FederatedLearning.ClientRegistered(1, client1);
        
        uint256 clientId1 = flContract.registerClient();
        vm.stopPrank();
        
        vm.startPrank(client2);
        uint256 clientId2 = flContract.registerClient();
        vm.stopPrank();
        
        // Verify client IDs
        assertEq(clientId1, 1, "First client ID should be 1");
        assertEq(clientId2, 2, "Second client ID should be 2");
        
        // Verify client count
        (uint256 totalClients, uint256 unused1, uint256 unused2, uint8 unused3) = flContract.getSystemStatus();
        assertEq(totalClients, 2, "Total clients should be 2");
        
        // Verify client info
        (
            address clientAddress,
            uint8 status,
            uint256 contributionScore,
            ,
            bool selectedForRound
        ) = flContract.getClientInfo(clientId1);
        
        assertEq(clientAddress, client1, "Client address should match");
        assertEq(status, uint8(FederatedLearning.ClientStatus.Registered), "Client status should be Registered");
        assertEq(contributionScore, 0, "Initial contribution score should be 0");
        assertEq(selectedForRound, false, "Client should not be selected for round initially");
        
        // Try to register again with same address
        vm.startPrank(client1);
        vm.expectRevert("Client already registered");
        flContract.registerClient();
        vm.stopPrank();
    }
    
    function testClientSelection() public {
        // Register clients
        vm.prank(client1);
        uint256 clientId1 = flContract.registerClient();
        
        vm.prank(client2);
        uint256 clientId2 = flContract.registerClient();
        
        // Create a task and start a round
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        flContract.startRound(taskId);
        
        // Select clients for the round
        uint256[] memory selectedClients = new uint256[](2);
        selectedClients[0] = clientId1;
        selectedClients[1] = clientId2;
        
        flContract.selectClients(1, selectedClients);
        
        // Verify clients are selected
        (
            ,
            ,
            ,
            ,
            bool selectedForRound1
        ) = flContract.getClientInfo(clientId1);
        
        (
            ,
            ,
            ,
            ,
            bool selectedForRound2
        ) = flContract.getClientInfo(clientId2);
        
        assertTrue(selectedForRound1, "Client 1 should be selected for round");
        assertTrue(selectedForRound2, "Client 2 should be selected for round");
    }
    
    function testClientContribution() public {
        // Register client
        vm.prank(client1);
        uint256 clientId = flContract.registerClient();
        
        // Create a task and start a round
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        flContract.startRound(taskId);
        
        // Select client for round
        uint256[] memory selectedClients = new uint256[](1);
        selectedClients[0] = clientId;
        flContract.selectClients(1, selectedClients);
        
        // Client submits an update
        vm.startPrank(client1);
        flContract.submitModelUpdate(clientId, 1, "QmModelUpdateHash123");
        vm.stopPrank();
        
        // Operator accepts the update
        flContract.acceptModelUpdate(clientId, 1);
        
        // Verify contribution score increased
        (
            address unused1,
            uint8 unused2,
            uint256 contributionScore,
            uint256 unused3,
            bool unused4
        ) = flContract.getClientInfo(clientId);
        
        assertGt(contributionScore, 0, "Contribution score should be increased");
    }
    
    function testKrumDefense() public {
        // Register 5 clients (need at least 2f+3=5 for Krum with f=1)
        address client3 = address(0x4);
        address client4 = address(0x5);
        address client5 = address(0x6);
        
        vm.prank(client1);
        uint256 clientId1 = flContract.registerClient();
        
        vm.prank(client2);
        uint256 clientId2 = flContract.registerClient();
        
        vm.prank(client3);
        uint256 clientId3 = flContract.registerClient();
        
        vm.prank(client4);
        uint256 clientId4 = flContract.registerClient();
        
        vm.prank(client5);
        uint256 clientId5 = flContract.registerClient();
        
        // Create a task and start a round
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        flContract.startRound(taskId);
        
        // Select all clients for the round
        uint256[] memory selectedClients = new uint256[](5);
        selectedClients[0] = clientId1;
        selectedClients[1] = clientId2;
        selectedClients[2] = clientId3;
        selectedClients[3] = clientId4;
        selectedClients[4] = clientId5;
        flContract.selectClients(1, selectedClients);
        
        // Clients submit updates:
        // - Clients 1, 2, 3 submit similar models (honest)
        // - Clients 4, 5 submit malicious models (very different)
        vm.prank(client1);
        flContract.submitModelUpdate(clientId1, 1, "QmHonestModel1");
        
        vm.prank(client2);
        flContract.submitModelUpdate(clientId2, 1, "QmHonestModel2");
        
        vm.prank(client3);
        flContract.submitModelUpdate(clientId3, 1, "QmHonestModel3");
        
        vm.prank(client4);
        flContract.submitModelUpdate(clientId4, 1, "QmMaliciousModel1");
        
        vm.prank(client5);
        flContract.submitModelUpdate(clientId5, 1, "QmMaliciousModel2");
        
        // Apply Krum defense
        uint256 selectedClientId = flContract.applyKrumDefense(1);
        
        // The selected client should be one of the honest clients
        assertTrue(
            selectedClientId == clientId1 || 
            selectedClientId == clientId2 || 
            selectedClientId == clientId3,
            "Krum should select one of the honest clients"
        );
        
        // Verify that the selected client's update was accepted
        string memory selectedModelHash = flContract.getModelUpdateHash(selectedClientId, 1);
        assertTrue(
            keccak256(abi.encodePacked(selectedModelHash)) == keccak256(abi.encodePacked("QmHonestModel1")) ||
            keccak256(abi.encodePacked(selectedModelHash)) == keccak256(abi.encodePacked("QmHonestModel2")) ||
            keccak256(abi.encodePacked(selectedModelHash)) == keccak256(abi.encodePacked("QmHonestModel3")),
            "Selected model should be one of the honest models"
        );
    }
    
    function testRewardSystem() public {
        // Register client
        vm.prank(client1);
        uint256 clientId = flContract.registerClient();
        
        // Create a task and start a round
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        flContract.startRound(taskId);
        
        // Select client for round
        uint256[] memory selectedClients = new uint256[](1);
        selectedClients[0] = clientId;
        flContract.selectClients(1, selectedClients);
        
        // Client submits an update
        vm.prank(client1);
        flContract.submitModelUpdate(clientId, 1, "QmModelUpdateHash123");
        
        // Operator accepts the update
        flContract.acceptModelUpdate(clientId, 1);
        
        // Update global model and complete the round
        flContract.updateGlobalModel(1, "QmNewGlobalModelHash");
        flContract.completeRound(1);
        
        // Check reward calculation
        uint256 calculatedReward = flContract.calculateReward(clientId);
        assertGt(calculatedReward, 0, "Reward calculation should return a positive amount");
        
        // Distribute reward to client
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.RewardDistributed(clientId, calculatedReward);
        
        uint256 rewardAmount = flContract.rewardClient(clientId, 1);
        assertEq(rewardAmount, calculatedReward, "Distributed reward should match calculated reward");
        
        // Check pending rewards
        uint256 pendingRewards = flContract.getPendingRewards(client1);
        assertEq(pendingRewards, calculatedReward, "Pending rewards should match distributed reward");
        
        // Client claims rewards
        vm.prank(client1);
        vm.expectEmit(true, false, false, true);
        emit FederatedLearning.RewardsClaimed(client1, pendingRewards);
        
        uint256 claimedAmount = flContract.claimRewards();
        assertEq(claimedAmount, calculatedReward, "Claimed amount should match distributed reward");
        
        // Verify rewards are reset after claiming
        uint256 remainingRewards = flContract.getPendingRewards(client1);
        assertEq(remainingRewards, 0, "Rewards should be zero after claiming");
    }
    
    function testRewardDistribution() public {
        // Register multiple clients
        address client3 = address(0x4);
        address client4 = address(0x5);
        
        vm.prank(client1);
        uint256 clientId1 = flContract.registerClient();
        
        vm.prank(client2);
        uint256 clientId2 = flContract.registerClient();
        
        vm.prank(client3);
        uint256 clientId3 = flContract.registerClient();
        
        vm.prank(client4);
        uint256 clientId4 = flContract.registerClient();
        
        // Create a task and start a round
        uint256 taskId = flContract.createTask("QmInitialModelHash", 5);
        flContract.startRound(taskId);
        
        // Select all clients for the round
        uint256[] memory selectedClients = new uint256[](4);
        selectedClients[0] = clientId1;
        selectedClients[1] = clientId2;
        selectedClients[2] = clientId3;
        selectedClients[3] = clientId4;
        flContract.selectClients(1, selectedClients);
        
        // Clients submit updates
        vm.prank(client1);
        flContract.submitModelUpdate(clientId1, 1, "QmModelUpdate1");
        
        vm.prank(client2);
        flContract.submitModelUpdate(clientId2, 1, "QmModelUpdate2");
        
        vm.prank(client3);
        flContract.submitModelUpdate(clientId3, 1, "QmModelUpdate3");
        
        // Client 4 does not submit
        
        // Accept some updates
        flContract.acceptModelUpdate(clientId1, 1);
        flContract.acceptModelUpdate(clientId3, 1);
        
        // Complete the round
        flContract.updateGlobalModel(1, "QmNewGlobalModelHash");
        flContract.completeRound(1);
        
        // Distribute rewards to all clients
        uint256[] memory clientIds = new uint256[](4);
        clientIds[0] = clientId1;
        clientIds[1] = clientId2;
        clientIds[2] = clientId3;
        clientIds[3] = clientId4;
        
        flContract.distributeRewards(clientIds, 1);
        
        // Check pending rewards - only clients 1 and 3 should have rewards
        assertGt(flContract.getPendingRewards(client1), 0, "Client 1 should have rewards");
        assertEq(flContract.getPendingRewards(client2), 0, "Client 2 should not have rewards");
        assertGt(flContract.getPendingRewards(client3), 0, "Client 3 should have rewards");
        assertEq(flContract.getPendingRewards(client4), 0, "Client 4 should not have rewards");
    }
}