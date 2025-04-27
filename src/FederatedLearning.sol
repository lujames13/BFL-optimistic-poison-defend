// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./libraries/KrumDefense.sol";

/**
 * @title FederatedLearning
 * @dev Smart contract for managing decentralized federated learning tasks on Arbitrum
 */
contract FederatedLearning is AccessControl, ReentrancyGuard {
    // Roles
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    
    // Public state variables
    address public admin;
    uint256 public minClientParticipation;
    uint256 public roundDuration;
    uint256 public challengePeriod;
    uint256 public baseReward; // Base reward amount
    mapping(address => uint256) public pendingRewards; // Rewards waiting to be claimed
    
    // Enums
    enum ClientStatus { Unregistered, Registered, Active, Suspended }
    enum RoundStatus { Inactive, Active, Aggregating, Completed, Challenged }
    enum TaskStatus { Inactive, Active, Completed, Terminated }
    
    // Structs
    struct Client {
        address clientAddress;
        ClientStatus status;
        uint256 contributionScore;
        uint256 lastUpdateTimestamp;
        bool selectedForRound;
        mapping(uint256 => bool) roundParticipation; // Round ID => participated
    }
    
    struct ModelUpdate {
        uint256 clientId;
        uint256 roundId;
        string modelUpdateHash;
        uint256 timestamp;
        bool accepted;
    }
    
    struct Round {
        uint256 roundId;
        RoundStatus status;
        uint256 startTime;
        uint256 endTime;
        uint256 participantCount;
        uint256 completedUpdates;
        string globalModelHash;
        mapping(uint256 => bool) clientParticipation; // Client ID => participated
        mapping(uint256 => ModelUpdate) updates; // Client ID => update
    }
    
    struct Task {
        uint256 taskId;
        TaskStatus status;
        uint256 startTime;
        uint256 completedRounds;
        uint256 totalRounds;
        string initialModelHash;
        string currentModelHash;
    }
    
    // Mappings
    mapping(uint256 => Client) public clients;
    mapping(uint256 => Round) public rounds;
    mapping(uint256 => Task) public tasks;
    
    // Counters
    uint256 public clientCount;
    uint256 public roundCount;
    uint256 public taskCount;
    uint256 public currentTaskId;
    
    // System state
    uint256 public currentRound;
    
    // Krum parameters
    uint256 public byzantineClientsToTolerate;
    
    // Events
    event ClientRegistered(uint256 indexed clientId, address indexed clientAddress);
    event RoundStarted(uint256 indexed roundId, uint256 startTime);
    event RoundCompleted(uint256 indexed roundId, uint256 endTime);
    event ModelUpdateSubmitted(uint256 indexed clientId, uint256 indexed roundId, string modelUpdateHash);
    event ModelUpdateAccepted(uint256 indexed clientId, uint256 indexed roundId);
    event GlobalModelUpdated(uint256 indexed roundId, string globalModelHash);
    event TaskCreated(uint256 indexed taskId, string initialModelHash);
    event TaskCompleted(uint256 indexed taskId, string finalModelHash);
    event TaskTerminated(uint256 indexed taskId, string reason);
    event RewardDistributed(uint256 indexed clientId, uint256 amount);
    event RewardsClaimed(address indexed clientAddress, uint256 amount);
    
    /**
     * @dev Constructor that sets the admin role to the deployer
     */
    constructor() {
        admin = msg.sender;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }
    
    /**
     * @dev Initialize contract with default settings
     */
    function initialize() external onlyRole(ADMIN_ROLE) {
        minClientParticipation = 3;
        roundDuration = 1 days;
        challengePeriod = 7 days;
        byzantineClientsToTolerate = 1; // Default: tolerate 1 Byzantine client
        baseReward = 10; // Base reward amount
        
        // Reset counters (in case of re-initialization)
        clientCount = 0;
        roundCount = 0;
        taskCount = 0;
        currentRound = 0;
        currentTaskId = 0;
    }
    
    /**
     * @dev Get system status information
     * @return totalClients Number of registered clients
     * @return totalRounds Number of completed rounds
     * @return currentRoundId Current round ID
     * @return currentRoundStatus Status of current round
     */
    function getSystemStatus() external view returns (
        uint256 totalClients,
        uint256 totalRounds,
        uint256 currentRoundId,
        uint8 currentRoundStatus
    ) {
        totalClients = clientCount;
        totalRounds = roundCount;
        currentRoundId = currentRound;
        
        if (currentRound > 0) {
            currentRoundStatus = uint8(rounds[currentRound].status);
        } else {
            currentRoundStatus = uint8(RoundStatus.Inactive);
        }
    }
    
    /**
     * @dev Get client information
     * @param clientId ID of the client
     * @return clientAddress Address of the client
     * @return status Status of the client
     * @return contributionScore Contribution score of the client
     * @return lastUpdateTimestamp Last time the client submitted an update
     * @return selectedForRound Whether the client is selected for the current round
     */
    function getClientInfo(uint256 clientId) external view returns (
        address clientAddress,
        uint8 status,
        uint256 contributionScore,
        uint256 lastUpdateTimestamp,
        bool selectedForRound
    ) {
        Client storage client = clients[clientId];
        clientAddress = client.clientAddress;
        status = uint8(client.status);
        contributionScore = client.contributionScore;
        lastUpdateTimestamp = client.lastUpdateTimestamp;
        selectedForRound = client.selectedForRound;
    }
    
    /**
     * @dev Get task information
     * @param taskId ID of the task
     * @return taskId ID of the task
     * @return status Status of the task
     * @return startTime Start time of the task
     * @return completedRounds Number of completed rounds
     * @return totalRounds Total number of rounds for the task
     * @return initialModelHash Hash of the initial model
     * @return currentModelHash Hash of the current global model
     */
    function getTaskInfo(uint256 taskId) external view returns (
        uint256,
        uint8 status,
        uint256 startTime,
        uint256 completedRounds,
        uint256 totalRounds,
        string memory initialModelHash,
        string memory currentModelHash
    ) {
        Task storage task = tasks[taskId];
        require(task.taskId == taskId, "Task does not exist");
        
        return (
            task.taskId,
            uint8(task.status),
            task.startTime,
            task.completedRounds,
            task.totalRounds,
            task.initialModelHash,
            task.currentModelHash
        );
    }
    
    /**
     * @dev Create a new federated learning task
     * @param initialModelHash IPFS hash of the initial model
     * @param totalRounds Total number of rounds for the task
     * @return taskId ID of the created task
     */
    function createTask(string calldata initialModelHash, uint256 totalRounds) 
        external 
        onlyRole(ADMIN_ROLE) 
        returns (uint256) 
    {
        require(bytes(initialModelHash).length > 0, "Initial model hash cannot be empty");
        require(totalRounds > 0, "Total rounds must be greater than 0");
        
        // Increment task count
        taskCount++;
        uint256 taskId = taskCount;
        
        // Create new task
        Task storage task = tasks[taskId];
        task.taskId = taskId;
        task.status = TaskStatus.Active;
        task.startTime = block.timestamp;
        task.completedRounds = 0;
        task.totalRounds = totalRounds;
        task.initialModelHash = initialModelHash;
        task.currentModelHash = initialModelHash; // Initially same as initial model
        
        // Update current task ID
        currentTaskId = taskId;
        
        emit TaskCreated(taskId, initialModelHash);
        
        return taskId;
    }
    
    /**
     * @dev Complete a federated learning task
     * @param taskId ID of the task to complete
     * @param finalModelHash IPFS hash of the final global model
     */
    function completeTask(uint256 taskId, string calldata finalModelHash) 
        external 
        onlyRole(ADMIN_ROLE) 
        taskIsActive(taskId)
    {
        require(bytes(finalModelHash).length > 0, "Final model hash cannot be empty");
        
        Task storage task = tasks[taskId];
        
        // Update task status
        task.status = TaskStatus.Completed;
        task.currentModelHash = finalModelHash;
        
        emit TaskCompleted(taskId, finalModelHash);
    }
    
    /**
     * @dev Terminate a federated learning task (for emergency or when issues are detected)
     * @param taskId ID of the task to terminate
     * @param reason Reason for termination
     */
    function terminateTask(uint256 taskId, string calldata reason) 
        external 
        onlyRole(ADMIN_ROLE) 
    {
        Task storage task = tasks[taskId];
        require(task.taskId == taskId, "Task does not exist");
        
        // Update task status
        task.status = TaskStatus.Terminated;
        
        emit TaskTerminated(taskId, reason);
    }
    
    /**
     * @dev Get round information
     * @param roundId ID of the round
     * @return roundId ID of the round
     * @return status Status of the round
     * @return startTime Start time of the round
     * @return endTime End time of the round
     * @return participantCount Number of participants
     * @return completedUpdates Number of completed updates
     * @return globalModelHash Hash of the global model for this round
     */
    function getRoundInfo(uint256 roundId) external view returns (
        uint256,
        uint8 status,
        uint256 startTime,
        uint256 endTime,
        uint256 participantCount,
        uint256 completedUpdates,
        string memory globalModelHash
    ) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        
        return (
            round.roundId,
            uint8(round.status),
            round.startTime,
            round.endTime,
            round.participantCount,
            round.completedUpdates,
            round.globalModelHash
        );
    }
    
    /**
     * @dev Start a new training round
     * @param taskId ID of the task
     * @return roundId ID of the new round
     */
    function startRound(uint256 taskId) external onlyRole(OPERATOR_ROLE) returns (uint256) {
        Task storage task = tasks[taskId];
        require(task.taskId == taskId, "Task does not exist");
        require(task.status == TaskStatus.Active, "Task is not active");
        
        // Check if we've reached the total rounds
        require(task.completedRounds < task.totalRounds, "All rounds completed");
        
        // Ensure no active round exists
        if (currentRound > 0) {
            require(rounds[currentRound].status != RoundStatus.Active, "Active round exists");
        }
        
        // Increment round count
        roundCount++;
        uint256 roundId = roundCount;
        
        // Create new round
        Round storage round = rounds[roundId];
        round.roundId = roundId;
        round.status = RoundStatus.Active;
        round.startTime = block.timestamp;
        round.endTime = block.timestamp + roundDuration;
        round.participantCount = 0;
        round.completedUpdates = 0;
        round.globalModelHash = task.currentModelHash;
        
        // Update current round
        currentRound = roundId;
        
        emit RoundStarted(roundId, block.timestamp);
        
        return roundId;
    }
    
    /**
     * @dev Update the global model for a round
     * @param roundId ID of the round
     * @param newGlobalModelHash IPFS hash of the new global model
     */
    function updateGlobalModel(uint256 roundId, string calldata newGlobalModelHash) 
        external 
        onlyRole(OPERATOR_ROLE) 
    {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.status == RoundStatus.Active, "Round is not active");
        require(bytes(newGlobalModelHash).length > 0, "Global model hash cannot be empty");
        
        // Update global model hash
        round.globalModelHash = newGlobalModelHash;
        
        emit GlobalModelUpdated(roundId, newGlobalModelHash);
    }
    
    /**
     * @dev Complete a training round
     * @param roundId ID of the round to complete
     */
    function completeRound(uint256 roundId) external onlyRole(OPERATOR_ROLE) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.status == RoundStatus.Active, "Round is not active");
        
        // Find associated task
        uint256 taskId = currentTaskId;
        Task storage task = tasks[taskId];
        require(task.taskId == taskId, "Task does not exist");
        
        // Update round status
        round.status = RoundStatus.Completed;
        round.endTime = block.timestamp;
        
        // Update task
        task.completedRounds++;
        task.currentModelHash = round.globalModelHash;
        
        emit RoundCompleted(roundId, block.timestamp);
        
        // If all rounds completed, complete the task
        if (task.completedRounds >= task.totalRounds) {
            task.status = TaskStatus.Completed;
            emit TaskCompleted(taskId, task.currentModelHash);
        }
    }
    
    /**
     * @dev Register a new client
     * @return clientId ID of the registered client
     */
    function registerClient() external returns (uint256) {
        // Check that client is not already registered
        for (uint256 i = 1; i <= clientCount; i++) {
            if (clients[i].clientAddress == msg.sender) {
                revert("Client already registered");
            }
        }
        
        // Increment client count
        clientCount++;
        uint256 clientId = clientCount;
        
        // Create new client
        Client storage client = clients[clientId];
        client.clientAddress = msg.sender;
        client.status = ClientStatus.Registered;
        client.contributionScore = 0;
        client.lastUpdateTimestamp = 0;
        client.selectedForRound = false;
        
        emit ClientRegistered(clientId, msg.sender);
        
        return clientId;
    }
    
    /**
     * @dev Select clients for the current round
     * @param roundId ID of the round
     * @param clientIds Array of client IDs to select
     */
    function selectClients(uint256 roundId, uint256[] calldata clientIds) external onlyRole(OPERATOR_ROLE) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.status == RoundStatus.Active, "Round is not active");
        
        // Reset all clients' selection status
        for (uint256 i = 1; i <= clientCount; i++) {
            clients[i].selectedForRound = false;
        }
        
        // Select specified clients
        for (uint256 i = 0; i < clientIds.length; i++) {
            uint256 clientId = clientIds[i];
            require(clientId > 0 && clientId <= clientCount, "Invalid client ID");
            require(clients[clientId].status == ClientStatus.Registered || 
                    clients[clientId].status == ClientStatus.Active, 
                    "Client is not in active status");
            
            clients[clientId].selectedForRound = true;
        }
    }
    
    /**
     * @dev Submit a model update for the current round
     * @param clientId ID of the client
     * @param roundId ID of the round
     * @param modelUpdateHash IPFS hash of the model update
     */
    function submitModelUpdate(
        uint256 clientId, 
        uint256 roundId, 
        string calldata modelUpdateHash
    ) 
        external 
        onlyRegisteredClient(clientId)
        roundIsActive(roundId)
        clientSelectedForRound(clientId, roundId)
    {
        require(bytes(modelUpdateHash).length > 0, "Model update hash cannot be empty");
        
        Round storage round = rounds[roundId];
        
        // Create model update
        ModelUpdate storage update = round.updates[clientId];
        update.clientId = clientId;
        update.roundId = roundId;
        update.modelUpdateHash = modelUpdateHash;
        update.timestamp = block.timestamp;
        update.accepted = false;
        
        // Mark client as participated in this round
        round.clientParticipation[clientId] = true;
        clients[clientId].roundParticipation[roundId] = true;
        clients[clientId].lastUpdateTimestamp = block.timestamp;
        
        // Update round participation count
        round.participantCount++;
        
        emit ModelUpdateSubmitted(clientId, roundId, modelUpdateHash);
    }
    
    /**
     * @dev Accept a model update (after validation)
     * @param clientId ID of the client
     * @param roundId ID of the round
     */
    function acceptModelUpdate(uint256 clientId, uint256 roundId) 
        external 
        onlyRole(OPERATOR_ROLE)
    {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.clientParticipation[clientId], "Client did not participate in this round");
        
        ModelUpdate storage update = round.updates[clientId];
        require(!update.accepted, "Update already accepted");
        
        // Mark update as accepted
        update.accepted = true;
        
        // Update round completed updates count
        round.completedUpdates++;
        
        // Update client contribution score
        clients[clientId].contributionScore += 10; // Simple increment, can be more complex
        clients[clientId].status = ClientStatus.Active;
        
        emit ModelUpdateAccepted(clientId, roundId);
    }
    
    /**
     * @dev Check if a client participated in a specific round
     * @param clientId ID of the client
     * @param roundId ID of the round
     * @return participated Whether the client participated
     */
    function didClientParticipate(uint256 clientId, uint256 roundId) external view returns (bool) {
        return clients[clientId].roundParticipation[roundId];
    }
    
    /**
     * @dev Set the number of Byzantine clients to tolerate in Krum
     * @param f Number of Byzantine clients to tolerate
     */
    function setByzantineClientsToTolerate(uint256 f) external onlyRole(ADMIN_ROLE) {
        byzantineClientsToTolerate = f;
    }
    
    /**
     * @dev Apply Krum defense to select the most representative model update
     * @param roundId ID of the round
     * @return selectedClientId ID of the selected client
     */
    function applyKrumDefense(uint256 roundId) external onlyRole(OPERATOR_ROLE) returns (uint256) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.status == RoundStatus.Active, "Round is not active");
        require(round.participantCount > 0, "No participants in this round");
        
        // Need at least 2f+3 clients for Krum
        uint256 f = byzantineClientsToTolerate;
        require(round.participantCount >= 2 * f + 3, "Not enough clients for Krum defense");
        
        // Collect model hashes and client IDs
        string[] memory modelHashes = new string[](round.participantCount);
        uint256[] memory participantIds = new uint256[](round.participantCount);
        
        uint256 index = 0;
        for (uint256 i = 1; i <= clientCount; i++) {
            if (round.clientParticipation[i]) {
                modelHashes[index] = round.updates[i].modelUpdateHash;
                participantIds[index] = i;
                index++;
            }
        }
        
        // Execute Krum to select the most representative update
        KrumDefense.KrumResult memory result = KrumDefense.executeKrum(
            modelHashes,
            participantIds,
            f
        );
        
        // Automatically accept the selected update
        acceptModelUpdate(result.selectedClientId, roundId);
        
        // Return the selected client ID
        return result.selectedClientId;
    }
    
    /**
     * @dev Get the model update hash from a specific client in a round
     * @param clientId ID of the client
     * @param roundId ID of the round
     * @return modelUpdateHash Hash of the model update
     */
    function getModelUpdateHash(uint256 clientId, uint256 roundId) external view returns (string memory) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.clientParticipation[clientId], "Client did not participate in this round");
        
        return round.updates[clientId].modelUpdateHash;
    }
    
    // Custom Modifiers
    
    /**
     * @dev Modifier to check if caller is a registered client
     * @param clientId ID of the client to check
     */
    modifier onlyRegisteredClient(uint256 clientId) {
        require(clients[clientId].clientAddress == msg.sender, "Caller is not the registered client");
        require(clients[clientId].status == ClientStatus.Registered || 
                clients[clientId].status == ClientStatus.Active, 
                "Client is not in active status");
        _;
    }
    
    /**
     * @dev Modifier to check if a round is active
     * @param roundId ID of the round to check
     */
    modifier roundIsActive(uint256 roundId) {
        require(rounds[roundId].status == RoundStatus.Active, "Round is not active");
        require(block.timestamp < rounds[roundId].endTime, "Round has ended");
        _;
    }
    
    /**
     * @dev Modifier to check if a task is active
     * @param taskId ID of the task to check
     */
    modifier taskIsActive(uint256 taskId) {
        require(tasks[taskId].status == TaskStatus.Active, "Task is not active");
        _;
    }
    
    /**
     * @dev Modifier to check if client is selected for the round
     * @param clientId ID of the client to check
     * @param roundId ID of the round to check
     */
    modifier clientSelectedForRound(uint256 clientId, uint256 roundId) {
        require(clients[clientId].selectedForRound, "Client not selected for this round");
        require(!rounds[roundId].clientParticipation[clientId], "Client already participated in this round");
        _;
    }
    function getModelUpdateHash(uint256 clientId, uint256 roundId) external view returns (string memory) {
        Round storage round = rounds[roundId];
        require(round.roundId == roundId, "Round does not exist");
        require(round.clientParticipation[clientId], "Client did not participate in this round");
        
        return round.updates[clientId].modelUpdateHash;
    }
    
    /**
     * @dev Set the base reward amount
     * @param amount New base reward amount
     */
    function setBaseReward(uint256 amount) external onlyRole(ADMIN_ROLE) {
        baseReward = amount;
    }
    
    /**
     * @dev Calculate rewards for a client based on contribution
     * @param clientId ID of the client
     * @return amount Reward amount
     */
    function calculateReward(uint256 clientId) public view returns (uint256) {
        Client storage client = clients[clientId];
        require(client.clientAddress != address(0), "Client does not exist");
        
        // Simple reward calculation based on contribution score
        // More complex formulas could be implemented
        return baseReward * (client.contributionScore + 10) / 10;
    }
    
    /**
     * @dev Reward a client for their contribution
     * @param clientId ID of the client
     * @param roundId ID of the round
     * @return amount Reward amount
     */
    function rewardClient(uint256 clientId, uint256 roundId) external onlyRole(OPERATOR_ROLE) returns (uint256) {
        Client storage client = clients[clientId];
        require(client.clientAddress != address(0), "Client does not exist");
        require(client.roundParticipation[roundId], "Client did not participate in this round");
        require(rounds[roundId].status == RoundStatus.Completed, "Round is not completed");
        
        // Calculate reward
        uint256 rewardAmount = calculateReward(clientId);
        
        // Add to pending rewards
        pendingRewards[client.clientAddress] += rewardAmount;
        
        emit RewardDistributed(clientId, rewardAmount);
        
        return rewardAmount;
    }
    
    /**
     * @dev Distribute rewards to multiple clients
     * @param clientIds Array of client IDs
     * @param roundId ID of the round
     */
    function distributeRewards(uint256[] calldata clientIds, uint256 roundId) external onlyRole(OPERATOR_ROLE) {
        for (uint256 i = 0; i < clientIds.length; i++) {
            // Skip if reward fails (don't revert the whole transaction)
            try this.rewardClient(clientIds[i], roundId) {
                // Reward successful
            } catch {
                // Skip and continue with next client
            }
        }
    }
    
    /**
     * @dev Claim pending rewards
     * @return amount Amount claimed
     */
    function claimRewards() external nonReentrant returns (uint256) {
        address clientAddress = msg.sender;
        uint256 amount = pendingRewards[clientAddress];
        require(amount > 0, "No rewards to claim");
        
        // Reset pending rewards
        pendingRewards[clientAddress] = 0;
        
        // In a real implementation, this would transfer tokens
        // For this MVP, we just emit an event
        emit RewardsClaimed(clientAddress, amount);
        
        return amount;
    }
    
    /**
     * @dev Check pending rewards for an address
     * @param clientAddress Address to check
     * @return amount Pending reward amount
     */
    function getPendingRewards(address clientAddress) external view returns (uint256) {
        return pendingRewards[clientAddress];
    }
}