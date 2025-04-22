# BFL-optimistic-poison-defend

A decentralized federated learning system leveraging blockchain technology to provide secure, verifiable, and poison-resistant model training.

## Project Overview

This project implements a blockchain-based federated learning system that combines the Flower framework for distributed machine learning with Arbitrum's Layer-2 rollup technology. The system uses IPFS for decentralized model storage and implements Krum defense mechanisms against poisoning attacks, all secured through an optimistic challenge mechanism.

## Architecture

The system follows a multi-layer architecture:

1. **Federated Learning Layer**: Implemented using Flower framework for client-server communication
2. **Storage Layer**: IPFS for decentralized storage of models and updates
3. **Execution Layer**: Arbitrum rollup for efficient, scalable computation
4. **Security Layer**: Krum aggregation for Byzantine-fault tolerance

## System Flow

1. A requester initiates a federated learning task via the Flower Server
2. The initial model is uploaded to IPFS and registered on-chain
3. Selected clients download the model, train locally, and submit updates
4. Updates are collected and processed by the Rollup Operator
5. Krum defense is applied to filter poisoned updates
6. Aggregated results are posted to Layer-2 with proofs
7. An optimistic challenge period allows validators to contest results
8. Finalized models are propagated for the next round or task completion

## Project Structure

```
/
├── contracts/
│   └── FederatedLearning.sol    # Main smart contract for task management
│
├── fl/
│   ├── blockchain_connector.py  # Interface between FL system and blockchain
│   ├── ipfs_connector.py        # IPFS integration for model storage
│   ├── client.py                # Federated learning client implementation
│   └── server.py                # Federated learning server implementation
│
├── defense/
│   └── krum.py                  # Krum defense algorithm implementation
│
├── rollup/
│   ├── operator.py              # Rollup operator implementation
│   └── validator.py             # Challenge validator implementation
│
├── ignition/
│   └── deploy.js                # Hardhat deployment scripts
│
├── test/
│   ├── contracts/               # Smart contract tests
│   └── fl/                      # Federated learning tests
│
├── hardhat.config.js            # Hardhat configuration
├── package.json                 # Dependencies
└── README.md                    # This file
```

## Key Components

### Flower Server (Requester)

The Flower Server acts as the main coordinator for the federated learning process:
- Initiates learning tasks
- Selects clients for each round
- Manages the global model
- Evaluates training progress
- Interfaces with blockchain through the connector

### Flower Clients

Clients participate in the federated learning process:
- Download the current global model from IPFS
- Perform local training on private data
- Upload model updates to IPFS
- Submit update references to the blockchain

### Blockchain Connector

Bridges the federated learning system with the blockchain:
- Creates tasks on-chain
- Registers model updates
- Monitors task state
- Handles event notifications

### IPFS Connector

Manages decentralized storage operations:
- Uploads models and updates to IPFS
- Retrieves models and updates from IPFS
- Verifies content integrity

### Rollup Operator

Handles the Layer-2 aggregation and validation:
- Collects update references from Layer-2
- Downloads actual updates from IPFS
- Applies Krum defense to filter poisoned updates
- Aggregates valid updates into a new global model
- Generates state proofs for verification
- Submits results back to Layer-2

### Smart Contracts

Two-layer contract system:
- **Layer-1 Contract**: Handles task creation, challenge resolution, and finalization
- **Layer-2 Contract**: Manages update submissions and batch processing

### Defense Mechanisms

Implements robust aggregation techniques:
- **Krum**: Selects the most representative update by minimizing the sum of distances to closest neighbors
- **Challenge Mechanism**: Allows validators to contest suspicious results during the challenge period

## Setup and Installation

```bash
# Clone the repository
git clone git@github.com:lujames13/BFL-optimistic-poison-defend.git
cd BFL-optimistic-poison-defend

# Install dependencies
npm install
pip install -r requirements.txt

# Compile smart contracts
npx hardhat compile

# Deploy to Arbitrum testnet
npx hardhat run ignition/deploy.js --network arbitrum-goerli
```

## Usage

```python
# Start the Flower server (Requester)
python fl/server.py --task_params "params.json" --initial_model "model.h5"

# Run a Flower client
python fl/client.py --client_id "client1" --data_path "data/"

# Run the Rollup Operator
python rollup/operator.py

# Deploy a validator
python rollup/validator.py
```

## Security Features

- **Anti-Poison Defense**: Krum algorithm filters out malicious updates
- **Optimistic Rollups**: Efficient computation with fraud proofs
- **Challenge Period**: 7-day window for contesting suspicious results
- **Penalties**: Economic incentives against malicious behavior
- **Verification Proofs**: Cryptographic validation of computation integrity

## License

[MIT License](LICENSE)
