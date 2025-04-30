# Blockchain Federated Learning (BFL)

A Byzantine-robust federated learning system with blockchain integration using Arbitrum rollup technology and IPFS decentralized storage.

## Overview

The BFL system combines federated learning with blockchain technology to provide secure, transparent, and Byzantine-robust model training. Key features include:

- **Secure Aggregation**: Krum defense mechanism protects against Byzantine attacks like model poisoning
- **Blockchain Integration**: Uses Arbitrum L2 rollups for efficient on-chain verification
- **IPFS Storage**: Decentralized storage for model weights and updates
- **Modular Design**: Separate components for client, server, defense, and blockchain communication
- **TDD Approach**: Extensive test suite covering all components

## Components

The system consists of several key components:

- **Client**: Blockchain-integrated Flower client for local training and model submission
- **Server**: Blockchain-integrated Flower server with Byzantine-robust aggregation
- **Defense**: Implementation of the Krum algorithm for defense against poisoning attacks
- **Blockchain Connector**: Interface to interact with the Arbitrum smart contract
- **IPFS Connector**: Interface for decentralized model storage

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bfl-optimistic-poison-defend.git
cd bfl-optimistic-poison-defend

# Install dependencies
pip install -r requirements.txt

# Compile the smart contracts
forge build
```

## Usage

### Running in Simulation Mode

```bash
# Run simulation with default settings
python -m fl.main mode=simulation

# Run with Krum defense and 3 Byzantine clients
python -m fl.main --config-name secure mode=simulation simulation.byzantine_clients="[2,5,8]"
```

### Running with Local Blockchain

```bash
# Start a local Anvil node (in a separate terminal)
anvil --fork-url https://arb-sepolia.g.alchemy.com/v2/YOUR_API_KEY

# Deploy smart contract
forge script script/Deploy.s.sol --fork-url http://localhost:8545 --broadcast

# Run server
python -m fl.main mode=server blockchain.contract_address=0x5FbDB2315678afecb367f032d93F642f64180aa3

# Run client
python -m fl.main mode=client client.id=1 blockchain.contract_address=0x5FbDB2315678afecb367f032d93F642f64180aa3
```

### Running the Tests

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run specific tests
python run_tests.py --pattern test_krum*
```

## Configuration

The system uses Hydra for configuration management. Key configuration files include:

- `fl/conf/base.yaml`: Default configuration
- `fl/conf/secure.yaml`: Configuration with enhanced security settings

You can override configuration parameters from the command line:

```bash
python -m fl.main fl.num_rounds=20 defense.byzantine_threshold=3
```

## Architecture

### Client-Server Communication

Clients perform local training and submit model updates to the blockchain. The server starts rounds, selects clients, applies defense mechanisms, and completes tasks.

### Blockchain Integration

The FederatedLearning contract manages tasks, rounds, client registration, and update verification. The Krum defense is implemented both on-chain and off-chain.

### IPFS Integration

Model weights are stored in IPFS, with only the content hash stored on the blockchain, enabling efficient storage of large models.

### Defense Mechanism

Krum selects the most representative update by computing the distances between each update and its n-f-2 nearest neighbors, where f is the number of Byzantine clients to tolerate.

## Example: Attack Mitigation

```python
# Create 5 clients (3 honest, 2 Byzantine)
clients = []
for i in range(3):
    clients.append(create_honest_client(i+1))
for i in range(2):
    clients.append(create_byzantine_client(i+4, attack_type="model_replacement"))

# Run federated learning with Krum defense
server = create_server(use_krum=True, byzantine_threshold=2)
results = run_experiment(server, clients, rounds=10)

# Compare with no defense
server_no_defense = create_server(use_krum=False)
results_no_defense = run_experiment(server_no_defense, clients, rounds=10)

# Plot results
plot_comparison([results, results_no_defense], ["With Krum", "No Defense"])
```

## Research Applications

This system can be used to study:

1. Effectiveness of Krum against different attack types
2. Performance impact of blockchain integration
3. Scalability of Byzantine-robust aggregation
4. Trade-offs between security and efficiency
5. Novel defense mechanisms in a real-world setting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
