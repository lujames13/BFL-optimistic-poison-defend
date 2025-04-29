# Blockchain Federated Learning (BFL) MVP Test Plan

This document outlines a simplified test-driven development (TDD) approach for the Blockchain Federated Learning (BFL) MVP. The goal is to deploy the system to a local Arbitrum network and generate data visualizations for thesis research.

## Simplified Test Structure

```
test/
├── unit/                       # Basic unit tests for core components
│   ├── test_flower_client.py   # Tests for FL client functionality
│   ├── test_flower_server.py   # Tests for FL server setup
│   ├── test_blockchain.py      # Tests for blockchain connector
│   ├── test_ipfs.py            # Tests for IPFS storage
│   └── test_krum.py            # Tests for Krum defense algorithm
├── integration/                # Essential integration tests
│   ├── test_fl_blockchain.py   # Tests FL and blockchain integration
│   └── test_krum_defense.py    # Tests Krum defense effectiveness
├── experiment/                 # Thesis experiment tests
│   ├── test_honest_scenario.py # Baseline performance with honest clients
│   ├── test_attack_scenario.py # Performance under attack
│   └── test_metrics.py         # Data collection for thesis visualizations
├── fixtures/                   # Test data and setup
│   ├── model_fixtures.py       # Test models
│   └── data_fixtures.py        # Test datasets
└── README.md                   # This simplified test plan
```

## MVP Testing Focus

### 1. Unit Testing (Essential Components Only)

#### 1.1 Flower Client/Server Tests

- Test basic client initialization and configuration
- Test server setup and aggregation functionality
- Test local training with test datasets
- Test parameter serialization/deserialization

#### 1.2 Blockchain Connector Tests

- Test connection to local Arbitrum network
- Test smart contract interaction (task creation, update submission)
- Test event handling for training rounds
- Test transaction submission and confirmation

#### 1.3 IPFS Tests

- Test model storage and retrieval
- Test content addressing functionality
- Test handling of model parameters

#### 1.4 Krum Algorithm Tests

- Test distance calculation between updates
- Test selection of representative update
- Test Byzantine-tolerant properties with simple attack examples

### 2. Integration Testing (MVP Critical Paths)

#### 2.1 FL-Blockchain Integration

- Test end-to-end flow from model training to blockchain submission
- Test round initialization and completion via blockchain
- Test model update verification through the contract

#### 2.2 Krum Defense Integration

- Test Krum integration with Flower aggregation
- Test defense effectiveness against basic attack scenarios

### 3. Experiment Testing (For Thesis Data)

#### 3.1 Baseline Performance

- Test system convergence with honest clients
- Measure accuracy, loss, and training time
- Generate learning curves for thesis

#### 3.2 Attack Scenarios

- Test system under label flipping attack
- Test system under model replacement attack
- Measure impact on model quality
- Generate comparative visualizations

#### 3.3 Defense Effectiveness

- Test system with Krum defense active
- Measure improvements in robustness
- Generate defense effectiveness visualizations

## Test Environment

### Local Testing Environment

- Local Foundry Anvil node for Arbitrum simulation
- Local IPFS node or mock
- Simulated Flower clients and server
- Small-scale test datasets (MNIST or similar)

## Essential Test Data

- **Simple Dataset**: MNIST or Fashion-MNIST (small subset)
- **Attack Dataset**: Modified datasets with flipped labels or corrupted samples
- **Validation Dataset**: Clean validation data for measuring performance

## Minimal Mocking Strategy

- **Blockchain Mocks**: Simple contract interface mocks for unit tests
- **IPFS Mocks**: Basic content addressable storage simulation
- **Client Mocks**: Simulate honest and Byzantine client behaviors

## Example Test Cases

### Unit Test Example

```python
def test_client_submits_update_to_blockchain():
    # Arrange
    client = FlowerClient(client_id=1, private_key=TEST_PRIVATE_KEY)
    model = get_simple_model()  # A small test model
    parameters = model.get_weights()

    # Act
    result = client.submit_update(parameters, round_id=1)

    # Assert
    assert result.success == True
    assert result.transaction_hash is not None
```

### Integration Test Example

```python
def test_krum_defends_against_label_flipping():
    # Arrange
    # Set up server with Krum defense
    server = FlowerServer(defense="krum", byzantine_tolerance=1)

    # Create clients (4 honest, 1 malicious)
    clients = [create_honest_client() for _ in range(4)]
    clients.append(create_label_flipping_client())

    # Act
    # Run federated training for 5 rounds
    results = run_federated_training(server, clients, rounds=5)

    # Assert
    # Check that final model accuracy is not significantly degraded
    assert results.final_accuracy > 0.8  # Or whatever baseline is reasonable
    # Check that malicious updates were rejected
    assert results.client_selection_history[4] == 0  # Client 4 (malicious) never selected
```

### Experiment Test Example

```python
def test_generate_defense_comparison_data():
    # Arrange
    # Configure experiment parameters
    experiment_configs = [
        {"defense": None, "attack": "label_flipping", "attack_ratio": 0.2},
        {"defense": "krum", "attack": "label_flipping", "attack_ratio": 0.2},
    ]

    # Act
    # Run experiments and collect results
    results = []
    for config in experiment_configs:
        result = run_experiment(config, rounds=10)
        results.append(result)

    # Generate visualizations
    accuracy_chart = generate_accuracy_comparison(results)
    convergence_chart = generate_convergence_comparison(results)

    # Assert
    # Ensure data was generated
    assert accuracy_chart is not None
    assert convergence_chart is not None
    # Save for thesis
    accuracy_chart.save("thesis_figures/accuracy_comparison.png")
    convergence_chart.save("thesis_figures/convergence_comparison.png")
```

## MVP Test Execution

```bash
# Run all MVP tests
pytest

# Run unit tests for quick development feedback
pytest test/unit/

# Run specific component test
pytest test/unit/test_blockchain.py

# Run integration tests
pytest test/integration/

# Run experiment tests (generates thesis data)
pytest test/experiment/
```

## Data Collection for Thesis

The test framework should collect the following metrics for thesis visualizations:

1. **Model Accuracy** (per round)
2. **Training Loss** (per round)
3. **Selected Clients** (for Krum defense analysis)
4. **Attack Impact** (comparison with/without defense)
5. **Convergence Rate** (rounds to reach target accuracy)
6. **Blockchain Transaction Data** (gas used, timing)

## Conclusion

This simplified MVP test plan focuses on the essential components needed to:

1. Verify the core functionality of the BFL system
2. Ensure proper integration between Flower and the local Arbitrum network
3. Validate the effectiveness of the Krum defense mechanism
4. Generate the necessary data visualizations for the thesis

By following this TDD approach, you'll build only what's needed for your MVP while ensuring the system works correctly and produces reliable research data.
