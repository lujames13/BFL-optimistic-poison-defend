"""
Test suite runner for the BFL-Optimistic-Poison-Defend project.
Runs all unit tests in a single session and provides a summary report.
"""

import unittest
import sys
import os
from pathlib import Path
import time
import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from test_blockchain_connector import TestBlockchainConnector
from test_ipfs_connector import TestModelIPFSConnector
from test_flower_server_unit import TestFlowerServer, TestKrumAggregationStrategy
from test_flower_client_unit import TestFlowerClient, TestFlowerClientFl
from test_attack_simulator import TestAttackSimulator, TestAttackStrategies
from test_defense_effectiveness import TestDefenseEvaluator


def run_tests():
    """Run all unit tests and generate a report."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBlockchainConnector))
    test_suite.addTest(unittest.makeSuite(TestModelIPFSConnector))
    test_suite.addTest(unittest.makeSuite(TestFlowerServer))
    test_suite.addTest(unittest.makeSuite(TestKrumAggregationStrategy))
    test_suite.addTest(unittest.makeSuite(TestFlowerClient))
    test_suite.addTest(unittest.makeSuite(TestFlowerClientFl))
    test_suite.addTest(unittest.makeSuite(TestAttackSimulator))
    test_suite.addTest(unittest.makeSuite(TestAttackStrategies))
    test_suite.addTest(unittest.makeSuite(TestDefenseEvaluator))
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Print header
    print("\n" + "="*80)
    print(f"BFL-Optimistic-Poison-Defend Test Suite")
    print(f"Run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Run the tests
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests run: {result.testsRun}")
    print(f"Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Tests failed: {len(result.failures)}")
    print(f"Tests with errors: {len(result.errors)}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("="*80 + "\n")
    
    # Return success/failure
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)