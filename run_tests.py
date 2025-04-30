#!/usr/bin/env python
"""Run the test suite for Blockchain Federated Learning.

This script runs the TDD tests for the BFL system, including unit tests,
integration tests, and experiment tests.
"""

import os
import sys
import unittest
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("run_tests")


def discover_and_run_tests(test_type: str = "all", verbose: bool = False, pattern: str = None):
    """Discover and run tests based on the specified type.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'experiment', 'all').
        verbose: Whether to run tests in verbose mode.
        pattern: Optional pattern to filter test files.
    
    Returns:
        TestResult object.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Set up pattern
    pattern = pattern or "test_*.py"
    
    # Map test types to directories
    test_dirs = {
        "unit": "test/unit",
        "integration": "test/integration",
        "experiment": "test/experiment",
    }
    
    # Determine which directories to search
    if test_type == "all":
        search_dirs = list(test_dirs.values())
    else:
        search_dirs = [test_dirs[test_type]]
    
    # Discover tests in each directory
    for directory in search_dirs:
        # Check if directory exists
        if not os.path.exists(directory):
            logger.warning(f"Test directory {directory} does not exist. Skipping.")
            continue
            
        logger.info(f"Discovering tests in {directory} with pattern {pattern}")
        discovered_suite = loader.discover(directory, pattern=pattern)
        suite.addTest(discovered_suite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run BFL tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "experiment", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--pattern", 
        type=str, 
        default=None, 
        help="Pattern to filter test files"
    )
    
    args = parser.parse_args()
    
    # Run tests
    logger.info(f"Running {args.type} tests...")
    result = discover_and_run_tests(args.type, args.verbose, args.pattern)
    
    # Print summary
    logger.info("=" * 70)
    logger.info(f"Test Summary: Ran {result.testsRun} tests")
    logger.info(f"  Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
    logger.info(f"  Failures: {len(result.failures)}")
    logger.info(f"  Errors: {len(result.errors)}")
    logger.info("=" * 70)
    
    # Return appropriate exit code
    if result.failures or result.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())