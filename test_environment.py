"""
Test script to verify the development environment setup.
"""
import os
import json
import requests
from web3 import Web3
# 在文件頂部添加
from dotenv import load_dotenv
load_dotenv()

def test_web3_connection():
    """Test connection to Ethereum node."""
    # Try local node first
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    if w3.is_connected():
        print("✅ Connected to local Ethereum node")
        return True
    
    # Try Arbitrum Sepolia
    rpc_url = os.getenv("ARBITRUM_SEPOLIA_RPC_URL")
    if not rpc_url:
        print("❌ ARBITRUM_SEPOLIA_RPC_URL not set in environment")
        return False
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if w3.is_connected():
        print(f"✅ Connected to Arbitrum Sepolia: {rpc_url}")
        return True
    else:
        print(f"❌ Failed to connect to Arbitrum Sepolia: {rpc_url}")
        return False

def test_ipfs_connection():
    """Test connection to IPFS node."""
    try:
        response = requests.post("http://localhost:5001/api/v0/id")
        if response.status_code == 200:
            node_id = response.json().get("ID")
            print(f"✅ Connected to IPFS node: {node_id}")
            return True
        else:
            print(f"❌ Failed to connect to IPFS API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to IPFS: {str(e)}")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import flwr
        print(f"✅ Flower version: {flwr.__version__}")
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Testing development environment setup...")
    
    web3_ok = test_web3_connection()
    ipfs_ok = test_ipfs_connection()
    deps_ok = check_dependencies()
    
    if web3_ok and ipfs_ok and deps_ok:
        print("\n✅ Development environment is properly set up!")
    else:
        print("\n❌ Some components of the development environment are not properly set up.")

if __name__ == "__main__":
    main()
