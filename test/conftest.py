"""
測試配置文件，提供共享的測試設置和功能
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 定義測試時使用的常量
TEST_CONTRACT_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
TEST_NODE_URL = "http://localhost:8545"
TEST_IPFS_URL = "http://localhost:5001/api/v0"

# 定義測試數據目錄
TEST_DATA_DIR = os.path.join(Path(__file__).parent, "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# 這裡可以添加更多測試用的通用函數