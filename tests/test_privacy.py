import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.sanitizer import sanitize_tool_output, PrivacyViolationError

class TestPrivacyFirewall(unittest.TestCase):
    
    def test_block_dataframe_leak(self):
        """Test that passing a DataFrame raises PrivacyViolationError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        print("\n[FIREWALL TEST 1] Attempting to leak DataFrame...")
        with self.assertRaises(PrivacyViolationError):
            sanitize_tool_output(df)
        print("✅ DataFrame leak blocked.")

    def test_block_large_list(self):
        """Test that large lists are blocked."""
        large_list = [i for i in range(200)] # Limit is 100
        print("\n[FIREWALL TEST 2] Attempting to leak Large List...")
        with self.assertRaises(PrivacyViolationError):
            sanitize_tool_output(large_list)
        print("✅ Large list leak blocked.")

    def test_allow_safe_metadata(self):
        """Test that safe metadata passes."""
        safe_meta = {"status": "ok", "stats": {"mean": 10.5}}
        print("\n[FIREWALL TEST 3] Passing safe metadata...")
        result = sanitize_tool_output(safe_meta)
        self.assertEqual(result, safe_meta)
        print("✅ Safe metadata allowed.")

if __name__ == "__main__":
    unittest.main()
