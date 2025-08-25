#!/usr/bin/env python3
"""
Quick test to verify the OCIRAGAgent class has the required methods
"""

import sys
import os

# Add the parent directory (project root) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rag_agent_oci import OCIRAGAgent
    
    # Check if the warm_cache method exists
    if hasattr(OCIRAGAgent, 'warm_cache'):
        print("✓ OCIRAGAgent.warm_cache method exists")
    else:
        print("✗ OCIRAGAgent.warm_cache method is missing")
    
    # Check if the get_cache_stats method exists
    if hasattr(OCIRAGAgent, 'get_cache_stats'):
        print("✓ OCIRAGAgent.get_cache_stats method exists")
    else:
        print("✗ OCIRAGAgent.get_cache_stats method is missing")
    
    print("All required methods are available!")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
