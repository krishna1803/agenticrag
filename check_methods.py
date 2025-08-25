#!/usr/bin/env python3
"""
Quick test to verify the OCIRAGAgent class structure without imports
"""

import ast
import sys

def check_class_methods(file_path):
    """Parse the Python file and check for required methods"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find the OCIRAGAgent class
        oci_rag_agent_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'OCIRAGAgent':
                oci_rag_agent_class = node
                break
        
        if not oci_rag_agent_class:
            print("✗ OCIRAGAgent class not found")
            return False
        
        # Check for required methods
        methods = [node.name for node in oci_rag_agent_class.body if isinstance(node, ast.FunctionDef)]
        
        required_methods = ['warm_cache', 'get_cache_stats', 'clear_cache']
        
        for method in required_methods:
            if method in methods:
                print(f"✓ {method} method found")
            else:
                print(f"✗ {method} method missing")
        
        print(f"\nAll methods in OCIRAGAgent: {methods}")
        return True
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return False

if __name__ == "__main__":
    file_path = "/Users/krshanmu/git/agenticrag/rag_agent_oci.py"
    check_class_methods(file_path)
