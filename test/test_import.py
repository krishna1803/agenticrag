#!/usr/bin/env python3
import sys
import os

# Add the parent directory (project root) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from provenance import ProvenanceStore
    print('✓ ProvenanceStore import successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
