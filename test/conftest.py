import sys, pathlib

# Ensure project root (parent of tests directory) is on sys.path for module imports
ROOT = pathlib.Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
