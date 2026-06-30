"""
Full-system experiment runner (delegates to main.py pipeline).
"""
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from main import main

if __name__ == "__main__":
    main()
