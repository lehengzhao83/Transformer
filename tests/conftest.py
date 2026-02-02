import os
import sys

# Ensure project root is on sys.path so `import src...` works in pytest
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
