import sys
import os

# Add the current directory to sys.path so we can import src
sys.path.append(os.getcwd())

try:
    from src.python import bake_model
    print("Successfully imported bake_model. Syntax is OK.")
except Exception as e:
    print(f"Failed to import bake_model: {e}")
    sys.exit(1)
