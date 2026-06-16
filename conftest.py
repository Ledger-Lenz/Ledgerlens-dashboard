import sys
from pathlib import Path

# Ensure the project root is on the Python path so all imports resolve
# when running pytest from any working directory.
sys.path.insert(0, str(Path(__file__).parent))
