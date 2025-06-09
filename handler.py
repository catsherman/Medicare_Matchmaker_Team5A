import sys
import os

# Add the current directory to the Python path
sys.path.append(os.getcwd())

# Now try importing again
from intent_extractor import parse_user_intent