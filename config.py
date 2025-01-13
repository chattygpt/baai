import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# OpenAI settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

# Project paths
PROJECT_ROOT = Path(__file__).parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DATA_DIR = PROJECT_ROOT / "data"

# Create necessary directories
UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True) 