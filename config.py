import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with defaults
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 