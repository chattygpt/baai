import os
import streamlit as st
from pathlib import Path
from config import DEBUG_MODE
from utils.logger import get_logger

logger = get_logger(__name__)

def debug(message: str, output_list: list = None):
    """Print debug message and optionally collect in output list."""
    if DEBUG_MODE:
        print(message)
    if output_list is not None:
        output_list.append(str(message))

def ensure_directories():
    """Create necessary project directories."""
    dirs = [
        "data",
        "uploads",
        "logs"
    ]
    
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")

def setup_project():
    """Initialize project setup."""
    try:
        logger.info("Starting project setup")
        ensure_directories()
        logger.info("Project setup complete")
    except Exception as e:
        logger.error(f"Error in project setup: {e}")
        raise 