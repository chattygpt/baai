import os
import streamlit as st
from pathlib import Path
import pkg_resources
import subprocess
import sys
from config import DEBUG_MODE
from utils.logger import get_logger

logger = get_logger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = {
        'langchain',
        'langchain-openai',
        'langchain-community',
        'faiss-cpu',
        'unstructured',
        'python-pptx',
        'pypdf'
    }
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        logger.warning(f"Missing dependencies: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            logger.info("Successfully installed missing dependencies")
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            raise RuntimeError(f"Failed to install required dependencies: {missing}")

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
        "logs",
        "docs",
        "vector_store"
    ]
    
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")

def setup_project():
    """Initialize project setup."""
    try:
        logger.info("Starting project setup")
        check_dependencies()
        ensure_directories()
        logger.info("Project setup complete")
    except Exception as e:
        logger.error(f"Error in project setup: {e}")
        raise 