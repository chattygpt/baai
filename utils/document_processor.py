from typing import List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

from agents.master_agent import MasterAgent
from config import UPLOADS_DIR

# Initialize environment and API key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Create master agent instance
master_agent = MasterAgent()

def analyze_documents(query: str, files: Optional[List[str]] = None) -> dict:
    """Main function to analyze documents and generate insights"""
    try:
        # Verify all files exist
        if files:
            for file_path in files:
                if not Path(file_path).exists():
                    raise ValueError(f"File not found: {file_path}")
        
        # Run analysis through master agent
        return master_agent.run(query, files)
    except Exception as e:
        return {"status": "error", "error": str(e)}

def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """Save uploaded file to disk and return its path"""
    try:
        # Create uploads directory if needed
        UPLOADS_DIR.mkdir(exist_ok=True)
        file_path = UPLOADS_DIR / Path(filename).name
        
        # Write file to disk
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)
    except Exception as e:
        raise ValueError(f"File upload failed: {str(e)}") 