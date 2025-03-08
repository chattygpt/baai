from .master_agent import run_analysis
from .python_agent import analyze_data
from typing import Dict, Any, Optional
from utils.setup import debug
from openai import OpenAI
from pathlib import Path

client = OpenAI()

def upload_file(file_path: str) -> str:
    """Upload file to OpenAI and return file ID."""
    with open(file_path, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose='assistants'
        )
        return response.id

__all__ = ['run_analysis', 'analyze_data']

def run_analysis(
    query: str,
    file_path: Optional[str] = None,
    debug_mode: bool = False,
    initialize: bool = False,
    thread_id: Optional[str] = None,
    file_id: Optional[str] = None,
    user_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Run analysis on data."""
    try:
        # Upload file if provided and not already uploaded
        if file_path and not file_id:
            file_id = upload_file(file_path)
            
        # Run analysis
        result = analyze_data(
            query=query,
            file_id=file_id,
            thread_id=thread_id,
            user_prompt=user_prompt
        )
        
        return result
        
    except Exception as e:
        debug(f"Analysis error: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        } 