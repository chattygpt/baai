# Main agent for coordinating analysis
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pathlib import Path
from config import MODEL_NAME, DEBUG_MODE, OPENAI_API_KEY
import pandas as pd
from .query_agent import analyze_query
from io import StringIO
from openai import OpenAI
from .python_agent import analyze_data
from utils.setup import debug

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def create_llm():
    # Create LangChain ChatOpenAI instance
    return ChatOpenAI(model_name=MODEL_NAME, temperature=0)

def process_data(state: Dict[str, Any]) -> Dict[str, Any]:
    # Process data from file
    # 
    # Args:
    #   state: Current application state containing file path and settings
    #
    # Returns:
    #   Updated state with loaded DataFrame
    try:
        if DEBUG_MODE:
            print(f"\n=== Processing Data ===")
            print(f"Loading file: {state['file_path']}")
            
        # Load and verify the data
        df = pd.read_csv(state['file_path'])
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if DEBUG_MODE:
            print(f"DataFrame loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            print("Column names:", df.columns.tolist())
            print("Sample data:")
            print(df.head(2).to_string())
            print("\nData types:")
            print(df.dtypes)
        
        # Store DataFrame in state
        state['df'] = df
        
        if DEBUG_MODE:
            print("\nData processing complete")
            print("=== Processing Complete ===\n")
            
        return state
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error processing data: {str(e)}")
        raise ValueError(f"Error processing file: {str(e)}")

def upload_file(file_path: str) -> str:
    """Upload file to OpenAI and return file ID."""
    with open(file_path, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose='assistants'
        )
        return response.id

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
        from .python_agent import analyze_data
        
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