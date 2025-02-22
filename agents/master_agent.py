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

def run_analysis(query: str, file_path: Optional[str] = None, debug_mode: bool = True, 
                thread_id: Optional[str] = None, file_id: Optional[str] = None,
                initialize: bool = False):
    """Run analysis using the Python agent."""
    try:
        if debug_mode:
            print(f"\n=== Starting new analysis ===")
            print(f"Query: {query}")
            print(f"File path: {file_path}")
            print(f"Thread ID: {thread_id}")
            print(f"File ID: {file_id}")
            print(f"Initialize: {initialize}")

        # Create thread if none exists
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            if debug_mode:
                print(f"Created new thread: {thread_id}")

        # Load data if file_path provided
        if file_path:
            df = pd.read_csv(file_path)
            
            # If no file_id, upload the file
            if not file_id:
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                file_content = csv_buffer.getvalue().encode('utf-8')
                file = client.files.create(
                    file=file_content,
                    purpose='assistants'
                )
                file_id = file.id
                if debug_mode:
                    print(f"Uploaded file with ID: {file_id}")

        # If initializing, return early with file info
        if initialize:
            return {
                'status': 'success',
                'thread_id': thread_id,
                'file_id': file_id,
                'message': 'File processed successfully'
            }

        # Run analysis through Python agent
        result = analyze_data(
            query=query,
            file_id=file_id,
            thread_id=thread_id
        )
        
        return result

    except Exception as e:
        if debug_mode:
            print(f"Error in analysis: {str(e)}")
        return {
            'status': 'error',
            'error': f"Error processing query: {str(e)}",
            'thread_id': thread_id,
            'file_id': file_id
        } 