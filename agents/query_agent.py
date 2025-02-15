# Query processing agent
from typing import Dict, Any
import pandas as pd
from config import DEBUG_MODE
from .python_agent import analyze_data

def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
    # Run query through Python analysis
    #
    # Args:
    #   state: Current application state containing:
    #     - query: User's question
    #     - df: Loaded DataFrame
    #     - debug_mode: Whether to show debug output
    #     - thread_id: Existing thread ID
    #     - file_id: Existing file ID
    #     - initialize: Whether this is initialization
    debug_mode = state.get('debug_mode', True)
    initialize = state.get('initialize', False)
    try:
        if debug_mode:
            print(f"Processing query: {state['query']}")
            
        df = state.get('df')
        if df is None:
            return {
                "status": "error",
                "error": "Please upload a CSV file first."
            }

        # Run the query using analyze_data function
        result = analyze_data(
            df=df, 
            query=state['query'], 
            debug_mode=debug_mode,
            thread_id=state.get('thread_id'),
            file_id=state.get('file_id'),
            initialize=initialize
        )
        
        # Return complete result
        return result
    except Exception as e:
        if debug_mode:
            print(f"Error analyzing query: {str(e)}")
        return {
            "status": "error",
            "error": f"Error processing query: {str(e)}"
        }

# Export the main function
__all__ = ['analyze_query'] 