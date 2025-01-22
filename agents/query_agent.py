from typing import Dict, Any
import pandas as pd
from config import DEBUG_MODE
from .python_agent import analyze_data

def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run query through Python analysis."""
    debug_mode = state.get('debug_mode', True)  # Get debug mode from state
    try:
        if debug_mode:
            print(f"Processing query: {state['query']}")
            
        df = state.get('df')
        if df is None:
            return {
                "response": {
                    "code": "",
                    "steps": [],
                    "results": [],
                    "final_answer": "Please upload a CSV file first."
                }
            }

        # Run the query using analyze_data function
        result = analyze_data(df=df, query=state['query'], debug_mode=debug_mode)
        
        # Return response
        return {
            "response": result['response'],
            "debug_output": result.get('debug_output')
        }
    except Exception as e:
        if debug_mode:
            print(f"Error analyzing query: {str(e)}")
        raise ValueError(f"Error processing query: {str(e)}")

# Export the main function
__all__ = ['analyze_query'] 