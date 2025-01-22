from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pathlib import Path
from config import MODEL_NAME, DEBUG_MODE
import pandas as pd
from .query_agent import analyze_query

def create_llm():
    """Create LangChain ChatOpenAI instance."""
    return ChatOpenAI(model_name=MODEL_NAME, temperature=0)

def process_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process data from file."""
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

def run_analysis(query: str, file_path: Optional[str] = None, debug_mode: bool = True) -> Dict[str, Any]:
    """Process query with optional file input."""
    try:
        if debug_mode:
            print(f"\n=== Starting new analysis ===")
            print(f"Query: {query}")
            print(f"File path: {file_path or 'None'}")
            
        # Initialize state
        state = {
            "query": query,
            "file_path": file_path,
            "llm": create_llm(),
            "conversation_history": [],
            "debug_mode": debug_mode  # Add debug mode to state
        }

        # Process file if provided
        if file_path and Path(file_path).exists():
            state = process_data(state)

        # Run query analysis
        result = analyze_query(state)
        
        # Format response
        response = {
            "status": "success",
            "response": result.get('response', ''),
            "debug_output": result.get('debug_output') if debug_mode else None,
            "conversation_history": state.get('conversation_history', [])
        }

        if debug_mode:
            print("=== Analysis complete ===\n")

        return response
    except Exception as e:
        if debug_mode:
            print(f"Error in analysis: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        } 