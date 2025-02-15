# Data processing agent
from typing import Dict, Any
import pandas as pd
from config import DEBUG_MODE
from .python_agent import analyze_data

class DataAgent:
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Process data using Python analysis
        #
        # Args:
        #   state: Current application state containing:
        #     - file_path: Path to CSV file
        #     - debug_mode: Whether to show debug output
        try:
            if DEBUG_MODE:
                print(f"\n=== DataAgent Processing ===")
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
            
            # Store DataFrame and analysis function in state
            state['df'] = df
            state['analyze_data'] = analyze_data
            
            if DEBUG_MODE:
                print("\nData processing complete")
                print("=== DataAgent Complete ===\n")
                
            return state
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in DataAgent: {str(e)}")
            raise ValueError(f"Error processing file: {str(e)}") 