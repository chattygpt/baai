from typing import Dict, Any
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
from config import DEBUG_MODE

class DataAgent:
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create pandas agent from CSV file."""
        try:
            if DEBUG_MODE:
                print(f"Loading file: {state['file_path']}")
                
            df = pd.read_csv(state['file_path'])
            state['agent'] = create_pandas_dataframe_agent(
                llm=state['llm'],
                df=df,
                verbose=DEBUG_MODE,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=True,  # Required for pandas operations
                max_iterations=50,
                max_execution_time=30,
                early_stopping_method="force"
            )
            
            if DEBUG_MODE:
                print(f"Created agent with DataFrame shape: {df.shape}")
                
            return state
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in DataAgent: {str(e)}")
            raise ValueError(f"Error processing file: {str(e)}") 