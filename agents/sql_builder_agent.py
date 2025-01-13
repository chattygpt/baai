from typing import Dict, Any
from pathlib import Path
from .base_agent import BaseAgent
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from utils.logger import get_logger

logger = get_logger(__name__)

class SQLBuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.agent = None

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the file using CSV agent."""
        try:
            self.logger.debug("Starting CSV agent process")
            file_path = state['file_path']
            
            # Validate file type
            if not file_path.endswith('.csv'):
                raise ValueError("Only CSV files are supported")
            
            # Create agent with safe configuration
            self.agent = create_csv_agent(
                llm=state['llm'],
                path=file_path,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=True,
                max_iterations=50,
                max_execution_time=30,
                early_stopping_method="force",
                pandas_kwargs={
                    "engine": "python",
                    "on_bad_lines": "skip"
                }
            )
            
            self.logger.debug(f"Created CSV agent for file: {file_path}")
            
            # Update state
            state['agent'] = self.agent
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in CSV agent: {str(e)}")
            raise

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate required state inputs"""
        return all(k in state for k in ['file_path', 'llm']) 