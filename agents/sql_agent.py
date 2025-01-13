from typing import Dict, Any
from .base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

class SQLAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.logger = logger

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using CSV agent."""
        try:
            self.logger.debug("Starting query processing")
            
            agent = state.get('agent')
            if not agent:
                raise ValueError("No agent found in state")
            
            # Run query through agent
            response = agent.run(state['query'])
            self.logger.debug(f"Generated response: {response}")
            
            # Update state
            state['response'] = response
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in query processing: {str(e)}")
            raise

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate required state inputs"""
        return all(k in state for k in ['agent', 'query']) 