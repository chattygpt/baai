from typing import Dict, Any
from .base_agent import BaseAgent

class CheckerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the response answers the question."""
        try:
            self.logger.debug("Starting response check")
            
            prompt = f"""
            Question: {state['query']}
            Response: {state['response']}
            
            Evaluate if the response fully answers the question. Consider:
            1. Does it address all parts of the question?
            2. Is the information accurate based on the data?
            3. Is anything missing or unclear?
            
            Return ONLY "complete" if the response is satisfactory, or "incomplete" with a brief explanation if not.
            """
            
            result = self.llm.invoke(prompt)
            is_complete = result.content.lower().startswith('complete')
            
            state['status'] = 'complete' if is_complete else 'incomplete'
            if not is_complete:
                state['feedback'] = result.content
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in Checker Agent: {str(e)}")
            raise

    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate required state inputs"""
        return all(k in state for k in ['query', 'response']) 