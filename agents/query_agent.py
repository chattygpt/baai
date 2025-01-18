from typing import Dict, Any
import pandas as pd
from config import DEBUG_MODE
import re

class QueryAgent:
    def __init__(self, llm):
        self.llm = llm

    def _format_analysis(self, raw_output: str, query: str) -> str:
        """Format the raw output into clear analysis steps."""
        if not raw_output:
            return "Analysis steps not available"

        # Clean the output
        clean_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', str(raw_output))
        clean_output = re.sub(r'> Entering new AgentExecutor chain...|> Finished chain.', '', clean_output)
        
        # Extract code and results
        steps = []
        current_step = None
        current_result = None
        
        for line in clean_output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if '=' in line or line.startswith('df['):
                if current_step and current_result:
                    steps.append((current_step, current_result))
                current_step = line
                current_result = None
            elif not line.startswith(('Thought:', 'Human:', 'Assistant:', '>')):
                current_result = line

        if current_step and current_result:
            steps.append((current_step, current_result))

        if not steps:
            return "No analysis steps available"

        # Format steps
        formatted_steps = []
        for i, (code, result) in enumerate(steps, 1):
            formatted_steps.append(
                f"Step {i}: Running calculation\n"
                f"Code: {code}\n"
                f"Ans = {result}"
            )

        if steps:
            formatted_steps.append(f"Final Answer = {steps[-1][1]}")

        return "\n\n".join(formatted_steps)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run query through Python agent."""
        try:
            if DEBUG_MODE:
                print(f"Processing query: {state['query']}")
                
            agent = state.get('agent')
            if not agent:
                state['response'] = "Please upload a CSV file first."
                return state

            # Run the query
            response = agent.run(state['query'])
            
            # Update state
            state['response'] = response
            state['reasoning'] = self._format_analysis(response, state['query'])
            
            return state
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in QueryAgent: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}") 