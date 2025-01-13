from typing import Dict, Any
import pandas as pd
from config import DEBUG_MODE
from io import StringIO
import sys
import re

class QueryAgent:
    def __init__(self, llm):
        self.llm = llm

    def _format_analysis(self, raw_output: str, query: str) -> str:
        """Format the raw output into clear analysis steps."""
        if not raw_output:
            return "Analysis steps not available"

        # Clean the output
        clean_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_output)
        clean_output = re.sub(r'> Entering new AgentExecutor chain...|> Finished chain.', '', clean_output)
        
        # Extract code and results
        steps = []
        current_step = None
        current_result = None
        
        for line in clean_output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('df[') or '=' in line:
                if current_step and current_result:
                    steps.append((current_step, current_result))
                current_step = line
                current_result = None
            elif not line.startswith('>') and not line.startswith('Thought:'):
                current_result = line

        if current_step and current_result:
            steps.append((current_step, current_result))

        if not steps:
            return "No analysis steps available"

        # Format steps
        formatted_steps = []
        for i, (code, result) in enumerate(steps, 1):
            # Interpret what the code is doing
            description = "Finding "
            if "sum()" in code:
                if "'sales'" in code.lower():
                    description += "total sales"
                else:
                    description += "sum"
            elif "mean()" in code:
                description += "average"
            elif "count()" in code:
                description += "count"
            else:
                description += "value"

            # Add time period if present
            if "'2015-08'" in code or "'aug'" in code.lower():
                description += " for August 2015"
            elif "'2015-07'" in code or "'jul'" in code.lower():
                description += " for July 2015"

            # Add SKU if present
            if "'sku'" in code.lower():
                sku_match = re.search(r"'([^']*)'", code)
                if sku_match:
                    description += f" for SKU {sku_match.group(1)}"

            # Format the step
            formatted_steps.append(
                f"Step {i}: {description}\n"
                f"Code: {code}\n"
                f"Ans = {result}"
            )

        # Add final answer
        if steps:
            formatted_steps.append(f"Final Answer = {steps[-1][1]}")

        return "\n\n".join(formatted_steps)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run query through pandas agent."""
        try:
            if DEBUG_MODE:
                print(f"Processing query: {state['query']}")
                
            agent = state.get('agent')
            if not agent:
                state['response'] = "Please upload a CSV file first."
                return state

            # Capture the agent's thought process
            thought_output = StringIO()
            sys.stdout = thought_output
            
            # Run the query
            response = agent.run(state['query'])
            
            # Restore stdout and get thoughts
            sys.stdout = sys.__stdout__
            reasoning = self._format_analysis(thought_output.getvalue(), state['query'])
            
            # Update state
            state['response'] = response
            state['reasoning'] = reasoning
            
            # Update history
            state.setdefault('conversation_history', []).append({
                'query': state['query'],
                'response': state['response'],
                'reasoning': state['reasoning'],
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            if DEBUG_MODE:
                print(f"Generated response: {state['response'][:200]}...")
                
            return state
        except Exception as e:
            sys.stdout = sys.__stdout__  # Restore stdout in case of error
            if DEBUG_MODE:
                print(f"Error in QueryAgent: {str(e)}")
            raise ValueError(f"Error processing query: {str(e)}") 