from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pathlib import Path
from .data_agent import DataAgent
from .query_agent import QueryAgent
from config import MODEL_NAME, DEBUG_MODE
import pandas as pd

class MasterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
        self.data_agent = DataAgent()
        self.query_agent = QueryAgent(self.llm)

    def run(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Process query with optional file input."""
        try:
            if DEBUG_MODE:
                print(f"\n=== Starting new analysis ===")
                print(f"Query: {query}")
                print(f"File path: {file_path or 'None'}")
                
            state = {
                "query": query,
                "file_path": file_path,
                "llm": self.llm,
                "conversation_history": []
            }

            if file_path and Path(file_path).exists():
                state = self.data_agent.process(state)

            # Get result from query agent
            result = self.query_agent.process(state)

            print("result", result)
            
            # Update state with result
            state['response'] = result.get('result', '')
            state['reasoning'] = result.get('analysis', '')
            
            # Update history
            state.setdefault('conversation_history', []).append({
                'query': query,
                'response': state['response'],
                'reasoning': state['reasoning'],
                'timestamp': pd.Timestamp.now().isoformat()
            })

            if DEBUG_MODE:
                print("=== Analysis complete ===\n")

            return {
                "status": "success",
                "response": state['response'],
                "reasoning": state['reasoning'],
                "conversation_history": state['conversation_history']
            }
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in MasterAgent: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 