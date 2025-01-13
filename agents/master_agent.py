from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from config import MODEL_NAME
from .sql_builder_agent import SQLBuilderAgent
from .sql_agent import SQLAgent
from .checker_agent import CheckerAgent
from utils.logger import get_logger

logger = get_logger(__name__)

class MasterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=0
        )
        self.tools = {
            'builder': SQLBuilderAgent(),
            'query': SQLAgent(self.llm),
            'checker': CheckerAgent(self.llm)
        }
        self.logger = logger

    def run(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute the analysis workflow."""
        try:
            self.logger.debug(f"Starting analysis for query: {query}")
            if file_path:
                self.logger.debug(f"With file: {file_path}")

            # Initialize state
            state = {
                "query": query,
                "file_path": file_path,
                "llm": self.llm,
                "attempts": 0
            }

            # Process file if provided
            if file_path:
                state = self.tools['builder'].process(state)

            # Process query
            state = self.tools['query'].process(state)

            # Check response
            state = self.tools['checker'].process(state)

            return {
                "status": "success",
                "response": state.get("response")
            }

        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 