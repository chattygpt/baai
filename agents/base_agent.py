from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm
        self.logger = logger

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        pass

    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate that the input state has required information."""
        pass

    def update_state(self, state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Safely update the state with new information."""
        state.update(updates)
        return state 