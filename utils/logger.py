import logging
import os
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level based on DEBUG_MODE env variable
        debug_mode = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    return logger 