# logger.py
import logging
import sys

def get_logger():
    """
    Configures and returns a standardized logger instance.
    """
    logger = logging.getLogger("personal_ai_assistant")
    if not logger.handlers: # Prevent adding handlers multiple times
        logger.setLevel(logging.INFO)
        
        # Create a handler to print to the console
        stream_handler = logging.StreamHandler(sys.stdout)
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s:%(lineno)d] (%(levelname)s) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(stream_handler)
        
    return logger

# Create a logger instance to be imported by other modules
logger = get_logger()