# Remove WandBLogger completely
# After cleanup, the file will only handle plain logging.

import pandas as pd
import logging

def setup_logger():
    logger = logging.getLogger("PAIR")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return logger

def set_logger_level(logger, verbosity):
    if verbosity == 0:
        level = logging.CRITICAL  # Disables logging
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

logger = setup_logger()
logger.set_level = lambda verbosity: set_logger_level(logger, verbosity)
