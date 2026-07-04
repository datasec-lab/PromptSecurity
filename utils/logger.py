import logging


def setup_logger():
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logs
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('experiment.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters with enhanced structure for readability
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n' + '-' * 80)
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n' + '-' * 80)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def format_log_text(text, max_length=2000):
    """
    Truncate text for logging if it exceeds max_length.

    Parameters:
        text (str): The text to format.
        max_length (int): Maximum length of text in log.

    Returns:
        str: Formatted (possibly truncated) text.
    """
    return text if len(text) <= max_length else f"{text[:max_length]}... [truncated]"


def log_experiment_step(logger, step, label, text):
    """
    Logs a formatted experiment step with truncation if needed.

    Parameters:
        logger (Logger): Logger instance.
        step (str): Name of the experiment step.
        label (str): Label for the content being logged.
        text (str): Text content (prompt/response) to log.
    """
    truncated_text = format_log_text(text)
    logger.info(f"\n--- {step}: {label} ---\n{truncated_text}\n")