import importlib
from utils.logger import setup_logger

def load_class(module_path, class_name):
    logger=setup_logger()
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading {class_name} from {module_path}: {e}")
        raise