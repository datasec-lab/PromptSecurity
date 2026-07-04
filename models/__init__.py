# models/__init__.py

def load_model(*args, **kwargs):
    """Lazy wrapper to avoid importing heavy model backends at package import time."""
    from .loader import load_model as _load_model
    return _load_model(*args, **kwargs)


def list_available_models(*args, **kwargs):
    """Lazy wrapper to avoid importing heavy model backends at package import time."""
    from .loader import list_available_models as _list_available_models
    return _list_available_models(*args, **kwargs)


# Backward compatibility
def load_model_from_config(*args, **kwargs):
    return load_model(*args, **kwargs)


def list_supported_models():
    return list_available_models()
