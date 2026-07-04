# attacks/black_box/ABJAttack/utils/clean_text.py

def clean_text(text: str) -> str:
    """Remove line breaks from text (basic cleaning)."""
    return text.replace("\r", "").replace("\n", "")
