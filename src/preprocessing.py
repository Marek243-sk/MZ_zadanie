import re

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
