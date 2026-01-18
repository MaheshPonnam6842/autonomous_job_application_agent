# NOTE:
# Resume normalization intentionally preserves line structure.
# Bullet-aware normalization is deferred to a later iteration
# after end-to-end graph validation.

def normalize_text(text: str) -> str:
    '''
    Normalize text:
    - Strip leading/trailing whitespace
    - Collapse multiple newlines
    - Remove excessive spaces
    '''
    if not text:
        return ""
    text= text.replace("\r","\n")
    lines= [line.strip() for line in text.split("\n") if line.strip()]
    normalized_text= "\n".join(lines)
    return normalized_text
