import re


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Apply minimal normalization for lexical TF-IDF features.

    URLs and email addresses are replaced with generic tokens so the
    lexical model does not simply memorize individual domains or addresses.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = EMAIL_PATTERN.sub(" EMAILTOKEN ", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text