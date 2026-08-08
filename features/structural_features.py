import re

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

CURRENCY_PATTERN = re.compile(
    r"(?<!\w)(?:PKR|USD|GBP|EUR|AED|SAR|CAD|AUD|INR|MYR|QAR)(?!\w)"
    r"|[$£€¥₹₨]",
    flags=re.IGNORECASE,
)

def currency_reference_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(CURRENCY_PATTERN.findall(text))

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(text.split())


def char_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(text)


def sentence_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    # Remove URLs and email addresses first so periods inside them
    # are not mistaken for sentence endings.
    cleaned_text = URL_PATTERN.sub(" URLTOKEN ", text)
    cleaned_text = EMAIL_PATTERN.sub(" EMAILTOKEN ", cleaned_text)

    # Treat punctuation or line breaks as sentence-like boundaries.
    segments = re.split(r"[.!?]+|\n+", cleaned_text)

    segments = [
        segment
        for segment in segments
        if segment.strip()
    ]

    return len(segments)


def url_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(URL_PATTERN.findall(text))


def email_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(EMAIL_PATTERN.findall(text))


def exclamation_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return text.count("!")


def question_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return text.count("?")


def caps_ratio(text: str) -> float:
    if not isinstance(text, str):
        return 0.0

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    uppercase_letters = [
        char
        for char in letters
        if char.isupper()
    ]

    return len(uppercase_letters) / len(letters)


def digit_ratio(text: str) -> float:
    if not isinstance(text, str):
        return 0.0

    if len(text) == 0:
        return 0.0

    digit_count_value = sum(
        char.isdigit()
        for char in text
    )

    return digit_count_value / len(text)


def currency_symbol_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return sum(
        text.count(symbol)
        for symbol in CURRENCY_SYMBOLS
    )

def extract_structural_features(text: str) -> dict:
    """
    Extract basic structural features from one recruitment text.
    """

    return {
        "word_count": word_count(text),
        "char_count": char_count(text),
        "sentence_count": sentence_count(text),
        "url_count": url_count(text),
        "question_count": question_count(text),
        "caps_ratio": caps_ratio(text),
        "digit_ratio": digit_ratio(text),
        "currency_reference_count": currency_reference_count(text),
    }