import re

from urllib.parse import urlparse

URL_PATTERN = re.compile(
    r"https?://[^\s)>]+",
    flags=re.IGNORECASE,
)

def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """
    Count how many behavioral patterns appear in a text.

    Each regex pattern contributes at most its number of actual matches.
    """
    if not isinstance(text, str):
        return 0

    text = text.lower()

    return sum(
        len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in patterns
    )


PAYMENT_REQUEST_PATTERNS = [
    # Applicant explicitly asked to pay/send/transfer a fee or charge
    r"\b(?:pay|send|transfer|deposit)\b.{0,40}"
    r"\b(?:fee|charge|costs?|deposit|money|funds|amount)\b",

    # Fee/charge followed by instructions to send or transfer it
    r"\b(?:fee|charge|costs?|deposit)\b.{0,40}"
    r"\b(?:pay|send|transfer|deposit|wallet)\b",

    # Recruitment-stage fees and charges
    r"\b(?:application|interview|scheduling|processing|registration|"
    r"onboarding|training|security|verification|visa|background[- ]?check)"
    r"\s+(?:fee|charge|costs?|deposit)\b",

    r"\bupfront\s+(?:fee|payment|charge|cost|deposit)\b",
]


CREDENTIAL_REQUEST_PATTERNS = [
    r"\b(?:send|share|provide|enter|submit)\b.{0,30}\b(?:password|otp|pin|cvv|credentials)\b",
    r"\bone[- ]?time\s+(?:password|code)\b",
    r"\blogin\s+credentials\b",
    r"\brecovery\s+code\b",
]


URGENCY_PATTERNS = [
    r"\burgent(?:ly)?\b",
    r"\bimmediately\b",
    r"\bact\s+now\b",
    r"\bwithin\s+(?:the\s+next\s+)?(?:24|48)\s+hours?\b",
    r"\btoday\s+only\b",
    r"\brespond\s+immediately\b",
    r"\bdeadline\s+(?:is\s+)?today\b",
]


IDENTITY_DOCUMENT_PATTERNS = [
    r"\bpassport\b",
    r"\bcnic\b",
    r"\bnational\s+id\b",
    r"\bidentity\s+(?:card|document)\b",
    r"\bdriver'?s?\s+licen[cs]e\b",
    r"\bbank\s+statement\b",
    r"\butility\s+bill\b",
    r"\blive\s+selfie\b",
]


EQUIPMENT_PURCHASE_PATTERNS = [
    r"\b(?:buy|purchase|order)\b.{0,35}\b(?:laptop|computer|equipment|software|device|workstation)\b",
    r"\bequipment\s+(?:purchase|deposit|payment|fee)\b",
    r"\bpurchase\b.{0,25}\bfrom\s+(?:our|the)\s+(?:vendor|supplier)\b",
]


MONEY_TRANSFER_PATTERNS = [
    r"\breceive\b.{0,40}\b(?:money|funds|payment|payments)\b.{0,50}\b(?:send|forward|transfer)\b",
    r"\b(?:send|forward|transfer)\b.{0,40}\b(?:money|funds|payment|payments)\b",
    r"\bpersonal\s+(?:bank\s+)?account\b.{0,50}\b(?:receive|process|transfer|forward)\b",
    r"\b(?:receive|process)\b.{0,50}\b(?:money|funds|payments?)\b.{0,50}\bpersonal\s+(?:bank\s+)?account\b",
]

PAID_TRAINING_PATTERNS = [
    # Explicit purchase of training/certificate
    r"\b(?:buy|purchase|bought|purchased)\b.{0,30}"
    r"\b(?:training|course|certificate|certification)s?\b",

    # Training/certificate explicitly described as something purchased
    r"\b(?:training|course|certificate|certification)s?\b.{0,30}"
    r"\b(?:buy|purchase|bought|purchased|fee)\b",

    # Mandatory certificate/training associated with a selected provider
    r"\bonly\b.{0,25}"
    r"\b(?:training|course|certificate|certification)s?\b.{0,60}"
    r"\b(?:bought|purchased|training\s+partner)\b",

    # Paid enrollment with a recruiter-selected training provider
    r"\btraining\s+partner\b.{0,100}"
    r"\b(?:PKR|USD|GBP|EUR|AED|SAR|CAD|AUD|INR|MYR|QAR)\b",
]

SENSITIVE_LINK_PATTERNS = [
    r"\bcard\s+details\b",
    r"\bpayment[- ]?card\b.{0,30}\bsecurity\s+code\b",
    r"\bsecurity\s+code\b",
    r"\bcvv\b",
    r"\bpassword\b",
    r"\botp\b",
    r"\bone[- ]?time\s+(?:password|code)\b",
    r"\bpin\b",
    r"\blogin\s+credentials\b",
    r"\baccount\s+access\b",
]

SELECTION_BYPASS_PATTERNS = [
    r"\bno\s+interviews?\b",
    r"\bwithout\s+(?:an?\s+)?interviews?\b",
    r"\binterviews?\s+(?:are|is)\s+not\s+required\b",
    r"\bno\s+further\s+application\s+stage\b",
    r"\bno\s+(?:interview|screening|selection)\s+(?:is\s+)?required\b",
    r"\brequires?\s+neither\s+interviews?\s+nor\s+prior\s+experience\b",
    r"\bdirect\s+onboarding\b",
]

CHEQUE_OVERPAYMENT_PATTERNS = [
    r"\b(?:cheque|check)\b.{0,120}"
    r"\b(?:more\s+than|exceed|exceeds|overpay|overpayment|"
    r"remainder|unused\s+balance)\b",

    r"\b(?:cheque|check)\b.{0,150}"
    r"\b(?:return|send|transfer)\b.{0,50}"
    r"\b(?:remainder|balance|difference|funds|money)\b",
]

NEGATED_MONEY_PATTERN = re.compile(
    r"\b(?:do|should|must|will)\s+not\s+"
    r"(?:pay|send|transfer|deposit)\b[^.!?\n]{0,80}"
    r"|\bnever\s+(?:pay|send|transfer|deposit)\b[^.!?\n]{0,80}",
    flags=re.IGNORECASE,
)

EMPLOYER_FUNDED_COST_PATTERN = re.compile(
    r"\b(?:employer|company|organization)\b"
    r"[^.!?\n]{0,25}"
    r"\b(?:pay|pays|cover|covers|fund|funds)\b"
    r"[^.!?\n]{0,60}"
    r"\b(?:fee|fees|cost|costs|charge|charges|visa|relocation|travel)\b"
    r"[^.!?\n]{0,30}",
    flags=re.IGNORECASE,
)


def cheque_overpayment_flag(text: str) -> int:
    return pattern_present(text, CHEQUE_OVERPAYMENT_PATTERNS)

def selection_bypass_flag(text: str) -> int:
    return pattern_present(text, SELECTION_BYPASS_PATTERNS)

def suspicious_application_link_flag(text: str) -> int:
    if not isinstance(text, str):
        return 0

    has_url = bool(URL_PATTERN.search(text))

    sensitive_request = any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in SENSITIVE_LINK_PATTERNS
    )

    return int(has_url and sensitive_request)

def paid_training_flag(text: str) -> int:
    return pattern_present(text, PAID_TRAINING_PATTERNS)

def pattern_present(text: str, patterns: list[str]) -> int:
    """
    Return 1 when at least one pattern is detected,
    otherwise return 0.
    """
    if not isinstance(text, str):
        return 0

    return int(
        any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in patterns
        )
    )


def payment_request_flag(text: str) -> int:
    if not isinstance(text, str):
        return 0

    cleaned_text = NEGATED_MONEY_PATTERN.sub(
        " ",
        text,
    )

    cleaned_text = EMPLOYER_FUNDED_COST_PATTERN.sub(
        " ",
        cleaned_text,
    )

    return pattern_present(
        cleaned_text,
        PAYMENT_REQUEST_PATTERNS,
    )


def credential_request_flag(text: str) -> int:
    return pattern_present(text, CREDENTIAL_REQUEST_PATTERNS)


def urgency_flag(text: str) -> int:
    return pattern_present(text, URGENCY_PATTERNS)


def identity_document_flag(text: str) -> int:
    return pattern_present(text, IDENTITY_DOCUMENT_PATTERNS)


def equipment_purchase_flag(text: str) -> int:
    return pattern_present(text, EQUIPMENT_PURCHASE_PATTERNS)


def money_transfer_flag(text: str) -> int:
    if not isinstance(text, str):
        return 0

    cleaned_text = NEGATED_MONEY_PATTERN.sub(" ", text)

    return pattern_present(
        cleaned_text,
        MONEY_TRANSFER_PATTERNS,
    )

def lookalike_domain_flag(text: str) -> int:
    """
    Detect simple digit substitutions such as:
    company -> c0mpany

    This is only a weak supporting signal because legitimate
    domains can also contain digits.
    """
    if not isinstance(text, str):
        return 0

    urls = URL_PATTERN.findall(text)

    for url in urls:
        cleaned_url = url.rstrip(".,;:!?")
        hostname = urlparse(cleaned_url).hostname or ""

        if re.search(
            r"[a-z][01][a-z]",
            hostname,
            flags=re.IGNORECASE,
        ):
            return 1

    return 0


def extract_behavioral_features(text: str) -> dict:
    """
    Extract binary recruitment-scam behavioral signals.

    A value of 1 means the pattern is present.
    It does not by itself mean the text is fraudulent.
    """

    return {
        "payment_request_flag": payment_request_flag(text),
        "credential_request_flag": credential_request_flag(text),
        "urgency_flag": urgency_flag(text),
        "identity_document_flag": identity_document_flag(text),
        "equipment_purchase_flag": equipment_purchase_flag(text),
        "money_transfer_flag": money_transfer_flag(text),
        "paid_training_flag": paid_training_flag(text),
        "suspicious_application_link_flag":
            suspicious_application_link_flag(text),
        "selection_bypass_flag": selection_bypass_flag(text),
        "cheque_overpayment_flag": cheque_overpayment_flag(text),
        "lookalike_domain_flag": lookalike_domain_flag(text),
    }