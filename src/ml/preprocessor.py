"""
Text Preprocessing, Sanitation, and PII Redaction
"""

import re
from typing import Dict, List, Tuple


class TextPreprocessor:
    """Preprocesses complaint texts and scrubs personally identifiable information (PII)."""

    # Regex patterns for PII scrubbing
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
    ACCOUNT_NUM_PATTERN = re.compile(r"\b(?:acct|account|acc)\s*#?\s*\d{6,12}\b", re.IGNORECASE)

    @classmethod
    def redact_pii(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Detects and redacts sensitive PII entities from customer complaint text."""
        if not text or not isinstance(text, str):
            return "", []

        entities_found = []
        redacted = text

        for match in cls.SSN_PATTERN.finditer(text):
            entities_found.append({"type": "SSN", "value": match.group()})
        redacted = cls.SSN_PATTERN.sub("[REDACTED_SSN]", redacted)

        for match in cls.CARD_PATTERN.finditer(text):
            entities_found.append({"type": "CARD_NUMBER", "value": match.group()})
        redacted = cls.CARD_PATTERN.sub("[REDACTED_CARD]", redacted)

        for match in cls.EMAIL_PATTERN.finditer(text):
            entities_found.append({"type": "EMAIL", "value": match.group()})
        redacted = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)

        for match in cls.PHONE_PATTERN.finditer(text):
            entities_found.append({"type": "PHONE", "value": match.group()})
        redacted = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)

        for match in cls.ACCOUNT_NUM_PATTERN.finditer(text):
            entities_found.append({"type": "ACCOUNT_NUMBER", "value": match.group()})
        redacted = cls.ACCOUNT_NUM_PATTERN.sub("[REDACTED_ACCOUNT]", redacted)

        return redacted, entities_found

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Standardizes text for NLP feature extraction."""
        if not text or not isinstance(text, str):
            return ""

        # Redact PII first
        sanitized, _ = cls.redact_pii(text)

        # Normalize whitespace and convert to lowercase
        cleaned = re.sub(r"\s+", " ", sanitized).strip().lower()
        # Keep letters, numbers, basic punctuation and currency markers
        cleaned = re.sub(r"[^a-z0-9\s$.,!?%'-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


def clean_complaint_text(text: str) -> str:
    """Convenience function for cleaning complaint text."""
    return TextPreprocessor.clean_text(text)
