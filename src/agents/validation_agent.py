"""
Data Validation Agent
Ensures text quality, schema adherence, and redacts PII before NLP processing.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.ml.preprocessor import TextPreprocessor


class DataValidationAgent(BaseAgent):
    """Validates raw incoming complaint payloads and masks sensitive information."""

    def __init__(self):
        super().__init__(
            name="ValidationAgent",
            role="Data Sanitation & PII Redaction",
            description="Validates payload structure, checks input sanity, and scrubs PII."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = state.get("complaint_text", "")
        if not raw_text or not isinstance(raw_text, str):
            return {
                "validation_status": "FAILED",
                "validation_errors": ["Complaint text is missing or not a string."],
                "is_valid": False,
            }

        stripped = raw_text.strip()
        errors: List[str] = []
        warnings: List[str] = []

        if len(stripped) < 10:
            errors.append("Complaint text is too short (< 10 characters).")
        if len(stripped) > 10000:
            warnings.append("Complaint text exceeds 10,000 characters; downstream latency may increase.")

        # Check alphanumeric content
        alpha_chars = sum(c.isalnum() for c in stripped)
        if len(stripped) > 0 and (alpha_chars / len(stripped)) < 0.4:
            warnings.append("Low alphanumeric character density; text may contain excessive symbols or noise.")

        # Perform PII scrubbing
        redacted_text, pii_entities = TextPreprocessor.redact_pii(stripped)

        if pii_entities:
            warnings.append(f"Detected and redacted {len(pii_entities)} sensitive PII item(s).")

        status = "FAILED" if errors else ("WARNING" if warnings else "PASSED")

        return {
            "validation_status": status,
            "is_valid": len(errors) == 0,
            "validation_errors": errors,
            "validation_warnings": warnings,
            "redacted_text": redacted_text,
            "pii_detected": len(pii_entities) > 0,
            "pii_entities": pii_entities,
        }
