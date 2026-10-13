"""
Tests for ML Preprocessing, Training, Registry, and Explainability
"""

import pytest
from src.ml.preprocessor import TextPreprocessor, clean_complaint_text
from src.ml.registry import registry
from src.ml.explainability import ModelExplainer
from src.ml.qlora_stub import QLORA_CONFIG


def test_pii_redaction():
    text = "My SSN is 123-45-6789, email is test.user@bank.com, and phone is (555) 234-5678."
    redacted, entities = TextPreprocessor.redact_pii(text)

    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert len(entities) >= 3


def test_text_cleaning():
    text = "  Unexplained $35.00 FEE on my ACCT! Please REFUND.  "
    cleaned = clean_complaint_text(text)
    assert cleaned == "unexplained $35.00 fee on my acct! please refund."


def test_model_registry_and_artifacts():
    active_version = registry.get_active_version()
    assert active_version is not None

    artifacts = registry.load_version(active_version)
    assert "vectorizer" in artifacts
    assert "category_model" in artifacts
    assert "sentiment_model" in artifacts
    assert "urgency_model" in artifacts
    assert "action_model" in artifacts
    assert "metadata" in artifacts


def test_model_explainability():
    artifacts = registry.load_version()
    vectorizer = artifacts["vectorizer"]
    cat_model = artifacts["category_model"]

    sample_text = "charged an unexpected overdraft fee on my checking account"
    explanation = ModelExplainer.explain_prediction(
        vectorizer=vectorizer,
        model=cat_model,
        cleaned_text=sample_text,
        predicted_class="Billing & Overdraft Fees",
        top_k=3
    )

    assert "top_features" in explanation
    assert len(explanation["top_features"]) > 0
    assert any("overdraft" in f["feature"] or "fee" in f["feature"] for f in explanation["top_features"])


def test_qlora_uses_smolLM2_360m_instruct_model():
    assert "SmolLM2-360M-Instruct" in QLORA_CONFIG["base_model"]
