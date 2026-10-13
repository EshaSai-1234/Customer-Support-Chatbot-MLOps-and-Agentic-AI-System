"""
Classification Agent
Coordinates multi-target TF-IDF + Logistic Regression inference and generates
feature attribution explanations.
"""

from typing import Dict, Any, Optional
import numpy as np

from src.agents.base import BaseAgent
from src.ml.explainability import ModelExplainer
from src.ml.preprocessor import clean_complaint_text
from src.ml.registry import registry


class ClassificationAgent(BaseAgent):
    """Executes multi-task machine learning predictions and explains results."""

    def __init__(self, model_version: Optional[str] = None):
        super().__init__(
            name="ClassificationAgent",
            role="Multi-Task ML Inference & Explainability",
            description="Predicts category, sentiment, urgency, and recommended action with confidence scores."
        )
        self.model_version = model_version
        self._artifacts = None

    def _ensure_artifacts_loaded(self):
        if self._artifacts is None:
            self._artifacts = registry.load_version(self.model_version)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("is_valid", True):
            return {"classification_skipped": True, "reason": "Validation failed"}

        self._ensure_artifacts_loaded()

        vectorizer = self._artifacts["vectorizer"]
        cat_model = self._artifacts["category_model"]
        sent_model = self._artifacts["sentiment_model"]
        urg_model = self._artifacts["urgency_model"]
        act_model = self._artifacts["action_model"]

        text = state.get("redacted_text") or state.get("complaint_text", "")
        clean_text = clean_complaint_text(text)
        tfidf_vec = vectorizer.transform([clean_text])

        # Category prediction
        cat_probs = cat_model.predict_proba(tfidf_vec)[0]
        cat_idx = np.argmax(cat_probs)
        predicted_category = cat_model.classes_[cat_idx]
        category_confidence = round(float(cat_probs[cat_idx]), 4)

        # Sentiment prediction
        sent_probs = sent_model.predict_proba(tfidf_vec)[0]
        sent_idx = np.argmax(sent_probs)
        predicted_sentiment = sent_model.classes_[sent_idx]
        sentiment_confidence = round(float(sent_probs[sent_idx]), 4)

        # Urgency prediction
        urg_probs = urg_model.predict_proba(tfidf_vec)[0]
        urg_idx = np.argmax(urg_probs)
        predicted_urgency = urg_model.classes_[urg_idx]
        urgency_confidence = round(float(urg_probs[urg_idx]), 4)

        # Action prediction
        act_probs = act_model.predict_proba(tfidf_vec)[0]
        act_idx = np.argmax(act_probs)
        predicted_action = act_model.classes_[act_idx]
        action_confidence = round(float(act_probs[act_idx]), 4)

        # Explain Category and Urgency
        category_explanation = ModelExplainer.explain_prediction(
            vectorizer=vectorizer,
            model=cat_model,
            cleaned_text=clean_text,
            predicted_class=predicted_category,
            top_k=5
        )

        urgency_explanation = ModelExplainer.explain_prediction(
            vectorizer=vectorizer,
            model=urg_model,
            cleaned_text=clean_text,
            predicted_class=predicted_urgency,
            top_k=5
        )

        return {
            "predicted_category": predicted_category,
            "category_confidence": category_confidence,
            "predicted_sentiment": predicted_sentiment,
            "sentiment_confidence": sentiment_confidence,
            "predicted_urgency": predicted_urgency,
            "urgency_confidence": urgency_confidence,
            "predicted_action": predicted_action,
            "action_confidence": action_confidence,
            "category_explanation": category_explanation,
            "urgency_explanation": urgency_explanation,
            "active_model_version": self._artifacts.get("version"),
        }
