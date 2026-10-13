"""
Explainability Module for TF-IDF + Logistic Regression Predictions
Computes feature attributions (token weights * TF-IDF values) to provide
clear, evidence-based explanations for model predictions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ModelExplainer:
    """Computes exact linear feature contributions for Logistic Regression with TF-IDF."""

    @staticmethod
    def explain_prediction(
        vectorizer,
        model,
        cleaned_text: str,
        predicted_class: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculates top positive driving features for the predicted class.
        Feature contribution = coefficient[class_idx, feature_idx] * tfidf_feature_value.
        """
        if not cleaned_text:
            return {"top_features": [], "summary": "No text provided for explanation."}

        # Transform single document
        tfidf_vec = vectorizer.transform([cleaned_text])
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Find class index
        classes = list(model.classes_)
        if predicted_class not in classes:
            return {"top_features": [], "summary": f"Class '{predicted_class}' not found in model."}

        class_idx = classes.index(predicted_class)

        # In binary vs multiclass LogisticRegression
        if len(classes) == 2 and model.coef_.shape[0] == 1:
            # Binary classification shape is (1, n_features)
            coefs = model.coef_[0] if class_idx == 1 else -model.coef_[0]
        else:
            coefs = model.coef_[class_idx]

        # Non-zero indices in tfidf_vec
        non_zero_indices = tfidf_vec.nonzero()[1]
        if len(non_zero_indices) == 0:
            return {"top_features": [], "summary": "No known vocabulary terms detected."}

        contributions = []
        for idx in non_zero_indices:
            weight = coefs[idx]
            tfidf_val = tfidf_vec[0, idx]
            score = weight * tfidf_val
            contributions.append({
                "feature": feature_names[idx],
                "score": round(float(score), 4),
                "tfidf": round(float(tfidf_val), 4),
                "coef": round(float(weight), 4),
            })

        # Sort descending by score
        contributions.sort(key=lambda x: x["score"], reverse=True)
        top_positive = [c for c in contributions if c["score"] > 0][:top_k]

        driver_words = [f"'{c['feature']}' (+{c['score']})" for c in top_positive]
        if driver_words:
            summary = f"Identified {len(top_positive)} primary driver term(s): {', '.join(driver_words)}."
        else:
            summary = "Model decision based on baseline class priors; no strong positive keyword features."

        return {
            "top_features": top_positive,
            "summary": summary,
            "all_active_tokens_count": len(non_zero_indices)
        }
