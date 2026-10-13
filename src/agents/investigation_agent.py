"""
Complaint Investigation Agent
Extracts financial/case entities, queries historical cases for pattern matching,
and identifies recurring systemic issues using semantic/TF-IDF similarity.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.base import BaseAgent
from src.config import RAW_DATA_PATH
from src.ml.preprocessor import clean_complaint_text


class ComplaintInvestigationAgent(BaseAgent):
    """Investigates root causes and links complaints to historical case patterns."""

    AMOUNT_REGEX = re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})*|\d+)(?:\.\d{2})?")
    DATE_REGEX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b", re.IGNORECASE)
    REF_REGEX = re.compile(r"\b(?:pr|acct|ticket|ref|case)\s*#?\s*([a-z0-9-]+)\b", re.IGNORECASE)

    def __init__(self, historical_data_path: Path = RAW_DATA_PATH):
        super().__init__(
            name="InvestigationAgent",
            role="Root Cause & Historical Pattern Investigation",
            description="Extracts entities, evaluates similar historical cases, and flags recurring failure modes."
        )
        self.historical_data_path = historical_data_path
        self._historical_df: Optional[pd.DataFrame] = None

    def _get_historical_df(self) -> Optional[pd.DataFrame]:
        if self._historical_df is None and self.historical_data_path.exists():
            try:
                self._historical_df = pd.read_csv(self.historical_data_path)
            except Exception:
                self._historical_df = None
        return self._historical_df

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("redacted_text") or state.get("complaint_text", "")
        clean_text = clean_complaint_text(text)

        # 1. Extract entities
        amounts = [float(m.replace(",", "")) for m in self.AMOUNT_REGEX.findall(text)]
        extracted_amount = max(amounts) if amounts else float(state.get("disputed_amount", 0.0) or 0.0)

        dates = self.DATE_REGEX.findall(text)
        refs = self.REF_REGEX.findall(text)

        extracted_entities = {
            "disputed_amounts": amounts,
            "primary_amount": extracted_amount,
            "dates_mentioned": dates,
            "reference_identifiers": refs,
        }

        # 2. Historical similarity retrieval
        historical_df = self._get_historical_df()
        similar_cases = []
        recurring_detected = False
        recurring_summary = "Isolated incident; no direct historical cluster match above similarity threshold."

        if historical_df is not None and not historical_df.empty:
            # Filter by matching category if available to increase relevance
            category = state.get("predicted_category")
            candidate_df = historical_df
            if category and "category" in historical_df.columns:
                subset = historical_df[historical_df["category"] == category]
                if len(subset) >= 5:
                    candidate_df = subset

            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=2000, stop_words="english")
            corpus = candidate_df["complaint_text"].astype(str).tolist()
            try:
                tfidf_mat = vec.fit_transform(corpus)
                query_vec = vec.transform([clean_text])
                sims = cosine_similarity(query_vec, tfidf_mat)[0]

                top_indices = sims.argsort()[::-1][:3]
                for idx in top_indices:
                    sim_score = round(float(sims[idx]), 3)
                    if sim_score > 0.15:
                        row = candidate_df.iloc[idx]
                        similar_cases.append({
                            "complaint_id": str(row.get("complaint_id", f"CMP-{idx}")),
                            "category": str(row.get("category", "")),
                            "similarity_score": sim_score,
                            "disputed_amount": float(row.get("disputed_amount", 0.0)),
                            "status": str(row.get("status", "Resolved")),
                            "snippet": str(row.get("complaint_text", ""))[:120] + "...",
                        })

                if similar_cases and similar_cases[0]["similarity_score"] >= 0.40:
                    recurring_detected = True
                    recurring_summary = (
                        f"High similarity ({similar_cases[0]['similarity_score']*100:.1f}%) with case "
                        f"{similar_cases[0]['complaint_id']} indicates a known recurring pattern in '{category}'."
                    )
            except Exception:
                pass

        return {
            "extracted_entities": extracted_entities,
            "similar_historical_cases": similar_cases,
            "recurring_issue_detected": recurring_detected,
            "investigation_summary": recurring_summary,
            "disputed_amount": extracted_amount,  # Update state with extracted amount
        }
