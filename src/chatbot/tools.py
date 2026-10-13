"""
Lightweight Tool Library for Complaint Intelligence Chatbot
Provides deterministic Python tools to inspect complaints, analyze text,
retrieve recurring issues, explain predictions, and suggest playbooks.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import RAW_DATA_PATH
from src.agents.orchestrator import orchestrator
from src.ml.explainability import ModelExplainer
from src.ml.preprocessor import clean_complaint_text
from src.ml.registry import registry


class ComplaintChatbotTools:
    """Tool execution layer for the chatbot."""

    @classmethod
    def _get_df(cls) -> pd.DataFrame:
        if RAW_DATA_PATH.exists():
            return pd.read_csv(RAW_DATA_PATH)
        return pd.DataFrame()

    @classmethod
    def get_overview_statistics(cls) -> Dict[str, Any]:
        """Calculates macro metrics and distribution statistics across complaints."""
        df = cls._get_df()
        if df.empty:
            return {"error": "No complaint data loaded"}

        total = len(df)
        cat_dist = df["category"].value_counts().to_dict()
        urg_dist = df["urgency"].value_counts().to_dict()
        status_dist = df["status"].value_counts().to_dict() if "status" in df.columns else {}
        total_disputed = float(df["disputed_amount"].sum()) if "disputed_amount" in df.columns else 0.0

        p1_count = urg_dist.get("Critical", 0)
        p1_rate = round((p1_count / total * 100), 1) if total > 0 else 0.0

        return {
            "total_complaints": total,
            "critical_p1_count": p1_count,
            "critical_p1_rate_pct": p1_rate,
            "total_disputed_amount": round(total_disputed, 2),
            "category_distribution": cat_dist,
            "urgency_distribution": urg_dist,
            "status_distribution": status_dist,
        }

    @classmethod
    def search_complaints(
        cls,
        query: Optional[str] = None,
        category: Optional[str] = None,
        urgency: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Searches complaints by keyword, category, or urgency."""
        df = cls._get_df()
        if df.empty:
            return []

        filtered = df.copy()

        if category and category.lower() != "all":
            filtered = filtered[filtered["category"].str.lower() == category.lower()]

        if urgency and urgency.lower() != "all":
            filtered = filtered[filtered["urgency"].str.lower() == urgency.lower()]

        if query:
            q = query.lower()
            filtered = filtered[
                filtered["complaint_text"].str.lower().str.contains(q, na=False) |
                filtered["customer_name"].str.lower().str.contains(q, na=False) |
                filtered["complaint_id"].str.lower().str.contains(q, na=False)
            ]

        results = filtered.head(limit).to_dict(orient="records")
        return results

    @classmethod
    def get_complaint_by_id(cls, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Fetches full complaint details by ID (e.g. CMP-1005)."""
        df = cls._get_df()
        if df.empty:
            return None

        match = df[df["complaint_id"].astype(str).str.upper() == complaint_id.strip().upper()]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    @classmethod
    def analyze_complaint_text(
        cls,
        text: str,
        customer_name: str = "Customer",
        customer_tier: str = "Standard",
        disputed_amount: float = 0.0
    ) -> Dict[str, Any]:
        """Runs the entire 5-agent pipeline on raw complaint text."""
        payload = {
            "complaint_text": text,
            "customer_name": customer_name,
            "customer_tier": customer_tier,
            "disputed_amount": disputed_amount,
        }
        return orchestrator.process(payload)

    @classmethod
    def identify_recurring_issues(cls, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Finds recurring complaint clusters and frequent failure patterns."""
        df = cls._get_df()
        if df.empty:
            return []

        filtered = df if not category else df[df["category"].str.lower() == category.lower()]
        if filtered.empty:
            return []

        patterns = []
        for cat, group in filtered.groupby("category"):
            total_in_cat = len(group)
            avg_amount = float(group["disputed_amount"].mean())
            high_urgency_pct = round((group["urgency"].isin(["High", "Critical"]).sum() / total_in_cat) * 100, 1)

            patterns.append({
                "category": cat,
                "case_count": total_in_cat,
                "avg_disputed_amount": round(avg_amount, 2),
                "high_urgency_rate_pct": high_urgency_pct,
                "most_common_action": group["recommended_action"].mode().iloc[0] if "recommended_action" in group else "N/A",
                "sample_case": group.iloc[0]["complaint_text"][:140] + "...",
            })

        return sorted(patterns, key=lambda x: x["case_count"], reverse=True)

    @classmethod
    def explain_prediction_for_text(cls, text: str) -> Dict[str, Any]:
        """Provides TF-IDF feature attribution breakdown for input text."""
        artifacts = registry.load_version()
        vectorizer = artifacts["vectorizer"]
        cat_model = artifacts["category_model"]

        clean_text = clean_complaint_text(text)
        tfidf = vectorizer.transform([clean_text])
        pred_class = cat_model.predict(tfidf)[0]

        return ModelExplainer.explain_prediction(
            vectorizer=vectorizer,
            model=cat_model,
            cleaned_text=clean_text,
            predicted_class=pred_class,
            top_k=7
        )
