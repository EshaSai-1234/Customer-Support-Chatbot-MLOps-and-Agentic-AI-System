"""
Priority Routing Agent
Calculates multi-criteria priority scores (0-100), assigns SLA tiers,
and directs complaints to specialized operational departments.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.config import SLA_HOURS, DEPARTMENTS, LEGAL_KEYWORDS


class PriorityRoutingAgent(BaseAgent):
    """Calculates objective priority scores and SLA routing based on multi-factor heuristics."""

    def __init__(self):
        super().__init__(
            name="PriorityRoutingAgent",
            role="Priority Scoring & Department Routing",
            description="Computes priority scores (0-100), SLA limits, and departmental assignments."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        urgency = state.get("predicted_urgency", "Medium")
        sentiment = state.get("predicted_sentiment", "Dissatisfied")
        category = state.get("predicted_category", "Customer Service & Delays")
        text = (state.get("redacted_text") or state.get("complaint_text", "")).lower()
        customer_tier = state.get("customer_tier") or "Standard"
        if customer_tier == "Standard":
            if "vip" in text or "private bank" in text:
                customer_tier = "VIP"
            elif "platinum" in text or "gold" in text or "premier" in text:
                customer_tier = "Gold"
            elif "silver" in text or "preferred" in text:
                customer_tier = "Silver"
        state["customer_tier"] = customer_tier
        disputed_amount = float(state.get("disputed_amount", 0.0) or 0.0)

        score = 0
        factors: List[str] = []

        # 1. Base Urgency Contribution
        urgency_weights = {
            "Critical": 55,
            "High": 35,
            "Medium": 20,
            "Low": 5
        }
        base_urgency = urgency_weights.get(urgency, 20)
        score += base_urgency
        factors.append(f"Base urgency [{urgency}]: +{base_urgency} pts")

        # 2. Sentiment Penalty
        if sentiment == "Extremely Frustrated / Hostile":
            score += 15
            factors.append("Extreme customer frustration / hostility: +15 pts")
        elif sentiment == "Dissatisfied":
            score += 5
            factors.append("Customer dissatisfaction: +5 pts")

        # 3. Customer Tier Multiplier
        tier_weights = {
            "VIP": 20,
            "Gold": 10,
            "Silver": 5,
            "Standard": 0
        }
        tier_bonus = tier_weights.get(customer_tier, 0)
        if tier_bonus > 0:
            score += tier_bonus
            factors.append(f"Customer relationship tier [{customer_tier}]: +{tier_bonus} pts")

        # 4. Financial Impact Threshold
        if disputed_amount >= 2000.0:
            score += 15
            factors.append(f"High financial impact (${disputed_amount:,.2f} >= $2,000): +15 pts")
        elif disputed_amount >= 500.0:
            score += 8
            factors.append(f"Moderate financial impact (${disputed_amount:,.2f} >= $500): +8 pts")

        # 5. Regulatory / Legal Risk Flags
        detected_legal_terms = [kw for kw in LEGAL_KEYWORDS if kw in text]
        legal_risk_flag = len(detected_legal_terms) > 0
        if legal_risk_flag:
            score += 25
            factors.append(f"Regulatory / legal escalation keywords found ({', '.join(detected_legal_terms)}): +25 pts")

        # Clamp score between 0 and 100
        priority_score = min(100, max(0, score))

        # Assign Priority Tier and SLA
        if priority_score >= 80 or urgency == "Critical":
            priority_tier = "P1-Critical"
            sla_key = "Critical"
        elif priority_score >= 60 or urgency == "High":
            priority_tier = "P2-High"
            sla_key = "High"
        elif priority_score >= 35:
            priority_tier = "P3-Medium"
            sla_key = "Medium"
        else:
            priority_tier = "P4-Low"
            sla_key = "Low"

        target_sla_hours = SLA_HOURS[sla_key]

        # Target Department Routing
        if legal_risk_flag:
            department = "Regulatory Compliance & Legal Escalations"
        elif priority_tier == "P1-Critical" and category == "Fraud & Unauthorized Transactions":
            department = "Fraud Emergency Response Unit"
        else:
            department = DEPARTMENTS.get(category, "Customer Experience Escalations")

        return {
            "priority_score": priority_score,
            "priority_tier": priority_tier,
            "target_sla_hours": target_sla_hours,
            "target_department": department,
            "legal_risk_flag": legal_risk_flag,
            "priority_scoring_factors": factors,
        }
