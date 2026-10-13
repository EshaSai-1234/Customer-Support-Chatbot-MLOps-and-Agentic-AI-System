"""
Recommendation Agent
Synthesizes agentic insights to formulate remediation action plans,
personalized customer response drafts, and compliance-aligned SOP steps.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent


class RecommendationAgent(BaseAgent):
    """Generates remediation playbooks, customer email drafts, and compliance advice."""

    def __init__(self):
        super().__init__(
            name="RecommendationAgent",
            role="Remediation Playbook & Response Synthesis",
            description="Synthesizes findings into actionable agent playbooks and drafted customer communications."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        category = state.get("predicted_category", "General Inquiries")
        action = state.get("predicted_action", "Standard Guidance & Self-Service FAQ")
        urgency = state.get("predicted_urgency", "Medium")
        priority_tier = state.get("priority_tier", "P3-Medium")
        sla_hours = state.get("target_sla_hours", 24)
        customer_name = state.get("customer_name") or "Valued Customer"
        disputed_amount = float(state.get("disputed_amount", 0.0) or 0.0)
        legal_flag = state.get("legal_risk_flag", False)
        recurring = state.get("recurring_issue_detected", False)

        # 1. Action Checklist SOP
        sop_steps: List[str] = []
        if category == "Fraud & Unauthorized Transactions":
            sop_steps = [
                "1. Immediately place a protective freeze on compromised debit/credit cards and affected account numbers.",
                f"2. Issue provisional credit of ${disputed_amount:,.2f} under Regulation E guidelines within 5 business days.",
                "3. Transmit security verification link to customer via out-of-band trusted SMS/Email for credential reset.",
                "4. File internal Suspicious Activity Report (SAR) with Fraud Investigation Unit."
            ]
        elif category == "Billing & Overdraft Fees":
            sop_steps = [
                "1. Review transaction history for the past 60 days to verify account balance timeline and deposit timing.",
                f"2. Apply automatic courtesy fee reversal of ${disputed_amount:,.2f} under customer retention policy.",
                "3. Offer enrollment in Overdraft Protection grace period to prevent future automated charges.",
                "4. Send billing adjustment confirmation receipt to customer."
            ]
        elif category == "Credit Reporting & Inquiries":
            sop_steps = [
                "1. Retrieve official Metro 2 data reporting logs for the disputed trade-line.",
                "2. Initiate Automated Credit Dispute Verification (ACDV) through e-OSCAR portal.",
                "3. Send 30-day formal FCRA compliance resolution notice to customer and credit bureaus.",
                "4. Update loan status to 'Account Closed / Paid in Full with Zero Balance'."
            ]
        elif category == "Account Access & Security":
            sop_steps = [
                "1. Verify customer identity via two-factor biometric callback or secondary knowledge-based authentication.",
                "2. Clear security lockout flag in IAM directory and reset multi-factor device pairing.",
                "3. Guide customer through secure portal onboarding and mobile application re-authentication.",
                "4. Monitor session for 24 hours to confirm stable account access."
            ]
        elif category == "Loan & Mortgage Servicing":
            sop_steps = [
                "1. Route file to Senior Escrow & Loan Servicing Officer for comprehensive ledger recalculation.",
                "2. Place temporary 30-day hold on any adverse delinquency notices or late payment reporting.",
                "3. Audit property tax and insurance disbursements against municipal records.",
                "4. Issue updated monthly amortization schedule and escrow reconciliation summary."
            ]
        else:
            sop_steps = [
                "1. Review previous customer touchpoints in CRM to understand interaction history.",
                "2. Initiate senior representative callback to acknowledge frustration and resolve inquiry directly.",
                "3. Apply goodwill customer retention credit if appropriate.",
                "4. Log customer feedback for quality assurance and training review."
            ]

        # 2. Personalized Customer Email Draft
        amount_mention = f" regarding the disputed amount of ${disputed_amount:,.2f}" if disputed_amount > 0 else ""
        draft_response = (
            f"Dear {customer_name},\n\n"
            f"Thank you for contacting our customer resolutions team. We have received your inquiry{amount_mention} "
            f"and classified it under our {priority_tier} priority response tier. Our dedicated team is actively "
            f"reviewing your case.\n\n"
            f"Action in progress: {action}.\n"
            f"Our commitment: A resolution specialist will follow up with you within {sla_hours} hours with an update "
            f"or final determination.\n\n"
            f"We deeply appreciate your patience and the opportunity to make this right.\n\n"
            f"Sincerely,\n"
            f"Executive Customer Solutions Team"
        )

        # 3. Compensation / Fee Adjustment Advice
        if category == "Billing & Overdraft Fees" and disputed_amount > 0:
            compensation_advice = f"Full fee waiver approved: Credit ${disputed_amount:,.2f} to customer account balance."
        elif category == "Fraud & Unauthorized Transactions" and disputed_amount > 0:
            compensation_advice = f"Provisional credit recommended: Issue ${disputed_amount:,.2f} pending fraud determination."
        elif state.get("customer_tier") == "VIP":
            compensation_advice = "Goodwill gesture recommended: Issue $50.00 customer appreciation credit for VIP account."
        else:
            compensation_advice = "No direct monetary compensation required; standard issue remediation."

        # 4. Regulatory & Compliance Notice
        if legal_flag or "cfpb" in str(state.get("complaint_text", "")).lower():
            compliance_notice = "CRITICAL: Regulatory escalation flag active. Mandatory 15-day written CFPB response requirement applies."
        elif category == "Credit Reporting & Inquiries":
            compliance_notice = "FCRA Mandate: Consumer dispute investigation must conclude within 30 days of initial notice."
        elif category == "Fraud & Unauthorized Transactions":
            compliance_notice = "Regulation E: Consumer liability capped at $50 if notified within 2 business days."
        else:
            compliance_notice = "Standard internal service level agreement applies."

        return {
            "recommended_sop": sop_steps,
            "draft_customer_response": draft_response,
            "compensation_recommendation": compensation_advice,
            "compliance_notice": compliance_notice,
        }
