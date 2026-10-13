"""
Conversational Agentic Chatbot Engine
Interprets customer intelligence queries, invokes specialized Python tools,
explains predictions, and provides formatted, evidence-based responses.
"""

import re
from typing import Dict, Any, List, Optional
from src.chatbot.tools import ComplaintChatbotTools
from src.utils.logger import get_logger

logger = get_logger("chatbot")


class ComplaintIntelligenceBot:
    """Chatbot for customer complaint investigation, prediction explanation, and triage."""

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def clear_history(self):
        self.history = []

    def respond(self, user_message: str) -> str:
        """Processes user input, dispatches tools, and formats a markdown response."""
        msg = user_message.strip()
        if not msg:
            return "Please provide a query, complaint ID, or paste complaint text to analyze."

        self.history.append({"role": "user", "content": msg})
        lower_msg = msg.lower()

        # 1. Direct Complaint ID Lookup (e.g. CMP-1025)
        id_match = re.search(r"\b(CMP-\d{4})\b", msg, re.IGNORECASE)
        if id_match and ("analyze" in lower_msg or "show" in lower_msg or "view" in lower_msg or "what is" in lower_msg or len(msg.split()) <= 3):
            complaint_id = id_match.group(1).upper()
            record = ComplaintChatbotTools.get_complaint_by_id(complaint_id)
            if record:
                response = (
                    f"### 📋 Complaint Record: `{complaint_id}`\n\n"
                    f"- **Customer:** {record.get('customer_name')} ({record.get('customer_tier')} Tier)\n"
                    f"- **Category:** `{record.get('category')}`\n"
                    f"- **Urgency:** `{record.get('urgency')}` | **Sentiment:** `{record.get('sentiment')}`\n"
                    f"- **Disputed Amount:** ${float(record.get('disputed_amount', 0.0)):,.2f}\n"
                    f"- **Status:** `{record.get('status')}` | **Channel:** `{record.get('channel')}`\n"
                    f"- **Submission Date:** {record.get('created_at')}\n\n"
                    f"**Complaint Narrative:**\n> *\"{record.get('complaint_text')}\"*\n\n"
                    f"**Recommended Action:**\n> {record.get('recommended_action')}\n\n"
                    f"💡 *Tip: To run the full 5-agent pipeline on this text, type `Analyze: \"{record.get('complaint_text')[:50]}...\"`*"
                )
            else:
                response = f"⚠️ Complaint ID `{complaint_id}` was not found in the historical records."
            self.history.append({"role": "assistant", "content": response})
            return response

        # 2. Text Analysis Intent ("Analyze: ...", "Predict: ...", or long complaint pasted)
        if lower_msg.startswith("analyze:") or lower_msg.startswith("analyze this") or lower_msg.startswith("predict:") or len(msg) > 120:
            clean_input = re.sub(r"^(analyze:?|predict:?|analyze this:?)\s*", "", msg, flags=re.IGNORECASE).strip()
            res = ComplaintChatbotTools.analyze_complaint_text(clean_input)

            if not res.get("is_valid", True):
                errors = ", ".join(res.get("validation_errors", ["Invalid text"]))
                response = f"❌ **Input Validation Failed:** {errors}"
            else:
                traces = res.get("execution_traces", [])
                trace_summary = " ➔ ".join([f"`{t['agent']}` ({t['latency_ms']}ms)" for t in traces])
                top_features = res.get("category_explanation", {}).get("top_features", [])
                keywords_str = ", ".join([f"`{f['feature']}` (+{f['score']})" for f in top_features[:4]]) or "Baseline prior"

                response = (
                    f"### ⚡ Agentic Complaint Investigation Report\n\n"
                    f"| Metric | Assessment | Confidence |\n"
                    f"| :--- | :--- | :--- |\n"
                    f"| **Category** | `{res.get('predicted_category')}` | {res.get('category_confidence')*100:.1f}% |\n"
                    f"| **Urgency** | `{res.get('predicted_urgency')}` | {res.get('urgency_confidence')*100:.1f}% |\n"
                    f"| **Sentiment** | `{res.get('predicted_sentiment')}` | {res.get('sentiment_confidence')*100:.1f}% |\n"
                    f"| **Priority Tier** | **{res.get('priority_tier')}** (Score: {res.get('priority_score')}/100) | Target SLA: {res.get('target_sla_hours')} hrs |\n"
                    f"| **Routing** | `{res.get('target_department')}` | Legal Risk: {'🚨 YES' if res.get('legal_risk_flag') else 'No'} |\n\n"
                    f"**Driver Keywords (Explainability):**\n{keywords_str}\n\n"
                    f"**Action Recommendation:**\n> {res.get('predicted_action')}\n\n"
                    f"**Agent Pipeline Execution Trace:**\n{trace_summary}\n\n"
                    f"**Drafted Customer Response:**\n```\n{res.get('draft_customer_response')}\n```"
                )
            self.history.append({"role": "assistant", "content": response})
            return response

        # 3. Overview / Metrics Intent
        if any(w in lower_msg for w in ["overview", "summary", "kpi", "metrics", "stats", "statistics", "how many"]):
            stats = ComplaintChatbotTools.get_overview_statistics()
            cat_list = "\n".join([f"- **{cat}:** {count} cases" for cat, count in stats.get("category_distribution", {}).items()])
            urg_list = "\n".join([f"- **{urg}:** {count} cases" for urg, count in stats.get("urgency_distribution", {}).items()])

            response = (
                f"### 📊 Customer Complaint Intelligence Overview\n\n"
                f"- **Total Ingested Complaints:** `{stats.get('total_complaints'):,}`\n"
                f"- **P1-Critical Escalations:** `{stats.get('critical_p1_count')}` ({stats.get('critical_p1_rate_pct')}%\n"
                f"- **Total Disputed Value:** `${stats.get('total_disputed_amount'):,.2f}`\n\n"
                f"#### Breakdown by Category:\n{cat_list}\n\n"
                f"#### Breakdown by Urgency:\n{urg_list}"
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # 4. Recurring Issues Intent
        if any(w in lower_msg for w in ["recurring", "patterns", "clusters", "repeat", "frequent", "trend"]):
            patterns = ComplaintChatbotTools.identify_recurring_issues()
            pattern_rows = []
            for p in patterns[:5]:
                pattern_rows.append(
                    f"| **{p['category']}** | {p['case_count']} | ${p['avg_disputed_amount']:,.2f} | {p['high_urgency_rate_pct']}% | {p['most_common_action']} |"
                )
            table_body = "\n".join(pattern_rows)

            response = (
                f"### 🔍 Recurring Systemic Issues & Failure Modes\n\n"
                f"| Category | Volume | Avg Disputed | High Urgency Rate | Default Action |\n"
                f"| :--- | :--- | :--- | :--- | :--- |\n"
                f"{table_body}\n\n"
                f"💡 *Key Finding: Fraud and Billing issues account for the largest volume of escalated claims.*"
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # 5. Search Complaints Intent
        if any(w in lower_msg for w in ["search", "find", "show complaints", "list complaints", "lookup", "filter"]):
            query_term = re.sub(r"(search|find|show complaints|list complaints|lookup|filter)\s*", "", msg, flags=re.IGNORECASE).strip()
            results = ComplaintChatbotTools.search_complaints(query=query_term if query_term else None, limit=4)
            if results:
                rows = []
                for r in results:
                    snippet = r['complaint_text'][:90] + "..."
                    rows.append(f"- **`{r['complaint_id']}`** | `{r['category']}` | *{r['urgency']}* | ${float(r.get('disputed_amount', 0)):,.2f}\n  > \"{snippet}\"")
                response = f"### 🔎 Search Results ({len(results)} matches):\n\n" + "\n\n".join(rows)
            else:
                response = f"No complaints found matching '{query_term}'."
            self.history.append({"role": "assistant", "content": response})
            return response

        # Default Guide / Assistant Help
        response = (
            f"👋 Hello! I am your **Customer Complaint Intelligence AI Assistant**.\n\n"
            f"Here are ways I can help you investigate and triage complaints:\n\n"
            f"1. **Analyze a live complaint:**\n"
            f"   > Type `Analyze: Someone stole $1,200 via unauthorized wire transfer!`\n\n"
            f"2. **Look up a specific complaint ID:**\n"
            f"   > Type `Show CMP-1015`\n\n"
            f"3. **Dataset Overview & Metrics:**\n"
            f"   > Type `Show complaint metrics overview`\n\n"
            f"4. **Identify recurring issues & trends:**\n"
            f"   > Type `What are the recurring issues?`\n\n"
            f"5. **Search database:**\n"
            f"   > Type `Search overdraft fees` or `Search unauthorized`"
        )
        self.history.append({"role": "assistant", "content": response})
        return response


# Global bot instance
chatbot = ComplaintIntelligenceBot()
