"""
Multi-Agent Orchestrator Pipeline
Coordinates execution across Validation, Classification, Priority Routing,
Investigation, and Recommendation agents, collecting comprehensive execution traces.
"""

import time
import uuid
from typing import Dict, Any, List, Optional

from src.agents.validation_agent import DataValidationAgent
from src.agents.classification_agent import ClassificationAgent
from src.agents.priority_routing_agent import PriorityRoutingAgent
from src.agents.investigation_agent import ComplaintInvestigationAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.utils.logger import get_logger

logger = get_logger("orchestrator")


class ComplaintAgentOrchestrator:
    """Manages sequential execution and state propagation across specialized AI agents."""

    def __init__(self, model_version: Optional[str] = None):
        self.validation_agent = DataValidationAgent()
        self.classification_agent = ClassificationAgent(model_version=model_version)
        self.priority_agent = PriorityRoutingAgent()
        self.investigation_agent = ComplaintInvestigationAgent()
        self.recommendation_agent = RecommendationAgent()

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full agentic analysis pipeline on a complaint record.
        """
        trace_id = f"TRC-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.perf_counter()

        # Initialize pipeline state
        state = {
            "trace_id": trace_id,
            "complaint_id": payload.get("complaint_id") or f"CMP-{uuid.uuid4().hex[:6].upper()}",
            "customer_id": payload.get("customer_id") or "CUST-ANON",
            "customer_name": payload.get("customer_name") or "Customer",
            "customer_tier": payload.get("customer_tier") or "Standard",
            "channel": payload.get("channel") or "Web Portal",
            "complaint_text": payload.get("complaint_text") or "",
            "disputed_amount": float(payload.get("disputed_amount") or 0.0),
            "execution_traces": [],
        }

        # Step 1: Validation & PII Redaction
        state = self.validation_agent(state)
        if not state.get("is_valid", True):
            total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
            state["total_pipeline_latency_ms"] = total_latency
            logger.warning(f"Pipeline terminated early at validation step for trace {trace_id}")
            return state

        # Step 2: Multi-target Classification & Explainability
        state = self.classification_agent(state)

        # Step 3: Investigation & Historical Pattern Search
        state = self.investigation_agent(state)

        # Step 4: Multi-factor Priority & SLA Routing
        state = self.priority_agent(state)

        # Step 5: Remediation Recommendations & Draft Generation
        state = self.recommendation_agent(state)

        total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
        state["total_pipeline_latency_ms"] = total_latency
        logger.info(f"Pipeline completed successfully for trace {trace_id} in {total_latency}ms")

        return state

    def batch_process(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a collection of complaint dictionaries."""
        results = []
        for record in records:
            results.append(self.process(record))
        return results


# Default singleton instance
orchestrator = ComplaintAgentOrchestrator()
