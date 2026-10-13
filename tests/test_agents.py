"""
Tests for Modular Multi-Agent Pipeline
"""

import pytest
from src.agents.validation_agent import DataValidationAgent
from src.agents.classification_agent import ClassificationAgent
from src.agents.priority_routing_agent import PriorityRoutingAgent
from src.agents.investigation_agent import ComplaintInvestigationAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.orchestrator import ComplaintAgentOrchestrator


def test_validation_agent_rejects_empty():
    agent = DataValidationAgent()
    state = {"complaint_text": "Hi"}
    res = agent(state)
    assert res["is_valid"] is False
    assert res["validation_status"] == "FAILED"


def test_validation_agent_scrubs_pii():
    agent = DataValidationAgent()
    state = {"complaint_text": "Please fix my account. My SSN is 987-65-4321 and card is 4532 8901 2345 6789."}
    res = agent(state)
    assert res["is_valid"] is True
    assert res["pii_detected"] is True
    assert "[REDACTED_SSN]" in res["redacted_text"]


def test_priority_routing_agent_escalates_legal():
    agent = PriorityRoutingAgent()
    state = {
        "complaint_text": "I will file a lawsuit and report your bank to the CFPB for fraud!",
        "predicted_urgency": "High",
        "predicted_sentiment": "Extremely Frustrated / Hostile",
        "customer_tier": "Gold",
        "disputed_amount": 1500.0,
    }
    res = agent(state)
    assert res["legal_risk_flag"] is True
    assert res["priority_score"] >= 80
    assert res["priority_tier"] == "P1-Critical"
    assert res["target_sla_hours"] == 2


def test_full_orchestrator_pipeline():
    orchestrator = ComplaintAgentOrchestrator()
    payload = {
        "complaint_text": "I was assessed an unauthorized $45 overdraft fee on my account.",
        "customer_name": "Test Customer",
        "customer_tier": "Silver",
        "disputed_amount": 45.0,
    }
    result = orchestrator.process(payload)

    assert result["is_valid"] is True
    assert "predicted_category" in result
    assert "predicted_urgency" in result
    assert "priority_tier" in result
    assert "recommended_sop" in result
    assert "draft_customer_response" in result
    assert len(result["execution_traces"]) == 5
