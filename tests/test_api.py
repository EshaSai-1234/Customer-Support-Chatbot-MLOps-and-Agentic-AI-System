"""
Tests for FastAPI REST API Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model_version" in data


def test_predict_endpoint(client):
    payload = {"complaint_text": "I was charged an unexpected $35 overdraft fee on my account."}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "category_confidence" in data
    assert "predicted_urgency" in data


def test_analyze_endpoint(client):
    payload = {
        "complaint_text": "Someone compromised my credit card and charged $2,500. CFPB complaint pending.",
        "customer_name": "Jane Doe",
        "customer_tier": "VIP",
        "disputed_amount": 2500.0
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["priority_tier"] == "P1-Critical"
    assert len(data["execution_traces"]) == 5


def test_chat_endpoint(client):
    response = client.post("/api/v1/chat", json={"message": "Show overview statistics"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["history"]) >= 1


def test_models_list_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_version" in data
    assert len(data["versions"]) > 0


def test_feedback_submission(client):
    fb_payload = {
        "complaint_id": "CMP-TEST-99",
        "complaint_text": "Test complaint for feedback endpoint",
        "predicted_category": "Customer Service & Delays",
        "corrected_category": "Billing & Overdraft Fees",
        "predicted_urgency": "Low",
        "corrected_urgency": "Medium",
        "reviewer_notes": "Misclassified fee complaint as customer service"
    }
    response = client.post("/api/v1/feedback", json=fb_payload)
    assert response.status_code == 201
    assert response.json()["status"] == "SUCCESS"
