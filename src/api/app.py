"""
FastAPI REST API for Agentic Customer Insight Platform
Serves ML predictions, agentic multi-stage analyses, chatbot interactions,
human feedback collection, and model retraining pipelines.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.config import (
    FEEDBACK_DATA_PATH,
    RAW_DATA_PATH,
    DEFAULT_MODEL_VERSION,
)
from src.agents.orchestrator import orchestrator
from src.agents.classification_agent import ClassificationAgent
from src.chatbot.bot import chatbot
from src.ml.registry import registry
from src.ml.train import train_pipeline
from src.utils.logger import get_logger
from src.utils.metrics_tracker import global_tracker
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    FullAnalysisRequest,
    FullAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    ChatRequest,
    ChatResponse,
    HumanFeedbackRequest,
    RetrainRequest,
)

logger = get_logger("fastapi_app")

app = FastAPI(
    title="Agentic Customer Insight Platform API",
    description="Production-ready REST API for ML Complaint Intelligence, Multi-Agent Triage, and Chatbot Assistance.",
    version="1.0.0"
)

# Enable CORS for frontends (e.g. Streamlit or web apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()
_fast_classifier = ClassificationAgent()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint exposing active model version and system uptime."""
    return {
        "status": "healthy",
        "service": "Agentic Customer Insight API",
        "active_model_version": registry.get_active_version(),
        "uptime_seconds": round(time.time() - _start_time, 2)
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_single(req: PredictionRequest):
    """Low-latency machine learning prediction across Category, Sentiment, Urgency, and Action."""
    t0 = time.perf_counter()
    state = {"complaint_text": req.complaint_text}
    result = _fast_classifier(state)

    if result.get("classification_skipped"):
        global_tracker.record_error()
        raise HTTPException(status_code=400, detail="Invalid text input for classification.")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    global_tracker.record_inference(
        latency_ms=latency_ms,
        category=result["predicted_category"],
        urgency=result["predicted_urgency"]
    )

    return {
        "predicted_category": result["predicted_category"],
        "category_confidence": result["category_confidence"],
        "predicted_sentiment": result["predicted_sentiment"],
        "sentiment_confidence": result["sentiment_confidence"],
        "predicted_urgency": result["predicted_urgency"],
        "urgency_confidence": result["urgency_confidence"],
        "predicted_action": result["predicted_action"],
        "action_confidence": result["action_confidence"],
        "category_explanation": result.get("category_explanation", {}),
        "model_version": result.get("active_model_version", DEFAULT_MODEL_VERSION),
    }


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
def analyze_complaint(req: FullAnalysisRequest):
    """Full 5-Agent orchestration analysis pipeline returning comprehensive audit trace."""
    t0 = time.perf_counter()
    payload = req.model_dump()
    result = orchestrator.process(payload)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if result.get("predicted_category") and result.get("predicted_urgency"):
        global_tracker.record_inference(
            latency_ms=latency_ms,
            category=result["predicted_category"],
            urgency=result["predicted_urgency"]
        )

    return result


@app.post("/api/v1/batch-predict", response_model=BatchAnalysisResponse)
def batch_analyze(req: BatchAnalysisRequest):
    """Batch analysis of multiple complaints."""
    records = [c.model_dump() for c in req.complaints]
    results = orchestrator.batch_process(records)
    return {
        "total_processed": len(results),
        "results": results
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Interactive natural language chatbot endpoint for complaint querying and assistance."""
    answer = chatbot.respond(req.message)
    return {
        "response": answer,
        "history": chatbot.history
    }


@app.get("/api/v1/models")
def list_models():
    """Lists registered model versions, active state, and metadata."""
    return {
        "active_version": registry.get_active_version(),
        "versions": registry.list_versions()
    }


@app.get("/api/v1/metrics")
def get_metrics():
    """Returns real-time inference latency percentiles and active model metrics."""
    metadata = registry.get_metadata()
    operational_stats = global_tracker.get_stats()
    return {
        "operational_metrics": operational_stats,
        "active_model_metadata": metadata
    }


@app.get("/api/v1/complaints")
def list_complaints(
    category: Optional[str] = Query(None),
    urgency: Optional[str] = Query(None),
    limit: int = Query(50, le=500)
):
    """Retrieves complaints from historical database with optional filters."""
    if not RAW_DATA_PATH.exists():
        return []
    df = pd.read_csv(RAW_DATA_PATH)
    if category and category.lower() != "all":
        df = df[df["category"].str.lower() == category.lower()]
    if urgency and urgency.lower() != "all":
        df = df[df["urgency"].str.lower() == urgency.lower()]
    return df.head(limit).to_dict(orient="records")


@app.post("/api/v1/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(fb: HumanFeedbackRequest):
    """Collects human analyst feedback / label corrections for active learning retraining."""
    FEEDBACK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = []
    if FEEDBACK_DATA_PATH.exists():
        try:
            with open(FEEDBACK_DATA_PATH, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            items = []

    record = fb.model_dump()
    record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    items.append(record)

    with open(FEEDBACK_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    logger.info(f"Human feedback recorded for complaint {fb.complaint_id}")
    return {"status": "SUCCESS", "message": "Feedback submitted successfully", "total_feedback_items": len(items)}


@app.get("/api/v1/feedback")
def get_feedback_items():
    """Retrieves all submitted human feedback records."""
    if not FEEDBACK_DATA_PATH.exists():
        return []
    try:
        with open(FEEDBACK_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.post("/api/v1/retrain")
def retrain_model(req: RetrainRequest):
    """Triggers model retraining, optionally integrating human review feedback."""
    feedback_df = None
    if req.incorporate_feedback and FEEDBACK_DATA_PATH.exists():
        try:
            with open(FEEDBACK_DATA_PATH, "r", encoding="utf-8") as f:
                fb_list = json.load(f)
            if fb_list:
                reformatted = []
                for idx, item in enumerate(fb_list):
                    reformatted.append({
                        "complaint_id": item.get("complaint_id", f"FB-{idx}"),
                        "customer_id": "CUST-FEEDBACK",
                        "customer_name": "Reviewer Sample",
                        "customer_tier": "Standard",
                        "channel": "Human Review",
                        "complaint_text": item.get("complaint_text"),
                        "category": item.get("corrected_category"),
                        "sentiment": "Dissatisfied",
                        "urgency": item.get("corrected_urgency"),
                        "recommended_action": "Standard Guidance & Self-Service FAQ",
                        "disputed_amount": 0.0,
                        "created_at": item.get("timestamp"),
                        "status": "Reviewed",
                    })
                feedback_df = pd.DataFrame(reformatted)
                logger.info(f"Loaded {len(feedback_df)} human feedback samples for retraining.")
        except Exception as e:
            logger.warning(f"Failed loading feedback dataset: {e}")

    # Generate version string if not provided
    version = req.new_version
    if not version:
        all_versions = registry.list_versions()
        version = f"v1.0.{len(all_versions)}"

    metadata = train_pipeline(version=version, feedback_df=feedback_df)
    return {
        "status": "SUCCESS",
        "version": version,
        "samples_trained": metadata["train_samples"],
        "metrics": metadata["metrics"]
    }
