"""
Global Configuration and Constants for Agentic Customer Insight Platform
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "complaints_synthetic.csv"
SAMPLE_UPLOAD_PATH = DATA_DIR / "sample_upload.csv"
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DATA_PATH = FEEDBACK_DIR / "human_feedback.json"

MODELS_DIR = BASE_DIR / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiment_log.json"
CURRENT_POINTER_PATH = REGISTRY_DIR / "current_model_pointer.json"

# Target Categories & Classes
CATEGORIES = [
    "Billing & Overdraft Fees",
    "Account Access & Security",
    "Credit Reporting & Inquiries",
    "Fraud & Unauthorized Transactions",
    "Loan & Mortgage Servicing",
    "Customer Service & Delays",
]

SENTIMENTS = [
    "Positive / Neutral",
    "Dissatisfied",
    "Extremely Frustrated / Hostile",
]

URGENCIES = [
    "Low",
    "Medium",
    "High",
    "Critical",
]

ACTIONS = [
    "Immediate Security Freeze & Fraud Review",
    "Fee Waiver & Account Balance Adjustment",
    "Credit Bureau Dispute Correction Workflow",
    "Loan Servicing Specialist Outreach",
    "Priority Tier 2 Callback & Investigation",
    "Standard Guidance & Self-Service FAQ",
]

# SLA Standards (Hours)
SLA_HOURS = {
    "Critical": 2,
    "High": 8,
    "Medium": 24,
    "Low": 48,
}

# Routing Department Targets
DEPARTMENTS = {
    "Billing & Overdraft Fees": "Billing & Revenue Operations",
    "Account Access & Security": "Identity & Security Desk",
    "Credit Reporting & Inquiries": "Credit Bureau Compliance",
    "Fraud & Unauthorized Transactions": "Fraud Emergency Response",
    "Loan & Mortgage Servicing": "Lending & Mortgage Servicing",
    "Customer Service & Delays": "Customer Experience Escalations",
}

# Regulatory / High-Risk Triggers
LEGAL_KEYWORDS = [
    "lawyer", "attorney", "legal action", "sue", "lawsuit", "court",
    "cfpb", "consumer financial protection", "complaint to regulator",
    "ftc", "police report", "fraud report", "attorney general"
]

FINANCIAL_IMPACT_KEYWORDS = [
    "stolen", "unauthorized transfer", "drained", "overdraft", "repossession",
    "foreclosure", "identity theft", "credit ruined", "eviction"
]

# ML Pipeline Hyperparameters
TFIDF_CONFIG = {
    "max_features": 4000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "stop_words": "english",
    "min_df": 2,
}

LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.5,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
}

# Service Defaults
DEFAULT_MODEL_VERSION = "v1.0.0"
API_HOST = "127.0.0.1"
API_PORT = 8000
