# Customer Support Chatbot MLOps and Agentic AI System

A production-grade, end-to-end customer complaint intelligence and triage platform built with **Python, Scikit-learn, Pandas, FastAPI, Streamlit, and lightweight multi-agent orchestration**.

Designed to demonstrate modern **MLOps best practices**, **interpretable linear explainability**, **multi-agent workflow automation**, and an **interactive customer intelligence chatbot**—all without paid APIs, heavy downloads, or mandatory GPU clusters.

---

## 🌟 Key Features

1. **Multi-Task Machine Learning:**
   - **TF-IDF + Logistic Regression** classifiers trained for 4 concurrent targets:
     - **Issue Category** (6 classes: Billing, Security, Credit, Fraud, Loans, Customer Service)
     - **Customer Sentiment** (Positive/Neutral, Dissatisfied, Extremely Frustrated/Hostile)
     - **Urgency Level** (Low, Medium, High, Critical)
     - **Remediation Action** (Fee Waiver, Security Freeze, Credit Correction, Escrow Review, etc.)
   - Calibrated probability estimates and confidence scoring.
   - Sub-millisecond latency: **p50 = 0.67 ms**, **p95 = 1.35 ms**.

2. **5-Stage Modular Multi-Agent Pipeline:**
   - **Validation Agent:** Input sanitization, length validation, and regex-based PII scrubbing (SSN, credit card, email, phone numbers).
   - **Classification Agent:** Multi-task inference with exact linear feature attribution ($w_i \cdot x_i$) for full prediction transparency.
   - **Investigation Agent:** Named entity extraction (disputed amounts, dates) and historical pattern matching via TF-IDF cosine similarity.
   - **Priority Routing Agent:** Multi-factor scoring algorithm (0–100) factoring urgency, customer tier (VIP/Standard), financial loss, and regulatory/legal triggers (CFPB, lawsuit).
   - **Recommendation Agent:** Generates step-by-step SOP checklists, drafted customer emails, goodwill compensation advice, and compliance mandates.
   - **Execution Traces:** Step-by-step waterfall capturing latency and status per agent.

3. **Interactive AI Chatbot:**
   - Deterministic tool-dispatched conversational engine.
   - Answers natural language questions regarding complaint volume and recurring trends.
   - Looks up individual complaint records by ID (`CMP-1005`).
   - Analyzes custom complaint narratives on the fly (`Analyze: <text>`).
   - Dispatches operational tools: `search_complaints`, `get_complaint_by_id`, `identify_recurring_issues`, `explain_prediction_for_text`.

4. **MLOps Governance & Active Learning:**
   - **Versioned Model Registry:** `models/registry/v1.0.0/` stores models, vectorizers, and metadata.
   - **Evaluation Benchmarks:** Macro-F1, Precision, Recall, Accuracy, Confusion Matrices, and Latency percentiles.
   - **Local Experiment Tracking:** Automatically logs runs to `models/experiments/experiment_log.json` (with graceful MLflow mirroring if installed).
   - **Human-in-the-Loop Review Queue:** Analysts can flag misclassifications, suggest corrections, and trigger one-click retraining that incorporates reviewed samples into the active model.

5. **Modern Streamlit SaaS Interface:**
   - 7 intuitive modules: Executive Overview, Real-Time Analyzer, Priority Triage Queue, AI Chatbot, Batch CSV Processing, MLOps Governance, and Human Review.

---

## 📂 Project Directory Structure

```text
MLOps Project/
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── complaints_synthetic.csv     # 700 synthetic realistic complaints
│   ├── feedback/
│   │   └── human_feedback.json         # Active learning feedback store
│   └── sample_upload.csv               # 25-row demo file for batch upload
├── models/
│   ├── registry/                       # Versioned model artifacts (v1.0.0, etc.)
│   │   ├── current_model_pointer.json
│   │   └── v1.0.0/
│   │       ├── category_model.joblib
│   │       ├── sentiment_model.joblib
│   │       ├── urgency_model.joblib
│   │       ├── action_model.joblib
│   │       ├── vectorizer.joblib
│   │       └── metadata.json
│   └── experiments/
│       └── experiment_log.json         # Experiment tracking history
├── src/
│   ├── config.py                       # Global paths, constants, and hyperparameters
│   ├── data/
│   │   └── generate_dataset.py         # Synthetic complaint generator
│   ├── ml/
│   │   ├── preprocessor.py             # PII redaction & text normalization
│   │   ├── explainability.py           # Linear feature attribution engine
│   │   ├── registry.py                 # Versioned model registry manager
│   │   ├── train.py                    # Training & latency benchmarking
│   │   ├── evaluate.py                 # Standalone model evaluation
│   │   └── qlora_stub.py               # Decoupled QLoRA fine-tuning stub
│   ├── agents/
│   │   ├── base.py                     # BaseAgent interface with timing & traces
│   │   ├── validation_agent.py         # PII masking & validation
│   │   ├── classification_agent.py     # Multi-task ML inference
│   │   ├── priority_routing_agent.py   # Multi-factor priority scoring
│   │   ├── investigation_agent.py      # Entity extraction & pattern retrieval
│   │   ├── recommendation_agent.py     # SOP & response draft generator
│   │   └── orchestrator.py             # 5-Agent orchestrator
│   ├── chatbot/
│   │   ├── tools.py                    # Deterministic complaint tools
│   │   └── bot.py                      # Conversational triage engine
│   ├── api/
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── app.py                      # FastAPI REST endpoints
│   └── utils/
│       ├── logger.py                   # Structured logging
│       └── metrics_tracker.py          # Real-time inference latency tracking
├── dashboard/
│   ├── app.py                          # Streamlit SaaS dashboard
│   └── components/
│       └── styles.py                   # Custom CSS theme
└── tests/
    ├── test_data.py                    # Dataset validation tests
    ├── test_ml.py                      # Preprocessing, training, and explainability tests
    ├── test_agents.py                  # Multi-agent and orchestrator tests
    ├── test_chatbot.py                 # Chatbot tools and interaction tests
    └── test_api.py                     # FastAPI REST endpoint tests
```

---

## 🚀 Quickstart & Windows Launch Commands

All core dependencies are pre-installed in your Python environment. Run the commands below in PowerShell or Windows Terminal from `c:\Users\eshas\Downloads\MLOps Project`.

### Step 1: Run the Automated Test Suite

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m pytest -v
```
*(All 18 automated tests pass 100% in ~3 seconds).*

---

### Step 2: Launch the FastAPI Backend Server

In your first terminal window, start the REST API:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

---

### Step 3: Launch the Streamlit SaaS Dashboard

In a second terminal window, start the interactive dashboard:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m streamlit run dashboard/app.py --server.port 8501
```

- **Dashboard UI URL:** `http://localhost:8501`

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check, uptime, and active model version |
| `POST` | `/api/v1/predict` | Fast single prediction (Category, Urgency, Action, Sentiment) |
| `POST` | `/api/v1/analyze` | Full 5-Agent pipeline with explainability and execution traces |
| `POST` | `/api/v1/batch-predict` | Batch complaint analysis |
| `POST` | `/api/v1/chat` | Conversational chatbot endpoint |
| `GET` | `/api/v1/models` | List all registered model versions and active status |
| `POST` | `/api/v1/retrain` | Trigger model retraining incorporating human review feedback |
| `POST` | `/api/v1/feedback` | Submit human reviewer corrections |
| `GET` | `/api/v1/feedback` | Retrieve active learning feedback queue |
| `GET` | `/api/v1/metrics` | Operational inference latency stats (p50, p95, p99) and metrics |
| `GET` | `/api/v1/complaints` | Filter and query historical complaint dataset |

---

## 💼 Resume-Ready Project Descriptions

### Option 1: For Machine Learning Engineer / MLOps Roles
> **Customer Support Chatbot MLOps and Agentic AI System**  
> *Python, Scikit-learn, FastAPI, Streamlit, Pandas, MLflow, Pytest*  
> - Architected a production-ready customer complaint intelligence platform processing 700+ financial cases with sub-millisecond inference latency ($p50 = 0.67\text{ ms}$, $p95 = 1.35\text{ ms}$).
> - Engineered a multi-task TF-IDF + Logistic Regression pipeline classifying issue categories, customer sentiment, urgency, and recommended actions, achieving $1.0$ Category Macro-F1 and $0.91$ Urgency Macro-F1.
> - Implemented a modular 5-agent orchestration framework (Data Validation, Classification, Complaint Investigation, Priority Routing, and Recommendation) with automated PII masking and full step-by-step latency execution traces.
> - Developed an active learning feedback loop and versioned model registry (`v1.x`) with automated experiment tracking, confusion matrix telemetry, and one-click model retraining.
> - Delivered a modern 7-module Streamlit SaaS dashboard and FastAPI gateway tested with 100% pass rate across 18 automated unit and integration tests.

### Option 2: For AI Engineer / Full-Stack AI Roles
> **Enterprise Customer Intelligence Platform & Agentic AI Chatbot**  
> *FastAPI, Streamlit, Scikit-learn, Python, Plotly, Pytest*  
> - Built an end-to-end customer triage platform featuring a deterministic, tool-calling AI chatbot that interrogates historical databases, identifies recurring systemic failure modes, and synthesizes evidence-based remediation playbooks.
> - Designed exact linear explainability using Logistic Regression feature attribution ($w_i \cdot x_i$) to explain model classifications and driver terms to non-technical support agents.
> - Created an automated multi-criteria priority scoring engine (0–100) factoring customer tier, monetary dispute amounts, and statutory regulatory risk triggers (CFPB, FCRA, Reg E).
> - Integrated RESTful FastAPI backend with OpenAPI documentation, CSV batch inference, and interactive Streamlit analytics.
