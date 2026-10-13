"""
Model Training and Experiment Tracking Pipeline
Trains TF-IDF + Logistic Regression models for Category, Sentiment, Urgency, and Action.
Computes evaluation metrics (macro-F1, precision, recall, confusion matrix, latency)
and logs to the versioned Model Registry and Experiment Tracker (with optional MLflow support).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    TFIDF_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    EXPERIMENT_LOG_PATH,
    DEFAULT_MODEL_VERSION,
)
from src.ml.preprocessor import clean_complaint_text
from src.ml.registry import registry
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")


def _compute_metrics(y_true, y_pred, labels) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for a single prediction task."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
    }


def _benchmark_latency(vectorizer, models: Dict[str, Any], sample_texts: list, n_iterations: int = 100) -> Dict[str, float]:
    """Measures end-to-end inference latency percentiles in milliseconds."""
    if not sample_texts:
        sample_texts = ["Unauthorized wire transfer drained funds from my account."]
    latencies = []
    
    # Warm-up
    _ = vectorizer.transform([sample_texts[0]])

    for i in range(n_iterations):
        sample = sample_texts[i % len(sample_texts)]
        t0 = time.perf_counter()
        vec = vectorizer.transform([sample])
        for model in models.values():
            _ = model.predict(vec)
            _ = model.predict_proba(vec)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
        "avg_latency_ms": round(float(np.mean(arr)), 2),
    }


def _log_experiment(experiment_record: Dict[str, Any]):
    """Appends experiment record to local JSON log and optional MLflow tracking."""
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if EXPERIMENT_LOG_PATH.exists():
        try:
            with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(experiment_record)
    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Experiment logged to {EXPERIMENT_LOG_PATH}")

    # Optional MLflow integration (graceful fallback if not installed)
    try:
        import mlflow
        mlflow.set_experiment("Agentic_Customer_Insight")
        with mlflow.start_run(run_name=experiment_record["version"]):
            mlflow.log_params(experiment_record["hyperparameters"])
            for task, metrics in experiment_record["metrics"].items():
                mlflow.log_metric(f"{task}_macro_f1", metrics["macro_f1"])
                mlflow.log_metric(f"{task}_accuracy", metrics["accuracy"])
            mlflow.log_metric("p50_latency_ms", experiment_record["latency"]["p50_latency_ms"])
            mlflow.log_metric("p95_latency_ms", experiment_record["latency"]["p95_latency_ms"])
            logger.info("Successfully mirrored experiment metrics to MLflow.")
    except ImportError:
        logger.info("MLflow not installed; used built-in local experiment tracking.")
    except Exception as e:
        logger.warning(f"MLflow logging bypassed due to error: {e}")


def train_pipeline(
    data_path: Path = RAW_DATA_PATH,
    version: str = DEFAULT_MODEL_VERSION,
    feedback_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end training, evaluation, latency benchmarking, and version registration.
    """
    logger.info(f"Starting training pipeline for version {version}...")

    if not data_path.exists():
        from src.data.generate_dataset import generate_dataset
        generate_dataset(output_path=data_path)

    df = pd.read_csv(data_path)

    # Merge feedback data if provided (Active Learning / Retraining)
    if feedback_df is not None and not feedback_df.empty:
        logger.info(f"Incorporating {len(feedback_df)} human feedback samples into training set.")
        df = pd.concat([df, feedback_df], ignore_index=True)

    # Data hash for lineage tracking
    data_bytes = df["complaint_text"].str.cat().encode("utf-8")
    data_hash = hashlib.sha256(data_bytes).hexdigest()[:12]

    # Preprocess text
    df["clean_text"] = df["complaint_text"].astype(str).apply(clean_complaint_text)

    # Split dataset
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"]
    )

    logger.info(f"Training set: {len(train_df)} samples | Test set: {len(test_df)} samples")

    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    targets = {
        "category": "category",
        "sentiment": "sentiment",
        "urgency": "urgency",
        "action": "recommended_action",
    }

    trained_models = {}
    metrics_report = {}

    for target_key, col_name in targets.items():
        logger.info(f"Training Logistic Regression for task: {target_key} ({col_name})...")
        model = LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)
        model.fit(X_train, train_df[col_name])
        trained_models[f"{target_key}_model"] = model

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        unique_labels = sorted(df[col_name].unique())
        metrics_report[target_key] = _compute_metrics(
            y_true=test_df[col_name],
            y_pred=y_test_pred,
            labels=unique_labels
        )
        logger.info(
            f"Task '{target_key}' - Macro F1: {metrics_report[target_key]['macro_f1']} | "
            f"Accuracy: {metrics_report[target_key]['accuracy']}"
        )

    # Benchmark Latency
    latency_stats = _benchmark_latency(
        vectorizer=vectorizer,
        models=trained_models,
        sample_texts=test_df["clean_text"].tolist(),
        n_iterations=100
    )
    logger.info(f"Latency benchmarks: p50={latency_stats['p50_latency_ms']}ms, p95={latency_stats['p95_latency_ms']}ms")

    # Compile Metadata
    metadata = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "total_samples": len(df),
        "data_hash": data_hash,
        "hyperparameters": {
            "tfidf": TFIDF_CONFIG,
            "logistic_regression": LOGISTIC_REGRESSION_CONFIG,
            "test_size": test_size,
        },
        "metrics": metrics_report,
        "latency": latency_stats,
    }

    # Save to Registry
    artifacts = {
        "vectorizer": vectorizer,
        **trained_models
    }
    registry.save_version(version=version, artifacts=artifacts, metadata=metadata)

    # Log Experiment
    _log_experiment(metadata)

    logger.info(f"Pipeline complete! Version {version} successfully deployed as active model.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register customer complaint models")
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION, help="Model version identifier")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    train_pipeline(version=args.version, test_size=args.test_size)
