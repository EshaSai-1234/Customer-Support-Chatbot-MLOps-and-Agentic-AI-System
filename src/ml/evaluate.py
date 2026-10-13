"""
Independent Model Evaluation and Benchmark Runner
Evaluates an active or specific model version against validation or test datasets.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from sklearn.metrics import classification_report

from src.config import RAW_DATA_PATH
from src.ml.preprocessor import clean_complaint_text
from src.ml.registry import registry
from src.ml.train import _compute_metrics, _benchmark_latency
from src.utils.logger import get_logger

logger = get_logger("model_evaluation")


def evaluate_model_version(
    version: Optional[str] = None,
    data_path: Path = RAW_DATA_PATH
) -> Dict[str, Any]:
    """Runs evaluation on a specific model version and returns full metrics."""
    active_version = version or registry.get_active_version()
    logger.info(f"Evaluating model version: {active_version}")

    artifacts = registry.load_version(active_version)
    vectorizer = artifacts["vectorizer"]
    models = {
        "category": artifacts["category_model"],
        "sentiment": artifacts["sentiment_model"],
        "urgency": artifacts["urgency_model"],
        "action": artifacts["action_model"],
    }

    df = pd.read_csv(data_path)
    clean_texts = df["complaint_text"].astype(str).apply(clean_complaint_text).tolist()
    X = vectorizer.transform(clean_texts)

    targets = {
        "category": ("category", models["category"]),
        "sentiment": ("sentiment", models["sentiment"]),
        "urgency": ("urgency", models["urgency"]),
        "action": ("recommended_action", models["action"]),
    }

    results = {}
    for task_name, (col, model) in targets.items():
        y_true = df[col]
        y_pred = model.predict(X)
        labels = sorted(y_true.unique())
        results[task_name] = _compute_metrics(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results[task_name]["detailed_report"] = report

    # Benchmark latency
    latency = _benchmark_latency(vectorizer, models, clean_texts[:100], n_iterations=50)

    summary = {
        "version": active_version,
        "eval_samples": len(df),
        "metrics": results,
        "latency": latency,
        "metadata": artifacts.get("metadata", {})
    }
    logger.info(f"Evaluation complete for {active_version}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None, help="Version to evaluate")
    args = parser.parse_args()

    eval_results = evaluate_model_version(version=args.version)
    print("\n--- Evaluation Summary ---")
    for task, res in eval_results["metrics"].items():
        print(f"[{task.upper()}] Macro-F1: {res['macro_f1']} | Accuracy: {res['accuracy']}")
