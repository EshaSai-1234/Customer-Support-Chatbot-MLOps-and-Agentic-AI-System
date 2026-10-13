"""
Inference and Operational Metrics Tracker
"""

import threading
import time
from typing import Dict, List, Any
import numpy as np


class MetricsTracker:
    """Thread-safe latency and prediction metrics tracker for MLOps monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latencies: List[float] = []
        self.total_requests: int = 0
        self.category_counts: Dict[str, int] = {}
        self.urgency_counts: Dict[str, int] = {}
        self.errors_count: int = 0
        self.start_time: float = time.time()

    def record_inference(self, latency_ms: float, category: str, urgency: str):
        with self._lock:
            self.latencies.append(latency_ms)
            if len(self.latencies) > 2000:
                self.latencies.pop(0)
            self.total_requests += 1
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.urgency_counts[urgency] = self.urgency_counts.get(urgency, 0) + 1

    def record_error(self):
        with self._lock:
            self.errors_count += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self.latencies:
                return {
                    "total_requests": self.total_requests,
                    "errors_count": self.errors_count,
                    "uptime_seconds": round(time.time() - self.start_time, 2),
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "avg_latency_ms": 0.0,
                    "category_distribution": self.category_counts,
                    "urgency_distribution": self.urgency_counts,
                }

            arr = np.array(self.latencies)
            return {
                "total_requests": self.total_requests,
                "errors_count": self.errors_count,
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
                "avg_latency_ms": round(float(np.mean(arr)), 2),
                "category_distribution": dict(self.category_counts),
                "urgency_distribution": dict(self.urgency_counts),
            }


# Singleton tracker
global_tracker = MetricsTracker()
