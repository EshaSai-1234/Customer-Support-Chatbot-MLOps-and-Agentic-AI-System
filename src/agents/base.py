"""
Base Agent Interface for Multi-Agent Complaint Analysis System
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("agents")


class BaseAgent(ABC):
    """Abstract Base Class for modular agents."""

    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Core execution logic to be implemented by each specialized agent."""
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes agent with latency measurement, logging, and state enrichment."""
        logger.info(f"[{self.name}] Beginning step: {self.role}")
        t0 = time.perf_counter()
        
        try:
            output = self.run(state)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            
            trace_entry = {
                "agent": self.name,
                "role": self.role,
                "status": "SUCCESS",
                "latency_ms": latency_ms,
                "output_keys": list(output.keys()),
            }
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.error(f"[{self.name}] Error during execution: {e}")
            output = {"error": str(e)}
            trace_entry = {
                "agent": self.name,
                "role": self.role,
                "status": "FAILED",
                "error": str(e),
                "latency_ms": latency_ms,
            }

        # Append to execution trace in state
        traces = state.get("execution_traces", [])
        traces.append(trace_entry)
        state["execution_traces"] = traces

        # Merge output into state
        state.update(output)
        return state
