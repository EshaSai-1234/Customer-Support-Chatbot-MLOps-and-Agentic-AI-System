"""
Model Registry and Artifact Versioning System
Manages versioned model artifacts, active version pointers, and metadata.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib

from src.config import REGISTRY_DIR, CURRENT_POINTER_PATH, DEFAULT_MODEL_VERSION
from src.utils.logger import get_logger

logger = get_logger("model_registry")


class ModelRegistry:
    """Handles storage, versioning, metadata tracking, and retrieval of trained model artifacts."""

    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_file = CURRENT_POINTER_PATH

    def get_active_version(self) -> str:
        """Retrieves the active model version identifier."""
        if self.pointer_file.exists():
            try:
                with open(self.pointer_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("active_version", DEFAULT_MODEL_VERSION)
            except Exception as e:
                logger.warning(f"Error reading pointer file: {e}. Defaulting to {DEFAULT_MODEL_VERSION}")
        return DEFAULT_MODEL_VERSION

    def set_active_version(self, version: str) -> bool:
        """Promotes a given version to be the active production model."""
        target_dir = self.registry_dir / version
        if not target_dir.exists():
            logger.error(f"Cannot set active version: {version} does not exist.")
            return False

        with open(self.pointer_file, "w", encoding="utf-8") as f:
            json.dump({
                "active_version": version,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        logger.info(f"Active model version updated to: {version}")
        return True

    def save_version(
        self,
        version: str,
        artifacts: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Path:
        """Saves models, vectorizer, and metadata into a versioned directory."""
        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        for name, artifact in artifacts.items():
            artifact_path = version_dir / f"{name}.joblib"
            joblib.dump(artifact, artifact_path)

        metadata["version"] = version
        metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Set as active version
        self.set_active_version(version)
        logger.info(f"Model version '{version}' successfully registered at {version_dir}")
        return version_dir

    def load_version(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Loads all model artifacts and metadata for the requested (or active) version."""
        if version is None:
            version = self.get_active_version()

        version_dir = self.registry_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version '{version}' not found in registry at {version_dir}")

        artifacts = {}
        for file in version_dir.glob("*.joblib"):
            key = file.stem
            artifacts[key] = joblib.load(file)

        metadata_path = version_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        artifacts["metadata"] = metadata
        artifacts["version"] = version
        return artifacts

    def list_versions(self) -> List[Dict[str, Any]]:
        """Returns a list of all available versions and their metadata."""
        versions = []
        active = self.get_active_version()

        if not self.registry_dir.exists():
            return []

        for sub in sorted(self.registry_dir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                meta_file = sub / "metadata.json"
                meta = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                versions.append({
                    "version": sub.name,
                    "is_active": (sub.name == active),
                    "created_at": meta.get("saved_at", "Unknown"),
                    "macro_f1": meta.get("metrics", {}).get("category", {}).get("macro_f1", 0.0),
                    "metadata": meta
                })
        return versions

    def get_metadata(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves metadata dict for a specific or active model version."""
        if version is None:
            version = self.get_active_version()
        version_dir = self.registry_dir / version
        meta_file = version_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


# Global registry instance
registry = ModelRegistry()
