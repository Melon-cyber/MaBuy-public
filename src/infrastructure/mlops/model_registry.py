import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized Model Registry for MaBuy MLOps Pipeline.
    Manages versioning, metadata logging, deployment promotion, and model discovery.
    """

    def __init__(self, registry_dir: str = "models", manifest_file: str = "registry_manifest.json"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.registry_dir, manifest_file)
        self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry manifest: {e}. Re-initializing...")
                self.manifest = {"models": [], "active_models": {}}
        else:
            self.manifest = {"models": [], "active_models": {}}
            self._save_manifest()

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        model_type: str,
        metrics: Dict[str, float],
        symbol: str = "BTCUSDT",
        set_active: bool = True
    ) -> Dict[str, Any]:
        """Registers a new model version and optionally promotes it to active production status."""
        model_entry = {
            "model_name": model_name,
            "version": version,
            "artifact_path": artifact_path,
            "model_type": model_type,
            "symbol": symbol,
            "metrics": metrics,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status": "ACTIVE" if set_active else "ARCHIVED"
        }

        # Check if version exists
        self.manifest["models"] = [m for m in self.manifest["models"] if not (m["model_name"] == model_name and m["version"] == version)]
        self.manifest["models"].append(model_entry)

        if set_active:
            # Archive all previous ACTIVE models for this symbol AND model_type
            for m in self.manifest["models"]:
                if (m["symbol"].upper() == symbol.upper()
                        and m.get("model_type", "").upper() == model_type.upper()
                        and m["version"] != version
                        and m.get("status") == "ACTIVE"):
                    m["status"] = "ARCHIVED"
                    logger.info(f"Auto-archived [{m['model_name']}] version [{m['version']}] for [{symbol}] ({model_type})")

            # Store both specific type key and symbol default
            type_key = f"{symbol.upper()}:{model_type.upper()}"
            self.manifest["active_models"][type_key] = version
            self.manifest["active_models"][symbol.upper()] = version

        self._save_manifest()
        logger.info(f"Registered model [{model_name}] version [{version}] for symbol [{symbol}] ({model_type}). Active: {set_active}")
        return model_entry

    def get_active_model_info(self, symbol: str = "BTCUSDT", model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves active model metadata for the specified asset symbol and optional model type."""
        symbol_upper = symbol.upper()
        if model_type:
            type_key = f"{symbol_upper}:{model_type.upper()}"
            active_version = self.manifest["active_models"].get(type_key)
            if active_version:
                for model in self.manifest["models"]:
                    if (model["symbol"].upper() == symbol_upper
                            and model.get("model_type", "").upper() == model_type.upper()
                            and model["version"] == active_version):
                        return model
            for model in reversed(self.manifest["models"]):
                if (model["symbol"].upper() == symbol_upper
                        and model.get("model_type", "").upper() == model_type.upper()
                        and model.get("status") == "ACTIVE"):
                    return model

        active_version = self.manifest["active_models"].get(symbol_upper)
        if not active_version:
            return None

        for model in self.manifest["models"]:
            if model["symbol"].upper() == symbol_upper and model["version"] == active_version:
                return model
        return None

    def list_registered_models(self) -> List[Dict[str, Any]]:
        return self.manifest.get("models", [])
