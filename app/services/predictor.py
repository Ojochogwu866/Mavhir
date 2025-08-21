import logging
import pickle
import joblib
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# ML imports
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from ..core.config import get_settings
from ..core.models import ToxicityEndpoint, PredictionClass, ConfidenceLevel
from ..core.exceptions import (
    MavhirModelLoadError,
    MavhirModelPredictionError,
    MavhirError,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Enhanced prediction result with metadata."""

    endpoint: ToxicityEndpoint
    prediction: PredictionClass
    probability: float
    confidence: ConfidenceLevel
    model_version: str
    processing_time: float
    feature_count: int


@dataclass
class CompoundPredictions:
    """Enhanced predictions for a single compound."""

    smiles: str
    predictions: Dict[ToxicityEndpoint, PredictionResult]
    descriptors_used: int
    processing_time: float
    request_id: Optional[str] = None
    cached: bool = False


class ModelMetrics:
    """Track model performance metrics."""

    def __init__(self):
        self.prediction_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0.0
        self.last_prediction_time = None
        self._lock = threading.Lock()

    def record_prediction(self, processing_time: float, success: bool = True):
        """Record prediction metrics thread-safely."""
        with self._lock:
            self.prediction_count += 1
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            self.total_processing_time += processing_time
            self.last_prediction_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            avg_time = (
                self.total_processing_time / self.prediction_count
                if self.prediction_count > 0
                else 0.0
            )
            success_rate = (
                self.success_count / self.prediction_count
                if self.prediction_count > 0
                else 0.0
            )

            return {
                "total_predictions": self.prediction_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": success_rate,
                "average_processing_time": avg_time,
                "last_prediction_time": self.last_prediction_time,
            }


class EnhancedToxicityModel:
    """
    Production-ready toxicity model with comprehensive error handling and monitoring.
    """

    def __init__(
        self,
        endpoint: ToxicityEndpoint,
        model_path: str,
        scaler_path: Optional[str] = None,
        threshold: float = 0.5,
        version: str = "1.0",
    ):
        self.endpoint = endpoint
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = threshold
        self.version = version
        self.metrics = ModelMetrics()

        # Thread safety
        self._prediction_lock = threading.Lock()

        # Load model components
        self._load_model_components()

        logger.info(f"Enhanced {endpoint.value} model loaded successfully")

    def _load_model_components(self):
        """Load all model components with comprehensive validation."""

        # Load main model
        self.model = self._load_model()
        self.expected_features = self._determine_expected_features()

        # Load and validate scaler
        self.scaler = self._load_and_validate_scaler()

        # Validate model compatibility
        self._validate_model_components()

        logger.info(
            f"Model {self.endpoint.value}: {self.expected_features} features, "
            f"scaler: {'enabled' if self.scaler else 'disabled'}"
        )

    def _load_model(self) -> BaseEstimator:
        """Load ML model with enhanced error handling."""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise MavhirModelLoadError(self.model_path, "Model file does not exist")

            if model_file.suffix == ".pkl":
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
            elif model_file.suffix == ".joblib":
                model = joblib.load(model_file)
            else:
                raise MavhirModelLoadError(
                    self.model_path, f"Unsupported file format: {model_file.suffix}"
                )

            # Validate model has required methods
            required_methods = ["predict"]
            for method in required_methods:
                if not hasattr(model, method):
                    raise MavhirModelLoadError(
                        self.model_path, f"Model missing required method: {method}"
                    )

            logger.debug(f"Successfully loaded model from {self.model_path}")
            return model

        except Exception as e:
            raise MavhirModelLoadError(self.model_path, str(e))

    def _determine_expected_features(self) -> int:
        """Determine expected number of features with multiple fallbacks."""

        if hasattr(self.model, "n_features_in_"):
            return self.model.n_features_in_

        try:
            settings = get_settings()
            metadata_path = Path(settings.models_dir) / "model_metadata.json"

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                endpoint_data = metadata.get(self.endpoint.value, {})
                if "n_features" in endpoint_data:
                    return endpoint_data["n_features"]
        except Exception as e:
            logger.debug(f"Could not load features from metadata: {e}")

        defaults = {
            ToxicityEndpoint.AMES_MUTAGENICITY: 896,
            ToxicityEndpoint.CARCINOGENICITY: 808,
        }

        fallback = defaults.get(self.endpoint, 1000)
        logger.warning(
            f"Using fallback feature count for {self.endpoint.value}: {fallback}"
        )
        return fallback

    def _load_and_validate_scaler(self) -> Optional[StandardScaler]:
        """Load scaler with comprehensive validation."""
        if not self.scaler_path:
            return None

        scaler_file = Path(self.scaler_path)
        if not scaler_file.exists():
            logger.warning(f"Scaler file not found: {self.scaler_path}")
            return None

        try:
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

            if not isinstance(scaler, StandardScaler):
                logger.warning(f"Invalid scaler type: {type(scaler)}")
                return None

            if hasattr(scaler, "n_features_in_"):
                if scaler.n_features_in_ != self.expected_features:
                    logger.warning(
                        f"Scaler feature mismatch: expected {self.expected_features}, "
                        f"got {scaler.n_features_in_}"
                    )
                    return None

            if not all(hasattr(scaler, attr) for attr in ["mean_", "scale_"]):
                logger.warning("Scaler not properly fitted")
                return None

            if any(arr is None for arr in [scaler.mean_, scaler.scale_]):
                logger.warning("Scaler has None arrays")
                return None

            return scaler

        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None

    def _validate_model_components(self):
        """Validate all model components work together."""
        try:
            dummy_features = np.random.random((1, self.expected_features))

            if self.scaler:
                scaled_features = self.scaler.transform(dummy_features)
                if np.any(np.isnan(scaled_features)) or np.any(
                    np.isinf(scaled_features)
                ):
                    logger.warning("Scaler produces invalid values, disabling")
                    self.scaler = None
                    scaled_features = dummy_features
            else:
                scaled_features = dummy_features

            # Test model prediction
            if hasattr(self.model, "predict_proba"):
                _ = self.model.predict_proba(scaled_features)
            else:
                _ = self.model.predict(scaled_features)

            logger.debug(
                f"Model components validation passed for {self.endpoint.value}"
            )

        except Exception as e:
            raise MavhirModelLoadError(
                self.model_path, f"Model component validation failed: {e}"
            )

    def predict(
        self, descriptors: Dict[str, float], request_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Make prediction with comprehensive error handling and monitoring.
        """
        start_time = time.time()

        try:
            with self._prediction_lock:
                feature_array = self._prepare_features(descriptors)
                scaled_features = self._apply_scaling_safely(feature_array)
                probability = self._make_model_prediction(scaled_features)
                prediction_class = self._get_prediction_class(probability)
                confidence = self._get_confidence_level(probability)

                processing_time = time.time() - start_time

                self.metrics.record_prediction(processing_time, success=True)

                result = PredictionResult(
                    endpoint=self.endpoint,
                    prediction=prediction_class,
                    probability=float(probability),
                    confidence=confidence,
                    model_version=self.version,
                    processing_time=processing_time,
                    feature_count=len(descriptors),
                )

                logger.debug(
                    f"{self.endpoint.value}: {prediction_class.value} "
                    f"(prob: {probability:.3f}, time: {processing_time:.3f}s)"
                )

                return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.record_prediction(processing_time, success=False)

            logger.error(f" Prediction failed for {self.endpoint.value}: {e}")
            raise MavhirModelPredictionError(self.endpoint.value, "unknown", str(e))

    def _prepare_features(self, descriptors: Dict[str, float]) -> np.ndarray:
        """Prepare feature array with validation."""
        sorted_items = sorted(descriptors.items())
        feature_values = [value for name, value in sorted_items]

        feature_array = np.array(feature_values, dtype=np.float64).reshape(1, -1)
        actual_features = feature_array.shape[1]

        if actual_features != self.expected_features:
            raise ValueError(
                f"Feature count mismatch for {self.endpoint.value}: "
                f"got {actual_features}, expected {self.expected_features}"
            )

        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            logger.debug(f"Cleaning NaN/Inf values for {self.endpoint.value}")
            feature_array = np.nan_to_num(
                feature_array, nan=0.0, posinf=0.0, neginf=0.0
            )

        return feature_array

    def _apply_scaling_safely(self, feature_array: np.ndarray) -> np.ndarray:
        """Apply scaling with error handling."""
        if self.scaler is None:
            return feature_array

        try:
            scaled_array = self.scaler.transform(feature_array)

            if np.any(np.isnan(scaled_array)) or np.any(np.isinf(scaled_array)):
                logger.warning(
                    f"Scaling produced invalid values for {self.endpoint.value}"
                )
                return feature_array

            return scaled_array

        except Exception as e:
            logger.warning(f"Scaling failed for {self.endpoint.value}: {e}")
            return feature_array

    def _make_model_prediction(self, feature_array: np.ndarray) -> float:
        """Make model prediction with fallback methods."""
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(feature_array)
                if proba.shape[1] >= 2:
                    return float(proba[0, 1])
                else:
                    return float(proba[0, 0])

            elif hasattr(self.model, "decision_function"):
                decision = self.model.decision_function(feature_array)[0]
                probability = 1 / (1 + np.exp(-decision))
                return float(probability)

            else:
                binary_pred = self.model.predict(feature_array)[0]
                return 0.8 if binary_pred == 1 else 0.2

        except Exception as e:
            logger.error(
                f"All prediction methods failed for {self.endpoint.value}: {e}"
            )
            raise

    def _get_prediction_class(self, probability: float) -> PredictionClass:
        """Convert probability to prediction class."""
        is_positive = probability >= self.threshold

        if self.endpoint == ToxicityEndpoint.AMES_MUTAGENICITY:
            return (
                PredictionClass.MUTAGENIC
                if is_positive
                else PredictionClass.NON_MUTAGENIC
            )
        elif self.endpoint == ToxicityEndpoint.CARCINOGENICITY:
            return (
                PredictionClass.CARCINOGENIC
                if is_positive
                else PredictionClass.NON_CARCINOGENIC
            )

    def _get_confidence_level(self, probability: float) -> ConfidenceLevel:
        """Determine confidence level based on probability."""
        if probability < 0.25 or probability > 0.75:
            return ConfidenceLevel.HIGH
        elif probability < 0.4 or probability > 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status."""
        stats = self.metrics.get_stats()

        return {
            "endpoint": self.endpoint.value,
            "status": "healthy" if stats["success_rate"] > 0.95 else "degraded",
            "model_path": self.model_path,
            "scaler_enabled": self.scaler is not None,
            "expected_features": self.expected_features,
            "threshold": self.threshold,
            "version": self.version,
            "metrics": stats,
        }


class EnhancedToxicityPredictor:
    """
    Production-ready toxicity prediction service with async support and monitoring.
    """

    def __init__(self):
        """Initialize predictor with enhanced capabilities."""
        self.settings = get_settings()
        self.models: Dict[ToxicityEndpoint, EnhancedToxicityModel] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._global_metrics = ModelMetrics()

        from ..services.descriptor_calculator import create_descriptor_calculator

        self.descriptor_calculator = create_descriptor_calculator()

        # Load models
        self._load_models()

        logger.info(
            f"EnhancedToxicityPredictor initialized with {len(self.models)} models"
        )

    def _load_models(self):
        """Load all available models with error handling."""
        model_config = self.settings.get_model_config()
        loaded_models = []
        failed_models = []

        for endpoint_name, config in model_config.items():
            try:
                endpoint = ToxicityEndpoint(endpoint_name)

                logger.info(f"Loading {endpoint_name} model...")

                model = EnhancedToxicityModel(
                    endpoint=endpoint,
                    model_path=config["model_path"],
                    scaler_path=config["scaler_path"],
                    threshold=config["threshold"],
                    version="2.0",  # Enhanced version
                )

                self.models[endpoint] = model
                loaded_models.append(endpoint_name)

            except Exception as e:
                logger.error(f"Failed to load {endpoint_name} model: {e}")
                failed_models.append((endpoint_name, str(e)))
                continue

        if not loaded_models:
            raise MavhirError("No models could be loaded")

        logger.info(f"Loaded models: {loaded_models}")
        if failed_models:
            logger.warning(f"Failed models: {[name for name, _ in failed_models]}")

    async def predict_async(
        self,
        descriptors: Optional[Dict[str, float]] = None,
        smiles: str = None,
        endpoints: Optional[List[ToxicityEndpoint]] = None,
        request_id: Optional[str] = None,
    ) -> CompoundPredictions:
        """
        Async prediction with concurrent model execution.
        """
        start_time = time.time()

        if endpoints is None:
            endpoints = list(self.models.keys())

        unavailable = [ep for ep in endpoints if ep not in self.models]
        if unavailable:
            available = list(self.models.keys())
            raise MavhirError(
                f"Unsupported endpoints: {unavailable}. Available: {[ep.value for ep in available]}"
            )

        if not smiles and not descriptors:
            raise ValueError("Either SMILES or descriptors must be provided")

        model_descriptors = {}
        if descriptors is None:
            for endpoint in endpoints:
                model_descriptors[
                    endpoint
                ] = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.descriptor_calculator.calculate_for_model,
                    smiles,
                    endpoint.value,
                )
        else:
            for endpoint in endpoints:
                model_descriptors[endpoint] = descriptors

        prediction_tasks = []
        for endpoint in endpoints:
            task = asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.models[endpoint].predict,
                model_descriptors[endpoint],
                request_id,
            )
            prediction_tasks.append((endpoint, task))

        predictions = {}
        total_descriptors_used = 0

        for endpoint, task in prediction_tasks:
            try:
                result = await task
                predictions[endpoint] = result
                total_descriptors_used = max(
                    total_descriptors_used, result.feature_count
                )
            except Exception as e:
                logger.error(f" Async prediction failed for {endpoint.value}: {e}")
                continue

        processing_time = time.time() - start_time
        self._global_metrics.record_prediction(
            processing_time, success=len(predictions) > 0
        )

        return CompoundPredictions(
            smiles=smiles or "unknown",
            predictions=predictions,
            descriptors_used=total_descriptors_used,
            processing_time=processing_time,
            request_id=request_id,
            cached=False,
        )

    def predict(
        self,
        descriptors: Optional[Dict[str, float]] = None,
        smiles: str = None,
        endpoints: Optional[List[ToxicityEndpoint]] = None,
        request_id: Optional[str] = None,
    ) -> CompoundPredictions:
        """
        Synchronous prediction (wrapper for async method).
        """
        return asyncio.run(
            self.predict_async(descriptors, smiles, endpoints, request_id)
        )

    async def predict_batch_async(
        self,
        compounds: List[Dict[str, Any]],
        endpoints: Optional[List[ToxicityEndpoint]] = None,
        max_concurrent: int = 10,
        request_id: Optional[str] = None,
    ) -> List[CompoundPredictions]:
        """
        Async batch prediction with concurrency control.
        """
        if endpoints is None:
            endpoints = list(self.models.keys())

        # Process in batches to control concurrency
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def predict_single(compound_data):
            async with semaphore:
                try:
                    smiles = compound_data.get("smiles")
                    descriptors = compound_data.get("descriptors")
                    compound_id = compound_data.get("id", "unknown")

                    result = await self.predict_async(
                        descriptors=descriptors,
                        smiles=smiles,
                        endpoints=endpoints,
                        request_id=(
                            f"{request_id}_{compound_id}" if request_id else compound_id
                        ),
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"Batch prediction failed for compound {compound_data}: {e}"
                    )
                    return None

        tasks = [predict_single(compound) for compound in compounds]

        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)
            completed += 1

            if completed % 10 == 0:
                logger.info(f"Batch progress: {completed}/{len(tasks)} completed")

        logger.info(
            f"Batch prediction complete: {len(results)}/{len(compounds)} successful"
        )
        return results

    def get_available_endpoints(self) -> List[ToxicityEndpoint]:
        """Get available prediction endpoints."""
        return list(self.models.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = {}

        for endpoint, model in self.models.items():
            health_status = model.get_health_status()
            info[endpoint.value] = {
                "version": model.version,
                "threshold": model.threshold,
                "expected_features": model.expected_features,
                "has_scaler": model.scaler is not None,
                "model_path": model.model_path,
                "scaler_path": model.scaler_path,
                "health": health_status,
                "status": health_status["status"],
            }

        return info

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        model_statuses = {}
        overall_healthy = True

        for endpoint, model in self.models.items():
            health = model.get_health_status()
            model_statuses[endpoint.value] = health
            if health["status"] != "healthy":
                overall_healthy = False

        global_stats = self._global_metrics.get_stats()

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "models_loaded": len(self.models),
            "available_endpoints": [ep.value for ep in self.get_available_endpoints()],
            "model_statuses": model_statuses,
            "global_metrics": global_stats,
            "system_info": {
                "descriptor_calculator_ready": hasattr(self, "descriptor_calculator"),
                "thread_pool_active": not self._executor._shutdown,
                "settings_loaded": self.settings is not None,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            test_result = await self.predict_async(
                smiles="CCO", request_id="health_check"
            )

            pipeline_healthy = len(test_result.predictions) > 0

            health_info = self.get_system_health()
            health_info["pipeline_test"] = {
                "success": pipeline_healthy,
                "predictions_made": len(test_result.predictions),
                "processing_time": test_result.processing_time,
            }

            return health_info

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_info = self.get_system_health()
            health_info["status"] = "unhealthy"
            health_info["pipeline_test"] = {"success": False, "error": str(e)}
            return health_info

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=True)
        except:
            pass


def create_predictor() -> EnhancedToxicityPredictor:
    """Factory function to create EnhancedToxicityPredictor."""
    return EnhancedToxicityPredictor()
