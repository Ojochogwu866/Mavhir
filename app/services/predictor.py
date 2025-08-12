import logging
import pickle
import joblib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# ML imports
from sklearn.base import BaseEstimator

from ..core.config import get_settings
from ..core.models import ToxicityEndpoint, PredictionClass, ConfidenceLevel
from ..core.exceptions import (
    ModelLoadError,
    ModelPredictionError,
    UnsupportedModelError,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single model prediction result."""

    endpoint: ToxicityEndpoint
    prediction: PredictionClass
    probability: float
    confidence: ConfidenceLevel
    model_version: str


@dataclass
class CompoundPredictions:
    """All predictions for a single compound."""

    smiles: str
    predictions: Dict[ToxicityEndpoint, PredictionResult]
    descriptors_used: int
    processing_time: float


class ToxicityModel:
    """
    Wrapper for individual toxicity prediction models.
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

        self.model = self._load_model()
        self.scaler = self._load_scaler() if scaler_path else None

        logger.info(f"Loaded {endpoint.value} model (version {version})")

    def _load_model(self) -> BaseEstimator:
        """Load ML model from file."""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise ModelLoadError(self.model_path, "File does not exist")

            if model_file.suffix == ".pkl":
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
            elif model_file.suffix == ".joblib":
                model = joblib.load(model_file)
            else:
                raise ModelLoadError(
                    self.model_path, f"Unsupported file format: {model_file.suffix}"
                )

            logger.debug(f"Successfully loaded model from {self.model_path}")
            return model

        except Exception as e:
            raise ModelLoadError(self.model_path, str(e))

    def _load_scaler(self) -> Optional[Any]:
        """Load feature scaler from file."""
        if not self.scaler_path:
            return None

        try:
            scaler_file = Path(self.scaler_path)
            if not scaler_file.exists():
                logger.warning(f"Scaler file not found: {self.scaler_path}")
                return None

            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

            logger.debug(f"Successfully loaded scaler from {self.scaler_path}")
            return scaler

        except Exception as e:
            logger.warning(f"Failed to load scaler from {self.scaler_path}: {e}")
            return None

    def predict(self, descriptors: Dict[str, float]) -> PredictionResult:
        """
        Make toxicity prediction from molecular descriptors.

        PARAMETERS:
            descriptors: Dictionary of descriptor name -> value

        RETURNS:
            PredictionResult with prediction, probability, confidence
        """
        try:
            feature_array = self._prepare_features(descriptors)
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)

            try:
                probability = self.model.predict_proba(feature_array)[0, 1]
            except AttributeError:
                decision = self.model.decision_function(feature_array)[0]
                probability = 1 / (1 + np.exp(-decision))  # Sigmoid transformation

            prediction_class = self._get_prediction_class(probability)

            # Determine confidence level
            confidence = self._get_confidence_level(probability)

            return PredictionResult(
                endpoint=self.endpoint,
                prediction=prediction_class,
                probability=float(probability),
                confidence=confidence,
                model_version=self.version,
            )

        except Exception as e:
            raise ModelPredictionError(self.endpoint.value, "unknown", str(e))

    def _prepare_features(self, descriptors: Dict[str, float]) -> np.ndarray:
        """
        Convert descriptor dictionary to feature array.
        """

        sorted_items = sorted(descriptors.items())
        feature_values = [value for name, value in sorted_items]

        feature_array = np.array(feature_values).reshape(1, -1)

        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            raise ModelPredictionError(
                self.endpoint.value,
                "feature_preparation",
                "Invalid values (NaN or Inf) in descriptors",
            )

        return feature_array

    def _get_prediction_class(self, probability: float) -> PredictionClass:
        """Convert probability to prediction class based on endpoint and threshold."""

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
        else:
            # Future endpoints
            return PredictionClass.POSITIVE if is_positive else PredictionClass.NEGATIVE

    def _get_confidence_level(self, probability: float) -> ConfidenceLevel:
        """
        Determine confidence level based on probability.
        """

        if probability < 0.25 or probability > 0.75:
            return ConfidenceLevel.HIGH
        elif probability < 0.4 or probability > 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class ToxicityPredictor:
    """
    Main service for toxicity predictions.
    """

    def __init__(self):
        """Initialize predictor with all available models."""

        self.settings = get_settings()
        self.models: Dict[ToxicityEndpoint, ToxicityModel] = {}

        # Load all configured models
        self._load_models()

        logger.info(f"ToxicityPredictor initialized with {len(self.models)} models")

    def _load_models(self):
        """Load all toxicity models from configuration."""

        model_config = self.settings.get_model_config()

        for endpoint_name, config in model_config.items():
            try:
                endpoint = ToxicityEndpoint(endpoint_name)

                model = ToxicityModel(
                    endpoint=endpoint,
                    model_path=config["model_path"],
                    scaler_path=config["scaler_path"],
                    threshold=config["threshold"],
                    version="1.0",
                )

                self.models[endpoint] = model

            except Exception as e:
                logger.error(f"Failed to load model for {endpoint_name}: {e}")
                # Continue loading other models

    def predict(
        self,
        descriptors: Dict[str, float],
        smiles: str,
        endpoints: Optional[List[ToxicityEndpoint]] = None,
    ) -> CompoundPredictions:
        """
        Predict toxicity for a single compound.
        """

        import time

        start_time = time.time()

        if endpoints is None:
            endpoints = list(self.models.keys())

        unavailable = [ep for ep in endpoints if ep not in self.models]
        if unavailable:
            available = list(self.models.keys())
            raise UnsupportedModelError(
                str(unavailable), [ep.value for ep in available]
            )

        predictions = {}

        for endpoint in endpoints:
            try:
                model = self.models[endpoint]
                result = model.predict(descriptors)
                predictions[endpoint] = result

            except Exception as e:
                logger.error(f"Prediction failed for {endpoint.value} on {smiles}: {e}")
                continue

        processing_time = time.time() - start_time

        return CompoundPredictions(
            smiles=smiles,
            predictions=predictions,
            descriptors_used=len(descriptors),
            processing_time=processing_time,
        )

    def predict_batch(
        self,
        compounds: List[Tuple[str, Dict[str, float]]],
        endpoints: Optional[List[ToxicityEndpoint]] = None,
    ) -> List[CompoundPredictions]:
        """
        Predict toxicity for multiple compounds.

        PARAMETERS:
            compounds: List of (smiles, descriptors) tuples
            endpoints: Which endpoints to predict

        RETURNS:
            List of CompoundPredictions
        """

        results = []

        for smiles, descriptors in compounds:
            try:
                prediction = self.predict(descriptors, smiles, endpoints)
                results.append(prediction)

            except Exception as e:
                logger.error(f"Batch prediction failed for {smiles}: {e}")
                results.append(
                    CompoundPredictions(
                        smiles=smiles,
                        predictions={},
                        descriptors_used=0,
                        processing_time=0.0,
                    )
                )

        return results

    def get_available_endpoints(self) -> List[ToxicityEndpoint]:
        """Get list of available prediction endpoints."""
        return list(self.models.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""

        info = {}

        for endpoint, model in self.models.items():
            info[endpoint.value] = {
                "version": model.version,
                "threshold": model.threshold,
                "model_path": model.model_path,
                "scaler_path": model.scaler_path,
                "has_scaler": model.scaler is not None,
            }

        return info


def create_predictor() -> ToxicityPredictor:
    """Factory function to create ToxicityPredictor."""
    return ToxicityPredictor()


def _test_predictor():
    """Test predictor with dummy data."""

    print("Testing ToxicityPredictor:")
    print("=" * 40)

    try:
        predictor = create_predictor()

        print(
            f"Available endpoints: {[ep.value for ep in predictor.get_available_endpoints()]}"
        )
        print()

        dummy_descriptors = {
            f"desc_{i}": float(i) for i in range(100)  # 100 dummy descriptors
        }

        test_smiles = "CCO"  # Ethanol

        print(f"Testing prediction for {test_smiles}")

        # This will fail if models aren't actually loaded, which is expected
        # prediction = predictor.predict(dummy_descriptors, test_smiles)
        # print(f"Predictions: {prediction.predictions}")

        print("✅ Predictor initialized successfully")
        print("Note: Actual predictions require trained model files")

    except Exception as e:
        print(f"Expected error (no model files): {e}")


if __name__ == "__main__":
    _test_predictor()
