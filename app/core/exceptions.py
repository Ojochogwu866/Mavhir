from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import time


class ToxicityPredictorError(Exception):
    """
    Base exception for all toxicity predictor errors
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert Exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class ChemicalProcessingError(ToxicityPredictorError):
    """Raised when chemical structure processing fails"""
    pass


class InvalidSMILESError(ChemicalProcessingError):
    """Raised when SMILES string is invalid or cannot be parsed"""

    def __init__(self, smiles: str, details: Optional[str] = None) -> None:
        message = f"Invalid SMILES: '{smiles}'"
        if details:
            message += f" - {details}"

        context = {"smiles": smiles}
        if details:
            context["details"] = details

        super().__init__(message, context=context)


class MolecularStandardizationError(ChemicalProcessingError):
    """Raised when molecular standardization fails"""

    def __init__(self, smiles: str, step: str, details: Optional[str] = None) -> None:
        message = f"Standardization failed at step '{step}' for SMILES: '{smiles}'"
        if details:
            message += f" - {details}"

        super().__init__(
            message, context={"smiles": smiles, "failed_step": step, "details": details}
        )


class DescriptorCalculationError(ToxicityPredictorError):
    """Base class for descriptor calculation errors"""
    pass


class DescriptorTimeoutError(DescriptorCalculationError):
    """Raised when descriptor calculation times out"""
    
    def __init__(self, smiles: str, timeout_seconds: int) -> None:
        message = f"Descriptor calculation timed out after {timeout_seconds}s for SMILES: {smiles}"
        super().__init__(message, context={"smiles": smiles, "timeout_seconds": timeout_seconds})


class MissingDescriptorsError(DescriptorCalculationError):
    """Raised when required descriptors are missing from calculation results."""

    def __init__(self, missing_descriptors: List[str]) -> None:
        message = f"Missing required descriptors: {', '.join(missing_descriptors)}"
        super().__init__(message, context={"missing_descriptors": missing_descriptors})


class ModelError(ToxicityPredictorError):
    """Base class for ML model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when ML model loading fails"""

    def __init__(self, model_path: str, details: Optional[str] = None) -> None:
        message = f"Failed to load model from: {model_path}"
        if details:
            message += f" - {details}"

        super().__init__(
            message, context={"model_path": model_path, "details": details}
        )


class ModelPredictionError(ModelError):
    """Raised when model prediction fails"""

    def __init__(
        self, model_name: str, smiles: str, details: Optional[str] = None
    ) -> None:
        message = f"Prediction failed for model '{model_name}' with SMILES: '{smiles}'"
        if details:
            message += f" - {details}"
        super().__init__(
            message,
            context={"model_name": model_name, "smiles": smiles, "details": details},
        )


class UnsupportedModelError(ModelError):
    """Raised when requested model is not supported or available."""

    def __init__(self, model_name: str, supported_models: List[str]) -> None:
        message = f"Unsupported model: '{model_name}'. Supported models: {', '.join(supported_models)}"
        super().__init__(
            message,
            context={
                "requested_model": model_name,
                "supported_models": supported_models,
            },
        )


class PubChemError(ToxicityPredictorError):
    """Base class for PubChem API Errors"""
    pass


class PubChemAPIError(PubChemError):
    """Raised when PubChem API request fails"""

    def __init__(
        self,
        query: str,
        status_code: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        message = f"PubChem API request failed for query: '{query}'"
        if status_code:
            message += f" (HTTP {status_code})"
        if details:
            message += f" - {details}"

        super().__init__(
            message,
            context={"query": query, "status_code": status_code, "details": details},
        )


class PubChemNotFoundError(PubChemError):
    """Raised when compound is not found in PubChem."""

    def __init__(self, query: str) -> None:
        message = f"Compound not found in PubChem: '{query}'"
        super().__init__(message, context={"query": query})


class PubChemRateLimitError(PubChemError):
    """Raised when PubChem rate limit is exceeded."""

    def __init__(self, retry_after: Optional[int] = None) -> None:
        message = "PubChem rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        super().__init__(message, context={"retry_after": retry_after})


class BatchProcessingError(ToxicityPredictorError):
    """Raised when batch processing fails"""
    pass


class BatchSizeError(BatchProcessingError):
    """Raised when batch size exceeds limits."""

    def __init__(self, batch_size: int, max_size: int) -> None:
        message = f"Batch size {batch_size} exceeds maximum allowed size {max_size}"
        super().__init__(
            message, context={"batch_size": batch_size, "max_size": max_size}
        )


class BatchValidationError(BatchProcessingError):
    """Raised when batch input validation fails."""

    def __init__(self, errors: Dict[int, str]) -> None:
        message = f"Batch validation failed for {len(errors)} items"
        super().__init__(
            message, context={"validation_errors": errors, "error_count": len(errors)}
        )


class ConfigurationError(ToxicityPredictorError):
    """Raised when configuration is invalid or missing."""
    pass


class ServiceUnavailableError(ToxicityPredictorError):
    """Raised when a required service is unavailable."""

    def __init__(self, service_name: str, details: Optional[str] = None) -> None:
        message = f"Service '{service_name}' is unavailable"
        if details:
            message += f" - {details}"

        super().__init__(
            message, context={"service_name": service_name, "details": details}
        )


def handle_rdkit_errors(func: Callable) -> Callable:
    """
    Decorator to convert RDKit errors to our custom exceptions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "SMILES" in str(e).upper():
                smiles_arg = str(args[0]) if args else "unknown"
                raise InvalidSMILESError(smiles_arg, str(e)) from e
            else:
                raise ChemicalProcessingError(f"RDKit operation failed: {e}") from e
    return wrapper


def create_error_response(exception: ToxicityPredictorError) -> Dict[str, Any]:
    """
    Create standardized error response for API endpoints.
    """
    return {
        "success": False,
        "error": exception.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

MoleculeStandardizationError = MolecularStandardizationError

__all__ = [
    "ToxicityPredictorError",
    "ChemicalProcessingError", 
    "InvalidSMILESError",
    "MolecularStandardizationError",
    "MoleculeStandardizationError",
    "DescriptorCalculationError",
    "DescriptorTimeoutError", 
    "MissingDescriptorsError",
    "ModelError",
    "ModelLoadError",
    "ModelPredictionError",
    "UnsupportedModelError",
    "PubChemError",
    "PubChemAPIError",
    "PubChemNotFoundError", 
    "PubChemRateLimitError",
    "BatchProcessingError",
    "BatchSizeError",
    "BatchValidationError",
    "ConfigurationError",
    "ServiceUnavailableError",
    "handle_rdkit_errors",
    "create_error_response",
]