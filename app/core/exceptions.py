from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import time

class MavhirError(Exception):
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
        return {
            "error": self.error_code,
            "message": self.message,
            "context": self.context,
        }

class MavhirChemicalProcessingError(MavhirError):
    """Raised when chemical structure processing fails"""
    pass

class MavhirInvalidSMILESError(MavhirChemicalProcessingError):
    """Raised when SMILES string is invalid or cannot be parsed"""

    def __init__(self, smiles: str, details: Optional[str] = None) -> None:
        message = f"Invalid SMILES: '{smiles}'"
        if details:
            message += f" - {details}"

        context = {"smiles": smiles}
        if details:
            context["details"] = details

        super().__init__(message, context=context)


class MavhirMolecularStandardizationError(MavhirChemicalProcessingError):
    """Raised when molecular standardization fails"""

    def __init__(self, smiles: str, step: str, details: Optional[str] = None) -> None:
        message = f"Standardization failed at step '{step}' for SMILES: '{smiles}'"
        if details:
            message += f" - {details}"

        super().__init__(
            message, context={"smiles": smiles, "failed_step": step, "details": details}
        )


class MavhirDescriptorCalculationError(MavhirError):
    """Base class for descriptor calculation errors"""
    pass

class MavhirDescriptorTimeoutError(MavhirDescriptorCalculationError):
    """Raised when descriptor calculation times out"""

    def __init__(self, smiles: str, timeout_seconds: int) -> None:
        message = f"Descriptor calculation timed out after {timeout_seconds}s for SMILES: {smiles}"
        super().__init__(
            message, context={"smiles": smiles, "timeout_seconds": timeout_seconds}
        )

class MavhirMissingDescriptorsError(MavhirDescriptorCalculationError):
    """Raised when required descriptors are missing from calculation results."""

    def __init__(self, missing_descriptors: List[str]) -> None:
        message = f"Missing required descriptors: {', '.join(missing_descriptors)}"
        super().__init__(message, context={"missing_descriptors": missing_descriptors})

class MavhirModelError(MavhirError):
    """Base class for ML model-related errors."""
    pass
class MavhirModelLoadError(MavhirModelError):
    """Raised when ML model loading fails"""

    def __init__(self, model_path: str, details: Optional[str] = None) -> None:
        message = f"Failed to load model from: {model_path}"
        if details:
            message += f" - {details}"

        super().__init__(
            message, context={"model_path": model_path, "details": details}
        )

class MavhirModelPredictionError(MavhirModelError):
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

class MavhirPubChemError(MavhirError):
    """Base class for PubChem API Errors"""
    pass

class MavhirPubChemAPIError(MavhirPubChemError):
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

class MavhirBatchProcessingError(MavhirError):
    """Raised when batch processing fails"""
    pass

class MavhirBatchSizeError(MavhirBatchProcessingError):
    """Raised when batch size exceeds limits."""

    def __init__(self, batch_size: int, max_size: int) -> None:
        message = f"Batch size {batch_size} exceeds maximum allowed size {max_size}"
        super().__init__(
            message, context={"batch_size": batch_size, "max_size": max_size}
        )


def handle_rdkit_errors(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "SMILES" in str(e).upper():
                smiles_arg = str(args[0]) if args else "unknown"
                raise MavhirInvalidSMILESError(smiles_arg, str(e)) from e
            else:
                raise MavhirChemicalProcessingError(
                    f"RDKit operation failed: {e}"
                ) from e
    return wrapper


def create_error_response(exception: MavhirError) -> Dict[str, Any]:
    return {
        "success": False,
        "error": exception.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
