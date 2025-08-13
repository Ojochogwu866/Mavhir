from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import re


class ToxicityEndpoint(str, Enum):
    AMES_MUTAGENICITY = "ames_mutagenicity"
    CARCINOGENICITY = "carcinogenicity"


class PredictionClass(str, Enum):
    MUTAGENIC = "mutagenic"
    NON_MUTAGENIC = "non_mutagenic"
    CARCINOGENIC = "carcinogenic"
    NON_CARCINOGENIC = "non_carcinogenic"


class ConfidenceLevel(str, Enum):
    LOW = "low"  # Probability value of 0.4-0.6
    MEDIUM = "medium"  # Probability level of 0.25-0.4 or 0.6-0.75
    HIGH = "high"  # Probability < 0.25 or > 0.75


class BaseResponse(BaseModel):
    """Base response model with common fields."""

    success: bool = Field(description="Whether the request was successful")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ErrorDetail(BaseModel):
    """Error detail information."""

    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )


class ErrorResponse(BaseResponse):
    """Standardized error response."""

    success: bool = Field(default=False)
    error: ErrorDetail = Field(description="Error details")


# Molecular property models
class MolecularProperties(BaseModel):
    """Base molecular properties calculated from chemical structure"""

    molecular_weight: float = Field(description="Molecular weight in Da", ge=0)
    logp: float = Field(description="Lipophilicity (LogP)", ge=-10, le=10)
    tpsa: float = Field(description="Topological polar surface area in Ų", ge=0)
    num_heavy_atoms: int = Field(
        description="Number of heavy (non-hydrogen) atoms", ge=0
    )
    num_aromatic_rings: int = Field(description="Number of aromatic rings", ge=0)
    num_rotatable_bonds: int = Field(description="Number of rotatable bonds", ge=0)
    num_hbd: int = Field(description="Number of hydrogen bond donors", ge=0)
    num_hba: int = Field(description="Number of hydrogen bond acceptors", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "molecular_weight": 46.07,
                "logp": -0.31,
                "tpsa": 20.23,
                "num_heavy_atoms": 3,
                "num_aromatic_rings": 0,
                "num_rotatable_bonds": 0,
                "num_hbd": 1,
                "num_hba": 1,
            }
        }
    }


class DrugLikenessAssessment(BaseModel):
    """Drug-likeness assessment based on established rules."""

    lipinski_violations: int = Field(
        description="Number of Lipinski Rule of 5 violations", ge=0, le=4
    )
    lipinski_passed: bool = Field(
        description="Passes Lipinski rules (≤1 violation allowed)"
    )
    veber_violations: int = Field(
        description="Number of Veber rule violations", ge=0, le=2
    )
    veber_passed: bool = Field(description="Passes Veber rules (0 violations required)")
    overall_drug_like: bool = Field(description="Overall drug-likeness assessment")
    violation_details: Optional[Dict[str, str]] = Field(
        None, description="Details of rule violations"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "lipinski_violations": 0,
                "lipinski_passed": True,
                "veber_violations": 0,
                "veber_passed": True,
                "overall_drug_like": True,
                "violation_details": {},
            }
        }
    }


class SinglePrediction(BaseModel):
    """Single toxicity endpoint result"""

    endpoint: ToxicityEndpoint = Field(description="Toxicity endpoint")
    prediction: PredictionClass = Field(description="Predicted class")
    probability: float = Field(description="Probability score", ge=0.0, le=1.0)
    confidence: ConfidenceLevel = Field(description="Prediction Confidence Level")

    model_config = {
        "json_schema_extra": {
            "example": {
                "endpoint": "ames_mutagenicity",
                "prediction": "non_mutagenic",
                "probability": 0.92,
                "confidence": "high",
            }
        }
    }


class PredictionResults(BaseModel):
    """Complete prediction results for all endpoints."""

    ames_mutagenicity: Optional[SinglePrediction] = Field(
        None, description="Ames mutagenicity prediction"
    )
    carcinogenicity: Optional[SinglePrediction] = Field(
        None, description="Carcinogenicity prediction"
    )

    @model_validator(mode='after')
    @classmethod
    def ensure_at_least_one_prediction(cls, model_instance: 'PredictionResults') -> 'PredictionResults':
        """Ensure at least one prediction endpoint is provided."""
        predictions = [
            getattr(model_instance, field_name) 
            for field_name in ['ames_mutagenicity', 'carcinogenicity']
            if getattr(model_instance, field_name) is not None
        ]
        if not predictions:
            raise ValueError("At least one prediction must be provided")
        return model_instance


class SMILESPredictionRequest(BaseModel):
    """Request model for single SMILES prediction."""

    smiles: str = Field(
        description="SMILES string representing the chemical structure",
        min_length=1,
        max_length=1000,
    )
    endpoints: Optional[List[ToxicityEndpoint]] = Field(
        None, description="Specific toxicity endpoints to predict (default: all)"
    )
    include_descriptors: bool = Field(
        default=False, description="Include molecular descriptors in response"
    )
    include_properties: bool = Field(
        default=True, description="Include molecular properties in response"
    )
    include_drug_likeness: bool = Field(
        default=True, description="Include drug-likeness assessment"
    )

    @field_validator("smiles")
    @classmethod
    def validate_smiles_format(cls, v: str) -> str:
        """Basic SMILES format validation."""
        v = v.strip()
        if not v:
            raise ValueError("SMILES cannot be empty")

        # Valid SMILES characters pattern
        pattern = r"^[A-Za-z0-9@+\-\[\]()=#%/\\.]+$"
        if not re.match(pattern, v):
            raise ValueError("SMILES contains invalid characters")

        return v

    @field_validator("endpoints")
    @classmethod
    def validate_endpoints(cls, v: Optional[List[ToxicityEndpoint]]) -> Optional[List[ToxicityEndpoint]]:
        """Validate endpoint list."""
        if v is not None:
            if not v:
                raise ValueError("Endpoints list cannot be empty if provided")
            if len(set(v)) != len(v):
                raise ValueError("Duplicate endpoints not allowed")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "smiles": "CCO",
                "endpoints": ["ames_mutagenicity", "carcinogenicity"],
                "include_descriptors": True,
                "include_properties": True,
                "include_drug_likeness": True,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request model for batch SMILES prediction."""

    smiles_list: List[str] = Field(
        description="List of SMILES strings",
        min_length=1,
        max_length=1000,  # Will be overridden by config
    )
    endpoints: Optional[List[ToxicityEndpoint]] = Field(
        None, description="Specific toxicity endpoints to predict (default: all)"
    )
    include_descriptors: bool = Field(
        default=False, description="Include molecular descriptors in response"
    )
    include_properties: bool = Field(
        default=True, description="Include molecular properties in response"
    )
    include_drug_likeness: bool = Field(
        default=False, description="Include drug-likeness assessment"
    )
    fail_on_error: bool = Field(
        default=False,
        description="Stop processing on first error (default: continue with errors)",
    )

    @field_validator("smiles_list")
    @classmethod
    def validate_smiles_list(cls, v: List[str]) -> List[str]:
        """Validate SMILES list."""
        if not v:
            raise ValueError("SMILES list cannot be empty")

        validated_smiles = []
        for i, smiles in enumerate(v):
            if not isinstance(smiles, str) or not smiles.strip():
                raise ValueError(f"SMILES at index {i} must be a non-empty string")
            validated_smiles.append(smiles.strip())

        return validated_smiles

    model_config = {
        "json_schema_extra": {
            "example": {
                "smiles_list": ["CCO", "CC(=O)O", "c1ccccc1"],
                "endpoints": ["ames_mutagenicity"],
                "include_descriptors": False,
                "include_properties": True,
                "include_drug_likeness": False,
                "fail_on_error": False,
            }
        }
    }


class CompoundPredictionResponse(BaseModel):
    """Single compound prediction"""

    smiles: str = Field(description="Input smiles string")
    canonical_smiles: str = Field(description="Canonical smiles representation")
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")

    predictions: PredictionResults = Field(description="Toxicity predictions")

    molecular_properties: Optional[MolecularProperties] = Field(
        None, description="Basic molecular properties"
    )
    drug_likeness: Optional[DrugLikenessAssessment] = Field(
        None, description="Drug likeness assessment"
    )
    descriptors: Optional[Dict[str, float]] = Field(
        None, description="Molecular descriptors used by ML models"
    )

    success: bool = Field(description="Whether processing was successful")
    errors: Optional[List[str]] = Field(None, description="Processing errors if any")

    model_config = {
        "json_schema_extra": {
            "example": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "molecular_formula": "C2H6O",
                "predictions": {
                    "ames_mutagenicity": {
                        "endpoint": "ames_mutagenicity",
                        "prediction": "non_mutagenic",
                        "probability": 0.92,
                        "confidence": "high",
                    },
                    "carcinogenicity": {
                        "endpoint": "carcinogenicity",
                        "prediction": "non_carcinogenic",
                        "probability": 0.88,
                        "confidence": "high",
                    },
                },
                "molecular_properties": {
                    "molecular_weight": 46.07,
                    "logp": -0.31,
                    "tpsa": 20.23,
                    "num_heavy_atoms": 3,
                    "num_aromatic_rings": 0,
                    "num_rotatable_bonds": 0,
                    "num_hbd": 1,
                    "num_hba": 1,
                },
                "success": True,
                "errors": None,
            }
        }
    }


class SMILESPredictionResponse(BaseResponse):
    """Response model for single SMILES prediction."""

    data: CompoundPredictionResponse = Field(description="Prediction results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "processing_time": 0.15,
                "data": {
                    "smiles": "CCO",
                    "canonical_smiles": "CCO",
                    "predictions": {
                        "ames_mutagenicity": {
                            "endpoint": "ames_mutagenicity",
                            "prediction": "non_mutagenic",
                            "probability": 0.92,
                            "confidence": "high",
                        }
                    },
                    "success": True,
                },
            }
        }
    }


class BatchResultItem(BaseModel):
    """Single item in batch processing results."""

    index: int = Field(description="Index in original batch")
    input_smiles: str = Field(description="Original input SMILES")
    result: Optional[CompoundPredictionResponse] = Field(
        None, description="Prediction result if successful"
    )
    error: Optional[str] = Field(None, description="Error message if processing failed")
    success: bool = Field(description="Whether this item was processed successfully")


class BatchPredictionSummary(BaseModel):
    """Summary statistics for batch processing."""

    total_compounds: int = Field(
        description="Total number of compounds processed", ge=0
    )
    successful: int = Field(description="Number of successful predictions", ge=0)
    failed: int = Field(description="Number of failed predictions", ge=0)
    success_rate: float = Field(description="Success rate as percentage", ge=0, le=100)


class BatchPredictionResponse(BaseResponse):
    """Response model for batch SMILES prediction."""

    results: List[BatchResultItem] = Field(description="Individual prediction results")
    summary: BatchPredictionSummary = Field(description="Batch processing summary")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "processing_time": 2.45,
                "results": [
                    {
                        "index": 0,
                        "input_smiles": "CCO",
                        "result": {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                            "predictions": {
                                "ames_mutagenicity": {
                                    "endpoint": "ames_mutagenicity",
                                    "prediction": "non_mutagenic",
                                    "probability": 0.92,
                                    "confidence": "high",
                                }
                            },
                            "success": True,
                        },
                        "success": True,
                    }
                ],
                "summary": {
                    "total_compounds": 1,
                    "successful": 1,
                    "failed": 0,
                    "success_rate": 100.0,
                },
            }
        }
    }


class ChemicalLookupResponse(BaseResponse):
    """Response model for chemical database lookup."""

    query: str = Field(description="Original search query")
    compound_name: Optional[str] = Field(None, description="Chemical name")
    cid: Optional[int] = Field(None, description="PubChem Compound ID")
    canonical_smiles: Optional[str] = Field(
        None, description="Canonical SMILES from database"
    )
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")
    molecular_weight: Optional[float] = Field(None, description="Molecular weight")
    synonyms: Optional[List[str]] = Field(None, description="Alternative names")
    found: bool = Field(description="Whether compound was found in database")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "processing_time": 0.85,
                "query": "ethanol",
                "compound_name": "ethanol",
                "cid": 702,
                "canonical_smiles": "CCO",
                "molecular_formula": "C2H6O",
                "molecular_weight": 46.07,
                "synonyms": ["ethyl alcohol", "grain alcohol"],
                "found": True,
            }
        }
    }


class ServiceStatus(BaseModel):
    """Individual service health status."""

    name: str = Field(description="Service name")
    status: Literal["healthy", "unhealthy", "unknown"] = Field(
        description="Service status"
    )
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional status details"
    )


class HealthCheckResponse(BaseResponse):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system status"
    )
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="System uptime in seconds")
    services: List[ServiceStatus] = Field(description="Individual service statuses")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.0,
                "services": [
                    {
                        "name": "chemical_processor",
                        "status": "healthy",
                        "response_time": 0.01,
                    },
                    {"name": "ames_model", "status": "healthy", "response_time": 0.005},
                ],
            }
        }
    }


class ModelInfo(BaseModel):
    """Information about a prediction model."""

    name: str = Field(description="Model name")
    endpoint: ToxicityEndpoint = Field(description="Toxicity endpoint")
    version: str = Field(description="Model version")
    description: str = Field(description="Model description")
    threshold: float = Field(description="Classification threshold")
    performance_metrics: Optional[Dict[str, float]] = Field(
        None, description="Model performance metrics"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Ames Mutagenicity Predictor",
                "endpoint": "ames_mutagenicity",
                "version": "1.0",
                "description": "Predicts Ames mutagenicity based on molecular descriptors",
                "threshold": 0.5,
                "performance_metrics": {
                    "accuracy": 0.88,
                    "precision": 0.85,
                    "recall": 0.91,
                    "auc_roc": 0.94,
                },
            }
        }
    }


class ModelInfoResponse(BaseResponse):
    """Response with information about available models."""

    models: List[ModelInfo] = Field(description="Available prediction models")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "models": [
                    {
                        "name": "Ames Mutagenicity Predictor",
                        "endpoint": "ames_mutagenicity",
                        "version": "1.0",
                        "description": "Predicts Ames mutagenicity",
                        "threshold": 0.5,
                    }
                ],
            }
        }
    }