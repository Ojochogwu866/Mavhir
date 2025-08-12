import time
import logging
from typing import Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException, status

# Our imports
from ..core.config import get_settings, Settings
from ..core.models import HealthCheckResponse, ServiceStatus
from ..services.chemical_processor import create_chemical_processor
from ..services.descriptor_calculator import create_descriptor_calculator
from ..services.predictor import create_predictor
from ..services.pubchem_client import create_pubchem_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

# Track when the service started
START_TIME = time.time()


@router.get("/", response_model=HealthCheckResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Basic health check - just confirms API is running.
    """

    uptime = time.time() - START_TIME

    return HealthCheckResponse(
        success=True,
        status="healthy",
        version=settings.version,
        uptime_seconds=uptime,
        services=[],
    )


@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """
    Detailed health check - tests all major components.
    """

    uptime = time.time() - START_TIME
    service_statuses = []

    chemical_status = await _test_chemical_processor()
    service_statuses.append(chemical_status)

    descriptor_status = await _test_descriptor_calculator()
    service_statuses.append(descriptor_status)

    predictor_status = await _test_predictor()
    service_statuses.append(predictor_status)

    pubchem_status = await _test_pubchem_client()
    service_statuses.append(pubchem_status)

    overall_status = _determine_overall_status(service_statuses)

    return HealthCheckResponse(
        success=True,
        status=overall_status,
        version=settings.version,
        uptime_seconds=uptime,
        services=service_statuses,
    )


@router.get("/models", response_model=Dict[str, Any])
async def models_health_check():
    """
    Check ML model loading and basic functionality.

    """

    try:
        # Try to create predictor (loads all models)
        predictor = create_predictor()

        model_info = predictor.get_model_info()
        available_endpoints = [ep.value for ep in predictor.get_available_endpoints()]

        dummy_descriptors = {f"desc_{i}": float(i) for i in range(100)}

        model_tests = {}
        for endpoint in available_endpoints:
            try:
                result = predictor.predict(
                    descriptors=dummy_descriptors,
                    smiles="CCO",
                    endpoints=[predictor.get_available_endpoints()[0]],
                )
                model_tests[endpoint] = "healthy"
            except Exception as e:
                model_tests[endpoint] = f"error: {str(e)[:100]}"

        return {
            "success": True,
            "models_loaded": len(available_endpoints),
            "available_endpoints": available_endpoints,
            "model_info": model_info,
            "model_tests": model_tests,
        }

    except Exception as e:
        logger.error(f"Model health check failed: {e}")

        return {
            "success": False,
            "error": str(e),
            "models_loaded": 0,
            "available_endpoints": [],
        }


async def _test_chemical_processor() -> ServiceStatus:
    """Test chemical processor service."""

    start_time = time.time()

    try:
        processor = create_chemical_processor()

        result = processor.process_smiles("CCO")

        response_time = time.time() - start_time

        if result.is_valid:
            return ServiceStatus(
                name="chemical_processor",
                status="healthy",
                response_time=response_time,
                details={
                    "test_molecule": "CCO",
                    "canonical_smiles": result.canonical_smiles,
                },
            )
        else:
            return ServiceStatus(
                name="chemical_processor",
                status="unhealthy",
                response_time=response_time,
                details={
                    "error": "Failed to process test molecule",
                    "errors": result.errors,
                },
            )

    except Exception as e:
        response_time = time.time() - start_time
        return ServiceStatus(
            name="chemical_processor",
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e)},
        )


async def _test_descriptor_calculator() -> ServiceStatus:
    """Test descriptor calculator service."""

    start_time = time.time()

    try:
        calculator = create_descriptor_calculator()

        descriptors = calculator.calculate_cached("CCO")

        response_time = time.time() - start_time

        if descriptors and len(descriptors) > 0:
            return ServiceStatus(
                name="descriptor_calculator",
                status="healthy",
                response_time=response_time,
                details={
                    "descriptors_calculated": len(descriptors),
                    "test_molecule": "CCO",
                },
            )
        else:
            return ServiceStatus(
                name="descriptor_calculator",
                status="unhealthy",
                response_time=response_time,
                details={"error": "No descriptors calculated"},
            )

    except Exception as e:
        response_time = time.time() - start_time
        return ServiceStatus(
            name="descriptor_calculator",
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e)},
        )


async def _test_predictor() -> ServiceStatus:
    """Test ML predictor service."""

    start_time = time.time()

    try:
        predictor = create_predictor()

        available_endpoints = predictor.get_available_endpoints()

        response_time = time.time() - start_time

        if available_endpoints:
            return ServiceStatus(
                name="ml_predictor",
                status="healthy",
                response_time=response_time,
                details={
                    "models_loaded": len(available_endpoints),
                    "available_endpoints": [ep.value for ep in available_endpoints],
                },
            )
        else:
            return ServiceStatus(
                name="ml_predictor",
                status="unhealthy",
                response_time=response_time,
                details={"error": "No models loaded"},
            )

    except Exception as e:
        response_time = time.time() - start_time
        return ServiceStatus(
            name="ml_predictor",
            status="unhealthy",
            response_time=response_time,
            details={"error": str(e)},
        )


async def _test_pubchem_client() -> ServiceStatus:
    """Test PubChem client (external service)."""

    start_time = time.time()

    try:
        client = create_pubchem_client()

        result = client.search_by_name("water")

        response_time = time.time() - start_time

        if result.found:
            return ServiceStatus(
                name="pubchem_client",
                status="healthy",
                response_time=response_time,
                details={
                    "test_compound": "water",
                    "cid": result.cid,
                    "smiles": result.smiles,
                },
            )
        else:
            return ServiceStatus(
                name="pubchem_client",
                status="degraded",
                response_time=response_time,
                details={"warning": "Could not find test compound 'water'"},
            )

    except Exception as e:
        response_time = time.time() - start_time

        return ServiceStatus(
            name="pubchem_client",
            status="degraded",
            response_time=response_time,
            details={"error": str(e), "note": "External service"},
        )


def _determine_overall_status(service_statuses: List[ServiceStatus]) -> str:
    """
    Determine overall system health from individual service statuses.


    CRITICAL SERVICES: chemical_processor, descriptor_calculator, ml_predictor
    OPTIONAL SERVICES: pubchem_client (external dependency)
    """

    critical_services = {"chemical_processor", "descriptor_calculator", "ml_predictor"}

    critical_unhealthy = []
    any_degraded = False

    for service in service_statuses:
        if service.name in critical_services and service.status == "unhealthy":
            critical_unhealthy.append(service.name)
        elif service.status in ["degraded", "unhealthy"]:
            any_degraded = True

    if critical_unhealthy:
        return "unhealthy"  # Critical services failing
    elif any_degraded:
        return "degraded"  # Some issues but functional
    else:
        return "healthy"  # All good
