import logging
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .api import health, chemical, predict
from .core.config import get_settings
from .core.exceptions import MavhirError, create_error_response
from .services.pubchem_client import create_pubchem_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        (
            logging.FileHandler("mavhir.log")
            if get_settings().environment != "development"
            else logging.NullHandler()
        ),
    ],
)
logger = logging.getLogger(__name__)

settings = get_settings()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(f"Starting Mavhir v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    startup_errors = []

    try:
        logger.info(" Validating configuration...")

        if not settings.models_dir:
            startup_errors.append("Models directory not configured")

        if settings.max_batch_size <= 0:
            startup_errors.append("Invalid batch size configuration")

        logger.info("Pre-warming and validating services...")

        try:
            from .services.chemical_processor import create_chemical_processor

            chemical_processor = create_chemical_processor()

            test_result = chemical_processor.process_smiles("CCO")
            if not test_result.is_valid:
                startup_errors.append("Chemical processor validation failed")
            else:
                logger.info("Chemical processor: OK")

        except Exception as e:
            startup_errors.append(f"Chemical processor initialization failed: {e}")

        try:
            from .services.descriptor_calculator import create_descriptor_calculator

            descriptor_calculator = create_descriptor_calculator()

            test_descriptors = descriptor_calculator.calculate_cached("CCO")
            if not test_descriptors:
                startup_errors.append("Descriptor calculator validation failed")
            else:
                logger.info("Descriptor calculator: OK")

        except Exception as e:
            startup_errors.append(f"Descriptor calculator initialization failed: {e}")

        try:
            from .services.predictor import create_predictor
            predictor = create_predictor()

            available_endpoints = predictor.get_available_endpoints()
            if not available_endpoints:
                startup_errors.append("No ML models available")
            else:
                logger.info(f"ML Predictor: {len(available_endpoints)} models loaded")
                test_prediction = predictor.predict(smiles="CCO")
                if not test_prediction.predictions:
                    startup_errors.append("Model prediction test failed")
                else:
                    logger.info("Model prediction test: OK")

        except Exception as e:
            startup_errors.append(f"ML predictor initialization failed: {e}")

        try:
            pubchem_client = create_pubchem_client()
            logger.info("PubChem client: OK")
        except Exception as e:
            logger.warning(f"PubChem client initialization failed: {e} (will continue)")

        if startup_errors:
            logger.error("Application startup failed with errors:")
            for error in startup_errors:
                logger.error(f"  - {error}")

            if settings.environment == "production":
                raise RuntimeError(f"Critical startup errors: {startup_errors}")
            else:
                logger.warning("Continuing in development mode despite errors")

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

    yield

    logger.info("Shutting down Mavhir...")

    try:
        if "chemical_processor" in locals():
            pass

        logger.info("Cleanup complete")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

    logger.info(" Mavhir shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application with comprehensive enhancements.
    """

    app = FastAPI(
        title="Mavhir",
        version=settings.version,
        description="""
        Mavhir - Toxicity Prediction API
        
        Predicts Ames mutagenicity and carcinogenicity using machine learning models
        trained on molecular descriptors.
        
        - Single SMILES prediction
        - Batch processing
        - SDF file upload support
        - Chemical database lookup via PubChem
        - Comprehensive molecular property calculation
        - Drug-likeness assessment
        
        Endpoints
        - Health: System diagnostics and model status
        - Chemical: SMILES validation and PubChem lookup
        - Predict: Toxicity predictions for compounds
        """,
        contact={
            "name": "Mavhir Team",
            "url": "https://github.com/Ojochogwu866/Mavhir",
            "email": "hello@ojochogwu.dev",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "Health",
                "description": "System health and diagnostics endpoints",
            },
            {
                "name": "Chemical Lookup",
                "description": "Chemical database lookup and SMILES validation",
            },
            {
                "name": "Toxicity Prediction",
                "description": "ML-based toxicity predictions for Ames mutagenicity and carcinogenicity",
            },
        ],
        lifespan=lifespan,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _setup_middleware(app)
    _register_routes(app)
    _setup_error_handlers(app)

    return app


def _setup_middleware(app: FastAPI) -> None:
    """application middleware."""

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.is_development():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.localhost"],
        )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time and request ID to response headers."""
        start_time = time.time()
        request_id = f"req_{int(time.time()*1000000)}"

        request.state.request_id = request_id

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id

        # Log slow requests
        if process_time > 5.0:
            logger.warning(
                f"Slow request {request_id}: {request.method} {request.url} took {process_time:.2f}s"
            )

        return response

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Request logging."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        if settings.debug:
            logger.info(f"[{request_id}] {request.method} {request.url}")

        try:
            response = await call_next(request)

            process_time = time.time() - start_time
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"[{request_id}] {response.status_code} - {process_time:.3f}s",
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] Request failed after {process_time:.3f}s: {e}"
            )
            raise

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers."""
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        if settings.is_production():
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers."""
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        if not settings.is_development():
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response


def _register_routes(app: FastAPI) -> None:
    """Register all API routes with consistent URL structure."""

    app.include_router(health.router)

    app.include_router(chemical.router, prefix=settings.api_v1_prefix)
    app.include_router(predict.router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["Root"])
    @limiter.limit("10/minute")
    async def root(request: Request):
        """Root endpoint with API information."""
        return {
            "service": "Mavhir",
            "description": "Advanced Toxicity Prediction API",
            "version": settings.version,
            "environment": settings.environment,
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "documentation": "/docs",
                "api_v1": settings.api_v1_prefix,
            },
            "features": [
                "Ames mutagenicity prediction",
                "Carcinogenicity prediction",
                "SMILES validation",
                "Batch processing",
                "SDF file upload",
                "PubChem lookup",
                "Molecular properties calculation",
            ],
            "contact": {
                "github": "https://github.com/Ojochogwu866/Mavhir",
                "email": "hello@ojochogwu.dev",
            },
        }

    @app.get("/status", tags=["Root"])
    @limiter.limit("30/minute")
    async def status(request: Request):
        """Quick status check."""
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": settings.version,
        }


def _setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers with proper Mavhir branding."""

    @app.exception_handler(MavhirError)
    async def mavhir_exception_handler(request: Request, exc: MavhirError):
        """Handle Mavhir application-specific errors."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"[{request_id}] Mavhir error: {exc.message} (Context: {exc.context})"
        )

        status_code = 400
        if "timeout" in exc.message.lower():
            status_code = 408
        elif "not found" in exc.message.lower():
            status_code = 404
        elif "load" in exc.message.lower() or "model" in exc.message.lower():
            status_code = 503

        return JSONResponse(status_code=status_code, content=create_error_response(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle request validation errors with feedback."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] Validation error: {exc}")

        simplified_errors = []
        for error in exc.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            simplified_errors.append(f"{field}: {message}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "error": "ValidationError",
                    "message": "Request validation failed",
                    "details": simplified_errors,
                    "raw_errors": exc.errors() if settings.debug else None,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {"error": "HTTPException", "message": exc.detail},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
            },
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] Rate limit exceeded: {request.client.host}")

        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": {
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {exc.detail}",
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected errors with proper logging."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"[{request_id}] Unexpected error: {exc}", exc_info=True)

        if settings.is_production():
            message = "Internal server error"
            details = None
        else:
            message = str(exc)
            details = {"type": type(exc).__name__, "args": exc.args}

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "error": "InternalServerError",
                    "message": message,
                    "details": details,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
            },
        )


app = create_application()

if __name__ == "__main__":
    import uvicorn

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default"],
        },
    }

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=settings.enable_access_logging,
        log_level=settings.log_level.lower(),
        log_config=log_config if not settings.debug else None,
        workers=1 if settings.debug else settings.workers,
    )
