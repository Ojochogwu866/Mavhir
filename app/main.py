import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .api import health, chemical, predict
from .core.config import get_settings
from .core.exceptions import ToxicityPredictorError, create_error_response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    """
    
    # STARTUP
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        logger.info("Validating configuration...")
        
        if not settings.is_development():
            logger.info("Pre-warming services...")
            
            from .services.chemical_processor import create_chemical_processor
            from .services.descriptor_calculator import create_descriptor_calculator  
            from .services.predictor import create_predictor
            
            chemical_processor = create_chemical_processor()
            descriptor_calculator = create_descriptor_calculator()
            predictor = create_predictor()
            
            logger.info(f"Pre-loaded {len(predictor.get_available_endpoints())} ML models")
        
        logger.info(" Application startup complete")
        
    except Exception as e:
        logger.error(f" Application startup failed: {e}")
        raise
    
    yield
    
    logger.info("Shutting down application...")
    logger.info(" Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    CONFIGURATION:
    - Metadata and documentation
    - CORS and security middleware
    - Error handling
    - API route registration
    """
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="""
        **Toxicity Predictor API** - ML-powered chemical toxicity prediction
        
        ## Features
        
        * **Chemical Processing**: SMILES validation and standardization
        * **Database Lookup**: PubChem compound search and properties
        * **Toxicity Prediction**: Ames mutagenicity and carcinogenicity models
        * **Batch Processing**: Handle multiple compounds efficiently
        * **File Upload**: SDF file processing support
        
        ## Endpoints
        
        * **Health**: System status and diagnostics
        * **Chemical**: Compound lookup and validation  
        * **Predict**: Toxicity predictions and model info
        
        ## Models
        
        * **Ames Mutagenicity**: Bacterial reverse mutation test prediction
        * **Carcinogenicity**: Rodent carcinogenicity study prediction
        
        Built with FastAPI, RDKit, and scikit-learn.
        """,
        contact={
            "name": "Toxicity Predictor Team",
            "url": "https://github.com/Ojochogwu866/amc-predictor",
            "email": "your-email@example.com"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        openapi_tags=[
            {
                "name": "Health",
                "description": "System health and diagnostics endpoints"
            },
            {
                "name": "Chemical Lookup", 
                "description": "Chemical database lookup and validation"
            },
            {
                "name": "Toxicity Prediction",
                "description": "ML-based toxicity predictions"
            }
        ],
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Configure middleware
    _setup_middleware(app)
    
    # Register API routes
    _register_routes(app)
    
    # Setup error handlers
    _setup_error_handlers(app)
    
    return app


def _setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.get_cors_methods(),
        allow_headers=["*"] if settings.cors_allow_headers == "*" else settings.cors_allow_headers.split(",")
    )
    
    if not settings.is_development():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
        )
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time to response headers."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(f"{process_time:.4f}")
        return response
    
    if settings.debug:
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all requests in debug mode."""
            logger.info(f"{request.method} {request.url}")
            response = await call_next(request)
            logger.info(f"Response: {response.status_code}")
            return response


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    # Health check routes
    app.include_router(health.router)
    
    # Chemical lookup routes  
    app.include_router(chemical.router, prefix=settings.api_v1_prefix)
    
    # Prediction routes
    app.include_router(predict.router, prefix=settings.api_v1_prefix)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with basic API information."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.version,
            "environment": settings.environment,
            "docs_url": "/docs",
            "health_check": "/health",
            "api_v1": settings.api_v1_prefix
        }

def _setup_error_handlers(app: FastAPI) -> None:
    """Setup custom error handlers."""
    
    @app.exception_handler(ToxicityPredictorError)
    async def toxicity_predictor_exception_handler(request: Request, exc: ToxicityPredictorError):
        """Handle application-specific errors."""
        logger.warning(f"Application error: {exc.message} (Context: {exc.context})")
        
        return JSONResponse(
            status_code=400,
            content=create_error_response(exc)
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc}")
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "error": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors()
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "error": "HTTPException", 
                    "message": exc.detail
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        if settings.is_production():
            message = "Internal server error"
        else:
            message = str(exc)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "error": "InternalServerError",
                    "message": message
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )

app = create_application()

if __name__ == "__main__":
    """
    Run the application directly for development.
    
    USAGE:
        python -m app.main
        
    FOR PRODUCTION:
        uvicorn app.main:app --host 0.0.0.0 --port 8000
    """
    
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=settings.enable_access_logging,
        log_level=settings.log_level.lower()
    )