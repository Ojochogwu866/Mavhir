from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator

from exceptions import MavhirError

class Settings(BaseSettings):
    """application settings with production-ready defaults."""

    app_name: str = Field(default="Mavhir", description="Application Name")
    version: str = Field(default="2.0.0", description="API version")
    environment: str = Field(default="development", description="Runtime Environment")
    debug: bool = Field(default=False, description="Enable debug mode")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(
        default=1, ge=1, le=16, description="Number of worker processes"
    )

    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    max_file_size_mb: int = Field(
        default=50, ge=1, le=1000, description="Maximum file size in MB"
    )

    rate_limit_per_minute: int = Field(
        default=100, ge=1, le=10000, description="Rate limit per minute"
    )
    burst_rate_limit: int = Field(
        default=20, ge=1, le=1000, description="Burst rate limit"
    )

    # CORS settings
    cors_origins: str = Field(default="*", description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORS credentials"
    )
    cors_allow_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS", description="Allowed CORS methods"
    )
    cors_allow_headers: str = Field(default="*", description="Allowed CORS headers")

    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )
    enable_trusted_hosts: bool = Field(
        default=True, description="Enable trusted host middleware"
    )
    trusted_hosts: str = Field(
        default="localhost,127.0.0.1,*.mavhir.com", description="Trusted hosts"
    )

    models_dir: str = Field(default="app/models", description="Models directory")

    ames_model_path: str = Field(
        default="app/models/ames_mutagenicity.pkl", description="Ames model path"
    )
    ames_scaler_path: str = Field(
        default="app/models/ames_mutagenicity_scaler.pkl",
        description="Ames scaler path",
    )
    ames_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Ames classification threshold"
    )

    carcinogenicity_model_path: str = Field(default="app/models/carcinogenicity.pkl")
    carcinogenicity_scaler_path: str = Field(
        default="app/models/carcinogenicity_scaler.pkl"
    )
    carcinogenicity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Carcinogenicity threshold"
    )

    enable_molecule_standardization: bool = Field(
        default=True, description="Enable molecule standardization"
    )
    standardization_timeout: int = Field(
        default=30, ge=1, le=300, description="Standardization timeout (seconds)"
    )
    max_smiles_length: int = Field(
        default=1000, ge=10, le=10000, description="Maximum SMILES length"
    )
    max_heavy_atoms: int = Field(
        default=200, ge=1, le=1000, description="Maximum heavy atoms"
    )

    descriptor_timeout: int = Field(
        default=60, ge=10, le=600, description="Descriptor calculation timeout"
    )
    enable_descriptor_caching: bool = Field(
        default=True, description="Enable descriptor caching"
    )
    descriptor_cache_size: int = Field(
        default=1000, ge=1, le=10000, description="Descriptor cache size"
    )
    descriptor_cache_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Cache TTL in seconds"
    )

    max_batch_size: int = Field(
        default=100, ge=1, le=10000, description="Maximum batch size"
    )
    batch_processing_timeout: int = Field(
        default=300, ge=30, le=3600, description="Batch timeout (seconds)"
    )
    max_concurrent_predictions: int = Field(
        default=10, ge=1, le=50, description="Max concurrent predictions"
    )

    pubchem_base_url: str = Field(
        default="https://pubchem.ncbi.nlm.nih.gov",
        description="PubChem API base URL",
    )
    pubchem_timeout: int = Field(
        default=10, ge=1, le=60, description="PubChem API timeout (seconds)"
    )
    pubchem_rate_limit_delay: float = Field(
        default=0.2, ge=0.1, le=2.0, description="Delay between PubChem requests"
    )
    pubchem_max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum PubChem API retries"
    )
    enable_pubchem: bool = Field(default=True, description="Enable PubChem integration")

    include_descriptors_default: bool = Field(
        default=False, description="Include descriptors by default"
    )
    include_confidence_default: bool = Field(
        default=True, description="Include confidence scores by default"
    )
    include_molecular_properties: bool = Field(
        default=True, description="Include molecular properties"
    )
    include_drug_likeness_default: bool = Field(
        default=True, description="Include drug-likeness by default"
    )

    probability_precision: int = Field(
        default=4, ge=2, le=10, description="Decimal places for probabilities"
    )
    descriptor_precision: int = Field(
        default=6, ge=2, le=10, description="Decimal places for descriptors"
    )

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="detailed", description="Log format: simple, detailed, or json"
    )
    enable_access_logging: bool = Field(
        default=True, description="Enable HTTP access logging"
    )
    enable_error_tracking: bool = Field(
        default=True, description="Enable error tracking"
    )
    log_file_path: Optional[str] = Field(
        default=None, description="Log file path (None for stdout only)"
    )
    max_log_file_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Max log file size in MB"
    )

    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    slow_request_threshold: float = Field(
        default=5.0, ge=0.1, le=60.0, description="Slow request threshold (seconds)"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Enable metrics collection"
    )

    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    database_url: Optional[str] = Field(
        default=None, description="Database URL for logging/analytics"
    )

    health_check_interval: int = Field(
        default=300, ge=30, le=3600, description="Health check interval (seconds)"
    )
    enable_startup_health_check: bool = Field(
        default=True, description="Enable startup health check"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_prefix": "MAVHIR_",
    }

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment settings."""
        allowed_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in allowed_environments:
            raise ValueError(
                f"Environment must be one of: {', '.join(allowed_environments)}"
            )
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of: {', '.join(allowed_levels)}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        allowed_formats = ["simple", "detailed", "json"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Log format must be one of: {', '.join(allowed_formats)}")
        return v.lower()

    @model_validator(mode="after")
    @classmethod
    def validate_model_files(cls, values: "Settings") -> "Settings":
        """Validate that model files exist or can be created."""
        if values.environment == "testing":
            return values

        model_files = [
            ("ames_model_path", "Ames mutagenicity model"),
            ("ames_scaler_path", "Ames scaler"),
            ("carcinogenicity_model_path", "Carcinogenicity model"),
            ("carcinogenicity_scaler_path", "Carcinogenicity scaler"),
        ]

        missing_files = []
        for field_name, description in model_files:
            file_path = getattr(values, field_name)
            if file_path and not Path(file_path).exists():
                missing_files.append(f"{description}: {file_path}")

        if missing_files:
            if values.environment == "development":
                models_dir = Path(values.models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)

                for field_name, _ in model_files:
                    file_path = Path(getattr(values, field_name, ""))
                    if file_path and not file_path.exists():
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.touch()
            else:
                raise MavhirError(
                    f"Missing model files in {values.environment} environment:\n"
                    + "\n".join(f"  - {f}" for f in missing_files)
                )

        return values

    @model_validator(mode="after")
    @classmethod
    def validate_performance_settings(cls, values: "Settings") -> "Settings":
        """Validate performance-related settings."""
        if values.environment == "production":
            if values.max_batch_size > 500:
                raise ValueError("max_batch_size too large for production (>500)")
            if values.max_concurrent_predictions > 20:
                raise ValueError(
                    "max_concurrent_predictions too large for production (>20)"
                )

        if values.batch_processing_timeout < values.descriptor_timeout:
            raise ValueError("batch_processing_timeout should be >= descriptor_timeout")

        return values

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def get_cors_methods(self) -> List[str]:
        """Get CORS methods as a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",")]

    def get_trusted_hosts(self) -> List[str]:
        """Get trusted hosts as a list."""
        return [host.strip() for host in self.trusted_hosts.split(",")]

    def get_model_config(self) -> Dict[str, Dict[str, Any]]:
        """Get model configuration dictionary."""
        return {
            "ames_mutagenicity": {
                "model_path": self.ames_model_path,
                "scaler_path": self.ames_scaler_path,
                "threshold": self.ames_threshold,
                "name": "Ames Mutagenicity Predictor",
                "description": "Predicts Ames mutagenicity (bacterial reverse mutation test)",
                "endpoint": "ames_mutagenicity",
            },
            "carcinogenicity": {
                "model_path": self.carcinogenicity_model_path,
                "scaler_path": self.carcinogenicity_scaler_path,
                "threshold": self.carcinogenicity_threshold,
                "name": "Carcinogenicity Predictor",
                "description": "Predicts rodent carcinogenicity based on long-term studies",
                "endpoint": "carcinogenicity",
            },
        }

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"

    def get_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def get_log_file_size_bytes(self) -> int:
        """Get maximum log file size in bytes."""
        return self.max_log_file_size_mb * 1024 * 1024

    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            "enabled": self.enable_descriptor_caching,
            "size": self.descriptor_cache_size,
            "ttl": self.descriptor_cache_ttl,
            "redis_url": self.redis_url,
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration."""
        return {
            "monitoring_enabled": self.enable_performance_monitoring,
            "slow_request_threshold": self.slow_request_threshold,
            "metrics_collection": self.enable_metrics_collection,
            "max_concurrent": self.max_concurrent_predictions,
        }


def _create_settings_instance() -> Settings:
    """Create settings instance with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        raise MavhirError(f"Failed to load Mavhir application settings: {e}") from e


settings = _create_settings_instance()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def create_sample_env_file(filename: str = ".env.example") -> None:
    env_content = """
# Basic Settings
MAVHIR_APP_NAME=Mavhir
MAVHIR_VERSION=2.0.0
MAVHIR_ENVIRONMENT=development
MAVHIR_DEBUG=false

# Server Settings
MAVHIR_HOST=0.0.0.0
MAVHIR_PORT=8000
MAVHIR_WORKERS=1

# API Configuration
MAVHIR_API_V1_PREFIX=/api/v1
MAVHIR_MAX_FILE_SIZE_MB=50

# Rate Limiting
MAVHIR_RATE_LIMIT_PER_MINUTE=100
MAVHIR_BURST_RATE_LIMIT=20

# CORS Settings
MAVHIR_CORS_ORIGINS=*
MAVHIR_CORS_ALLOW_CREDENTIALS=true
MAVHIR_CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
MAVHIR_CORS_ALLOW_HEADERS=*

# Security Settings
MAVHIR_ENABLE_SECURITY_HEADERS=true
MAVHIR_ENABLE_TRUSTED_HOSTS=true
MAVHIR_TRUSTED_HOSTS=localhost,127.0.0.1,*.mavhir.com

# Model Settings
MAVHIR_MODELS_DIR=app/models
MAVHIR_AMES_MODEL_PATH=app/models/ames_mutagenicity.pkl
MAVHIR_AMES_SCALER_PATH=app/models/ames_mutagenicity_scaler.pkl
MAVHIR_AMES_THRESHOLD=0.5
MAVHIR_CARCINOGENICITY_MODEL_PATH=app/models/carcinogenicity.pkl
MAVHIR_CARCINOGENICITY_SCALER_PATH=app/models/carcinogenicity_scaler.pkl
MAVHIR_CARCINOGENICITY_THRESHOLD=0.5

# Chemical Processing Settings
MAVHIR_ENABLE_MOLECULE_STANDARDIZATION=true
MAVHIR_STANDARDIZATION_TIMEOUT=30
MAVHIR_MAX_SMILES_LENGTH=1000
MAVHIR_MAX_HEAVY_ATOMS=200

# Descriptor Calculation Settings
MAVHIR_DESCRIPTOR_TIMEOUT=60
MAVHIR_ENABLE_DESCRIPTOR_CACHING=true
MAVHIR_DESCRIPTOR_CACHE_SIZE=1000
MAVHIR_DESCRIPTOR_CACHE_TTL=3600

# Batch Processing Settings
MAVHIR_MAX_BATCH_SIZE=100
MAVHIR_BATCH_PROCESSING_TIMEOUT=300
MAVHIR_MAX_CONCURRENT_PREDICTIONS=10

# PubChem API Settings
MAVHIR_PUBCHEM_BASE_URL=https://pubchem.ncbi.nlm.nih.gov
MAVHIR_PUBCHEM_TIMEOUT=10
MAVHIR_PUBCHEM_RATE_LIMIT_DELAY=0.2
MAVHIR_PUBCHEM_MAX_RETRIES=3
MAVHIR_ENABLE_PUBCHEM=true

# Response Settings
MAVHIR_INCLUDE_DESCRIPTORS_DEFAULT=false
MAVHIR_INCLUDE_CONFIDENCE_DEFAULT=true
MAVHIR_INCLUDE_MOLECULAR_PROPERTIES=true
MAVHIR_INCLUDE_DRUG_LIKENESS_DEFAULT=true
MAVHIR_PROBABILITY_PRECISION=4
MAVHIR_DESCRIPTOR_PRECISION=6

MAVHIR_LOG_LEVEL=INFO
MAVHIR_LOG_FORMAT=detailed
MAVHIR_ENABLE_ACCESS_LOGGING=true
MAVHIR_ENABLE_ERROR_TRACKING=true
MAVHIR_LOG_FILE_PATH=
MAVHIR_MAX_LOG_FILE_SIZE_MB=100

MAVHIR_ENABLE_PERFORMANCE_MONITORING=true
MAVHIR_SLOW_REQUEST_THRESHOLD=5.0
MAVHIR_ENABLE_METRICS_COLLECTION=true

MAVHIR_HEALTH_CHECK_INTERVAL=300
MAVHIR_ENABLE_STARTUP_HEALTH_CHECK=true
"""

    with open(filename, "w") as f:
        f.write(env_content)

    print(f"Sample environment file created: {filename}")
    print("Copy this to .env and customize as needed.")


def validate_configuration() -> None:
    """Validate the current configuration."""
    try:
        test_settings = Settings()

        models_dir = Path(test_settings.models_dir)
        if not models_dir.exists():
            if test_settings.is_development():
                models_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created models directory: {models_dir}")
            else:
                raise MavhirError(f"Models directory does not exist: {models_dir}")

        if not test_settings.is_development():
            model_config = test_settings.get_model_config()
            for endpoint, config in model_config.items():
                model_path = Path(config["model_path"])
                if not model_path.exists():
                    raise MavhirError(
                        f"Model file missing for {endpoint}: {model_path}"
                    )

        if test_settings.redis_url:
            print(f"Redis configured: {test_settings.redis_url}")

        if test_settings.database_url:
            print(f"Database configured: {test_settings.database_url}")

        print("Configuration validation passed")
        return True

    except Exception as e:
        raise MavhirError(f"Configuration validation failed: {e}") from e


def print_current_config() -> None:
    """Print current configuration for debugging."""
    current_settings = get_settings()

    print("MAVHIR CONFIGURATION")
    print("=" * 50)
    print(f"Environment: {current_settings.environment}")
    print(f"Debug Mode: {current_settings.debug}")
    print(f"Version: {current_settings.version}")
    print(f"API Prefix: {current_settings.api_v1_prefix}")
    print(f"Models Directory: {current_settings.models_dir}")
    print(f"Max Batch Size: {current_settings.max_batch_size}")
    print(f"PubChem Enabled: {current_settings.enable_pubchem}")
    print(f"Log Level: {current_settings.log_level}")

    print("\nModel Configuration:")
    model_config = current_settings.get_model_config()
    for endpoint, config in model_config.items():
        print(f"  {endpoint}:")
        print(f"    Model: {config['model_path']}")
        print(f"    Scaler: {config['scaler_path']}")
        print(f"    Threshold: {config['threshold']}")

    print("\nPerformance Settings:")
    perf_config = current_settings.get_performance_config()
    for key, value in perf_config.items():
        print(f"  {key}: {value}")


def create_requirements_file():
    requirements_content = """

fastapi
uvicorn[standard]
pydantic
pydantic-settings

slowapi
rdkit
mordred
scikit-learn
numpy
pandas
joblib

httpx
requests

python-dotenv

structlog

pytest
pytest-asyncio
pytest-cov
black
flake8
mypy
pre-commit

mkdocs
mkdocs-material
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

    print("Created comprehensive requirements.txt")


def create_docker_files():
    dockerfile_content = """
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAVHIR_ENVIRONMENT=production

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        libpq-dev \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash mavhir

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Change ownership to app user
RUN chown -R mavhir:mavhir /app

# Switch to app user
USER mavhir

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    docker_compose_content = """
version: '3.8'

services:
  mavhir:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MAVHIR_ENVIRONMENT=production
      - MAVHIR_DEBUG=false
      - MAVHIR_LOG_LEVEL=INFO
      - MAVHIR_REDIS_URL=redis://redis:6379/0
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: Database
  # postgres:
  #   image: postgres:15-alpine
  #   environment:
  #     POSTGRES_DB: mavhir
  #     POSTGRES_USER: mavhir
  #     POSTGRES_PASSWORD: your_password_here
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"
  #   restart: unless-stopped

volumes:
  redis_data:
  # postgres_data:

networks:
  default:
    name: mavhir_network
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)

    print("Created Docker configuration files")


if __name__ == "__main__":
    print("MAVHIR CONFIGURATION MANAGER")
    print("=" * 40)

    create_sample_env_file()
    create_requirements_file()
    create_docker_files()

    print()

    try:
        validate_configuration()
    except MavhirError as e:
        print(f"Configuration Error: {e}")
        exit(1)

    print()
    print_current_config()
    print()
    print("Configuration setup complete!")
