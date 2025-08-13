import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator

from .exceptions import ConfigurationError


class Settings(BaseSettings):
    """Application settings: hooked with .env"""

    app_name: str = Field(default="Toxicity prediction API", description="Application Name")
    version: str = Field(default="1.0.0", description="API version")
    environment: str = Field(default="development", description="Runtime Config")
    debug: bool = Field(default=False, description="Enable debug mode")

    # server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(
        default=1, ge=1, le=16, description="Number of worker processes"
    )

    # API settings
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    max_file_size_mb: int = Field(default=50, ge=1, le=1000, description="Maximum file size in MB")
    rate_limit_per_minute: int = Field(default=100, ge=1, le=10000, description="Rate limit per minute")

    # cors settings
    cors_origins: str = Field(default="*", description="Allowed cors origin")
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORs credentials"
    )
    cors_allow_methods: str = Field(
        default="GET,POST,PUT,DELETE", description="Allowed cors methods"
    )
    cors_allow_headers: str = Field(default="*", description="Allowed CORs headers")

    # model settings
    models_dir: str = Field(default="app/models", description="Models directory")

    # Ames mutagenicity model
    ames_model_path: str = Field(
        default="app/models/ames_mutagenicity.pkl", description="Ames model path"
    )
    ames_scaler_path: str = Field(
        default="app/models/ames_scaler.pkl", description="Ames scaler path"
    )
    ames_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Ames classification threshold"
    )

    # carcinogenicity model
    carcinogenicity_model_path: str = Field(default="app/models/carcinogenicity.pkl")
    carcinogenicity_scaler_path: str = Field(
        default="app/models/carcinogenicity_scaler.pkl",
        description="Carcinogenicity scaler path",
    )
    carcinogenicity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Carcinogenicity classification threshold",
    )

    # chemical processing
    enable_molecule_standardization: bool = Field(
        default=True, description="Enable molecule standardization"
    )
    standardization_timeout: int = Field(
        default=30, ge=1, le=30, description="Standardization timeout"
    )

    # descriptor calculation
    descriptor_timeout: int = Field(
        default=60, ge=10, le=600, description="Descriptor calculation"
    )
    enable_descriptor_caching: bool = Field(
        default=True, description="Enable descriptor caching"
    )
    descriptor_cache_size: int = Field(
        default=1000, ge=1, description="Descriptor cache size"
    )

    # Batch processing
    max_batch_size: int = Field(
        default=100, ge=1, le=10000, description="Maximum batch processing size"
    )
    batch_processing_timeout: int = Field(
        default=300, ge=30, le=3600, description="Batch processing timeout (seconds)"
    )

    # EXTERNAL API SETTINGS

    # PubChem API
    pubchem_base_url: str = Field(
        default="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        description="PubChem API base URL",
    )
    pubchem_timeout: int = Field(
        default=10, ge=1, le=60, description="PubChem API timeout (seconds)"
    )
    pubchem_rate_limit_delay: float = Field(
        default=0.2,
        ge=0.1,
        le=2.0,
        description="Delay between PubChem requests (seconds)",
    )
    pubchem_max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum PubChem API retries"
    )

    # Default response options
    include_descriptors_default: bool = Field(
        default=False, description="Include descriptors in response by default"
    )
    include_confidence_default: bool = Field(
        default=True, description="Include confidence scores by default"
    )
    include_molecular_properties: bool = Field(
        default=True, description="Include molecular properties in response"
    )

    # Precision settings
    probability_precision: int = Field(
        default=4, ge=2, le=10, description="Decimal places for probability values"
    )
    descriptor_precision: int = Field(
        default=6, ge=2, le=10, description="Decimal places for descriptor values"
    )

    # LOGGING SETTINGS
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="detailed", description="Log format: simple, detailed, or json"
    )
    enable_access_logging: bool = Field(
        default=True, description="Enable HTTP access logging"
    )
    enable_error_tracking: bool = Field(
        default=True, description="Enable error tracking and reporting"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # validators
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
        """Validate logging level"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of: {', '.join(allowed_levels)}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format"""
        allowed_formats = ["simple", "detailed", "json"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Log format must be one of: {', '.join(allowed_formats)}")
        return v.lower()

    @model_validator(mode='after')
    @classmethod
    def validate_model_files(cls, values: 'Settings') -> 'Settings':
        """
        Validate that model files exist.
        """
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
                raise ConfigurationError(
                    f"Missing model files:\n"
                    + "\n".join(f"  - {f}" for f in missing_files)
                )

        return values

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def get_cors_methods(self) -> List[str]:
        """Get CORS methods as a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",")]

    def get_model_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configuration as a dictionary.

        RETURNS:
            Dictionary with model configurations for each toxicity endpoint
        """
        return {
            "ames_mutagenicity": {
                "model_path": self.ames_model_path,
                "scaler_path": self.ames_scaler_path,
                "threshold": self.ames_threshold,
                "name": "Ames Mutagenicity Predictor",
                "description": "Predicts Ames mutagenicity (bacterial reverse mutation test)",
            },
            "carcinogenicity": {
                "model_path": self.carcinogenicity_model_path,
                "scaler_path": self.carcinogenicity_scaler_path,
                "threshold": self.carcinogenicity_threshold,
                "name": "Carcinogenicity Predictor",
                "description": "Predicts rodent carcinogenicity based on long-term studies",
            },
        }

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def get_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


def _create_settings_instance() -> Settings:
    """Create settings instance with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        raise ConfigurationError(f"Failed to load application settings: {e}") from e


# Global settings instance
settings = _create_settings_instance()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    """
    return settings


def create_sample_env_file(filename: str = ".env.example") -> None:
    """
    Create a sample .env file with all configuration options.

    Useful for deployment and development setup.
    """
    env_content = f"""# Toxicity Predictor API Configuration

# Basic App Settings
APP_NAME=Toxicity Predictor API
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=false

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1

# API Settings
API_V1_PREFIX=/api/v1
MAX_FILE_SIZE_MB=50
RATE_LIMIT_PER_MINUTE=100

# CORS Settings
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE
CORS_ALLOW_HEADERS=*

# Model Settings
MODELS_DIR=app/models
AMES_MODEL_PATH=app/models/ames_mutagenicity.pkl
AMES_SCALER_PATH=app/models/ames_scaler.pkl
AMES_THRESHOLD=0.5
CARCINOGENICITY_MODEL_PATH=app/models/carcinogenicity.pkl
CARCINOGENICITY_SCALER_PATH=app/models/carcinogenicity_scaler.pkl
CARCINOGENICITY_THRESHOLD=0.5

# Processing Settings
ENABLE_MOLECULE_STANDARDIZATION=true
STANDARDIZATION_TIMEOUT=30
DESCRIPTOR_TIMEOUT=60
ENABLE_DESCRIPTOR_CACHING=true
DESCRIPTOR_CACHE_SIZE=1000
MAX_BATCH_SIZE=100
BATCH_PROCESSING_TIMEOUT=300

# PubChem API Settings
PUBCHEM_BASE_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug
PUBCHEM_TIMEOUT=10
PUBCHEM_RATE_LIMIT_DELAY=0.2
PUBCHEM_MAX_RETRIES=3

# Response Settings
INCLUDE_DESCRIPTORS_DEFAULT=false
INCLUDE_CONFIDENCE_DEFAULT=true
INCLUDE_MOLECULAR_PROPERTIES=true
PROBABILITY_PRECISION=4
DESCRIPTOR_PRECISION=6

# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=detailed
ENABLE_ACCESS_LOGGING=true
ENABLE_ERROR_TRACKING=true
"""

    with open(filename, "w") as f:
        f.write(env_content)

    print(f"Sample environment file created: {filename}")
    print("Copy this to .env and customize as needed.")


def validate_configuration() -> None:
    """
    Validate the current configuration.

    Useful for startup checks and health monitoring.
    Raises ConfigurationError if any issues are found.
    """
    try:
        test_settings = Settings()
        models_dir = Path(test_settings.models_dir)
        if not models_dir.exists():
            raise ConfigurationError(f"Models directory does not exist: {models_dir}")

        model_config = test_settings.get_model_config()
        for endpoint, config in model_config.items():
            model_path = Path(config["model_path"])
            if not model_path.exists() and not test_settings.is_development():
                raise ConfigurationError(
                    f"Model file missing for {endpoint}: {model_path}"
                )

        print("Configuration validation passed")

    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


# development helpers


def print_current_config() -> None:
    """Print current configuration (useful for debugging)."""
    current_settings = get_settings()

    print("Current Configuration:")
    print("=" * 50)
    print(f"Environment: {current_settings.environment}")
    print(f"Debug Mode: {current_settings.debug}")
    print(f"API Prefix: {current_settings.api_v1_prefix}")
    print(f"Models Directory: {current_settings.models_dir}")
    print(f"Max Batch Size: {current_settings.max_batch_size}")
    print(f"PubChem Timeout: {current_settings.pubchem_timeout}s")
    print(f"Log Level: {current_settings.log_level}")

    print("\nModel Configuration:")
    model_config = current_settings.get_model_config()
    for endpoint, config in model_config.items():
        print(f"  {endpoint}:")
        print(f"    Model: {config['model_path']}")
        print(f"    Scaler: {config['scaler_path']}")
        print(f"    Threshold: {config['threshold']}")


if __name__ == "__main__":
    print("Toxicity Predictor Configuration Manager")
    print("=" * 40)

    create_sample_env_file()
    print()

    try:
        validate_configuration()
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        exit(1)

    print()
    print_current_config()