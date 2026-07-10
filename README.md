# Mavhir

> **ML-powered chemical toxicity prediction API built with FastAPI, RDKit, and scikit-learn**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Toxicity Predictor API provides machine learning-based predictions for chemical toxicity endpoints including:

- **Ames Mutagenicity**: Bacterial reverse mutation test (OECD 471)
- **Carcinogenicity**: Rodent carcinogenicity studies (OECD 451/453)

Built for **researchers**, **regulatory scientists**, and **pharmaceutical companies** who need fast, reliable toxicity predictions for chemical risk assessment.

## Key Features

- **Fast Predictions**: Sub-second response times for single compounds
- **Batch Processing**: Handle up to 1000 compounds per request
- **File Upload**: Direct SDF file processing
- **Chemical Lookup**: Integrated PubChem database search
- **Rich Metadata**: Molecular properties, descriptors, confidence scores
- **Health Monitoring**: Built-in system diagnostics
- **Auto Documentation**: Interactive API docs with Swagger UI
- **Docker Ready**: One-command deployment

## Quick Start

### Prerequisites
- Python 3.9+
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/Ojochogwu866/Mavhir.git
cd Mavhir

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train Models
```bash
python data/create_example_data.py
python app/models/train_models.py
```

### 3. Configure Environment
```bash
cp .env.example .env

# Edit .env file with your settings (optional)
```

### 4. Start the API
```bash
# Development server
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API
```bash
# Check health
curl http://localhost:8000/health

# Predict toxicity
curl -X POST "http://localhost:8000/api/v1/predict/smiles" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "endpoints": ["ames_mutagenicity"]}'
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/api/v1/predict/smiles` | POST | Single compound prediction |
| `/api/v1/predict/batch` | POST | Batch prediction |
| `/api/v1/predict/sdf` | POST | SDF file upload |
| `/api/v1/chemical/lookup/{name}` | GET | PubChem compound lookup |
| `/api/v1/chemical/validate` | GET | SMILES validation |

## Usage Examples

### Single Compound Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/smiles",
    json={
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "endpoints": ["ames_mutagenicity", "carcinogenicity"],
        "include_properties": True
    }
)

result = response.json()
print(f"Ames prediction: {result['data']['predictions']['ames_mutagenicity']['prediction']}")
```

### Batch Processing
```python
compounds = [
    "CCO",           # Ethanol
    "CC(=O)O",       # Acetic acid
    "c1ccccc1"       # Benzene
]

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    json={
        "smiles_list": compounds,
        "endpoints": ["ames_mutagenicity"]
    }
)

results = response.json()
print(f"Processed {results['summary']['successful']} compounds successfully")
```

### PubChem Lookup
```python
response = requests.get("http://localhost:8000/api/v1/chemical/lookup/aspirin")
compound_info = response.json()

if compound_info["found"]:
    print(f"Aspirin SMILES: {compound_info['canonical_smiles']}")
    print(f"Molecular weight: {compound_info['molecular_weight']}")
```

## Architecture

```
mavhir/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── health.py
│   │   ├── chemical.py
│   │   └── predict.py
│   ├── core/
│   │   ├── config.py
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── services/
│   │   ├── chemical_processor.py
│   │   ├── descriptor_calculator.py
│   │   ├── predictor.py
│   │   └── pubchem_client.py
│   └── models/
├── data/
├── tests/
├── docs/
└── requirements.txt
```

## Testing

```bash

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m "not benchmark"
pytest tests/test_api.py
```

## Docker Deployment

### Development
```bash
docker-compose up --build
```

### Production
```bash
# Build image
docker build -t mavhir .

# Run container
docker run -p 8000:8000 mavhir
```

## 📊 Model Information

### Ames Mutagenicity Model
- **Algorithm**: Random Forest Classifier
- **Features**: 552 Mordred molecular descriptors (variance/correlation-filtered on the training split only)
- **Training Data**: 6,506 compounds from the Hansen et al. (2009) benchmark (http://doc.ml.tu-berlin.de/toxbenchmark/)
- **Performance (held-out test — the authors' own predefined fold, not an arbitrary split)**: 81.3% accuracy, 0.884 AUC-ROC, 82.7% precision, 79.7% recall
- **Endpoint**: Bacterial reverse mutation (Salmonella typhimurium)

### Carcinogenicity Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 550 Mordred molecular descriptors (variance/correlation-filtered on the training split only)
- **Training Data**: 1,447 compounds from CPDB (Carcinogenic Potency Database), labeled per the database's own legend across all species sheets — see `data/raw/README.md` for full methodology
- **Performance (held-out test, fresh stratified split)**: 65.1% accuracy, 0.704 AUC-ROC, 68.2% precision, 63.5% recall
- **Endpoint**: 2-year rodent bioassays

These numbers replace an earlier version of this README that cited different figures (88%/0.91 AUC for Ames, 76%/0.82 AUC for carcinogenicity, on claimed but never-actually-wired-in datasets). The models were retrained from scratch on the real, documented datasets above once that discrepancy was found — see `docs/gnn_extension_design.md` Section 0 for the full account.

## 🔧 Configuration

Key environment variables:

```bash
# Basic settings
ENVIRONMENT=production
DEBUG=false
MAX_BATCH_SIZE=100

# Model settings
AMES_MODEL_PATH=app/models/ames_mutagenicity.pkl
CARCINOGENICITY_MODEL_PATH=app/models/carcinogenicity.pkl

# Processing settings
DESCRIPTOR_TIMEOUT=60
ENABLE_DESCRIPTOR_CACHING=true

# PubChem API
PUBCHEM_RATE_LIMIT_DELAY=0.2
PUBCHEM_MAX_RETRIES=3
```

## 📈 Performance

- **Single prediction**: ~150ms average
- **Batch processing**: ~100 compounds/minute
- **Descriptor calculation**: ~50ms per compound
- **Memory usage**: ~500MB base + ~1MB per cached compound

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black app/ tests/
isort app/ tests/

# Type checking
mypy app/