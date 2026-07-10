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
- **Features**: 549 Mordred molecular descriptors (variance/correlation-filtered on the training split only)
- **Training Data**: 6,506 compounds from the Hansen et al. (2009) benchmark (http://doc.ml.tu-berlin.de/toxbenchmark/), minus 14 organochlorine pesticides deliberately held out (see Probe Set below)
- **Performance (held-out test — the authors' own predefined fold, not an arbitrary split)**: 81.9% accuracy, 0.883 AUC-ROC, 82.8% precision, 81.1% recall
- **Endpoint**: Bacterial reverse mutation (Salmonella typhimurium)

### Carcinogenicity Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 552 Mordred molecular descriptors (variance/correlation-filtered on the training split only)
- **Training Data**: 1,447 compounds from CPDB (Carcinogenic Potency Database), labeled per the database's own legend across all species sheets, minus 22 organochlorine pesticides deliberately held out — see `data/raw/README.md` for full methodology
- **Performance (held-out test, fresh stratified split)**: 67.8% accuracy, 0.730 AUC-ROC, 69.3% precision, 69.9% recall
- **Endpoint**: 2-year rodent bioassays

These numbers replace an earlier version of this README that cited different figures (88%/0.91 AUC for Ames, 76%/0.82 AUC for carcinogenicity, on claimed but never-actually-wired-in datasets). The models were retrained from scratch on the real, documented datasets above once that discrepancy was found — see `docs/gnn_extension_design.md` Section 0 for the full account.

### GNN comparison and the organochlorine probe set

A GNN (PyTorch Geometric, `app/gnn/`) was trained on the identical splits above for direct comparison, evaluated across 5 seeds (mean ± std, not a single run):

| Task | Metric | RF / GBM baseline | GNN |
|---|---|---|---|
| Ames | Accuracy | 81.9% | 80.1% ± 0.5% |
| Ames | AUC-ROC | 0.883 | 0.874 ± 0.005 |
| Carcinogenicity | Accuracy | 67.8% | 66.6% ± 4.2% |
| Carcinogenicity | AUC-ROC | 0.730 | 0.739 ± 0.027 |

**The 14 (Ames) / 22 (carcinogenicity) organochlorine pesticide compounds above are not a random holdout — they were deliberately excluded from train/val entirely** (`data/processed/organochlorine_probe_set.json`), after an earlier check found that a random split had put 6 of 7 originally spot-checked pesticide compounds into training data by chance, invalidating a naive generalization test. Result on this genuinely unseen compound class:

| Task | Metric | Baseline (RF/GBM) | GNN |
|---|---|---|---|
| Ames probe (n=14, 12 neg/2 pos) | Accuracy | 85.7% | 85.7% |
| Ames probe | AUC-ROC | 0.250 | 0.625 |
| Carcinogenicity probe (n=22, 6 neg/16 pos) | Accuracy | 54.5% | 68.2% |
| Carcinogenicity probe | AUC-ROC | 0.651 | 0.547 |

Both the Ames RF and Ames GNN predict **non-mutagenic for every one of the 14 held-out organochlorines**, missing both true positives — the 85.7% accuracy is a base-rate artifact, not genuine discrimination. On carcinogenicity, the GNN's higher raw accuracy is similarly misleading (specificity of just 16.7% on a majority-positive set); by AUC-ROC and MCC the GBM baseline is actually more discriminative. **Neither architecture reliably generalizes to this compound class** — moving to a GNN does not clearly close the organochlorine under-prediction gap the original research identified. Given the small probe set sizes (particularly n=14 with only 2 positives for Ames), the specific AUC rankings should be read as suggestive, not conclusive; the robust part of the finding is the identical degenerate binary behavior on Ames.

**From-scratch validation**: a second GNN (`app/gnn/model_scratch.py`) was built without PyTorch Geometric's `MessagePassing`/conv layers — message passing and pooling implemented directly with `index_add_` scatter operations — to confirm the library model's results reflect real understanding of the mechanism. Standard-test performance is close (Ames: 79.2% acc / 0.869 AUC vs library's 80.1% / 0.874; CPDB: 65.0% / 0.734 vs 66.6% / 0.739), and on the Ames probe set it reproduces the **identical** degenerate behavior (non-mutagenic for all 14, MCC 0) — a third independent model landing on the same failure mode is stronger evidence this is a real property of the problem, not an artifact of one implementation.

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