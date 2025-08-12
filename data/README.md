# Data Directory

This directory contains training data and examples for the Toxicity Predictor API.

## Structure

```
data/
├── training/                    # Training datasets
│   ├── ames_mutagenicity_training.csv
│   └── carcinogenicity_training.csv
├── examples/                    # Example files for testing
│   ├── example_compounds.sdf    # SDF file with sample structures
│   └── test_compounds.csv       # Common compounds for testing
└── README.md                    # This file
```

## Usage

### Training Models
```bash
python app/models/train_models.py
```

### Testing API  
```bash
curl -X POST "http://localhost:8000/api/v1/predict/smiles" \
    -H "Content-Type: application/json" \
    -d '{"smiles": "CCO"}'
```

## Data Formats

### Training Data
- CSV files with compound identifiers, SMILES, labels, and metadata
- Labels: 0 (negative), 1 (positive)

### SDF Files
- Standard structure data format for chemical compounds
- Includes molecular structure and properties

### Test Compounds
- CSV file with diverse chemical compounds for API testing
- Includes pharmaceuticals, natural products, and common chemicals
