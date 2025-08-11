import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_sdf():
    """
    Create example SDF file with common chemicals.
    
    SDF (Structure Data Format) is the standard format for
    chemical structures with associated data.
    """
    
    sdf_content = """Ethanol
	-OEChem-01012400002D

	3  2  0     0  0  0  0  0  0999 V2000
		2.5369    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		1.0369    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		0.5369    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
	1  2  1  0  0  0  0
	2  3  1  0  0  0  0
M  END
>  <NAME>
Ethanol

>  <SMILES>
CCO

>  <CAS>
64-17-5

>  <MOLECULAR_WEIGHT>
46.07

$$$$
Benzene
	-OEChem-01012400002D

	6  6  0     0  0  0  0  0  0999 V2000
		2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		1.0000    1.7321    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	-1.0000    1.7321    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	-2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	-1.0000   -1.7321    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		1.0000   -1.7321    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	1  2  2  0  0  0  0
	2  3  1  0  0  0  0
	3  4  2  0  0  0  0
	4  5  1  0  0  0  0
	5  6  2  0  0  0  0
	6  1  1  0  0  0  0
M  END
>  <NAME>
Benzene

>  <SMILES>
c1ccccc1

>  <CAS>
71-43-2

>  <MOLECULAR_WEIGHT>
78.11

$$$$
Acetic_acid
	-OEChem-01012400002D

	4  3  0     0  0  0  0  0  0999 V2000
		1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	-0.7500    1.2990    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
	-0.7500   -1.2990    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
	1  2  1  0  0  0  0
	2  3  2  0  0  0  0
	2  4  1  0  0  0  0
M  END
>  <NAME>
Acetic acid

>  <SMILES>
CC(=O)O

>  <CAS>
64-19-7

>  <MOLECULAR_WEIGHT>
60.05

$$$$
"""
    
    return sdf_content


def create_training_datasets():
    """Create example training datasets in CSV format."""
    
    # Ames mutagenicity training data
    ames_data = pd.DataFrame({
        'compound_id': [f'AMES_{i:04d}' for i in range(1, 21)],
        'compound_name': [
            'Ethanol', 'Methanol', 'Benzene', 'Toluene', 'Phenol',
            'Aniline', 'Chloroform', 'Acetone', 'Hexane', 'Butanol',
            'Acetic acid', 'Formic acid', 'Nitrobenzene', 'Pyridine', 'Furan',
            'Thiophene', 'Quinoline', 'Naphthalene', 'Anthracene', 'Pyrene'
        ],
        'smiles': [
            'CCO', 'CO', 'c1ccccc1', 'Cc1ccccc1', 'c1ccc(cc1)O',
            'c1ccc(cc1)N', 'C(Cl)(Cl)Cl', 'CC(=O)C', 'CCCCCC', 'CCCCO',
            'CC(=O)O', 'C(=O)O', 'c1ccc(cc1)[N+](=O)[O-]', 'c1ccncc1', 'c1ccoc1',
            'c1ccsc1', 'c1ccnc2ccccc2c1', 'c1ccc2ccccc2c1', 'c1ccc2cc3ccccc3cc2c1', 'c1cc2ccc3cccc4ccc(c1)c2c34'
        ],
        'ames_mutagenic': [
            0, 0, 1, 0, 0,  # Ethanol-Phenol
            1, 1, 0, 0, 0,  # Aniline-Butanol  
            0, 0, 1, 0, 0,  # Acetic acid-Furan
            0, 1, 1, 1, 1   # Thiophene-Pyrene
        ],
        'test_organism': [
            'S. typhimurium TA98/TA100'] * 20,
        'reference': [
            'Sample dataset for demonstration'] * 20
    })
    
    # Carcinogenicity training data
    carcinogenicity_data = pd.DataFrame({
        'compound_id': [f'CARC_{i:04d}' for i in range(1, 16)],
        'compound_name': [
            'Ethanol', 'Benzene', 'Toluene', 'Phenol', 'Aniline',
            'Chloroform', 'Acetone', 'Acetic acid', 'Nitrobenzene', 'Quinoline',
            'Naphthalene', 'Anthracene', 'Benzo[a]pyrene', 'Formaldehyde', 'Methylene chloride'
        ],
        'smiles': [
            'CCO', 'c1ccccc1', 'Cc1ccccc1', 'c1ccc(cc1)O', 'c1ccc(cc1)N',
            'C(Cl)(Cl)Cl', 'CC(=O)C', 'CC(=O)O', 'c1ccc(cc1)[N+](=O)[O-]', 'c1ccnc2ccccc2c1',
            'c1ccc2ccccc2c1', 'c1ccc2cc3ccccc3cc2c1', 'c1cc2c3c(c1)ccc1c3c(cc3ccccc13)c1ccccc21', 'C=O', 'C(Cl)Cl'
        ],
        'carcinogenic': [
            0, 1, 0, 0, 1,  # Ethanol-Aniline
            1, 0, 0, 1, 1,  # Chloroform-Quinoline
            1, 1, 1, 1, 1   # Naphthalene-Methylene chloride
        ],
        'species': ['Rat/Mouse'] * 15,
        'study_duration': ['2-year chronic'] * 15,
        'reference': ['Sample dataset for demonstration'] * 15
    })
    
    return ames_data, carcinogenicity_data


def create_test_compounds():
    """Create test compounds for API testing."""
    
    test_compounds = pd.DataFrame({
        'name': [
            'Caffeine', 'Aspirin', 'Ibuprofen', 'Paracetamol', 'Nicotine',
            'Glucose', 'Sucrose', 'Citric acid', 'Vanillin', 'Menthol'
        ],
        'smiles': [
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(=O)OC1=CC=CC=C1C(=O)O',      # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C(=O)O)C',  # Ibuprofen
            'CC(=O)NC1=CC=C(C=C1)O',         # Paracetamol
            'CN1CCCC1C2=CN=CC=C2',           # Nicotine
            'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O',  # Glucose
            'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O',  # Sucrose
            'C(C(=O)O)C(CC(=O)O)(C(=O)O)O',  # Citric acid
            'COC1=C(C=CC(=C1)C=O)O',         # Vanillin
            'CC(C)[C@@H]1CC[C@@H](CC1)O'     # Menthol
        ],
        'cas_number': [
            '58-08-2', '50-78-2', '15687-27-1', '103-90-2', '54-11-5',
            '50-99-7', '57-50-1', '77-92-9', '121-33-5', '2216-51-5'
        ],
        'category': [
            'Drug', 'Drug', 'Drug', 'Drug', 'Alkaloid',
            'Sugar', 'Sugar', 'Acid', 'Flavor', 'Terpene'
        ]
    })
    
    return test_compounds


def create_all_example_data():
    """Create all example data files."""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    (data_dir / "training").mkdir(exist_ok=True)
    (data_dir / "examples").mkdir(exist_ok=True)
    
    logger.info("Creating example data files...")
    
    # Create SDF file
    sdf_content = create_example_sdf()
    with open(data_dir / "examples" / "example_compounds.sdf", 'w') as f:
        f.write(sdf_content)
    logger.info("Created example SDF file")
    
    # Create training datasets
    ames_data, carc_data = create_training_datasets()
    
    ames_data.to_csv(data_dir / "training" / "ames_mutagenicity_training.csv", index=False)
    logger.info("Created Ames training data")
    
    carc_data.to_csv(data_dir / "training" / "carcinogenicity_training.csv", index=False)
    logger.info("Created carcinogenicity training data")
    
    # Create test compounds
    test_data = create_test_compounds()
    test_data.to_csv(data_dir / "examples" / "test_compounds.csv", index=False)
    logger.info("Created test compounds file")
    
    # Create README for data folder
    readme_content = """# Data Directory

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
curl -X POST "http://localhost:8000/api/v1/predict/smiles" \\
	-H "Content-Type: application/json" \\
	-d '{"smiles": "CCO"}'
```
"""
    
    with open(data_dir / "README.md", 'w') as f:
        f.write(readme_content)
    logger.info("✅ Created data README")
    
    logger.info("🎉 All example data files created successfully!")


if __name__ == "__main__":
    create_all_example_data()