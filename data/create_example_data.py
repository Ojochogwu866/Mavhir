import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types for data creation."""

    CSV = "csv"
    SDF = "sdf"
    TXT = "txt"


@dataclass
class CompoundData:
    """Type-safe compound data structure."""

    name: str
    smiles: str
    cas_number: Optional[str] = None
    molecular_weight: Optional[float] = None
    properties: Optional[Dict[str, Union[str, int, float]]] = None


@dataclass
class TrainingData:
    """Type-safe training data structure."""

    compound_id: str
    compound_name: str
    smiles: str
    label: int
    metadata: Dict[str, str]


class DataCreator(ABC):
    """Abstract base class for data creators."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_data(self) -> None:
        """Create the specific type of data."""
        pass

    def _write_file_safely(
        self, filepath: Path, content: str, encoding: str = "utf-8"
    ) -> None:
        """Safely write file with proper encoding handling."""
        try:
            with open(filepath, "w", encoding=encoding, errors="replace") as f:
                f.write(content)
            logger.info(f"Successfully created: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write {filepath}: {e}")
            raise


class SDFCreator(DataCreator):
    """Creates SDF (Structure Data Format) files."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.compounds = self._get_example_compounds()

    def _get_example_compounds(self) -> List[CompoundData]:
        """Get example compound data."""
        return [
            CompoundData(
                name="Ethanol",
                smiles="CCO",
                cas_number="64-17-5",
                molecular_weight=46.07,
            ),
            CompoundData(
                name="Benzene",
                smiles="c1ccccc1",
                cas_number="71-43-2",
                molecular_weight=78.11,
            ),
            CompoundData(
                name="Acetic_acid",
                smiles="CC(=O)O",
                cas_number="64-19-7",
                molecular_weight=60.05,
            ),
        ]

    def _create_sdf_block(self, compound: CompoundData) -> str:
        """Create an SDF block for a single compound."""
        sdf_block = f"""{compound.name}
\t-OEChem-01012400002D

\t3  2  0     0  0  0  0  0  0999 V2000
\t\t2.5369    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
\t\t1.0369    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
\t\t0.5369    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
\t1  2  1  0  0  0  0
\t2  3  1  0  0  0  0
M  END
>  <NAME>
{compound.name}

>  <SMILES>
{compound.smiles}

>  <CAS>
{compound.cas_number}

>  <MOLECULAR_WEIGHT>
{compound.molecular_weight}

$$$$
"""
        return sdf_block

    def create_data(self) -> None:
        """Create SDF file with example compounds."""
        sdf_content = ""
        for compound in self.compounds:
            sdf_content += self._create_sdf_block(compound)

        filepath = self.output_dir / "example_compounds.sdf"
        self._write_file_safely(filepath, sdf_content)


class TrainingDataCreator(DataCreator):
    """Creates training datasets for toxicity prediction models."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)

    def _get_base_compounds(self) -> List[Dict[str, str]]:
        """Get base compound information - DRY principle."""
        return [
            {"name": "Ethanol", "smiles": "CCO"},
            {"name": "Methanol", "smiles": "CO"},
            {"name": "Benzene", "smiles": "c1ccccc1"},
            {"name": "Toluene", "smiles": "Cc1ccccc1"},
            {"name": "Phenol", "smiles": "c1ccc(cc1)O"},
            {"name": "Aniline", "smiles": "c1ccc(cc1)N"},
            {"name": "Chloroform", "smiles": "C(Cl)(Cl)Cl"},
            {"name": "Acetone", "smiles": "CC(=O)C"},
            {"name": "Hexane", "smiles": "CCCCCC"},
            {"name": "Butanol", "smiles": "CCCCO"},
            {"name": "Acetic acid", "smiles": "CC(=O)O"},
            {"name": "Formic acid", "smiles": "C(=O)O"},
            {"name": "Nitrobenzene", "smiles": "c1ccc(cc1)[N+](=O)[O-]"},
            {"name": "Pyridine", "smiles": "c1ccncc1"},
            {"name": "Furan", "smiles": "c1ccoc1"},
            {"name": "Thiophene", "smiles": "c1ccsc1"},
            {"name": "Quinoline", "smiles": "c1ccnc2ccccc2c1"},
            {"name": "Naphthalene", "smiles": "c1ccc2ccccc2c1"},
            {"name": "Anthracene", "smiles": "c1ccc2cc3ccccc3cc2c1"},
            {"name": "Pyrene", "smiles": "c1cc2ccc3cccc4ccc(c1)c2c34"},
        ]

    def _create_ames_data(self) -> pd.DataFrame:
        """Create Ames mutagenicity training data."""
        compounds = self._get_base_compounds()

        ames_labels = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]

        data = []
        for i, (compound, label) in enumerate(zip(compounds, ames_labels), 1):
            training_data = TrainingData(
                compound_id=f"AMES_{i:04d}",
                compound_name=compound["name"],
                smiles=compound["smiles"],
                label=label,
                metadata={
                    "test_organism": "S. typhimurium TA98/TA100",
                    "reference": "Sample dataset for demonstration",
                },
            )
            data.append(training_data)

        return self._training_data_to_dataframe(data, "ames_mutagenic")

    def _create_carcinogenicity_data(self) -> pd.DataFrame:
        """Create carcinogenicity training data."""
        compounds = self._get_base_compounds()[:15]

        carc_labels = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        data = []
        for i, (compound, label) in enumerate(zip(compounds, carc_labels), 1):
            training_data = TrainingData(
                compound_id=f"CARC_{i:04d}",
                compound_name=compound["name"],
                smiles=compound["smiles"],
                label=label,
                metadata={
                    "species": "Rat/Mouse",
                    "study_duration": "2-year chronic",
                    "reference": "Sample dataset for demonstration",
                },
            )
            data.append(training_data)

        return self._training_data_to_dataframe(data, "carcinogenic")

    def _training_data_to_dataframe(
        self, data: List[TrainingData], label_column: str
    ) -> pd.DataFrame:
        """Convert training data to DataFrame - DRY principle."""
        df_data = {
            "compound_id": [d.compound_id for d in data],
            "compound_name": [d.compound_name for d in data],
            "smiles": [d.smiles for d in data],
            label_column: [d.label for d in data],
        }

        if data:
            for key in data[0].metadata.keys():
                df_data[key] = [d.metadata[key] for d in data]

        return pd.DataFrame(df_data)

    def create_data(self) -> None:
        """Create all training datasets."""
        # Create Ames data
        ames_df = self._create_ames_data()
        ames_filepath = self.output_dir / "ames_mutagenicity_training.csv"
        ames_df.to_csv(ames_filepath, index=False, encoding="utf-8")
        logger.info(f"Created Ames training data: {ames_filepath}")

        # Create carcinogenicity data
        carc_df = self._create_carcinogenicity_data()
        carc_filepath = self.output_dir / "carcinogenicity_training.csv"
        carc_df.to_csv(carc_filepath, index=False, encoding="utf-8")
        logger.info(f"Created carcinogenicity training data: {carc_filepath}")


class TestCompoundsCreator(DataCreator):
    """Creates test compound datasets."""

    def _get_test_compounds(self) -> List[CompoundData]:
        """Get test compound data."""
        return [
            CompoundData(
                "Caffeine",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "58-08-2",
                properties={"category": "Drug"},
            ),
            CompoundData(
                "Aspirin",
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "50-78-2",
                properties={"category": "Drug"},
            ),
            CompoundData(
                "Ibuprofen",
                "CC(C)CC1=CC=C(C=C1)C(C(=O)O)C",
                "15687-27-1",
                properties={"category": "Drug"},
            ),
            CompoundData(
                "Paracetamol",
                "CC(=O)NC1=CC=C(C=C1)O",
                "103-90-2",
                properties={"category": "Drug"},
            ),
            CompoundData(
                "Nicotine",
                "CN1CCCC1C2=CN=CC=C2",
                "54-11-5",
                properties={"category": "Alkaloid"},
            ),
            CompoundData(
                "Glucose",
                "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
                "50-99-7",
                properties={"category": "Sugar"},
            ),
            CompoundData(
                "Sucrose",
                "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O",
                "57-50-1",
                properties={"category": "Sugar"},
            ),
            CompoundData(
                "Citric acid",
                "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
                "77-92-9",
                properties={"category": "Acid"},
            ),
            CompoundData(
                "Vanillin",
                "COC1=C(C=CC(=C1)C=O)O",
                "121-33-5",
                properties={"category": "Flavor"},
            ),
            CompoundData(
                "Menthol",
                "CC(C)[C@@H]1CC[C@@H](CC1)O",
                "2216-51-5",
                properties={"category": "Terpene"},
            ),
        ]

    def create_data(self) -> None:
        """Create test compounds CSV file."""
        compounds = self._get_test_compounds()

        df_data = {
            "name": [c.name for c in compounds],
            "smiles": [c.smiles for c in compounds],
            "cas_number": [c.cas_number for c in compounds],
            "category": [
                c.properties.get("category", "") if c.properties else ""
                for c in compounds
            ],
        }

        df = pd.DataFrame(df_data)
        filepath = self.output_dir / "test_compounds.csv"
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"Created test compounds: {filepath}")


class ReadmeCreator(DataCreator):
    """Creates README documentation."""

    def create_data(self) -> None:
        """Create README.md file."""
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
"""

        filepath = self.output_dir / "README.md"
        self._write_file_safely(filepath, readme_content)


class DataFactory:
    """Factory class for creating all data types."""

    def __init__(self, base_dir: Path = Path("data")) -> None:
        self.base_dir = base_dir
        self.training_dir = base_dir / "training"
        self.examples_dir = base_dir / "examples"

    def create_all_data(self) -> None:
        """Create all example data files with proper error handling."""
        try:
            logger.info("Starting data creation process...")

            # Create SDF files
            sdf_creator = SDFCreator(self.examples_dir)
            sdf_creator.create_data()

            # Create training data
            training_creator = TrainingDataCreator(self.training_dir)
            training_creator.create_data()

            # Create test compounds
            test_creator = TestCompoundsCreator(self.examples_dir)
            test_creator.create_data()

            # Create README
            readme_creator = ReadmeCreator(self.base_dir)
            readme_creator.create_data()

            logger.info("🎉 All example data files created successfully!")

        except Exception as e:
            logger.error(f"Failed to create data files: {e}")
            raise


def main() -> None:
    """Main function to create all example data."""
    factory = DataFactory()
    factory.create_all_data()


if __name__ == "__main__":
    main()
