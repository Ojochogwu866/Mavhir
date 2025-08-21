import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
from sklearn.base import BaseEstimator

from rdkit import Chem
from mordred import Calculator, descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("mavhir_training.log")],
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported ML model types."""

    RANDOM_FOREST = "RandomForestClassifier"
    GRADIENT_BOOSTING = "GradientBoostingClassifier"
    EXTRA_TREES = "ExtraTreesClassifier"


class ToxicityEndpoint(Enum):
    """Toxicity prediction endpoints."""

    AMES_MUTAGENICITY = "ames_mutagenicity"
    CARCINOGENICITY = "carcinogenicity"


@dataclass(frozen=True)
class ModelMetrics:
    """Comprehensive model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    f1_score: float

    specificity: float
    balanced_accuracy: float
    matthews_corrcoef: float

    cv_mean: float
    cv_std: float

    def __post_init__(self) -> None:
        """Validate metric values."""
        for metric_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                continue
            if (
                not 0.0 <= value <= 1.0
                and "std" not in metric_name
                and "matthews" not in metric_name
            ):
                raise ValueError(f"Invalid {metric_name}: {value} (must be 0-1)")


@dataclass
class ModelConfig:
    """ model configuration."""

    endpoint: ToxicityEndpoint
    model_type: ModelType
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Training parameters
    enable_feature_selection: bool = True
    max_features: Optional[int] = None
    class_weight: str = "balanced"

    # Validation parameters
    stratify: bool = True
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError(
                f"validation_size must be between 0 and 1, got {self.validation_size}"
            )
        if self.test_size + self.validation_size >= 0.8:
            raise ValueError("test_size + validation_size must be < 0.8")
        if self.cv_folds <= 1:
            raise ValueError(f"cv_folds must be > 1, got {self.cv_folds}")


@dataclass
class TrainingResult:
    """Comprehensive training results."""

    model: BaseEstimator
    scaler: StandardScaler
    descriptor_names: List[str]
    feature_names: List[str]
    metrics: ModelMetrics
    config: ModelConfig
    training_time: float

    # Additional data
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[Dict[str, List[float]]] = None
    validation_curves: Optional[Dict[str, Any]] = None


class ProductionDataGenerator:

    @staticmethod
    def get_realistic_ames_data() -> List[Dict[str, Any]]:
        return [
            # Well-established non-mutagens
            {
                "name": "Ethanol",
                "smiles": "CCO",
                "label": 0,
                "confidence": "high",
                "references": ["OECD TG 471", "Mortelmans 1986"],
                "study_type": "Regulatory",
            },
            {
                "name": "Glucose",
                "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "Sucrose",
                "smiles": "OC[C@H]1OC(O[C@]2(CO)OC[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "Acetic acid",
                "smiles": "CC(=O)O",
                "label": 0,
                "confidence": "medium",
                "references": ["OECD TG 471"],
                "study_type": "Regulatory",
            },
            {
                "name": "Citric acid",
                "smiles": "OC(=O)CC(O)(C(=O)O)CC(=O)O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "L-Ascorbic acid",
                "smiles": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "label": 0,
                "confidence": "medium",
                "references": ["Mortelmans 1986"],
                "study_type": "Academic",
            },
            {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "label": 0,
                "confidence": "medium",
                "references": ["Ishidate 1984"],
                "study_type": "Academic",
            },
            {
                "name": "Vanillin",
                "smiles": "COC1=C(C=CC(=C1)C=O)O",
                "label": 0,
                "confidence": "high",
                "references": ["NTP 1982"],
                "study_type": "NTP",
            },
            {
                "name": "Menthol",
                "smiles": "CC(C)[C@@H]1CC[C@@H](CC1)O",
                "label": 0,
                "confidence": "high",
                "references": ["RIFM 2001"],
                "study_type": "Industry",
            },
            {
                "name": "Sodium chloride",
                "smiles": "[Na+].[Cl-]",
                "label": 0,
                "confidence": "high",
                "references": ["OECD TG 471"],
                "study_type": "Regulatory",
            },
            {
                "name": "Potassium chloride",
                "smiles": "[K+].[Cl-]",
                "label": 0,
                "confidence": "high",
                "references": ["OECD TG 471"],
                "study_type": "Regulatory",
            },
            {
                "name": "Magnesium sulfate",
                "smiles": "[Mg+2].[O-]S(=O)(=O)[O-]",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            # Well-established mutagens
            {
                "name": "2-Aminoanthracene",
                "smiles": "NC1=CC2=CC3=CC=CC=C3C=C2C=C1",
                "label": 1,
                "confidence": "high",
                "references": ["Mortelmans 1986", "NTP positive control"],
                "study_type": "Positive Control",
            },
            {
                "name": "Benzo[a]pyrene",
                "smiles": "C1=CC2=C3C(=C1)C=CC4=CC=CC(=C43)C=C2",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1", "Mortelmans 1986"],
                "study_type": "IARC",
            },
            {
                "name": "4-Nitroquinoline N-oxide",
                "smiles": "[O-][N+](=O)C1=CC=NC2=CC=CC=C12",
                "label": 1,
                "confidence": "high",
                "references": ["NTP positive control"],
                "study_type": "Positive Control",
            },
            {
                "name": "2-Acetylaminofluorene",
                "smiles": "CC(=O)NC1=CC2=CC=CC=C2C3=CC=CC=C13",
                "label": 1,
                "confidence": "high",
                "references": ["Mortelmans 1986", "NTP"],
                "study_type": "NTP",
            },
            {
                "name": "Aflatoxin B1",
                "smiles": "COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C5=C1OC(CO5)=C4",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Mitomycin C",
                "smiles": "COC1=C(N)C(=O)C2=C(C1=O)N3CC4N(C3C2)C4(C)OC(=O)N",
                "label": 1,
                "confidence": "high",
                "references": ["NTP positive control"],
                "study_type": "Positive Control",
            },
            {
                "name": "9-Aminoacridine",
                "smiles": "NC1=CC=CC2=NC3=CC=CC=C3C=C12",
                "label": 1,
                "confidence": "high",
                "references": ["Mortelmans 1986"],
                "study_type": "Academic",
            },
            {
                "name": "2-Nitrofluorene",
                "smiles": "[O-][N+](=O)C1=CC2=CC=CC=C2C3=CC=CC=C13",
                "label": 1,
                "confidence": "high",
                "references": ["NTP 1988"],
                "study_type": "NTP",
            },
            {
                "name": "Methyl methanesulfonate",
                "smiles": "COS(=O)(=O)C",
                "label": 1,
                "confidence": "high",
                "references": ["NTP positive control"],
                "study_type": "Positive Control",
            },
            {
                "name": "ICR-191",
                "smiles": "CCN(CC)CCCCCCN(CC)CC.ClC1=CC=C2C3=CC=CC=C3C(=O)C4=CC=CC=C4C2=C1",
                "label": 1,
                "confidence": "high",
                "references": ["NTP positive control"],
                "study_type": "Positive Control",
            },
            # Borderline/moderate cases (important for model robustness)
            {
                "name": "Benzene",
                "smiles": "C1=CC=CC=C1",
                "label": 0,
                "confidence": "low",
                "references": ["Mortelmans 1986 - negative"],
                "study_type": "Academic",
            },
            {
                "name": "Toluene",
                "smiles": "CC1=CC=CC=C1",
                "label": 0,
                "confidence": "medium",
                "references": ["OECD TG 471"],
                "study_type": "Regulatory",
            },
            {
                "name": "Phenol",
                "smiles": "OC1=CC=CC=C1",
                "label": 0,
                "confidence": "medium",
                "references": ["OECD TG 471"],
                "study_type": "Regulatory",
            },
            {
                "name": "Aniline",
                "smiles": "NC1=CC=CC=C1",
                "label": 1,
                "confidence": "medium",
                "references": ["Zeiger 1987"],
                "study_type": "Academic",
            },
            {
                "name": "4-Aminobiphenyl",
                "smiles": "NC1=CC=C(C=C1)C2=CC=CC=C2",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "2-Naphthylamine",
                "smiles": "NC1=CC2=CC=CC=C2C=C1",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Quinoline",
                "smiles": "C1=CC=NC2=CC=CC=C12",
                "label": 0,
                "confidence": "low",
                "references": ["NTP 1979"],
                "study_type": "NTP",
            },
            {
                "name": "Isoquinoline",
                "smiles": "C1=CC=C2C=NC=CC2=C1",
                "label": 0,
                "confidence": "low",
                "references": ["Zeiger 1987"],
                "study_type": "Academic",
            },
            {
                "name": "2-Methylquinoline",
                "smiles": "CC1=NC2=CC=CC=C2C=C1",
                "label": 0,
                "confidence": "medium",
                "references": ["Zeiger 1987"],
                "study_type": "Academic",
            },
            {
                "name": "8-Hydroxyquinoline",
                "smiles": "OC1=CC=CC2=NC=CC=C12",
                "label": 1,
                "confidence": "medium",
                "references": ["Mortelmans 1986"],
                "study_type": "Academic",
            },
        ]

    @staticmethod
    def get_realistic_carcinogenicity_data() -> List[Dict[str, Any]]:
        """
        Realistic carcinogenicity data based on rodent studies and IARC classifications.
        """
        return [
            # Non-carcinogens (IARC Group 3 or negative studies)
            {
                "name": "Ethanol",
                "smiles": "CCO",
                "label": 0,
                "confidence": "medium",
                "references": ["IARC Group 3"],
                "study_type": "IARC",
            },
            {
                "name": "Glucose",
                "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "Sucrose",
                "smiles": "OC[C@H]1OC(O[C@]2(CO)OC[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
                "label": 0,
                "confidence": "high",
                "references": ["NTP negative"],
                "study_type": "NTP",
            },
            {
                "name": "Acetic acid",
                "smiles": "CC(=O)O",
                "label": 0,
                "confidence": "high",
                "references": ["IARC Group 3"],
                "study_type": "IARC",
            },
            {
                "name": "Citric acid",
                "smiles": "OC(=O)CC(O)(C(=O)O)CC(=O)O",
                "label": 0,
                "confidence": "high",
                "references": ["FDA GRAS"],
                "study_type": "GRAS",
            },
            {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "label": 0,
                "confidence": "medium",
                "references": ["NTP negative"],
                "study_type": "NTP",
            },
            {
                "name": "Vanillin",
                "smiles": "COC1=C(C=CC(=C1)C=O)O",
                "label": 0,
                "confidence": "high",
                "references": ["NTP negative"],
                "study_type": "NTP",
            },
            {
                "name": "Menthol",
                "smiles": "CC(C)[C@@H]1CC[C@@H](CC1)O",
                "label": 0,
                "confidence": "high",
                "references": ["NTP negative"],
                "study_type": "NTP",
            },
            {
                "name": "Sodium chloride",
                "smiles": "[Na+].[Cl-]",
                "label": 0,
                "confidence": "high",
                "references": ["IARC Group 3"],
                "study_type": "IARC",
            },
            # Carcinogens (IARC Group 1, 2A, 2B or positive NTP studies)
            {
                "name": "Benzo[a]pyrene",
                "smiles": "C1=CC2=C3C(=C1)C=CC4=CC=CC(=C43)C=C2",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Aflatoxin B1",
                "smiles": "COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C5=C1OC(CO5)=C4",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "2-Acetylaminofluorene",
                "smiles": "CC(=O)NC1=CC2=CC=CC=C2C3=CC=CC=C13",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 2B"],
                "study_type": "IARC",
            },
            {
                "name": "4-Aminobiphenyl",
                "smiles": "NC1=CC=C(C=C1)C2=CC=CC=C2",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "2-Naphthylamine",
                "smiles": "NC1=CC2=CC=CC=C2C=C1",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Benzidine",
                "smiles": "NC1=CC=C(C=C1)C2=CC=C(C=C2)N",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Vinyl chloride",
                "smiles": "C=CCl",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Formaldehyde",
                "smiles": "C=O",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "1,3-Butadiene",
                "smiles": "C=CC=C",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Ethylene oxide",
                "smiles": "C1CO1",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "N-Nitrosodiethylamine",
                "smiles": "CCN(CC)N=O",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 2A"],
                "study_type": "IARC",
            },
            {
                "name": "2,3,7,8-TCDD",
                "smiles": "ClC1=CC2=C(OC3=C(Cl)C=C(Cl)C4=C3OC(=O)C=4)C(Cl)=C1",
                "label": 1,
                "confidence": "high",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            # Borderline cases (important for model robustness)
            {
                "name": "Benzene",
                "smiles": "C1=CC=CC=C1",
                "label": 1,
                "confidence": "medium",
                "references": ["IARC Group 1"],
                "study_type": "IARC",
            },
            {
                "name": "Toluene",
                "smiles": "CC1=CC=CC=C1",
                "label": 0,
                "confidence": "medium",
                "references": ["IARC Group 3"],
                "study_type": "IARC",
            },
            {
                "name": "Phenol",
                "smiles": "OC1=CC=CC=C1",
                "label": 0,
                "confidence": "low",
                "references": ["NTP equivocal"],
                "study_type": "NTP",
            },
            {
                "name": "Aniline",
                "smiles": "NC1=CC=CC=C1",
                "label": 1,
                "confidence": "medium",
                "references": ["IARC Group 3", "NTP positive"],
                "study_type": "Mixed",
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "label": 0,
                "confidence": "low",
                "references": ["IARC Group 3"],
                "study_type": "IARC",
            },
            {
                "name": "Saccharin",
                "smiles": "C1=CC=C2C(=C1)C(=O)NS2(=O)=O",
                "label": 1,
                "confidence": "low",
                "references": ["IARC Group 2B"],
                "study_type": "IARC",
            },
            {
                "name": "Chloroform",
                "smiles": "C(Cl)(Cl)Cl",
                "label": 1,
                "confidence": "medium",
                "references": ["IARC Group 2B"],
                "study_type": "IARC",
            },
            {
                "name": "Methylene chloride",
                "smiles": "C(Cl)Cl",
                "label": 1,
                "confidence": "medium",
                "references": ["IARC Group 2A"],
                "study_type": "IARC",
            },
        ]

class DescriptorCalculator:

    def __init__(self):
        """Initialize with comprehensive descriptor sets."""
        self.calculator = Calculator()
        self._register_descriptors()
        self.all_descriptor_names = [str(d) for d in self.calculator.descriptors]

        logger.info(
            f"Initialized calculator with {len(self.all_descriptor_names)} descriptors"
        )

    def _register_descriptors(self):
        """Register comprehensive descriptor sets."""
        try:
            # Core descriptor groups
            core_groups = [
                descriptors.AtomCount,
                descriptors.BondCount,
                descriptors.RingCount,
                descriptors.Constitutional,
                descriptors.Weight,
                descriptors.SLogP,
            ]

            for group in core_groups:
                try:
                    self.calculator.register(group)
                except Exception as e:
                    logger.warning(f"Failed to register {group.__name__}: {e}")

            additional_groups = [
                "TopologicalIndex",
                "Polarizability",
                "FragmentComplexity",
                "Framework",
                "Autocorrelation",
                "BCUT",
                "DistanceMatrix",
                "EState",
                "Aromatic",
                "TopoPSA",
                "BalabanJ",
                "BertzCT",
            ]

            for group_name in additional_groups:
                try:
                    if hasattr(descriptors, group_name):
                        group = getattr(descriptors, group_name)
                        self.calculator.register(group)
                except Exception as e:
                    logger.debug(f"Could not register {group_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize descriptor calculator: {e}")
            raise

    def calculate_descriptors(
        self, smiles_list: List[str]
    ) -> Tuple[pd.DataFrame, List[int], List[str]]:
        logger.info(f"Calculating descriptors for {len(smiles_list)} compounds...")

        descriptor_matrix = []
        valid_indices = []
        failed_count = 0

        for i, smiles in enumerate(smiles_list):
            if i % 50 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(smiles_list)} compounds")

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                    failed_count += 1
                    continue

                desc_values = self.calculator(mol)

                clean_values = []
                for val in desc_values:
                    if val is None:
                        clean_values.append(0.0)
                    else:
                        try:
                            float_val = float(val)
                            if np.isfinite(float_val):
                                clean_values.append(float_val)
                            else:
                                clean_values.append(0.0)
                        except (ValueError, TypeError, OverflowError):
                            clean_values.append(0.0)

                descriptor_matrix.append(clean_values)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate descriptors for SMILES at index {i}: {e}"
                )
                failed_count += 1
                continue

        if not descriptor_matrix:
            raise RuntimeError("No valid descriptors calculated for any compounds")

        descriptor_df = pd.DataFrame(
            descriptor_matrix, columns=self.all_descriptor_names
        )

        descriptor_df = self._clean_descriptors(descriptor_df)

        success_rate = (len(descriptor_df) / len(smiles_list)) * 100
        logger.info(
            f"Descriptor calculation complete: {len(descriptor_df)} valid compounds "
            f"({success_rate:.1f}% success rate), {len(descriptor_df.columns)} features"
        )

        return descriptor_df, valid_indices, list(descriptor_df.columns)

    def _clean_descriptors(
        self, df: pd.DataFrame, variance_threshold: float = 1e-6
    ) -> pd.DataFrame:
        """Clean descriptor matrix by removing problematic features."""
        initial_features = len(df.columns)

        constant_features = df.columns[df.var() <= variance_threshold]
        df = df.drop(columns=constant_features)
        nan_threshold = 0.1
        nan_counts = df.isnull().sum() / len(df)
        high_nan_features = nan_counts[nan_counts > nan_threshold].index
        df = df.drop(columns=high_nan_features)

        df = self._remove_correlated_features(df)

        removed_features = initial_features - len(df.columns)
        if removed_features > 0:
            logger.info(f"Removed {removed_features} problematic features")

        return df

    def _remove_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column for column in upper.columns if any(upper[column] > threshold)
            ]

            if to_drop:
                logger.info(f"Removing {len(to_drop)} highly correlated features")
                df = df.drop(columns=to_drop)

            return df
        except Exception as e:
            logger.warning(f"Could not remove correlated features: {e}")
            return df


class ModelTrainer:
    def __init__(self, use_realistic_data: bool = True):
        """Initialize trainer."""
        self.use_realistic_data = use_realistic_data
        self.descriptor_calculator = DescriptorCalculator()
        self.data_generator = ProductionDataGenerator()

        logger.info("Model Trainer initialized")

    def train_model(self, config: ModelConfig) -> TrainingResult:
        """
        Train model with validation and monitoring.
        """
        start_time = time.time()
        logger.info(
            f"Training {config.endpoint.value} model with {config.model_type.value}"
        )

        if self.use_realistic_data:
            data = self._get_realistic_training_data(config.endpoint)
        else:
            data = self._generate_synthetic_data(config.endpoint, 1000)

        logger.info(f"Training data: {len(data)} compounds")

        descriptor_df, valid_indices, descriptor_names = (
            self.descriptor_calculator.calculate_descriptors(data["smiles"].tolist())
        )

        valid_data = data.iloc[valid_indices].reset_index(drop=True)

        X = descriptor_df.values
        y = valid_data[config.endpoint.value].values

        logger.info(f"Final dataset: {len(X)} compounds, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)} (negative/positive)")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y if config.stratify else None,
            shuffle=config.shuffle,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=config.validation_size / (1 - config.test_size),
            random_state=config.random_state,
            stratify=y_temp if config.stratify else None,
            shuffle=config.shuffle,
        )

        logger.info(
            f"Data splits - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}"
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model = self._create_model(config)

        logger.info("Training model xoxo")
        model.fit(X_train_scaled, y_train)

        cv_scores = []
        if config.cv_folds > 1:
            logger.info("Performing cross-validation...")
            cv = StratifiedKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
            )
            logger.info(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        y_train_pred = model.predict(X_train_scaled)
        y_train_proba = (
            model.predict_proba(X_train_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else y_train_pred
        )

        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = (
            model.predict_proba(X_val_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else y_val_pred
        )

        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = (
            model.predict_proba(X_test_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else y_test_pred
        )

        metrics = self._calculate_comprehensive_metrics(
            y_test, y_test_pred, y_test_proba, cv_scores
        )

        feature_importance = self._get_feature_importance(model, descriptor_names)

        self._log_model_performance(
            config.endpoint.value,
            metrics,
            y_train,
            y_train_pred,
            y_val,
            y_val_pred,
            y_test,
            y_test_pred,
        )

        training_time = time.time() - start_time

        return TrainingResult(
            model=model,
            scaler=scaler,
            descriptor_names=descriptor_names,
            feature_names=descriptor_names,
            metrics=metrics,
            config=config,
            training_time=training_time,
            cv_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
        )

    def _get_realistic_training_data(self, endpoint: ToxicityEndpoint) -> pd.DataFrame:
        """Get realistic training data for the specified endpoint."""
        if endpoint == ToxicityEndpoint.AMES_MUTAGENICITY:
            compounds = self.data_generator.get_realistic_ames_data()
            label_col = "ames_mutagenicity"
        elif endpoint == ToxicityEndpoint.CARCINOGENICITY:
            compounds = self.data_generator.get_realistic_carcinogenicity_data()
            label_col = "carcinogenicity"
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")

        # Convert to DataFrame
        df_data = {
            "compound_id": [f"{endpoint.value}_{i:04d}" for i in range(len(compounds))],
            "name": [c["name"] for c in compounds],
            "smiles": [c["smiles"] for c in compounds],
            label_col: [c["label"] for c in compounds],
            "confidence": [c["confidence"] for c in compounds],
            "references": ["; ".join(c["references"]) for c in compounds],
            "study_type": [c["study_type"] for c in compounds],
        }

        df = pd.DataFrame(df_data)

        if len(df) < 50:
            df = self._augment_training_data(df, label_col, target_size=100)

        return df

    def _augment_training_data(
        self, df: pd.DataFrame, label_col: str, target_size: int
    ) -> pd.DataFrame:
        """Augment training data with additional realistic compounds."""
        logger.info(
            f"Augmenting training data from {len(df)} to ~{target_size} compounds"
        )

        # Simple augmentation by adding some common compounds
        additional_compounds = [
            # More non-toxic compounds
            {"name": "Water", "smiles": "O", "label": 0, "confidence": "high"},
            {"name": "Methane", "smiles": "C", "label": 0, "confidence": "high"},
            {"name": "Ethane", "smiles": "CC", "label": 0, "confidence": "high"},
            {"name": "Propane", "smiles": "CCC", "label": 0, "confidence": "high"},
            {"name": "Butane", "smiles": "CCCC", "label": 0, "confidence": "medium"},
            {"name": "Methanol", "smiles": "CO", "label": 0, "confidence": "medium"},
            {
                "name": "Ethylene glycol",
                "smiles": "OCCO",
                "label": 0,
                "confidence": "medium",
            },
            {
                "name": "Glycerol",
                "smiles": "OCC(O)CO",
                "label": 0,
                "confidence": "high",
            },
            # More toxic compounds ( these are approximations)
            {"name": "Formamide", "smiles": "NC=O", "label": 1, "confidence": "low"},
            {"name": "Hydrazine", "smiles": "NN", "label": 1, "confidence": "medium"},
            {
                "name": "Acrylonitrile",
                "smiles": "C=CC#N",
                "label": 1,
                "confidence": "medium",
            },
        ]

        current_size = len(df)
        compounds_to_add = min(len(additional_compounds), target_size - current_size)

        if compounds_to_add > 0:
            for i, compound in enumerate(additional_compounds[:compounds_to_add]):
                new_row = {
                    "compound_id": f"aug_{i:04d}",
                    "name": compound["name"],
                    "smiles": compound["smiles"],
                    label_col: compound["label"],
                    "confidence": compound["confidence"],
                    "references": "Augmentation data",
                    "study_type": "Augmented",
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return df

    def _generate_synthetic_data(
        self, endpoint: ToxicityEndpoint, n_samples: int
    ) -> pd.DataFrame:
        """Generate synthetic training data (fallback method)."""
        logger.warning("Using synthetic data - not recommended for production!")

        np.random.seed(42)
        synthetic_smiles = [
            "CCO",
            "CC(=O)O",
            "c1ccccc1",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ] * (n_samples // 4 + 1)
        synthetic_labels = [0, 0, 1, 0] * (n_samples // 4 + 1)

        synthetic_smiles = synthetic_smiles[:n_samples]
        synthetic_labels = synthetic_labels[:n_samples]

        df_data = {
            "compound_id": [f"syn_{i:04d}" for i in range(len(synthetic_smiles))],
            "name": [f"Synthetic_compound_{i}" for i in range(len(synthetic_smiles))],
            "smiles": synthetic_smiles,
            endpoint.value: synthetic_labels,
        }

        return pd.DataFrame(df_data)

    def _create_model(self, config: ModelConfig) -> BaseEstimator:
        """Create ML model with optimized parameters."""
        default_params = {
            ModelType.RANDOM_FOREST: {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": config.random_state,
                "class_weight": config.class_weight,
                "n_jobs": -1,
                "oob_score": True,
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": config.random_state,
            },
            ModelType.EXTRA_TREES: {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": config.random_state,
                "class_weight": config.class_weight,
                "n_jobs": -1,
                "bootstrap": True,
            },
        }

        params = default_params[config.model_type].copy()
        params.update(config.model_params)

        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(**params)
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(**params)
        elif config.model_type == ModelType.EXTRA_TREES:
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        cv_scores: List[float],
    ) -> ModelMetrics:
        """Calculate comprehensive performance metrics."""
        from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        cv_mean = np.mean(cv_scores) if len(cv_scores) > 0 else 0.0
        cv_std = np.std(cv_scores) if len(cv_scores) > 0 else 0.0

        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_proba),
            specificity=specificity,
            balanced_accuracy=balanced_acc,
            matthews_corrcoef=mcc,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )

    def _get_feature_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                # Sort by importance
                return dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None

    def _log_model_performance(
        self,
        endpoint: str,
        metrics: ModelMetrics,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> None:
        """Log comprehensive model performance."""
        logger.info(f"\n{'='*60}")
        logger.info(f"{endpoint.upper()} MODEL PERFORMANCE")
        logger.info(f"{'='*60}")

        logger.info("TEST SET METRICS:")
        logger.info(f"  Accuracy:         {metrics.accuracy:.3f}")
        logger.info(f"  Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
        logger.info(f"  Precision:        {metrics.precision:.3f}")
        logger.info(f"  Recall:           {metrics.recall:.3f}")
        logger.info(f"  Specificity:      {metrics.specificity:.3f}")
        logger.info(f"  F1-Score:         {metrics.f1_score:.3f}")
        logger.info(f"  AUC-ROC:          {metrics.roc_auc:.3f}")
        logger.info(f"  Matthews Corr:    {metrics.matthews_corrcoef:.3f}")

        if metrics.cv_mean > 0:
            logger.info(
                f"  CV AUC:           {metrics.cv_mean:.3f} ± {metrics.cv_std:.3f}"
            )

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        logger.info(f"\nTRAINING METRICS:")
        logger.info(f"  Train Accuracy:   {train_acc:.3f}")
        logger.info(f"  Val Accuracy:     {val_acc:.3f}")
        logger.info(f"  Test Accuracy:    {metrics.accuracy:.3f}")

        if train_acc - metrics.accuracy > 0.1:
            logger.warning("Potential overfitting detected!")

        logger.info(f"{'='*60}\n")

def train_and_save_models():
    """Train models and save them to pickle files."""
    logger.info(" Training and saving models...")
    
    trainer = ModelTrainer(use_realistic_data=True)
    
    models_dir = Path("app/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [
        ModelConfig(
            endpoint=ToxicityEndpoint.AMES_MUTAGENICITY,
            model_type=ModelType.RANDOM_FOREST,
        ),
        ModelConfig(
            endpoint=ToxicityEndpoint.CARCINOGENICITY,
            model_type=ModelType.GRADIENT_BOOSTING,
        ),
    ]
    
    metadata = {}
    
    for config in configs:
        try:
            logger.info(f"Training {config.endpoint.value} model...")
            
            # Train the model
            result = trainer.train_model(config)
            
            # CRITICAL: Verify model is trained
            if not hasattr(result.model, 'classes_'):
                logger.error(f" Model not properly trained: {config.endpoint.value}")
                continue
            
            # Save model
            endpoint = config.endpoint.value
            model_path = models_dir / f"{endpoint}.pkl"
            
            with open(model_path, "wb") as f:
                pickle.dump(result.model, f, protocol=4)
            
            # Verify model file
            model_size = model_path.stat().st_size
            if model_size == 0:
                logger.error(f" Model file is empty: {model_path}")
                continue
            
            logger.info(f" Saved model: {model_path} ({model_size} bytes)")
            
            # Save scaler
            scaler_path = models_dir / f"{endpoint}_scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(result.scaler, f, protocol=4)
            
            scaler_size = scaler_path.stat().st_size
            logger.info(f"Saved scaler: {scaler_path} ({scaler_size} bytes)")
            
            # Save metadata
            metadata[endpoint] = {
                "model_type": config.model_type.value,
                "n_features": len(result.descriptor_names),
                "feature_names": result.descriptor_names,
                "training_date": datetime.now().isoformat(),
                "version": "1.0",
                "metrics": {
                    "accuracy": result.metrics.accuracy,
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1_score": result.metrics.f1_score,
                    "roc_auc": result.metrics.roc_auc,
                },
            }
            
            logger.info(f" {endpoint} model training completed")
            
        except Exception as e:
            logger.error(f" Failed to train {config.endpoint.value}: {e}")
            continue
    
    # Save metadata
    if metadata:
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    
    logger.info("Model training and saving complete!")


if __name__ == "__main__":
    train_and_save_models()