"""
Enhanced model training system for toxicity prediction.
Type-safe, modular, and DRY implementation with proper error handling.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import BaseEstimator

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from mordred import Calculator, descriptors

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    RANDOM_FOREST = "RandomForestClassifier"
    GRADIENT_BOOSTING = "GradientBoostingClassifier"


class ToxicityEndpoint(Enum):
    AMES_MUTAGENICITY = "ames_mutagenicity"
    CARCINOGENICITY = "carcinogenicity"


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    f1_score: float

    def __post_init__(self) -> None:
        for metric_name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {metric_name}: {value} (must be 0-1)")


@dataclass
class ModelConfig:
    endpoint: ToxicityEndpoint
    model_type: ModelType
    n_samples: int
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.cv_folds <= 1:
            raise ValueError(f"cv_folds must be > 1, got {self.cv_folds}")


@dataclass
class TrainingResult:
    model: BaseEstimator
    scaler: StandardScaler
    descriptor_names: List[str]
    metrics: ModelMetrics
    config: ModelConfig
    training_time: float
    cv_scores: Optional[List[float]] = None


class DescriptorCalculatorProtocol(Protocol):
    def calculate_descriptors(
        self, smiles_list: List[str]
    ) -> Tuple[pd.DataFrame, List[int], List[str]]: ...


class DataGeneratorProtocol(Protocol):
    def generate_data(self, n_samples: int) -> pd.DataFrame: ...


class SafeMordredCalculator:
    def __init__(self) -> None:
        self.calculator = Calculator()
        self._register_safe_descriptors()

    def _register_safe_descriptors(self) -> None:
        try:
            safe_descriptor_groups = [
                descriptors.AtomCount,
                descriptors.BondCount,
                descriptors.RingCount,
                descriptors.Weight,
                descriptors.SLogP,
                descriptors.Constitutional,
            ]

            optional_descriptors = [
                "ABC",
                "AcidBase",
                "BalabanJ",
                "BertzCT",
                "Chi",
                "FragmentComplexity",
                "Framework",
                "Polarizability",
                "Aromatic",
                "TopoPSA",
                "EState",
                "Autocorrelation",
                "BCUT",
                "BaryszMatrix",
                "DetourMatrix",
                "DistanceMatrix",
            ]

            for desc_group in safe_descriptor_groups:
                try:
                    self.calculator.register(desc_group)
                    logger.debug(f"Registered descriptor group: {desc_group}")
                except Exception as e:
                    logger.warning(f"Failed to register {desc_group}: {e}")

            for desc_name in optional_descriptors:
                try:
                    if hasattr(descriptors, desc_name):
                        desc_class = getattr(descriptors, desc_name)
                        self.calculator.register(desc_class)
                        logger.debug(f"Registered optional descriptor: {desc_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to register optional descriptor {desc_name}: {e}"
                    )

            logger.info(
                f"Initialized calculator with {len(self.calculator.descriptors)} descriptors"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Mordred calculator: {e}")
            raise RuntimeError(f"Cannot initialize descriptor calculator: {e}")

    def calculate_descriptors(
        self, smiles_list: List[str]
    ) -> Tuple[pd.DataFrame, List[int], List[str]]:
        logger.info(f"Calculating descriptors for {len(smiles_list)} compounds...")

        descriptor_names = [str(d) for d in self.calculator.descriptors]
        descriptor_matrix = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(smiles_list)} compounds")

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES at index {i}: {smiles}")
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
                    f"Failed to calculate descriptors for SMILES at index {i} ({smiles}): {e}"
                )
                continue

        if not descriptor_matrix:
            raise RuntimeError("No valid descriptors calculated for any compounds")

        descriptor_df = pd.DataFrame(descriptor_matrix, columns=descriptor_names)
        descriptor_df = self._remove_low_variance_features(descriptor_df)

        logger.info(
            f"Calculated {len(descriptor_df.columns)} descriptors for {len(descriptor_df)} valid compounds"
        )

        return descriptor_df, valid_indices, list(descriptor_df.columns)

    def _remove_low_variance_features(
        self, df: pd.DataFrame, threshold: float = 1e-6
    ) -> pd.DataFrame:
        initial_features = len(df.columns)
        variances = df.var()
        keep_features = variances[variances > threshold].index
        df_filtered = df[keep_features]

        removed_features = initial_features - len(df_filtered.columns)
        if removed_features > 0:
            logger.info(f"Removed {removed_features} low-variance features")

        return df_filtered


class RDKitCalculator:
    def calculate_descriptors(
        self, smiles_list: List[str]
    ) -> Tuple[pd.DataFrame, List[int], List[str]]:
        logger.info(
            f"Calculating RDKit descriptors for {len(smiles_list)} compounds..."
        )

        descriptor_data = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                desc_dict = {
                    "MolWt": Descriptors.MolWt(mol),
                    "LogP": Crippen.MolLogP(mol),
                    "TPSA": rdMolDescriptors.CalcTPSA(mol),
                    "NumHBA": rdMolDescriptors.CalcNumHBA(mol),
                    "NumHBD": rdMolDescriptors.CalcNumHBD(mol),
                    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
                    "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
                    "FractionCsp3": rdMolDescriptors.CalcFractionCsp3(mol),
                    "BalabanJ": rdMolDescriptors.BalabanJ(mol),
                    "BertzCT": rdMolDescriptors.BertzCT(mol),
                    "Chi0n": rdMolDescriptors.Chi0n(mol),
                    "Chi1n": rdMolDescriptors.Chi1n(mol),
                    "HallKierAlpha": rdMolDescriptors.HallKierAlpha(mol),
                    "Kappa1": rdMolDescriptors.Kappa1(mol),
                    "Kappa2": rdMolDescriptors.Kappa2(mol),
                    "LabuteASA": rdMolDescriptors.LabuteASA(mol),
                    "PEOE_VSA1": rdMolDescriptors.PEOE_VSA1(mol),
                    "PEOE_VSA2": rdMolDescriptors.PEOE_VSA2(mol),
                    "SMR_VSA1": rdMolDescriptors.SMR_VSA1(mol),
                    "SMR_VSA2": rdMolDescriptors.SMR_VSA2(mol),
                }

                for key, value in desc_dict.items():
                    if value is None or not np.isfinite(value):
                        desc_dict[key] = 0.0

                descriptor_data.append(desc_dict)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate RDKit descriptors for {smiles}: {e}"
                )
                continue

        if not descriptor_data:
            raise RuntimeError("No valid RDKit descriptors calculated")

        descriptor_df = pd.DataFrame(descriptor_data)
        descriptor_names = list(descriptor_df.columns)

        logger.info(
            f"Calculated {len(descriptor_names)} RDKit descriptors for {len(descriptor_df)} compounds"
        )

        return descriptor_df, valid_indices, descriptor_names


class SampleDataGenerator:
    @staticmethod
    def generate_ames_data(n_samples: int) -> pd.DataFrame:
        mutagenic_smiles = [
            "c1ccccc1",
            "c1ccc(cc1)N",
            "c1ccc2[nH]c3ccccc3c2c1",
            "c1ccc(cc1)[N+](=O)[O-]",
            "Nc1ccc(cc1)C(=O)O",
        ]

        non_mutagenic_smiles = [
            "CCO",
            "CC(=O)O",
            "CCN(CC)CC",
            "CCCCO",
            "CC(C)O",
            "c1ccc(cc1)C(=O)O",
        ]

        return SampleDataGenerator._generate_balanced_data(
            mutagenic_smiles, non_mutagenic_smiles, n_samples, "ames_mutagenicity"
        )

    @staticmethod
    def generate_carcinogenicity_data(n_samples: int) -> pd.DataFrame:
        carcinogenic_smiles = [
            "c1ccccc1",
            "c1ccc(cc1)N",
            "c1ccc(cc1)[N+](=O)[O-]",
            "ClCCl",
        ]

        non_carcinogenic_smiles = [
            "CCO",
            "CC(=O)O",
            "CCN(CC)CC",
            "CCCCO",
            "CC(C)O",
            "c1ccc(cc1)C(=O)O",
        ]

        return SampleDataGenerator._generate_balanced_data(
            carcinogenic_smiles, non_carcinogenic_smiles, n_samples, "carcinogenicity"
        )

    @staticmethod
    def _generate_balanced_data(
        positive_smiles: List[str],
        negative_smiles: List[str],
        n_samples: int,
        label_column: str,
    ) -> pd.DataFrame:
        smiles_data = []
        labels = []

        n_positive = n_samples // 2
        n_negative = n_samples - n_positive

        for _ in range(n_positive):
            smiles = np.random.choice(positive_smiles)
            smiles_data.append(smiles)
            labels.append(1)

        for _ in range(n_negative):
            smiles = np.random.choice(negative_smiles)
            smiles_data.append(smiles)
            labels.append(0)

        combined = list(zip(smiles_data, labels))
        np.random.shuffle(combined)
        smiles_data, labels = zip(*combined)

        return pd.DataFrame({"smiles": smiles_data, label_column: labels})


class ModelTrainer:
    def __init__(
        self,
        descriptor_calculator: Optional[DescriptorCalculatorProtocol] = None,
        use_mordred: bool = True,
    ) -> None:
        if descriptor_calculator:
            self.descriptor_calculator = descriptor_calculator
        else:
            if use_mordred:
                try:
                    self.descriptor_calculator = SafeMordredCalculator()
                    logger.info("Using Mordred descriptor calculator")
                except Exception as e:
                    logger.warning(f"Mordred initialization failed: {e}")
                    logger.info("Falling back to RDKit descriptor calculator")
                    self.descriptor_calculator = RDKitCalculator()
            else:
                self.descriptor_calculator = RDKitCalculator()
                logger.info("Using RDKit descriptor calculator")

    def train_model(self, config: ModelConfig) -> TrainingResult:
        start_time = datetime.now()
        logger.info(
            f"Training {config.endpoint.value} model with {config.model_type.value}"
        )

        data = self._generate_training_data(config)

        descriptor_df, valid_indices, descriptor_names = (
            self.descriptor_calculator.calculate_descriptors(data["smiles"].tolist())
        )

        valid_data = data.iloc[valid_indices].reset_index(drop=True)

        X = descriptor_df.values
        y = valid_data[config.endpoint.value].values

        logger.info(f"Training data: {len(X)} compounds, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)} (negative/positive)")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = self._create_model(config)
        model.fit(X_train_scaled, y_train)

        cv_scores = None
        if config.cv_folds > 1:
            cv = StratifiedKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
            )
            logger.info(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self._log_model_performance(config.endpoint.value, metrics)

        training_time = (datetime.now() - start_time).total_seconds()

        return TrainingResult(
            model=model,
            scaler=scaler,
            descriptor_names=descriptor_names,
            metrics=metrics,
            config=config,
            training_time=training_time,
            cv_scores=cv_scores.tolist() if cv_scores is not None else None,
        )

    def _generate_training_data(self, config: ModelConfig) -> pd.DataFrame:
        if config.endpoint == ToxicityEndpoint.AMES_MUTAGENICITY:
            return SampleDataGenerator.generate_ames_data(config.n_samples)
        elif config.endpoint == ToxicityEndpoint.CARCINOGENICITY:
            return SampleDataGenerator.generate_carcinogenicity_data(config.n_samples)
        else:
            raise ValueError(f"Unsupported endpoint: {config.endpoint}")

    def _create_model(self, config: ModelConfig) -> BaseEstimator:
        default_params = {
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": config.random_state,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": config.random_state,
            },
        }

        params = default_params[config.model_type].copy()
        params.update(config.model_params)

        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(**params)
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> ModelMetrics:
        from sklearn.metrics import f1_score

        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_pred_proba),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
        )

    def _log_model_performance(self, endpoint: str, metrics: ModelMetrics) -> None:
        logger.info(f"{endpoint.title()} Model Performance:")
        logger.info(f"  Accuracy:  {metrics.accuracy:.3f}")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall:    {metrics.recall:.3f}")
        logger.info(f"  F1-Score:  {metrics.f1_score:.3f}")
        logger.info(f"  AUC-ROC:   {metrics.roc_auc:.3f}")


class ModelPersistence:
    def __init__(self, models_dir: Path = Path("app/models")) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, result: TrainingResult) -> None:
        endpoint = result.config.endpoint.value

        model_path = self.models_dir / f"{endpoint}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(result.model, f)
        logger.info(f"Saved model: {model_path}")

        scaler_path = self.models_dir / f"{endpoint}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(result.scaler, f)
        logger.info(f"Saved scaler: {scaler_path}")

        self._update_metadata(result)

    def _update_metadata(self, result: TrainingResult) -> None:
        metadata_path = self.models_dir / "model_metadata.json"

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        endpoint_key = result.config.endpoint.value
        metadata[endpoint_key] = {
            "model_type": result.config.model_type.value,
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
            "training_config": {
                "n_samples": result.config.n_samples,
                "test_size": result.config.test_size,
                "cv_folds": result.config.cv_folds,
                "model_params": result.config.model_params,
            },
            "training_time_seconds": result.training_time,
            "cv_scores": result.cv_scores,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Updated metadata: {metadata_path}")


def train_all_models(use_mordred: bool = True) -> None:
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)

    trainer = ModelTrainer(use_mordred=use_mordred)
    persistence = ModelPersistence()

    configs = [
        ModelConfig(
            endpoint=ToxicityEndpoint.AMES_MUTAGENICITY,
            model_type=ModelType.RANDOM_FOREST,
            n_samples=1000,
            model_params={"n_estimators": 200, "max_depth": 15},
        ),
        ModelConfig(
            endpoint=ToxicityEndpoint.CARCINOGENICITY,
            model_type=ModelType.GRADIENT_BOOSTING,
            n_samples=800,
            model_params={"n_estimators": 150, "learning_rate": 0.05},
        ),
    ]

    results = []

    for config in configs:
        try:
            logger.info(f"Training {config.endpoint.value} model...")
            result = trainer.train_model(config)
            persistence.save_model(result)
            results.append(result)
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to train {config.endpoint.value} model: {e}")
            continue

    if results:
        logger.info(f" Successfully trained {len(results)} models!")
        logger.info("You can now start the API server.")
    else:
        logger.error(" No models were successfully trained!")


if __name__ == "__main__":
    try:
        train_all_models(use_mordred=True)
    except Exception as e:
        logger.warning(f"Training with Mordred failed: {e}")
        logger.info("Retrying with RDKit descriptors only...")
        train_all_models(use_mordred=False)
