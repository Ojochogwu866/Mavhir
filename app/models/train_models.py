
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from rdkit import Chem
from mordred import Calculator, descriptors

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_ames_data(n_samples=1000):
    """
    Generate sample Ames mutagenicity data for demonstration.
    """
    
    sample_smiles = [
        "CCO",                   # Ethanol - non-mutagenic
        "CC(=O)O",               # Acetic acid - non-mutagenic  
        "c1ccccc1",              # Benzene - mutagenic
        "CCN(CC)CC",             # Triethylamine - non-mutagenic
        "c1ccc2[nH]c3ccccc3c2c1", # Carbazole - mutagenic
        "CCCCO",                 # Butanol - non-mutagenic
        "c1ccc(cc1)N",           # Aniline - mutagenic
        "CC(C)O",                # Isopropanol - non-mutagenic
    ]
    
    smiles_data = []
    labels = []
    
    for _ in range(n_samples):
        base_smiles = np.random.choice(sample_smiles)
        smiles_data.append(base_smiles)
        
        if base_smiles in ["c1ccccc1", "c1ccc2[nH]c3ccccc3c2c1", "c1ccc(cc1)N"]:
            labels.append(1)  # Mutagenic
        else:
            labels.append(0)  # Non-mutagenic
    
    return pd.DataFrame({
        'smiles': smiles_data,
        'ames_mutagenic': labels
    })


def generate_sample_carcinogenicity_data(n_samples=800):
    """Generate sample carcinogenicity data."""
    
    sample_smiles = [
        "CCO",                    # Ethanol - non-carcinogenic
        "CC(=O)O",               # Acetic acid - non-carcinogenic
        "c1ccccc1",              # Benzene - carcinogenic
        "CCN(CC)CC",             # Triethylamine - non-carcinogenic
        "c1ccc(cc1)N",           # Aniline - carcinogenic
        "CCCCO",                 # Butanol - non-carcinogenic
        "CC(C)O",                # Isopropanol - non-carcinogenic
        "c1ccc(cc1)C(=O)O",      # Benzoic acid - non-carcinogenic
    ]
    
    smiles_data = []
    labels = []
    
    for _ in range(n_samples):
        base_smiles = np.random.choice(sample_smiles)
        smiles_data.append(base_smiles)
        
        if base_smiles in ["c1ccccc1", "c1ccc(cc1)N"]:
            labels.append(1)  # Carcinogenic
        else:
            labels.append(0)  # Non-carcinogenic
    
    return pd.DataFrame({
        'smiles': smiles_data,
        'carcinogenic': labels
    })


def calculate_descriptors_for_training(smiles_list):
    """
    Calculate molecular descriptors for training data.
    """
    
    logger.info(f"Calculating descriptors for {len(smiles_list)} compounds...")
    
    calc = Calculator()
    calc.register(descriptors.AtomCount)
    calc.register(descriptors.BondCount)
    calc.register(descriptors.RingCount)
    calc.register(descriptors.TopologicalIndex)
    calc.register(descriptors.ConnectivityIndex)
    calc.register(descriptors.InformationIndex)
    calc.register(descriptors.GeometricIndex)
    calc.register(descriptors.PartialCharge)
    calc.register(descriptors.Polarizability)
    calc.register(descriptors.FragmentComplexity)
    calc.register(descriptors.Framework)
    
    descriptor_names = [str(d) for d in calc.descriptors]
    
    descriptor_matrix = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc_values = calc(mol)
                
                clean_values = []
                for val in desc_values:
                    if val is None or np.isnan(val) or np.isinf(val):
                        clean_values.append(0.0)
                    else:
                        clean_values.append(float(val))
                
                descriptor_matrix.append(clean_values)
                valid_indices.append(i)
            
        except Exception as e:
            logger.warning(f"Failed to calculate descriptors for {smiles}: {e}")
            continue
    
    descriptor_df = pd.DataFrame(descriptor_matrix, columns=descriptor_names)
    
    logger.info(f"Calculated {len(descriptor_df.columns)} descriptors for {len(descriptor_df)} valid compounds")
    
    return descriptor_df, valid_indices, descriptor_names


def train_ames_model():
    """Train Ames mutagenicity prediction model."""
    
    logger.info("Training Ames mutagenicity model...")
    
    data = generate_sample_ames_data(1000)
    
    descriptor_df, valid_indices, descriptor_names = calculate_descriptors_for_training(data['smiles'].tolist())

    valid_data = data.iloc[valid_indices].reset_index(drop=True)
    
    X = descriptor_df.values
    y = valid_data['ames_mutagenic'].values
    
    logger.info(f"Training data: {len(X)} compounds, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)} (negative/positive)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Ames Model Performance:")
    logger.info(f"  Accuracy:  {accuracy:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall:    {recall:.3f}")
    logger.info(f"  AUC-ROC:   {auc:.3f}")
    
    return model, scaler, descriptor_names


def train_carcinogenicity_model():
    """Train carcinogenicity prediction model."""
    
    logger.info("Training carcinogenicity model...")
    
    data = generate_sample_carcinogenicity_data(800)
    
    descriptor_df, valid_indices, descriptor_names = calculate_descriptors_for_training(data['smiles'].tolist())
    
    valid_data = data.iloc[valid_indices].reset_index(drop=True)
    
    X = descriptor_df.values
    y = valid_data['carcinogenic'].values
    
    logger.info(f"Training data: {len(X)} compounds, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)} (negative/positive)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Carcinogenicity Model Performance:")
    logger.info(f"  Accuracy:  {accuracy:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall:    {recall:.3f}")
    logger.info(f"  AUC-ROC:   {auc:.3f}")
    
    return model, scaler, descriptor_names

def save_models():
    """Train and save all models."""
    
    models_dir = Path("app/models")
    models_dir.mkdir(exist_ok=True)
    
    logger.info("="*50)
    ames_model, ames_scaler, ames_descriptors = train_ames_model()

    with open(models_dir / "ames_mutagenicity.pkl", 'wb') as f:
        pickle.dump(ames_model, f)
    
    with open(models_dir / "ames_scaler.pkl", 'wb') as f:
        pickle.dump(ames_scaler, f)
    
    logger.info("Saved Ames mutagenicity model")
    
    logger.info("="*50)
    carc_model, carc_scaler, carc_descriptors = train_carcinogenicity_model()
    
    with open(models_dir / "carcinogenicity.pkl", 'wb') as f:
        pickle.dump(carc_model, f)
    
    with open(models_dir / "carcinogenicity_scaler.pkl", 'wb') as f:
        pickle.dump(carc_scaler, f)
    
    logger.info("Saved carcinogenicity model")
    
    metadata = {
        "ames_mutagenicity": {
            "model_type": "RandomForestClassifier",
            "n_features": len(ames_descriptors),
            "feature_names": ames_descriptors,
            "training_date": "2024-01-01",
            "version": "1.0"
        },
        "carcinogenicity": {
            "model_type": "GradientBoostingClassifier", 
            "n_features": len(carc_descriptors),
            "feature_names": carc_descriptors,
            "training_date": "2024-01-01",
            "version": "1.0"
        }
    }
    
    with open(models_dir / "model_metadata.json", 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info("Saved model metadata")
    logger.info("="*50)
    logger.info("All models trained and saved successfully!")
    logger.info("You can now start the API server.")


if __name__ == "__main__":
    save_models()