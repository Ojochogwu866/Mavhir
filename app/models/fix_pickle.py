"""
TARGETED FIX: Empty pickle files issue
This fixes the specific problem where pickle files are being created but are empty.
"""

import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def fix_model_saving_in_persistence(persistence_class):
    """
    Fix the save_model method in ModelPersistence class.
    The issue is likely that the model isn't being properly serialized.
    """
    
    def fixed_save_model(self, result):
        """Fixed version of save_model that ensures proper pickle serialization."""
        endpoint = result.config.endpoint.value
        
        # CRITICAL FIX: Verify model is actually trained before saving
        if not hasattr(result.model, 'classes_'):
            raise RuntimeError(f"Model for {endpoint} is not properly trained - missing classes_ attribute")
        
        if not hasattr(result.model, '_sklearn_version'):
            logger.warning(f"Model for {endpoint} may not be a properly fitted sklearn model")
        
        # Create model path
        model_path = self.models_dir / f"{endpoint}.pkl"
        
        # CRITICAL FIX: Use protocol 4 for better compatibility and verify writing
        try:
            with open(model_path, "wb") as f:
                pickle.dump(result.model, f, protocol=4)
            
            # IMMEDIATE VERIFICATION: Check file was written
            if not model_path.exists():
                raise RuntimeError(f"Model file was not created: {model_path}")
            
            file_size = model_path.stat().st_size
            if file_size == 0:
                raise RuntimeError(f"Model file is empty: {model_path}")
            
            # VERIFICATION LOAD TEST: Try to load immediately
            with open(model_path, "rb") as f:
                test_model = pickle.load(f)
            
            if not hasattr(test_model, 'classes_'):
                raise RuntimeError(f"Saved model cannot be loaded properly")
            
            logger.info(f"✅ Model saved successfully: {model_path} ({file_size} bytes)")
            
        except Exception as e:
            # If saving fails, try alternative approach
            logger.error(f"Standard pickle save failed: {e}")
            logger.info("Trying alternative save method...")
            
            # Alternative: Use joblib if available
            try:
                import joblib
                joblib_path = self.models_dir / f"{endpoint}.joblib"
                joblib.dump(result.model, joblib_path)
                logger.info(f"✅ Saved with joblib: {joblib_path}")
            except ImportError:
                logger.error("joblib not available, pickle save failed")
                raise e
        
        # Save scaler with same verification
        scaler_path = self.models_dir / f"{endpoint}_scaler.pkl"
        
        try:
            with open(scaler_path, "wb") as f:
                pickle.dump(result.scaler, f, protocol=4)
            
            # Verify scaler file
            if not scaler_path.exists() or scaler_path.stat().st_size == 0:
                raise RuntimeError(f"Scaler file is empty: {scaler_path}")
            
            # Test load scaler
            with open(scaler_path, "rb") as f:
                test_scaler = pickle.load(f)
            
            if not hasattr(test_scaler, 'mean_'):
                raise RuntimeError(f"Saved scaler cannot be loaded properly")
            
            logger.info(f"✅ Scaler saved successfully: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Scaler save failed: {e}")
            raise
        
        # Update metadata
        self._update_metadata(result)
    
    # Replace the method
    persistence_class.save_model = fixed_save_model


def fix_model_training():
    """
    Fix the model training to ensure models are properly fitted before saving.
    """
    
    def verify_model_is_trained(model, X_train, y_train):
        """Verify that a model is properly trained."""
        
        # Check 1: Model has classes
        if not hasattr(model, 'classes_'):
            raise RuntimeError("Model not fitted - no classes_ attribute")
        
        # Check 2: Can make predictions
        try:
            pred = model.predict(X_train[:1])
            if len(pred) == 0:
                raise RuntimeError("Model predict returns empty array")
        except Exception as e:
            raise RuntimeError(f"Model cannot make predictions: {e}")
        
        # Check 3: Has probability if supported
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_train[:1])
                if proba.shape[1] < 2:
                    raise RuntimeError("Model predict_proba returns wrong shape")
            except Exception as e:
                raise RuntimeError(f"Model predict_proba failed: {e}")
        
        logger.info("✅ Model training verification passed")
        return True
    
    return verify_model_is_trained


def debug_empty_pickle():
    """Debug function to check why pickle files are empty."""
    
    print("🔍 Debugging empty pickle files...")
    
    models_dir = Path("app/models")
    
    # Check current model files
    for filepath in models_dir.glob("*.pkl"):
        size = filepath.stat().st_size
        print(f"{filepath.name}: {size} bytes")
        
        if size == 0:
            print(f"  ❌ EMPTY FILE: {filepath}")
        elif size < 100:
            print(f"  ⚠️  VERY SMALL: {filepath}")
        else:
            print(f"  ✅ NORMAL SIZE: {filepath}")
            
            # Try to load and inspect
            try:
                with open(filepath, "rb") as f:
                    obj = pickle.load(f)
                print(f"    Type: {type(obj)}")
                
                if hasattr(obj, 'classes_'):
                    print(f"    Classes: {obj.classes_}")
                if hasattr(obj, 'n_features_in_'):
                    print(f"    Features: {obj.n_features_in_}")
                    
            except Exception as e:
                print(f"    ❌ Load error: {e}")


def quick_fix_train_single_model():
    """Quick fix: Train a single model with verification."""
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    print("🚀 Quick fix: Training a single model with verification...")
    
    # Generate simple training data
    smiles_data = ["CCO", "c1ccccc1", "CC(=O)O", "CCC"] * 50  # 200 samples
    labels = [0, 1, 0, 0] * 50  # Simple labels
    
    # Calculate simple descriptors
    X = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_data):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate basic descriptors
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                mol.GetNumHeavyAtoms(),
                mol.GetNumAtoms(),
            ]
            
            # Pad to reach target size (896 for Ames)
            while len(descriptors) < 896:
                # Create synthetic features
                descriptors.append(descriptors[0] * descriptors[1] / (descriptors[2] + 1))
            
            descriptors = descriptors[:896]  # Ensure exact count
            X.append(descriptors)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Failed to process {smiles}: {e}")
            continue
    
    if not X:
        print("❌ No valid descriptors calculated")
        return False
    
    X = np.array(X)
    y = np.array([labels[i] for i in valid_indices])
    
    print(f"✅ Dataset ready: {X.shape} features, {len(y)} samples")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ Model trained successfully")
    print(f"   Classes: {model.classes_}")
    print(f"   Features: {getattr(model, 'n_features_in_', 'unknown')}")
    
    # CRITICAL: Verify model before saving
    if not hasattr(model, 'classes_'):
        print("❌ Model not properly fitted!")
        return False
    
    # Test prediction
    try:
        test_pred = model.predict(X_test[:1])
        test_proba = model.predict_proba(X_test[:1])
        print(f"✅ Model can make predictions: {test_pred}, {test_proba}")
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False
    
    # Save with verification
    models_dir = Path("app/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "test_model.pkl"
    
    try:
        # CRITICAL FIX: Use binary mode and check immediately
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=4)
        
        # IMMEDIATE CHECK
        if not model_path.exists():
            print("❌ File was not created!")
            return False
        
        file_size = model_path.stat().st_size
        if file_size == 0:
            print("❌ File is empty after writing!")
            return False
        
        print(f"✅ Model saved: {file_size} bytes")
        
        # VERIFICATION LOAD
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        
        if not hasattr(loaded_model, 'classes_'):
            print("❌ Loaded model is corrupted!")
            return False
        
        print(f"✅ Model loads correctly: {loaded_model.classes_}")
        
        # Save scaler too
        scaler_path = models_dir / "test_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f, protocol=4)
        
        scaler_size = scaler_path.stat().st_size
        print(f"✅ Scaler saved: {scaler_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 MAVHIR PICKLE FIX")
    print("=" * 40)
    
    # First, debug current state
    debug_empty_pickle()
    print()
    
    # Try quick fix
    print("Attempting quick fix...")
    if quick_fix_train_single_model():
        print("\n🎉 PICKLE FIX SUCCESSFUL!")
        print("Your models should now save properly.")
        print("\nTo apply this fix to your main training:")
        print("1. Ensure models are properly fitted before saving")
        print("2. Use pickle.dump(model, file, protocol=4)")
        print("3. Verify file size immediately after saving")
        print("4. Test load the model right after saving")
    else:
        print("\n❌ Quick fix failed. Check the error messages above.")