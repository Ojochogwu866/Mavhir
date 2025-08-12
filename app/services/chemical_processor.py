"""
Chemical structure processing service using RDKit.

RDKit is the industry standard for cheminformatics in Python.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

from ..core.config import get_settings
from ..core.exceptions import (
    ChemicalProcessingError,
    InvalidSMILESError,
    MoleculeStandardizationError,
)

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


@dataclass
class MolecularProperties:
    """
    Basic molecular properties calculated from structure.
    """

    molecular_weight: float
    logp: float  # Lipophilicity
    tpsa: float  # Topological Polar Surface Area
    num_heavy_atoms: int
    num_aromatic_rings: int
    num_rotatable_bonds: int
    num_hbd: int  # Hydrogen bond donors
    num_hba: int  # Hydrogen bond acceptors


@dataclass
class ProcessedMolecule:
    """
    Result of chemical structure processing.
    """

    original_smiles: str
    canonical_smiles: str
    molecular_formula: str
    properties: MolecularProperties
    is_valid: bool
    errors: List[str]
    rdkit_mol: Optional[Chem.Mol] = None


class MoleculeStandardizer:
    """
    Handles molecular structure standardization using RDKit.
    """

    def __init__(self):
        """Initialize standardization components."""
        try:
            self.normalizer = rdMolStandardize.Normalizer()
            self.largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
            self.uncharger = rdMolStandardize.Uncharger()
            self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

            logger.debug("MoleculeStandardizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MoleculeStandardizer: {e}")
            raise ChemicalProcessingError(f"Standardizer initialization failed: {e}")

    def standardize(self, mol: Chem.Mol, smiles: str) -> Chem.Mol:
        """
        Apply full standardization pipeline.

        PARAMETERS:
            mol: RDKit molecule object
            smiles: Original SMILES (for error reporting)

        RETURNS:
            Standardized RDKit molecule

        RAISES:
            MoleculeStandardizationError: If any step fails
        """
        if mol is None:
            raise MoleculeStandardizationError(
                smiles, "parse", "Input molecule is None"
            )

        try:
            mol = self.normalizer.normalize(mol)
            if mol is None:
                raise MoleculeStandardizationError(
                    smiles, "normalize", "Normalization returned None"
                )

            mol = self.largest_fragment_chooser.choose(mol)
            if mol is None:
                raise MoleculeStandardizationError(
                    smiles, "fragment_selection", "Fragment selection returned None"
                )

            mol = self.uncharger.uncharge(mol)
            if mol is None:
                raise MoleculeStandardizationError(
                    smiles, "neutralize", "Charge neutralization returned None"
                )

            mol = self.tautomer_enumerator.Canonicalize(mol)
            if mol is None:
                raise MoleculeStandardizationError(
                    smiles,
                    "tautomer_canonicalization",
                    "Tautomer canonicalization returned None",
                )

            return mol

        except MoleculeStandardizationError:

            raise
        except Exception as e:
            # Catch any other RDKit errors
            raise MoleculeStandardizationError(
                smiles, "unknown", f"Unexpected error: {e}"
            )

    def quick_standardize(self, mol: Chem.Mol, smiles: str) -> Chem.Mol:
        """
        Fast standardization without tautomer canonicalization.
        """
        if mol is None:
            raise MoleculeStandardizationError(
                smiles, "parse", "Input molecule is None"
            )

        try:
            mol = self.normalizer.normalize(mol)
            mol = self.largest_fragment_chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)

            if mol is None:
                raise MoleculeStandardizationError(
                    smiles, "quick_standardize", "Quick standardization failed"
                )

            return mol

        except Exception as e:
            raise MoleculeStandardizationError(
                smiles, "quick_standardize", f"Quick standardization error: {e}"
            )


class PropertyCalculator:
    """
    Calculates molecular properties essential for toxicity prediction.
    """

    @staticmethod
    def calculate_properties(mol: Chem.Mol) -> MolecularProperties:
        """
        Calculate core molecular properties.
        """
        try:
            molecular_weight = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))

            num_heavy_atoms = int(mol.GetNumHeavyAtoms())
            num_aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            num_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

            num_hbd = int(rdMolDescriptors.CalcNumHBD(mol))
            num_hba = int(rdMolDescriptors.CalcNumHBA(mol))

            return MolecularProperties(
                molecular_weight=molecular_weight,
                logp=logp,
                tpsa=tpsa,
                num_heavy_atoms=num_heavy_atoms,
                num_aromatic_rings=num_aromatic_rings,
                num_rotatable_bonds=num_rotatable_bonds,
                num_hbd=num_hbd,
                num_hba=num_hba,
            )

        except Exception as e:
            raise ChemicalProcessingError(f"Property calculation failed: {e}")

    @staticmethod
    def assess_drug_likeness(properties: MolecularProperties) -> Dict[str, Any]:
        """
        Assess drug-likeness using established pharmaceutical rules.
        """

        lipinski_violations = []

        if properties.molecular_weight > 500:
            lipinski_violations.append(f"MW > 500 ({properties.molecular_weight:.1f})")

        if properties.logp > 5:
            lipinski_violations.append(f"LogP > 5 ({properties.logp:.2f})")

        if properties.num_hbd > 5:
            lipinski_violations.append(f"HBD > 5 ({properties.num_hbd})")

        if properties.num_hba > 10:
            lipinski_violations.append(f"HBA > 10 ({properties.num_hba})")

        lipinski_passed = len(lipinski_violations) <= 1

        veber_violations = []

        if properties.num_rotatable_bonds > 10:
            veber_violations.append(
                f"Rotatable bonds > 10 ({properties.num_rotatable_bonds})"
            )

        if properties.tpsa > 140:
            veber_violations.append(f"TPSA > 140 ({properties.tpsa:.1f})")

        veber_passed = len(veber_violations) == 0

        return {
            "lipinski": {
                "violations": len(lipinski_violations),
                "passed": lipinski_passed,
                "details": lipinski_violations,
            },
            "veber": {
                "violations": len(veber_violations),
                "passed": veber_passed,
                "details": veber_violations,
            },
            "overall_drug_like": lipinski_passed and veber_passed,
        }


class ChemicalProcessor:
    """
    Main service for chemical structure processing.
    """

    def __init__(self):
        """Initialize chemical processor with all components."""
        self.settings = get_settings()

        self.standardizer = MoleculeStandardizer()
        self.property_calculator = PropertyCalculator()

        self.enable_standardization = self.settings.enable_molecule_standardization
        self.timeout_seconds = self.settings.standardization_timeout

        logger.info(
            f"ChemicalProcessor initialized (standardization: {self.enable_standardization})"
        )

    def process_smiles(
        self, smiles: str, quick_mode: bool = False
    ) -> ProcessedMolecule:
        """
        Process a single SMILES string through the full pipeline.
        """

        smiles = smiles.strip()

        if not smiles:
            return self._create_invalid_molecule(smiles, ["Empty SMILES string"])

        try:
            # Step 1: Parse SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_invalid_molecule(smiles, ["Invalid SMILES format"])

            # Step 2: Standardize molecule (if enabled)
            if self.enable_standardization:
                try:
                    if quick_mode:
                        mol = self.standardizer.quick_standardize(mol, smiles)
                    else:
                        mol = self.standardizer.standardize(mol, smiles)
                except MoleculeStandardizationError as e:
                    return self._create_invalid_molecule(smiles, [e.message])

            # Step 3: Generate canonical SMILES
            try:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                return self._create_invalid_molecule(
                    smiles, [f"Failed to generate canonical SMILES: {e}"]
                )

            # Step 4: Calculate molecular formula
            try:
                molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            except Exception as e:
                molecular_formula = "Unknown"
                logger.warning(
                    f"Molecular formula calculation failed for {smiles}: {e}"
                )

            # Step 5: Calculate properties
            try:
                properties = self.property_calculator.calculate_properties(mol)
            except Exception as e:
                return self._create_invalid_molecule(
                    smiles, [f"Property calculation failed: {e}"]
                )

            # Step 6: Create successful result
            return ProcessedMolecule(
                original_smiles=smiles,
                canonical_smiles=canonical_smiles,
                molecular_formula=molecular_formula,
                properties=properties,
                is_valid=True,
                errors=[],
                rdkit_mol=mol,
            )

        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error processing SMILES '{smiles}': {e}")
            return self._create_invalid_molecule(
                smiles, [f"Unexpected processing error: {e}"]
            )

    def process_smiles_batch(
        self, smiles_list: List[str], quick_mode: bool = False
    ) -> List[ProcessedMolecule]:
        """
        Process a batch of SMILES strings.
        """
        if not smiles_list:
            return []

        logger.info(
            f"Processing batch of {len(smiles_list)} SMILES (quick_mode: {quick_mode})"
        )

        results = []
        successful = 0
        failed = 0

        for i, smiles in enumerate(smiles_list):
            try:
                result = self.process_smiles(smiles, quick_mode=quick_mode)
                results.append(result)

                if result.is_valid:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Unexpected error in batch processing at index {i}: {e}")
                results.append(
                    self._create_invalid_molecule(
                        smiles, [f"Batch processing error: {e}"]
                    )
                )
                failed += 1

        success_rate = (successful / len(smiles_list)) * 100
        logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed ({success_rate:.1f}% success rate)"
        )

        return results

    @lru_cache(maxsize=1000)
    def get_canonical_smiles(self, smiles: str) -> str:
        """
        Get canonical SMILES with caching.
        """
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                raise InvalidSMILESError(smiles, "Cannot parse SMILES")

            return Chem.MolToSmiles(mol, canonical=True)

        except Exception as e:
            raise InvalidSMILESError(smiles, str(e))

    def _create_invalid_molecule(
        self, smiles: str, errors: List[str]
    ) -> ProcessedMolecule:
        """Create ProcessedMolecule for invalid input."""

        empty_properties = MolecularProperties(
            molecular_weight=0.0,
            logp=0.0,
            tpsa=0.0,
            num_heavy_atoms=0,
            num_aromatic_rings=0,
            num_rotatable_bonds=0,
            num_hbd=0,
            num_hba=0,
        )

        return ProcessedMolecule(
            original_smiles=smiles,
            canonical_smiles="",
            molecular_formula="",
            properties=empty_properties,
            is_valid=False,
            errors=errors,
            rdkit_mol=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring."""
        cache_info = self.get_canonical_smiles.cache_info()

        return {
            "standardization_enabled": self.enable_standardization,
            "timeout_seconds": self.timeout_seconds,
            "canonical_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "current_size": cache_info.currsize,
                "max_size": cache_info.maxsize,
            },
        }


def create_chemical_processor() -> ChemicalProcessor:
    """
    Factory function to create ChemicalProcessor.

    Useful for dependency injection and testing.
    """
    return ChemicalProcessor()


def validate_smiles_simple(smiles: str) -> bool:
    """
    Simple SMILES validation without full processing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except:
        return False


def _test_chemical_processor():
    """Test function for development and debugging."""

    processor = create_chemical_processor()

    test_cases = [
        ("CCO", "Valid simple molecule - ethanol"),
        ("CC(=O)O", "Valid carboxylic acid - acetic acid"),
        ("c1ccccc1", "Valid aromatic - benzene"),
        ("[Na+].[Cl-]", "Salt - should be desalted to Cl"),
        ("CC(=O)[O-]", "Charged molecule - should be neutralized"),
        ("invalid_smiles", "Invalid SMILES"),
        ("", "Empty string"),
        ("C" * 1000, "Very long SMILES"),
    ]

    print("Testing ChemicalProcessor:")
    print("=" * 60)

    for smiles, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")

        try:
            result = processor.process_smiles(smiles)

            if result.is_valid:
                print(f"   SUCCESS")
                print(f"   Canonical: {result.canonical_smiles}")
                print(f"   Formula: {result.molecular_formula}")
                print(f"   MW: {result.properties.molecular_weight:.2f}")
                print(f"   LogP: {result.properties.logp:.2f}")

                # Test drug-likeness
                drug_assessment = processor.property_calculator.assess_drug_likeness(
                    result.properties
                )
                print(f"   Drug-like: {drug_assessment['overall_drug_like']}")

            else:
                print(f"FAILED")
                print(f"   Errors: {', '.join(result.errors)}")

        except Exception as e:
            print(f"EXCEPTION: {e}")

    # Test batch processing
    print(f"\n{'='*60}")
    print("Testing batch processing:")

    batch_smiles = ["CCO", "CC(=O)O", "invalid", "c1ccccc1"]
    results = processor.process_smiles_batch(batch_smiles)

    print(f"Batch size: {len(batch_smiles)}")
    print(f"Results: {len(results)}")

    for i, result in enumerate(results):
        status = "" if result.is_valid else "❌"
        print(f"  {i}: {status} {result.original_smiles} → {result.canonical_smiles}")

    # Show statistics
    print(f"\n{'='*60}")
    print("Processor statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _test_chemical_processor()
