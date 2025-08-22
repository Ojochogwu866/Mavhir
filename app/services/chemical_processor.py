import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
import time
import threading

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

from ..core.config import get_settings
from ..core.exceptions import (
    MavhirChemicalProcessingError,
    MavhirInvalidSMILESError,
    MavhirMolecularStandardizationError,
)

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


@dataclass
class MolecularProperties:

    molecular_weight: float
    logp: float  # Lipophilicity
    tpsa: float  # Topological Polar Surface Area
    num_heavy_atoms: int
    num_aromatic_rings: int
    num_rotatable_bonds: int
    num_hbd: int  # Hydrogen bond donors
    num_hba: int  # Hydrogen bond acceptors

    # Additional properties
    num_saturated_rings: int
    num_aliphatic_rings: int
    fraction_csp3: float
    num_radical_electrons: int
    formal_charge: int


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
    warnings: List[str]
    processing_time: float
    rdkit_mol: Optional[Chem.Mol] = None


@dataclass
class ValidationResult:
    """
    Comprehensive SMILES validation result.
    """

    is_valid: bool
    smiles: str
    canonical_smiles: Optional[str]
    errors: List[str]
    warnings: List[str]
    basic_properties: Optional[Dict[str, Any]]


class MoleculeStandardizer:
    """
    Molecular structure standardization using RDKit.
    """

    def __init__(self):
        """Initialize standardization components with error handling."""
        try:
            self.normalizer = rdMolStandardize.Normalizer()
            self.largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
            self.uncharger = rdMolStandardize.Uncharger()
            self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

            self._lock = threading.Lock()

            logger.debug("MoleculeStandardizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MoleculeStandardizer: {e}")
            raise MavhirChemicalProcessingError(
                f"Standardizer initialization failed: {e}"
            )

    def standardize(
        self, mol: Chem.Mol, smiles: str, quick_mode: bool = False
    ) -> Chem.Mol:
        """
        Apply standardization pipeline with thread safety.
        """
        if mol is None:
            raise MavhirMolecularStandardizationError(
                smiles, "parse", "Input molecule is None"
            )

        with self._lock: 
            try:
                mol = self.normalizer.normalize(mol)
                if mol is None:
                    raise MavhirMolecularStandardizationError(
                        smiles, "normalize", "Normalization returned None"
                    )

                mol = self.largest_fragment_chooser.choose(mol)
                if mol is None:
                    raise MavhirMolecularStandardizationError(
                        smiles, "fragment_selection", "Fragment selection returned None"
                    )

                mol = self.uncharger.uncharge(mol)
                if mol is None:
                    raise MavhirMolecularStandardizationError(
                        smiles, "neutralize", "Charge neutralization returned None"
                    )

                if not quick_mode:
                    mol = self.tautomer_enumerator.Canonicalize(mol)
                    if mol is None:
                        raise MavhirMolecularStandardizationError(
                            smiles,
                            "tautomer_canonicalization",
                            "Tautomer canonicalization returned None",
                        )

                return mol

            except MavhirMolecularStandardizationError:
                raise
            except Exception as e:
                raise MavhirMolecularStandardizationError(
                    smiles, "unknown", f"Unexpected standardization error: {e}"
                )


class PropertyCalculator:
    @staticmethod
    def calculate_properties(mol: Chem.Mol) -> MolecularProperties:
        try:
            molecular_weight = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))

            num_heavy_atoms = int(mol.GetNumHeavyAtoms())
            num_hbd = int(rdMolDescriptors.CalcNumHBD(mol))
            num_hba = int(rdMolDescriptors.CalcNumHBA(mol))

            num_aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            num_saturated_rings = int(rdMolDescriptors.CalcNumSaturatedRings(mol))
            num_aliphatic_rings = int(rdMolDescriptors.CalcNumAliphaticRings(mol))

            num_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

            try:
                fraction_csp3 = float(rdMolDescriptors.CalcFractionCsp3(mol))
            except:
                fraction_csp3 = 0.0

            formal_charge = int(Chem.rdmolops.GetFormalCharge(mol))
            num_radical_electrons = sum(
                atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()
            )

            return MolecularProperties(
                molecular_weight=molecular_weight,
                logp=logp,
                tpsa=tpsa,
                num_heavy_atoms=num_heavy_atoms,
                num_aromatic_rings=num_aromatic_rings,
                num_rotatable_bonds=num_rotatable_bonds,
                num_hbd=num_hbd,
                num_hba=num_hba,
                num_saturated_rings=num_saturated_rings,
                num_aliphatic_rings=num_aliphatic_rings,
                fraction_csp3=fraction_csp3,
                num_radical_electrons=num_radical_electrons,
                formal_charge=formal_charge,
            )

        except Exception as e:
            raise MavhirChemicalProcessingError(f"Property calculation failed: {e}")

    @staticmethod
    def assess_drug_likeness(
        self: str, properties: MolecularProperties
    ) -> Dict[str, Any]:
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

        lead_like_violations = []
        if properties.molecular_weight > 350:
            lead_like_violations.append(f"MW > 350 ({properties.molecular_weight:.1f})")
        if properties.logp > 3:
            lead_like_violations.append(f"LogP > 3 ({properties.logp:.2f})")

        lead_like_passed = len(lead_like_violations) == 0

        overall_drug_like = lipinski_passed and veber_passed

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
            "lead_like": {
                "violations": len(lead_like_violations),
                "passed": lead_like_passed,
                "details": lead_like_violations,
            },
            "overall_drug_like": overall_drug_like,
            "recommendation": self._get_drug_likeness_recommendation(
                lipinski_passed, veber_passed, lead_like_passed
            ),
        }

    @staticmethod
    def _get_drug_likeness_recommendation(
        lipinski: bool, veber: bool, lead_like: bool
    ) -> str:
        """Get recommendation based on rule compliance."""
        if lipinski and veber and lead_like:
            return "Excellent drug-like properties"
        elif lipinski and veber:
            return "Good drug-like properties"
        elif lipinski or veber:
            return "Moderate drug-like properties"
        else:
            return "Poor drug-like properties"


class ChemicalProcessor:
    def __init__(self):
        """Initialize chemical processor."""
        self.settings = get_settings()
        self.standardizer = MoleculeStandardizer()
        self.property_calculator = PropertyCalculator()

        self.enable_standardization = self.settings.enable_molecule_standardization
        self.timeout_seconds = self.settings.standardization_timeout
        self.max_smiles_length = getattr(self.settings, "max_smiles_length", 1000)
        self.max_heavy_atoms = getattr(self.settings, "max_heavy_atoms", 200)

        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "standardized": 0,
            "total_processing_time": 0.0,
        }
        self._stats_lock = threading.Lock()

        logger.info(f"ChemicalProcessor initialized")

    def validate_smiles_comprehensive(self, smiles: str) -> ValidationResult:
        start_time = time.time()
        errors = []
        warnings = []

        try:
            smiles = smiles.strip()
            if not smiles:
                errors.append("Empty SMILES string")
                return ValidationResult(False, smiles, None, errors, warnings, None)

            if len(smiles) > self.max_smiles_length:
                errors.append(f"SMILES too long (>{self.max_smiles_length} characters)")
                return ValidationResult(False, smiles, None, errors, warnings, None)

            import re

            pattern = r"^[A-Za-z0-9@+\-\[\]()=#%/\\.\\\\:]+$"
            if not re.match(pattern, smiles):
                errors.append("SMILES contains invalid characters")
                return ValidationResult(False, smiles, None, errors, warnings, None)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors.append("Cannot parse SMILES with RDKit")
                return ValidationResult(False, smiles, None, errors, warnings, None)

            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms == 0:
                errors.append("Molecule has no heavy atoms")
                return ValidationResult(False, smiles, None, errors, warnings, None)

            if num_heavy_atoms > self.max_heavy_atoms:
                warnings.append(f"Large molecule ({num_heavy_atoms} heavy atoms)")

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            basic_properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "num_heavy_atoms": num_heavy_atoms,
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "is_aromatic": any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
            }

            if num_heavy_atoms > 100:
                warnings.append("Very large molecule")
            if basic_properties["molecular_weight"] > 1000:
                warnings.append("High molecular weight")

            return ValidationResult(
                is_valid=True,
                smiles=smiles,
                canonical_smiles=canonical_smiles,
                errors=errors,
                warnings=warnings,
                basic_properties=basic_properties,
            )

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, smiles, None, errors, warnings, None)

        finally:
            processing_time = time.time() - start_time
            logger.debug(f"SMILES validation took {processing_time:.3f}s")

    def process_smiles(
        self, smiles: str, quick_mode: bool = False, validate_only: bool = False
    ) -> ProcessedMolecule:
        """
        SMILES processing with comprehensive validation and monitoring.
        """
        start_time = time.time()

        with self._stats_lock:
            self._stats["total_processed"] += 1

        try:
            validation = self.validate_smiles_comprehensive(smiles)

            if not validation.is_valid:
                processing_time = time.time() - start_time
                self._update_stats(success=False, processing_time=processing_time)
                return self._create_invalid_molecule(
                    smiles, validation.errors, validation.warnings, processing_time
                )

            if validate_only:
                processing_time = time.time() - start_time
                self._update_stats(success=True, processing_time=processing_time)

                return ProcessedMolecule(
                    original_smiles=smiles,
                    canonical_smiles=validation.canonical_smiles,
                    molecular_formula="",
                    properties=self._create_minimal_properties(
                        validation.basic_properties
                    ),
                    is_valid=True,
                    errors=[],
                    warnings=validation.warnings,
                    processing_time=processing_time,
                    rdkit_mol=None,
                )

            mol = Chem.MolFromSmiles(smiles)

            if self.enable_standardization:
                try:
                    mol = self.standardizer.standardize(
                        mol, smiles, quick_mode=quick_mode
                    )
                    with self._stats_lock:
                        self._stats["standardized"] += 1
                except MavhirMolecularStandardizationError as e:
                    processing_time = time.time() - start_time
                    self._update_stats(success=False, processing_time=processing_time)
                    return self._create_invalid_molecule(
                        smiles, [e.message], [], processing_time
                    )

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            try:
                molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            except:
                molecular_formula = "Unknown"
                logger.warning(f"Could not calculate molecular formula for {smiles}")

            properties = self.property_calculator.calculate_properties(mol)

            processing_time = time.time() - start_time
            self._update_stats(success=True, processing_time=processing_time)

            return ProcessedMolecule(
                original_smiles=smiles,
                canonical_smiles=canonical_smiles,
                molecular_formula=molecular_formula,
                properties=properties,
                is_valid=True,
                errors=[],
                warnings=validation.warnings,
                processing_time=processing_time,
                rdkit_mol=mol,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(success=False, processing_time=processing_time)
            logger.error(f"Processing failed for '{smiles}': {e}")
            return self._create_invalid_molecule(
                smiles, [f"Processing error: {e}"], [], processing_time
            )

    def process_smiles_batch(
        self,
        smiles_list: List[str],
        quick_mode: bool = False,
        validate_only: bool = False,
        max_workers: int = 4,
    ) -> List[ProcessedMolecule]:
        """
        batch processing with optional parallelization.
        """
        if not smiles_list:
            return []

        logger.info(
            f"Processing batch of {len(smiles_list)} SMILES "
            f"(quick_mode: {quick_mode}, validate_only: {validate_only})"
        )

        results = []
        successful = 0
        failed = 0

        if len(smiles_list) <= 10 or max_workers == 1:
            for i, smiles in enumerate(smiles_list):
                try:
                    result = self.process_smiles(
                        smiles, quick_mode=quick_mode, validate_only=validate_only
                    )
                    results.append(result)

                    if result.is_valid:
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"Batch processing error at index {i}: {e}")
                    error_result = self._create_invalid_molecule(
                        smiles, [f"Batch processing error: {e}"], [], 0.0
                    )
                    results.append(error_result)
                    failed += 1
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_smiles = {
                    executor.submit(
                        self.process_smiles, smiles, quick_mode, validate_only
                    ): (i, smiles)
                    for i, smiles in enumerate(smiles_list)
                }

                indexed_results = {}
                for future in concurrent.futures.as_completed(future_to_smiles):
                    i, smiles = future_to_smiles[future]
                    try:
                        result = future.result()
                        indexed_results[i] = result
                        if result.is_valid:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(
                            f"Parallel processing error for SMILES at index {i}: {e}"
                        )
                        error_result = self._create_invalid_molecule(
                            smiles, [f"Parallel processing error: {e}"], [], 0.0
                        )
                        indexed_results[i] = error_result
                        failed += 1

                results = [indexed_results[i] for i in range(len(smiles_list))]

        success_rate = (successful / len(smiles_list)) * 100
        logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed "
            f"({success_rate:.1f}% success rate)"
        )

        return results

    @lru_cache(maxsize=1000)
    def get_canonical_smiles_cached(self, smiles: str) -> str:
        """
        Get canonical SMILES with caching.
        """
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                raise MavhirInvalidSMILESError(smiles, "Cannot parse SMILES")
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            raise MavhirInvalidSMILESError(smiles, str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        with self._stats_lock:
            stats = self._stats.copy()

        cache_info = self.get_canonical_smiles_cached.cache_info()

        if stats["total_processed"] > 0:
            success_rate = (stats["successful"] / stats["total_processed"]) * 100
            avg_processing_time = (
                stats["total_processing_time"] / stats["total_processed"]
            )
            standardization_rate = (
                stats["standardized"] / stats["total_processed"]
            ) * 100
        else:
            success_rate = 0.0
            avg_processing_time = 0.0
            standardization_rate = 0.0

        return {
            "processing_stats": {
                **stats,
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time,
                "standardization_rate": standardization_rate,
            },
            "configuration": {
                "standardization_enabled": self.enable_standardization,
                "timeout_seconds": self.timeout_seconds,
                "max_smiles_length": self.max_smiles_length,
                "max_heavy_atoms": self.max_heavy_atoms,
            },
            "canonical_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "current_size": cache_info.currsize,
                "max_size": cache_info.maxsize,
                "hit_rate": (
                    cache_info.hits / (cache_info.hits + cache_info.misses) * 100
                    if (cache_info.hits + cache_info.misses) > 0
                    else 0.0
                ),
            },
        }

    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Update processing statistics thread-safely."""
        with self._stats_lock:
            if success:
                self._stats["successful"] += 1
            else:
                self._stats["failed"] += 1
            self._stats["total_processing_time"] += processing_time

    def _create_invalid_molecule(
        self,
        smiles: str,
        errors: List[str],
        warnings: List[str],
        processing_time: float,
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
            num_saturated_rings=0,
            num_aliphatic_rings=0,
            fraction_csp3=0.0,
            num_radical_electrons=0,
            formal_charge=0,
        )

        return ProcessedMolecule(
            original_smiles=smiles,
            canonical_smiles="",
            molecular_formula="",
            properties=empty_properties,
            is_valid=False,
            errors=errors,
            warnings=warnings,
            processing_time=processing_time,
            rdkit_mol=None,
        )

    def _create_minimal_properties(
        self, basic_props: Dict[str, Any]
    ) -> MolecularProperties:
        """Create minimal properties for validation-only mode."""
        return MolecularProperties(
            molecular_weight=basic_props.get("molecular_weight", 0.0),
            logp=0.0,
            tpsa=0.0,
            num_heavy_atoms=basic_props.get("num_heavy_atoms", 0),
            num_aromatic_rings=0,
            num_rotatable_bonds=0,
            num_hbd=0,
            num_hba=0,
            num_saturated_rings=0,
            num_aliphatic_rings=0,
            fraction_csp3=0.0,
            num_radical_electrons=0,
            formal_charge=0,
        )


def create_chemical_processor() -> ChemicalProcessor:
    return ChemicalProcessor()


def validate_smiles_simple(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except:
        return False


def create_chemical_processor_legacy():
    return create_chemical_processor()


if __name__ == "__main__":
    processor = create_chemical_processor()

    test_cases = [
        ("CCO", "Valid simple molecule - ethanol"),
        ("CC(=O)O", "Valid carboxylic acid - acetic acid"),
        ("c1ccccc1", "Valid aromatic - benzene"),
        ("[Na+].[Cl-]", "Salt - should be desalted"),
        ("invalid_smiles", "Invalid SMILES"),
        ("", "Empty string"),
    ]

    print("Testing ChemicalProcessor:")
    print("=" * 60)

    for smiles, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {smiles}")

        try:
            result = processor.process_smiles(smiles)

            if result.is_valid:
                print(f"   SUCCESS")
                print(f"   Canonical: {result.canonical_smiles}")
                print(f"   Formula: {result.molecular_formula}")
                print(f"   MW: {result.properties.molecular_weight:.2f}")
                print(f"   Processing time: {result.processing_time:.3f}s")
                if result.warnings:
                    print(f"   Warnings: {', '.join(result.warnings)}")
            else:
                print(f"   FAILED")
                print(f"   Errors: {', '.join(result.errors)}")
                if result.warnings:
                    print(f"   Warnings: {', '.join(result.warnings)}")

        except Exception as e:
            print(f"   EXCEPTION: {e}")

    print(f"\n{'='*60}")
    print("Processor Statistics:")
    stats = processor.get_statistics()

    print("Processing Stats:")
    for key, value in stats["processing_stats"].items():
        print(f"  {key}: {value}")

    print("Cache Stats:")
    for key, value in stats["canonical_cache"].items():
        print(f"  {key}: {value}")
