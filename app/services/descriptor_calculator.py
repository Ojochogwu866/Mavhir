"""
Molecular descriptor calculator using Mordred.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from functools import lru_cache

from mordred import Calculator, descriptors
from rdkit import Chem

# Our imports
from ..core.config import get_settings
from ..core.exceptions import DescriptorCalculationError, DescriptorTimeoutError

logger = logging.getLogger(__name__)


class DescriptorCalculator:
    """
    Calculates molecular descriptors for ML models.
    """

    def __init__(self):
        """Initialize the calculator with our chosen descriptors."""

        self.settings = get_settings()
        self.timeout = self.settings.descriptor_timeout

        self._setup_calculator()

        logger.info(
            f"DescriptorCalculator initialized with {len(self.descriptor_names)} descriptors"
        )

    def _setup_calculator(self):
        calc = Calculator()
        safe_descriptors = []
        
        try:
            calc.register(descriptors.AtomCount)
            safe_descriptors.append("AtomCount")
        except Exception as e:
            logger.warning(f"Failed to register AtomCount: {e}")
        
        try:
            calc.register(descriptors.BondCount)
            safe_descriptors.append("BondCount")
        except Exception as e:
            logger.warning(f"Failed to register BondCount: {e}")		
		
        try:
            calc.register(descriptors.RingCount)
            safe_descriptors.append("RingCount")
        except Exception as e:
            logger.warning(f"Failed to register RingCount: {e}")
    
        optional_descriptors = [
			"Constitutional",
			"TopologicalIndex", 
			"InformationIndex",
			"GeometricIndex",
			"PartialCharge",
			"Polarizability",
			"FragmentComplexity",
			"Framework"
		]
    
        for desc_name in optional_descriptors:
            try:
                if hasattr(descriptors, desc_name):
                    desc_class = getattr(descriptors, desc_name)
                    calc.register(desc_class)
                    safe_descriptors.append(desc_name)
                    logger.debug(f"Registered: {desc_name}")
            except Exception as e:
                logger.warning(f"Failed to register {desc_name}: {e}")
    
        self.calculator = calc
        self.descriptor_names = [str(d) for d in calc.descriptors]
			
        logger.info(f"Successfully registered {len(safe_descriptors)} descriptor groups: {safe_descriptors}")
        logger.debug(f"Total descriptors: {len(self.descriptor_names)}")


    def calculate(self, mol: Chem.Mol, smiles: str) -> Dict[str, float]:
        """
        Calculate descriptors for a molecule.
        """

        if mol is None:
            raise DescriptorCalculationError(
                f"Cannot calculate descriptors for None molecule: {smiles}"
            )

        try:
            logger.debug(f"Calculating descriptors for: {smiles}")

            descriptor_values = self.calculator(mol)
            result = {}

            for name, value in zip(self.descriptor_names, descriptor_values):

                if value is None or np.isnan(value) or np.isinf(value):
                    result[name] = 0.0
                else:
                    result[name] = float(value)

            logger.debug(f"Calculated {len(result)} descriptors successfully")
            return result

        except Exception as e:
            logger.error(f"Descriptor calculation failed for {smiles}: {e}")
            raise DescriptorCalculationError(
                f"Failed to calculate descriptors for {smiles}: {e}"
            )

    @lru_cache(maxsize=1000)
    def calculate_cached(self, smiles: str) -> Dict[str, float]:
        """
        Calculate descriptors with caching.
        """

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise DescriptorCalculationError(
                f"Invalid SMILES for descriptor calculation: {smiles}"
            )

        return self.calculate(mol, smiles)

    def get_descriptor_names(self) -> List[str]:
        """Get list of all calculated descriptor names."""
        return self.descriptor_names.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculation statistics for monitoring."""

        cache_info = self.calculate_cached.cache_info()

        return {
            "num_descriptors": len(self.descriptor_names),
            "timeout_seconds": self.timeout,
            "cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "current_size": cache_info.currsize,
                "max_size": cache_info.maxsize,
                "hit_rate": (
                    cache_info.hits / (cache_info.hits + cache_info.misses)
                    if (cache_info.hits + cache_info.misses) > 0
                    else 0
                ),
            },
        }


def create_descriptor_calculator() -> DescriptorCalculator:
    """Factory function to create DescriptorCalculator."""
    return DescriptorCalculator()


def quick_descriptor_test(smiles: str) -> Dict[str, float]:
    """
    Quick test function to see descriptors for a molecule.

    USAGE:
        descriptors = quick_descriptor_test("CCO")
        print(f"Ethanol has {descriptors['nHeavyAtom']} heavy atoms")
    """

    calculator = create_descriptor_calculator()
    return calculator.calculate_cached(smiles)


def _test_descriptor_calculator():
    """Test the descriptor calculator with simple molecules."""

    calculator = create_descriptor_calculator()

    test_molecules = [
        ("CCO", "Ethanol"),
        ("CC(=O)O", "Acetic acid"),
        ("c1ccccc1", "Benzene"),
        ("CCN(CC)CC", "Triethylamine"),
    ]

    print("Testing Descriptor Calculator:")
    print("=" * 50)
    print(f"Calculator has {len(calculator.descriptor_names)} descriptors")
    print()

    for smiles, name in test_molecules:
        try:
            descriptors = calculator.calculate_cached(smiles)

            print(f" {name} ({smiles}):")

            for i, (desc_name, value) in enumerate(list(descriptors.items())[:10]):
                print(f"   {desc_name}: {value:.3f}")

            print(f"   ... and {len(descriptors) - 10} more descriptors")
            print()

        except Exception as e:
            print(f" {name} ({smiles}): {e}")
            print()

    # Show statistics
    stats = calculator.get_statistics()
    print("Calculator Statistics:")
    print(f"  Descriptors: {stats['num_descriptors']}")
    print(f"  Cache hits: {stats['cache']['hits']}")
    print(f"  Cache misses: {stats['cache']['misses']}")
    print(f"  Hit rate: {stats['cache']['hit_rate']:.1%}")


if __name__ == "__main__":
    _test_descriptor_calculator()
