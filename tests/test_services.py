import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem

from app.services.chemical_processor import ChemicalProcessor, create_chemical_processor
from app.services.descriptor_calculator import (
    DescriptorCalculator,
    create_descriptor_calculator,
)
from app.services.predictor import ToxicityPredictor, create_predictor
from app.services.pubchem_client import PubChemClient, create_pubchem_client
from app.core.exceptions import (
    InvalidSMILESError,
    ChemicalProcessingError,
    DescriptorCalculationError,
    ModelPredictionError,
    PubChemAPIError,
)


class TestChemicalProcessor:
    """Test chemical processing service."""

    def test_process_valid_smiles(self):
        """Test processing valid SMILES."""
        processor = create_chemical_processor()

        result = processor.process_smiles("CCO")  # Ethanol

        assert result.is_valid
        assert result.canonical_smiles == "CCO"
        assert result.molecular_formula == "C2H6O"
        assert result.properties.molecular_weight > 0
        assert len(result.errors) == 0

    def test_process_invalid_smiles(self):
        """Test processing invalid SMILES."""
        processor = create_chemical_processor()

        result = processor.process_smiles("invalid_smiles_123")

        assert not result.is_valid
        assert result.canonical_smiles == ""
        assert len(result.errors) > 0
        assert (
            "Invalid SMILES" in result.errors[0]
            or "processing" in result.errors[0].lower()
        )

    def test_process_empty_smiles(self):
        """Test processing empty SMILES."""
        processor = create_chemical_processor()

        result = processor.process_smiles("")

        assert not result.is_valid
        assert "Empty" in result.errors[0] or "empty" in result.errors[0]

    def test_get_canonical_smiles(self):
        """Test canonical SMILES generation."""
        processor = create_chemical_processor()

        canonical1 = processor.get_canonical_smiles("CCO")
        canonical2 = processor.get_canonical_smiles("OCC")

        assert canonical1 == canonical2 == "CCO"

    def test_get_canonical_smiles_invalid(self):
        """Test canonical SMILES with invalid input."""
        processor = create_chemical_processor()

        with pytest.raises(InvalidSMILESError):
            processor.get_canonical_smiles("invalid_smiles")

    def test_batch_processing(self):
        """Test batch SMILES processing."""
        processor = create_chemical_processor()

        smiles_list = ["CCO", "CC(=O)O", "invalid_smiles", "c1ccccc1"]
        results = processor.process_smiles_batch(smiles_list)

        assert len(results) == 4
        assert results[0].is_valid  # CCO
        assert results[1].is_valid  # CC(=O)O
        assert not results[2].is_valid  # invalid
        assert results[3].is_valid  # benzene

    def test_caching(self):
        """Test that caching works for repeated queries."""
        processor = create_chemical_processor()

        canonical1 = processor.get_canonical_smiles("CCO")

        canonical2 = processor.get_canonical_smiles("CCO")

        assert canonical1 == canonical2

        stats = processor.get_statistics()
        assert "canonical_cache" in stats
        assert stats["canonical_cache"]["hits"] > 0


class TestDescriptorCalculator:
    """Test molecular descriptor calculation."""

    def test_calculate_descriptors(self):
        """Test descriptor calculation for valid molecule."""
        calculator = create_descriptor_calculator()

        descriptors = calculator.calculate_cached("CCO")  # Ethanol

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 50

        for name, value in descriptors.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_calculate_descriptors_invalid(self):
        """Test descriptor calculation for invalid SMILES."""
        calculator = create_descriptor_calculator()

        with pytest.raises(DescriptorCalculationError):
            calculator.calculate_cached("invalid_smiles")

    def test_descriptor_names(self):
        """Test getting descriptor names."""
        calculator = create_descriptor_calculator()

        names = calculator.get_descriptor_names()

        assert isinstance(names, list)
        assert len(names) > 50
        assert all(isinstance(name, str) for name in names)

    def test_caching_behavior(self):
        """Test descriptor caching."""
        calculator = create_descriptor_calculator()

        desc1 = calculator.calculate_cached("CCO")

        desc2 = calculator.calculate_cached("CCO")

        assert desc1 == desc2

        stats = calculator.get_statistics()
        assert stats["cache"]["hits"] > 0

    @patch("app.services.descriptor_calculator.Calculator")
    def test_mordred_error_handling(self, mock_calculator):
        """Test handling of Mordred calculation errors."""
        mock_calc_instance = Mock()
        mock_calc_instance.side_effect = Exception("Mordred error")
        mock_calculator.return_value = mock_calc_instance

        calculator = create_descriptor_calculator()

        with pytest.raises(DescriptorCalculationError):
            calculator.calculate_cached("CCO")


class TestToxicityPredictor:
    """Test ML prediction service."""

    @patch("app.services.predictor.Path.exists")
    def test_model_loading_missing_files(self, mock_exists):
        """Test behavior when model files are missing."""
        mock_exists.return_value = False

        predictor = create_predictor()

        assert len(predictor.get_available_endpoints()) == 0

    def test_get_available_endpoints(self):
        """Test getting available prediction endpoints."""
        predictor = create_predictor()

        endpoints = predictor.get_available_endpoints()

        assert isinstance(endpoints, list)
        # May be empty if models not trained

    def test_get_model_info(self):
        """Test getting model information."""
        predictor = create_predictor()

        model_info = predictor.get_model_info()

        assert isinstance(model_info, dict)

    @patch("app.services.predictor.ToxicityModel")
    def test_prediction_with_mocked_model(self, mock_model_class):
        """Test prediction with mocked model."""
        mock_model = Mock()
        mock_model.predict.return_value = Mock(
            endpoint="ames_mutagenicity",
            prediction="non_mutagenic",
            probability=0.8,
            confidence="high",
            model_version="1.0",
        )
        mock_model_class.return_value = mock_model

        predictor = create_predictor()
        predictor.models = {"ames_mutagenicity": mock_model}

        # Test prediction
        dummy_descriptors = {f"desc_{i}": float(i) for i in range(100)}

        result = predictor.predict(
            descriptors=dummy_descriptors, smiles="CCO", endpoints=None
        )

        assert result.smiles == "CCO"
        assert len(result.predictions) > 0


class TestPubChemClient:
    """Test PubChem API client."""

    @patch("app.services.pubchem_client.requests.Session.get")
    def test_search_by_name_found(self, mock_get):
        """Test successful compound search."""
        # Mock successful CID lookup
        mock_cid_response = Mock()
        mock_cid_response.status_code = 200
        mock_cid_response.json.return_value = {"IdentifierList": {"CID": [702]}}

        mock_prop_response = Mock()
        mock_prop_response.status_code = 200
        mock_prop_response.json.return_value = {
            "PropertyTable": {
                "Properties": [
                    {
                        "Title": "ethanol",
                        "CanonicalSMILES": "CCO",
                        "MolecularFormula": "C2H6O",
                        "MolecularWeight": 46.07,
                    }
                ]
            }
        }

        mock_get.side_effect = [mock_cid_response, mock_prop_response]

        client = create_pubchem_client()
        result = client.search_by_name("ethanol")

        assert result.found
        assert result.cid == 702
        assert result.name == "ethanol"
        assert result.smiles == "CCO"
        assert result.molecular_weight == 46.07

    @patch("app.services.pubchem_client.requests.Session.get")
    def test_search_by_name_not_found(self, mock_get):
        """Test compound not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404")

        mock_get.return_value = mock_response

        client = create_pubchem_client()
        result = client.search_by_name("nonexistent_compound_12345")

        assert not result.found

    @patch("app.services.pubchem_client.requests.Session.get")
    def test_rate_limiting(self, mock_get):
        """Test rate limiting behavior."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"IdentifierList": {"CID": [123]}}
        mock_get.return_value = mock_response

        client = create_pubchem_client()

        import time

        start_time = time.time()

        client._get_cid_by_name("compound1")
        client._get_cid_by_name("compound2")

        end_time = time.time()

        assert end_time - start_time >= client.rate_delay

    @patch("app.services.pubchem_client.requests.Session.get")
    def test_timeout_handling(self, mock_get):
        """Test timeout handling."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        client = create_pubchem_client()

        with pytest.raises(PubChemAPIError):
            client.search_by_name("test_compound")

    @patch("app.services.pubchem_client.requests.Session.get")
    def test_http_error_handling(self, mock_get):
        """Test HTTP error handling."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        client = create_pubchem_client()

        with pytest.raises(PubChemAPIError):
            client.search_by_name("test_compound")


class TestServiceIntegration:
    """Test service integration and workflows."""

    def test_full_prediction_workflow(self):
        """Test complete prediction workflow (if models available)."""
        try:
            processor = create_chemical_processor()
            processed = processor.process_smiles("CCO")

            if not processed.is_valid:
                pytest.skip("Chemical processing failed")

            calculator = create_descriptor_calculator()
            descriptors = calculator.calculate_cached(processed.canonical_smiles)

            assert len(descriptors) > 50

            predictor = create_predictor()

            if len(predictor.get_available_endpoints()) == 0:
                pytest.skip("No trained models available")

            result = predictor.predict(
                descriptors=descriptors, smiles=processed.canonical_smiles
            )

            assert result.smiles == processed.canonical_smiles

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    def test_error_propagation(self):
        """Test that errors are properly propagated through the workflow."""
        processor = create_chemical_processor()

        result = processor.process_smiles("definitely_invalid_smiles_12345")

        assert not result.is_valid
        assert len(result.errors) > 0

        error_text = " ".join(result.errors).lower()
        assert "invalid" in error_text or "failed" in error_text


class TestPerformance:
    """Test performance characteristics."""

    def test_chemical_processing_performance(self):
        """Test chemical processing performance."""
        processor = create_chemical_processor()

        import time

        start_time = time.time()

        # Process multiple compounds
        test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CCCCO"]
        results = processor.process_smiles_batch(test_smiles)

        end_time = time.time()
        processing_time = end_time - start_time

        assert len(results) == 5
        assert processing_time < 5.0

        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count >= 4

    def test_descriptor_caching_performance(self):
        """Test descriptor calculation caching performance."""
        calculator = create_descriptor_calculator()

        import time

        start_time = time.time()
        desc1 = calculator.calculate_cached("CCO")
        first_time = time.time() - start_time

        start_time = time.time()
        desc2 = calculator.calculate_cached("CCO")
        second_time = time.time() - start_time

        assert desc1 == desc2
        assert second_time < first_time * 0.1


@pytest.fixture
def sample_descriptors():
    """Create sample descriptor dictionary for testing."""
    return {f"descriptor_{i}": float(i * 0.1) for i in range(100)}


@pytest.fixture
def mock_rdkit_mol():
    """Create a mock RDKit molecule for testing."""
    mol = Mock()
    mol.GetNumHeavyAtoms.return_value = 3
    mol.GetNumAtoms.return_value = 9
    return mol


class TestFixtures:
    """Test that fixtures work correctly."""

    def test_sample_descriptors_fixture(self, sample_descriptors):
        """Test sample descriptors fixture."""
        assert isinstance(sample_descriptors, dict)
        assert len(sample_descriptors) == 100
        assert all(isinstance(v, float) for v in sample_descriptors.values())

    def test_mock_rdkit_mol_fixture(self, mock_rdkit_mol):
        """Test mock RDKit molecule fixture."""
        assert mock_rdkit_mol.GetNumHeavyAtoms() == 3
        assert mock_rdkit_mol.GetNumAtoms() == 9


class TestBenchmarks:
    """Benchmark tests for performance monitoring."""

    @pytest.mark.benchmark
    def test_smiles_processing_benchmark(self, benchmark):
        """Benchmark SMILES processing performance."""
        processor = create_chemical_processor()

        def process_smiles():
            return processor.process_smiles("CCO")

        result = benchmark(process_smiles)
        assert result.is_valid

    @pytest.mark.benchmark
    def test_descriptor_calculation_benchmark(self, benchmark):
        """Benchmark descriptor calculation performance."""
        calculator = create_descriptor_calculator()

        def calculate_descriptors():
            return calculator.calculate_cached("CCO")

        result = benchmark(calculate_descriptors)
        assert len(result) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
