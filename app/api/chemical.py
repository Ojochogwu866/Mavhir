import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Query

from ..core.config import get_settings, Settings
from ..core.models import ChemicalLookupResponse, ErrorResponse
from ..services.pubchem_client import create_pubchem_client
from ..services.chemical_processor import create_chemical_processor
from ..core.exceptions import MavhirPubChemAPIError, MavhirInvalidSMILESError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chemical", tags=["Chemical Lookup"])


@router.get("/lookup/{query}", response_model=ChemicalLookupResponse)
async def lookup_compound(query: str, settings: Settings = Depends(get_settings)):
    start_time = time.time()

    try:
        client = create_pubchem_client()

        logger.info(f"Looking up compound: {query}")
        result = client.search_by_name(query)

        processing_time = time.time() - start_time

        return ChemicalLookupResponse(
            success=True,
            processing_time=processing_time,
            query=query,
            compound_name=result.name,
            cid=result.cid,
            canonical_smiles=result.smiles,
            molecular_formula=result.molecular_formula,
            molecular_weight=result.molecular_weight,
            synonyms=result.synonyms,
            found=result.found,
        )

    except MavhirPubChemAPIError:
        processing_time = time.time() - start_time

        return ChemicalLookupResponse(
            success=True, processing_time=processing_time, query=query, found=False
        )

    except MavhirPubChemAPIError as e:
        logger.error(f"PubChem API error for query '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chemical database temporarily unavailable: {e.message}",
        )

    except Exception as e:
        logger.error(f"Unexpected error in chemical lookup '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chemical lookup",
        )


@router.get("/validate", response_model=Dict[str, Any])
async def validate_smiles(
    smiles: str = Query(..., description="SMILES string to validate"),
    standardize: bool = Query(default=True, description="Return standardized form"),
):
    start_time = time.time()

    try:
        processor = create_chemical_processor()

        if standardize:
            result = processor.process_smiles(smiles)

            processing_time = time.time() - start_time

            if result.is_valid:
                return {
                    "success": True,
                    "processing_time": processing_time,
                    "input_smiles": smiles,
                    "is_valid": True,
                    "canonical_smiles": result.canonical_smiles,
                    "molecular_formula": result.molecular_formula,
                    "molecular_properties": {
                        "molecular_weight": result.properties.molecular_weight,
                        "logp": result.properties.logp,
                        "tpsa": result.properties.tpsa,
                        "heavy_atoms": result.properties.num_heavy_atoms,
                    },
                }
            else:
                return {
                    "success": True,
                    "processing_time": processing_time,
                    "input_smiles": smiles,
                    "is_valid": False,
                    "errors": result.errors,
                }
        else:
            canonical_smiles = processor.get_canonical_smiles(smiles)

            processing_time = time.time() - start_time

            return {
                "success": True,
                "processing_time": processing_time,
                "input_smiles": smiles,
                "is_valid": True,
                "canonical_smiles": canonical_smiles,
            }

    except MavhirInvalidSMILESError as e:
        processing_time = time.time() - start_time

        return {
            "success": True,
            "processing_time": processing_time,
            "input_smiles": smiles,
            "is_valid": False,
            "error": e.message,
        }

    except Exception as e:
        logger.error(f"SMILES validation error for '{smiles}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during SMILES validation",
        )


@router.get("/properties", response_model=Dict[str, Any])
async def get_chemical_properties(
    smiles: str = Query(..., description="SMILES string"),
    include_drug_likeness: bool = Query(
        default=True, description="Include drug-likeness assessment"
    ),
):

    start_time = time.time()

    try:
        processor = create_chemical_processor()

        result = processor.process_smiles(smiles)

        if not result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid SMILES: {'; '.join(result.errors)}",
            )

        processing_time = time.time() - start_time

        response_data = {
            "success": True,
            "processing_time": processing_time,
            "smiles": smiles,
            "canonical_smiles": result.canonical_smiles,
            "molecular_formula": result.molecular_formula,
            "properties": {
                "molecular_weight": result.properties.molecular_weight,
                "logp": result.properties.logp,
                "tpsa": result.properties.tpsa,
                "num_heavy_atoms": result.properties.num_heavy_atoms,
                "num_aromatic_rings": result.properties.num_aromatic_rings,
                "num_rotatable_bonds": result.properties.num_rotatable_bonds,
                "num_hbd": result.properties.num_hbd,
                "num_hba": result.properties.num_hba,
            },
        }

        if include_drug_likeness:
            drug_assessment = processor.property_calculator.assess_drug_likeness(
                result.properties
            )
            response_data["drug_likeness"] = drug_assessment

        return response_data

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Property calculation error for '{smiles}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during property calculation",
        )


@router.post("/validate/batch", response_model=Dict[str, Any])
async def validate_smiles_batch(smiles_list: list[str], max_batch_size: int = 100):
    start_time = time.time()

    if len(smiles_list) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(smiles_list)} exceeds maximum {max_batch_size}",
        )

    if not smiles_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SMILES list cannot be empty",
        )

    try:
        processor = create_chemical_processor()

        results = processor.process_smiles_batch(smiles_list, quick_mode=True)

        processing_time = time.time() - start_time
        valid_results = []
        invalid_results = []

        for i, result in enumerate(results):
            if result.is_valid:
                valid_results.append(
                    {
                        "index": i,
                        "input_smiles": result.original_smiles,
                        "canonical_smiles": result.canonical_smiles,
                        "molecular_formula": result.molecular_formula,
                    }
                )
            else:
                invalid_results.append(
                    {
                        "index": i,
                        "input_smiles": result.original_smiles,
                        "errors": result.errors,
                    }
                )

        return {
            "success": True,
            "processing_time": processing_time,
            "summary": {
                "total": len(smiles_list),
                "valid": len(valid_results),
                "invalid": len(invalid_results),
                "success_rate": len(valid_results) / len(smiles_list) * 100,
            },
            "valid_compounds": valid_results,
            "invalid_compounds": invalid_results,
        }

    except Exception as e:
        logger.error(f"Batch SMILES validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch validation",
        )
