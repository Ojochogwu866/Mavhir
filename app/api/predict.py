import time
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from ..core.config import get_settings, Settings
from ..core.models import (
    SMILESPredictionRequest,
    SMILESPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchResultItem,
    BatchPredictionSummary,
    CompoundPredictionResponse,
    ToxicityEndpoint,
    ModelInfoResponse,
    ModelInfo,
    PredictionResults,
    SinglePrediction,
)
from ..services.chemical_processor import create_chemical_processor
from ..services.descriptor_calculator import create_descriptor_calculator
from ..services.predictor import create_predictor
from ..core.exceptions import (
    InvalidSMILESError,
    DescriptorCalculationError,
    ModelPredictionError,
    BatchSizeError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Toxicity Prediction"])


@router.post("/smiles", response_model=SMILESPredictionResponse)
async def predict_smiles(
    request: SMILESPredictionRequest, settings: Settings = Depends(get_settings)
):
    """
    Predict toxicity for a single SMILES string.
    """

    start_time = time.time()

    try:
        logger.info(f"Processing SMILES: {request.smiles}")
        chemical_processor = create_chemical_processor()

        processed_mol = chemical_processor.process_smiles(request.smiles)

        if not processed_mol.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid SMILES: {'; '.join(processed_mol.errors)}",
            )

        logger.debug("Calculating molecular descriptors")
        descriptor_calculator = create_descriptor_calculator()

        try:
            descriptors = descriptor_calculator.calculate_cached(
                processed_mol.canonical_smiles
            )
        except DescriptorCalculationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Descriptor calculation failed: {e.message}",
            )

        logger.debug("Making toxicity predictions")
        predictor = create_predictor()

        try:
            predictions = predictor.predict(
                descriptors=descriptors,
                smiles=processed_mol.canonical_smiles,
                endpoints=request.endpoints,
            )
        except ModelPredictionError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {e.message}",
            )

        processing_time = time.time() - start_time

        api_predictions = {}
        for endpoint, pred_result in predictions.predictions.items():
            api_predictions[endpoint.value] = SinglePrediction(
                endpoint=endpoint,
                prediction=pred_result.prediction,
                probability=pred_result.probability,
                confidence=pred_result.confidence,
            )

        compound_response = CompoundPredictionResponse(
            smiles=request.smiles,
            canonical_smiles=processed_mol.canonical_smiles,
            molecular_formula=processed_mol.molecular_formula,
            predictions=PredictionResults(**api_predictions),
            molecular_properties=(
                processed_mol.properties if request.include_properties else None
            ),
            descriptors=descriptors if request.include_descriptors else None,
            success=True,
        )

        return SMILESPredictionResponse(
            success=True, processing_time=processing_time, data=compound_response
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in SMILES prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction",
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest, settings: Settings = Depends(get_settings)
):
    """
    Predict toxicity for multiple SMILES strings.
    """

    start_time = time.time()

    # Validate batch size
    if len(request.smiles_list) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.smiles_list)} exceeds maximum {settings.max_batch_size}",
        )

    logger.info(f"Processing batch of {len(request.smiles_list)} compounds")

    # Initialize services
    chemical_processor = create_chemical_processor()
    descriptor_calculator = create_descriptor_calculator()
    predictor = create_predictor()

    results = []
    successful = 0
    failed = 0

    for idx, smiles in enumerate(request.smiles_list):
        try:
            result_item = await _process_single_compound(
                idx=idx,
                smiles=smiles,
                endpoints=request.endpoints,
                include_descriptors=request.include_descriptors,
                include_properties=request.include_properties,
                chemical_processor=chemical_processor,
                descriptor_calculator=descriptor_calculator,
                predictor=predictor,
            )

            results.append(result_item)

            if result_item.success:
                successful += 1
            else:
                failed += 1

                if request.fail_on_error:
                    break

        except Exception as e:
            logger.error(f"Batch processing error at index {idx}: {e}")

            error_item = BatchResultItem(
                index=idx, input_smiles=smiles, result=None, error=str(e), success=False
            )
            results.append(error_item)
            failed += 1

            if request.fail_on_error:
                break

    processing_time = time.time() - start_time
    success_rate = (successful / len(results)) * 100 if results else 0

    summary = BatchPredictionSummary(
        total_compounds=len(results),
        successful=successful,
        failed=failed,
        success_rate=success_rate,
    )

    return BatchPredictionResponse(
        success=True, processing_time=processing_time, results=results, summary=summary
    )


@router.post("/sdf", response_model=BatchPredictionResponse)
async def predict_sdf(
    file: UploadFile = File(...),
    endpoints: Optional[List[ToxicityEndpoint]] = None,
    include_descriptors: bool = False,
    include_properties: bool = True,
    settings: Settings = Depends(get_settings),
):
    """
    Predict toxicity from SDF (Structure Data Format) file upload.
    """

    start_time = time.time()

    if not file.filename.lower().endswith((".sdf", ".mol")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be SDF (.sdf) or MOL (.mol) format",
        )

    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if file.size and file.size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file.size} bytes exceeds maximum {max_size_bytes} bytes",
        )

    try:
        content = await file.read()
        smiles_list = _parse_sdf_content(content.decode("utf-8"))

        if len(smiles_list) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SDF contains {len(smiles_list)} compounds, exceeds maximum {settings.max_batch_size}",
            )

        batch_request = BatchPredictionRequest(
            smiles_list=[smiles for smiles, _ in smiles_list],
            endpoints=endpoints,
            include_descriptors=include_descriptors,
            include_properties=include_properties,
            fail_on_error=False,
        )

        # Process batch
        logger.info(f"Processing SDF file with {len(smiles_list)} compounds")
        batch_response = await predict_batch(batch_request, settings)

        for i, (result, (_, name)) in enumerate(
            zip(batch_response.results, smiles_list)
        ):
            if result.result and name:
                pass

        return batch_response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"SDF processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process SDF file: {str(e)}",
        )


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about available prediction models.

    """

    try:
        predictor = create_predictor()

        model_info_dict = predictor.get_model_info()
        available_endpoints = predictor.get_available_endpoints()

        models = []
        for endpoint in available_endpoints:
            info = model_info_dict.get(endpoint.value, {})

            model = ModelInfo(
                name=info.get("name", f"{endpoint.value} model"),
                endpoint=endpoint,
                version=info.get("version", "unknown"),
                description=f"Predicts {endpoint.value.replace('_', ' ')} toxicity",
                threshold=info.get("threshold", 0.5),
                performance_metrics=None,
            )
            models.append(model)

        return ModelInfoResponse(success=True, models=models)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information",
        )


async def _process_single_compound(
    idx: int,
    smiles: str,
    endpoints: Optional[List[ToxicityEndpoint]],
    include_descriptors: bool,
    include_properties: bool,
    chemical_processor,
    descriptor_calculator,
    predictor,
) -> BatchResultItem:
    """Process a single compound for batch prediction."""

    try:
        processed_mol = chemical_processor.process_smiles(smiles)

        if not processed_mol.is_valid:
            return BatchResultItem(
                index=idx,
                input_smiles=smiles,
                result=None,
                error=f"Invalid SMILES: {'; '.join(processed_mol.errors)}",
                success=False,
            )

        descriptors = descriptor_calculator.calculate_cached(
            processed_mol.canonical_smiles
        )

        predictions = predictor.predict(
            descriptors=descriptors,
            smiles=processed_mol.canonical_smiles,
            endpoints=endpoints,
        )

        api_predictions = {}
        for endpoint, pred_result in predictions.predictions.items():
            api_predictions[endpoint.value] = SinglePrediction(
                endpoint=endpoint,
                prediction=pred_result.prediction,
                probability=pred_result.probability,
                confidence=pred_result.confidence,
            )

        compound_response = CompoundPredictionResponse(
            smiles=smiles,
            canonical_smiles=processed_mol.canonical_smiles,
            molecular_formula=processed_mol.molecular_formula,
            predictions=PredictionResults(**api_predictions),
            molecular_properties=(
                processed_mol.properties if include_properties else None
            ),
            descriptors=descriptors if include_descriptors else None,
            success=True,
        )

        return BatchResultItem(
            index=idx,
            input_smiles=smiles,
            result=compound_response,
            error=None,
            success=True,
        )

    except Exception as e:
        return BatchResultItem(
            index=idx, input_smiles=smiles, result=None, error=str(e), success=False
        )


def _parse_sdf_content(content: str) -> List[tuple]:
    molecules = content.split("$$$$")

    results = []
    for mol_block in molecules:
        if not mol_block.strip():
            continue

        lines = mol_block.strip().split("\n")
        if len(lines) < 4:
            continue

        name = lines[0].strip() if lines else f"compound_{len(results)}"

        smiles = "CCO"  # Placeholder

        results.append((smiles, name))

    return results
