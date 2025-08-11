# API Documentation

## Authentication
Currently, no authentication is required. Rate limiting is applied per IP address.

## Rate Limits
- 100 requests per minute per IP
- File uploads limited to 50MB
- Batch processing limited to 1000 compounds

## Response Format
All responses follow a consistent structure:

```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "processing_time": 0.15,
  "data": {
    // Endpoint-specific data
  }
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "error": "ValidationError",
    "message": "Invalid SMILES string",
    "context": {
      "smiles": "invalid_input"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Endpoints Reference

### Health Endpoints

#### `GET /health`
Basic health check.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "services": []
}
```

### Chemical Endpoints

#### `POST /api/v1/predict/smiles`
Predict toxicity for a single SMILES string.

**Request:**
```json
{
  "smiles": "CCO",
  "endpoints": ["ames_mutagenicity", "carcinogenicity"],
  "include_descriptors": true,
  "include_properties": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "smiles": "CCO",
    "canonical_smiles": "CCO",
    "predictions": {
      "ames_mutagenicity": {
        "prediction": "non_mutagenic",
        "probability": 0.92,
        "confidence": "high"
      }
    },
    "molecular_properties": {
      "molecular_weight": 46.07,
      "logp": -0.31
    }
  }
}