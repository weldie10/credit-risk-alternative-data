"""
FastAPI application for credit risk prediction API.

This module provides REST API endpoints for making credit risk predictions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.pydantic_models import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse
)
from src.predict import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using alternative data",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    try:
        # Try to load model from default location
        model_path = Path("models/model_xgboost.joblib")
        if not model_path.exists():
            model_path = Path("models/model_lightgbm.joblib")
        
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            predictor = Predictor(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning("No model found. API will be available but predictions will fail.")
            logger.warning("Please train a model first or specify model path via environment variable.")
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        logger.warning("API will start without model. Use /load_model endpoint to load model.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=predictor is not None and predictor.model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with prediction, probability, and risk score
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load a model first."
            )
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Prepare features
        X = predictor.prepare_features(features_df)
        
        # Make prediction
        predictions, probabilities = predictor.predict(X, return_proba=True)
        
        # Get first prediction (since we only have one row)
        prediction = int(predictions[0])
        probability = float(probabilities[0])
        risk_score = int(probability * 1000)
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_score=risk_score
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request with list of features
        
    Returns:
        Batch prediction response with list of predictions
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load a model first."
            )
        
        if not request.features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Features list cannot be empty"
            )
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(request.features)
        
        # Prepare features
        X = predictor.prepare_features(features_df)
        
        # Make predictions
        predictions, probabilities = predictor.predict(X, return_proba=True)
        
        # Create response list
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                risk_score=int(prob * 1000)
            ))
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results)
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/load_model")
async def load_model(model_path: str):
    """
    Load a model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Success message
    """
    global predictor
    try:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {model_path}"
            )
        
        logger.info(f"Loading model from {model_path}")
        predictor = Predictor(str(model_path))
        logger.info("Model loaded successfully")
        
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "model_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model information
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded"
            )
        
        model_type = type(predictor.model).__name__
        n_features = len(predictor.feature_columns) if predictor.feature_columns else "unknown"
        
        return {
            "model_type": model_type,
            "num_features": n_features,
            "feature_columns": predictor.feature_columns[:10] if predictor.feature_columns else None,
            "model_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
