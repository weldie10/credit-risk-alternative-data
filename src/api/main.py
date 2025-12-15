"""
FastAPI application for credit risk prediction API.

This module provides REST API endpoints for making credit risk predictions.
The API loads the best model from MLflow Model Registry.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
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

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

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

# Global model instance
model = None
feature_columns = []


def load_model_from_mlflow(
    model_name: str = "credit-risk-model",
    stage: str = "Production"
) -> Any:
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, etc.)
        
    Returns:
        Loaded model
    """
    try:
        # Set MLflow tracking URI
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        logger.info(f"Loading model '{model_name}' from stage '{stage}'...")
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"✅ Model loaded successfully from MLflow registry")
        return loaded_model
        
    except Exception as e:
        logger.warning(f"Could not load from MLflow registry: {str(e)}")
        logger.info("Attempting to load from local file...")
        
        # Fallback: Try loading from local file
        try:
            model_path = Path("models")
            model_files = list(model_path.glob("model_*.joblib"))
            
            if model_files:
                import joblib
                model_path = model_files[0]
                logger.info(f"Loading model from {model_path}")
                loaded_model = joblib.load(model_path)
                
                # Load feature columns if available
                feature_path = model_path.parent / f"{model_path.stem}_features.joblib"
                if feature_path.exists():
                    global feature_columns
                    feature_columns = joblib.load(feature_path)
                    logger.info(f"Loaded {len(feature_columns)} feature columns")
                
                return loaded_model
            else:
                raise FileNotFoundError("No model files found")
                
        except Exception as e2:
            logger.error(f"Error loading model: {str(e2)}")
            raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, feature_columns
    
    try:
        # Get model name and stage from environment variables
        model_name = os.getenv("MLFLOW_MODEL_NAME", "credit-risk-model")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
        
        # Try to load from MLflow registry
        model = load_model_from_mlflow(model_name, model_stage)
        logger.info("✅ Model loaded successfully on startup")
        
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        logger.warning("API will start without model. Use /load_model endpoint to load model.")
        model = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "model_status": "loaded" if model is not None else "not_loaded"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=model is not None
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
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load a model first."
            )
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Prepare features
        X = _prepare_features(features_df)
        
        # Make prediction
        if hasattr(model, 'predict'):
            # Standard sklearn model
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions
        else:
            # MLflow pyfunc model
            predictions = model.predict(X)
            if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                probabilities = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
            else:
                probabilities = predictions
        
        # Get first prediction
        prediction = int(predictions[0]) if isinstance(predictions, np.ndarray) else int(predictions)
        probability = float(probabilities[0]) if isinstance(probabilities, np.ndarray) else float(probabilities)
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
        if model is None:
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
        X = _prepare_features(features_df)
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions
        else:
            predictions = model.predict(X)
            if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                probabilities = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
            else:
                probabilities = predictions
        
        # Create response list
        results = []
        for i in range(len(request.features)):
            pred = int(predictions[i]) if isinstance(predictions, np.ndarray) else int(predictions)
            prob = float(probabilities[i]) if isinstance(probabilities, np.ndarray) else float(probabilities)
            results.append(PredictionResponse(
                prediction=pred,
                probability=prob,
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
async def load_model_endpoint(
    model_name: str = "credit-risk-model",
    stage: str = "Production"
):
    """
    Load a model from MLflow registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, etc.)
        
    Returns:
        Success message
    """
    global model
    try:
        model = load_model_from_mlflow(model_name, stage)
        logger.info("Model loaded successfully via endpoint")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded from stage '{stage}'",
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
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded"
            )
        
        model_type = type(model).__name__
        n_features = len(feature_columns) if feature_columns else "unknown"
        
        return {
            "model_type": model_type,
            "num_features": n_features,
            "feature_columns": feature_columns[:10] if feature_columns else None,
            "model_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    Args:
        df: Input DataFrame with features
        
    Returns:
        Prepared features DataFrame
    """
    try:
        # If feature columns are known, use them
        if feature_columns:
            # Add missing columns with zeros
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only feature columns in correct order
            X = df[[col for col in feature_columns if col in df.columns]].copy()
        else:
            # Use all numeric columns
            exclude_cols = [
                'TransactionId', 'BatchId', 'AccountId',
                'SubscriptionId', 'CustomerId', 'TransactionStartTime'
            ]
            numeric_cols = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in [np.number, 'int64', 'float64']
            ]
            X = df[numeric_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if len(X) > 0 else 0)
        
        return X
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
