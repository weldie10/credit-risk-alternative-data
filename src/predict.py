"""
Inference script for making predictions.

This module handles loading trained models and making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Union, Tuple
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """Class for making predictions using trained models."""
    
    def __init__(self, model_path: str):
        """
        Initialize Predictor.
        
        Args:
            model_path: Path to saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Try to load feature columns
            feature_path = model_path.parent / f"{model_path.stem}_features.joblib"
            if feature_path.exists():
                self.feature_columns = joblib.load(feature_path)
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                self.feature_columns = None
                logger.warning("Feature columns file not found. Will use all numeric columns.")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data for prediction.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with features
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            logger.info(f"Loading data from {file_path}")
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        try:
            if self.feature_columns is None:
                # Use all numeric columns if feature columns not specified
                exclude_cols = [
                    'TransactionId', 'BatchId', 'AccountId',
                    'SubscriptionId', 'CustomerId', 'TransactionStartTime'
                ]
                feature_columns = [
                    col for col in df.columns
                    if col not in exclude_cols and df[col].dtype in [np.number, 'int64', 'float64']
                ]
            else:
                # Use specified feature columns
                missing_cols = [col for col in self.feature_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing feature columns: {missing_cols}")
                    logger.warning("These columns will be filled with zeros")
                
                feature_columns = [col for col in self.feature_columns if col in df.columns]
            
            # Select and prepare features
            X = df[feature_columns].copy()
            
            # Fill missing columns with zeros
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in X.columns:
                        X[col] = 0
            
            # Ensure correct column order
            if self.feature_columns:
                X = X[[col for col in self.feature_columns if col in X.columns]]
            
            # Handle missing values
            X = X.fillna(X.median())
            
            logger.info(f"Prepared {X.shape[1]} features for prediction")
            return X
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features DataFrame
            return_proba: Whether to return probability scores
            
        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            logger.info(f"Making predictions on {len(X)} samples...")
            
            predictions = self.model.predict(X)
            
            if return_proba:
                probabilities = self.model.predict_proba(X)[:, 1]
                logger.info("Predictions completed with probabilities")
                return predictions, probabilities
            else:
                logger.info("Predictions completed")
                return predictions
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        output_path: str = "predictions.csv",
        index: Optional[pd.Index] = None
    ) -> None:
        """
        Save predictions to file.
        
        Args:
            predictions: Predicted classes
            probabilities: Predicted probabilities (optional)
            output_path: Path to save predictions
            index: Index to use for the output DataFrame
        """
        try:
            results = pd.DataFrame({'prediction': predictions})
            
            if probabilities is not None:
                results['probability'] = probabilities
                results['risk_score'] = (probabilities * 1000).astype(int)  # Scale to 0-1000
            
            if index is not None:
                results.index = index
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to data file for prediction"
    )
    parser.add_argument(
        "--output-path", type=str, default="predictions.csv",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--no-proba", action="store_true",
        help="Don't include probability scores"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = Predictor(args.model_path)
        
        # Load data
        df = predictor.load_data(args.data_path)
        
        # Prepare features
        X = predictor.prepare_features(df)
        
        # Make predictions
        if args.no_proba:
            predictions = predictor.predict(X, return_proba=False)
            predictor.save_predictions(predictions, None, args.output_path, df.index)
        else:
            predictions, probabilities = predictor.predict(X, return_proba=True)
            predictor.save_predictions(predictions, probabilities, args.output_path, df.index)
        
        logger.info("Prediction complete!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
