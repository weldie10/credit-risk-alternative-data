"""
Model training script.

This module handles loading processed data, training machine learning models,
evaluating performance, and saving trained models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import sys
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and evaluating credit risk models."""
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: Type of model ('xgboost' or 'lightgbm')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        if self.model_type not in ['xgboost', 'lightgbm']:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'xgboost' or 'lightgbm'")
    
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """
        Load processed data for training.
        
        Args:
            file_path: Path to processed data file
            
        Returns:
            DataFrame with processed features
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            logger.info(f"Loading processed data from {file_path}")
            
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
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Select feature columns (exclude target and ID columns)
            exclude_cols = [
                target_column, 'TransactionId', 'BatchId', 'AccountId',
                'SubscriptionId', 'CustomerId', 'TransactionStartTime'
            ]
            feature_columns = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in [np.number, 'int64', 'float64']
            ]
            
            X = df[feature_columns].fillna(df[feature_columns].median())
            y = df[target_column]
            
            # Handle class imbalance - check if target is binary
            if y.nunique() == 2:
                logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=y_train
            )
            
            self.feature_columns = feature_columns
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model hyperparameters
        """
        try:
            logger.info(f"Training {self.model_type} model...")
            
            if self.model_type == 'xgboost':
                if params is None:
                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc',
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'random_state': self.random_state,
                        'n_jobs': -1
                    }
                
                self.model = xgb.XGBClassifier(**params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
            elif self.model_type == 'lightgbm':
                if params is None:
                    params = {
                        'objective': 'binary',
                        'metric': 'auc',
                        'boosting_type': 'gbdt',
                        'num_leaves': 31,
                        'learning_rate': 0.05,
                        'n_estimators': 100,
                        'random_state': self.random_state,
                        'n_jobs': -1
                    }
                
                self.model = lgb.LGBMClassifier(**params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'accuracy': (y_pred == y_test).mean()
            }
            
            logger.info("Model Evaluation Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Print classification report
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            # Print confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nConfusion Matrix:\n{cm}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, file_path: str) -> None:
        """
        Save trained model to file.
        
        Args:
            file_path: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, file_path)
            
            # Save feature columns
            feature_path = file_path.parent / f"{file_path.stem}_features.joblib"
            joblib.dump(self.feature_columns, feature_path)
            
            logger.info(f"Model saved to {file_path}")
            logger.info(f"Feature columns saved to {feature_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to processed data file"
    )
    parser.add_argument(
        "--target-column", type=str, required=True,
        help="Name of target column"
    )
    parser.add_argument(
        "--model-type", type=str, default="xgboost",
        choices=["xgboost", "lightgbm"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory to save model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data for testing"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2,
        help="Proportion of training data for validation"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(model_type=args.model_type)
        
        # Load data
        df = trainer.load_processed_data(args.data_path)
        
        # Prepare data
        data_splits = trainer.prepare_data(
            df, args.target_column,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        # Train model
        trainer.train(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val']
        )
        
        # Evaluate model
        metrics = trainer.evaluate(
            data_splits['X_test'], data_splits['y_test']
        )
        
        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"model_{args.model_type}.joblib"
        trainer.save_model(str(model_path))
        
        logger.info(f"\nTraining complete! Model saved to {model_path}")
        logger.info(f"Final AUC Score: {metrics['auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
