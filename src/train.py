"""
Model training script with MLflow tracking.

This module handles:
- Training multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning (Grid Search, Random Search)
- MLflow experiment tracking
- Model evaluation with comprehensive metrics
- Model registry
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import sys
import argparse
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

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


class ModelTrainer:
    """Class for training and evaluating credit risk models with MLflow tracking."""
    
    def __init__(
        self,
        experiment_name: str = "credit-risk-modeling",
        random_state: int = 42,
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            experiment_name: MLflow experiment name
            random_state: Random state for reproducibility
            mlflow_tracking_uri: MLflow tracking URI (default: local)
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.mlflow_tracking_uri = mlflow_tracking_uri or "file:./mlruns"
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Error setting MLflow experiment: {str(e)}")
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.feature_columns = []
        
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """Load processed data for training."""
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
    ) -> Dict[str, Any]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with data splits
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Select feature columns
            exclude_cols = [
                target_column, 'TransactionId', 'BatchId', 'AccountId',
                'SubscriptionId', 'CustomerId', 'TransactionStartTime',
                'last_transaction_date'
            ]
            feature_columns = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in [np.number, 'int64', 'float64']
            ]
            
            X = df[feature_columns].fillna(df[feature_columns].median())
            y = df[target_column]
            
            self.feature_columns = feature_columns
            
            # Split data with random_state for reproducibility
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=y_train
            )
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
            
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
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True
    ) -> LogisticRegression:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        
        with mlflow.start_run(run_name="LogisticRegression", nested=True):
            if use_grid_search:
                # Grid Search
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
                
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc',
                    n_jobs=-1, random_state=self.random_state
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                
                mlflow.log_params(grid_search.best_params_)
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model = LogisticRegression(
                    C=1.0, penalty='l2', solver='lbfgs',
                    random_state=self.random_state, max_iter=1000
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            self._log_metrics(metrics, "val")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['LogisticRegression'] = model
            return model
    
    def train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True
    ) -> DecisionTreeClassifier:
        """Train Decision Tree model."""
        logger.info("Training Decision Tree...")
        
        with mlflow.start_run(run_name="DecisionTree", nested=True):
            if use_grid_search:
                # Grid Search
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
                
                model = DecisionTreeClassifier(random_state=self.random_state)
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc',
                    n_jobs=-1, random_state=self.random_state
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                
                mlflow.log_params(grid_search.best_params_)
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model = DecisionTreeClassifier(
                    max_depth=5, random_state=self.random_state
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            self._log_metrics(metrics, "val")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['DecisionTree'] = model
            return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_random_search: bool = True
    ) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        
        with mlflow.start_run(run_name="RandomForest", nested=True):
            if use_random_search:
                # Random Search
                param_dist = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                
                model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
                random_search = RandomizedSearchCV(
                    model, param_dist, n_iter=20, cv=5, scoring='roc_auc',
                    n_jobs=-1, random_state=self.random_state
                )
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                
                mlflow.log_params(random_search.best_params_)
                logger.info(f"Best parameters: {random_search.best_params_}")
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    random_state=self.random_state, n_jobs=-1
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            self._log_metrics(metrics, "val")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['RandomForest'] = model
            return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        with mlflow.start_run(run_name="XGBoost", nested=True):
            if use_grid_search:
                # Grid Search
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.8, 1.0]
                }
                
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc',
                    n_jobs=-1, random_state=self.random_state
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                
                mlflow.log_params(grid_search.best_params_)
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            self._log_metrics(metrics, "val")
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            self.models['XGBoost'] = model
            return model
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_random_search: bool = True
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")
        
        with mlflow.start_run(run_name="LightGBM", nested=True):
            if use_random_search:
                # Random Search
                param_dist = {
                    'num_leaves': [31, 50, 70],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
                
                model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                random_search = RandomizedSearchCV(
                    model, param_dist, n_iter=20, cv=5, scoring='roc_auc',
                    n_jobs=-1, random_state=self.random_state
                )
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                
                mlflow.log_params(random_search.best_params_)
                logger.info(f"Best parameters: {random_search.best_params_}")
            else:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            self._log_metrics(metrics, "val")
            
            # Log model
            mlflow.lightgbm.log_model(model, "model")
            
            self.models['LightGBM'] = model
            return model
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to MLflow."""
        for metric_name, metric_value in metrics.items():
            metric_key = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(metric_key, metric_value)
    
    def train_all_models(
        self,
        data_splits: Dict[str, Any],
        models_to_train: Optional[List[str]] = None,
        use_hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Train all models and track in MLflow.
        
        Args:
            data_splits: Dictionary with data splits
            models_to_train: List of models to train (None = all)
            use_hyperparameter_tuning: Whether to use hyperparameter tuning
            
        Returns:
            Dictionary with all trained models
        """
        if models_to_train is None:
            models_to_train = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM']
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        with mlflow.start_run(run_name=f"All_Models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("models_trained", ", ".join(models_to_train))
            mlflow.log_param("hyperparameter_tuning", use_hyperparameter_tuning)
            
            # Train each model
            for model_name in models_to_train:
                try:
                    if model_name == 'LogisticRegression':
                        self.train_logistic_regression(
                            X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                        )
                    elif model_name == 'DecisionTree':
                        self.train_decision_tree(
                            X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                        )
                    elif model_name == 'RandomForest':
                        self.train_random_forest(
                            X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                        )
                    elif model_name == 'XGBoost':
                        self.train_xgboost(
                            X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                        )
                    elif model_name == 'LightGBM':
                        self.train_lightgbm(
                            X_train, y_train, X_val, y_val, use_hyperparameter_tuning
                        )
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Find best model
            self._identify_best_model(data_splits['X_val'], data_splits['y_val'])
            
            return self.models
    
    def _identify_best_model(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Identify best model based on ROC-AUC score."""
        best_score = 0.0
        best_model_name = None
        
        for model_name, model in self.models.items():
            metrics = self._evaluate_model(model, X_val, y_val)
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.best_score = best_score
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        logger.info(f"{'='*80}")
    
    def evaluate_best_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate best model on test set."""
        if self.best_model is None:
            raise ValueError("No best model identified. Train models first.")
        
        logger.info(f"\nEvaluating best model ({self.best_model_name}) on test set...")
        
        metrics = self._evaluate_model(self.best_model, X_test, y_test)
        
        logger.info("\nTest Set Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Classification report
        y_pred = self.best_model.predict(X_test)
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return metrics
    
    def register_best_model(
        self,
        model_name: str = "credit-risk-model",
        stage: str = "Production"
    ):
        """Register best model in MLflow Model Registry."""
        if self.best_model is None:
            raise ValueError("No best model to register. Train models first.")
        
        try:
            logger.info(f"Registering {self.best_model_name} as {model_name}...")
            
            # Get the latest run for the best model
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{self.best_model_name}'",
                order_by=["metrics.val_roc_auc DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{run_id}/model"
                
                # Register model
                mlflow.register_model(model_uri, model_name)
                
                # Transition to stage
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage=stage
                )
                
                logger.info(f"✅ Model registered: {model_name} (version {latest_version}, stage: {stage})")
            else:
                logger.warning("Could not find run for best model. Skipping registration.")
                
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def save_model(self, file_path: str):
        """Save best model to file."""
        if self.best_model is None:
            raise ValueError("No model to save. Train models first.")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, file_path)
        
        # Save feature columns
        feature_path = file_path.parent / f"{file_path.stem}_features.joblib"
        joblib.dump(self.feature_columns, feature_path)
        
        logger.info(f"Model saved to {file_path}")
        logger.info(f"Feature columns saved to {feature_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow tracking")
    parser.add_argument("--data-path", type=str, required=True, help="Path to processed data file")
    parser.add_argument("--target-column", type=str, required=True, help="Name of target column")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       choices=['LogisticRegression', 'DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM'],
                       help="Models to train (default: all)")
    parser.add_argument("--no-tuning", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model")
    parser.add_argument("--register-model", type=str, default=None, help="Register best model with this name")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(random_state=args.random_state)
        
        # Load data
        df = trainer.load_processed_data(args.data_path)
        
        # Prepare data
        data_splits = trainer.prepare_data(df, args.target_column)
        
        # Train all models
        trainer.train_all_models(
            data_splits,
            models_to_train=args.models,
            use_hyperparameter_tuning=not args.no_tuning
        )
        
        # Evaluate best model
        test_metrics = trainer.evaluate_best_model(
            data_splits['X_test'], data_splits['y_test']
        )
        
        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"model_{trainer.best_model_name.lower()}.joblib"
        trainer.save_model(str(model_path))
        
        # Register model if requested
        if args.register_model:
            trainer.register_best_model(args.register_model)
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"Best model: {trainer.best_model_name}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"\nView MLflow UI: mlflow ui --backend-store-uri {trainer.mlflow_tracking_uri}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
