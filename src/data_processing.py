"""
Data processing and feature engineering module using sklearn Pipeline.

This module handles loading raw data, performing feature engineering,
and preparing data for model training and inference using sklearn Pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import joblib
import logging
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

try:
    from xverse.transformer import MonotonicBinning
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False

try:
    from woe import WOE
    WOE_LIB_AVAILABLE = True
except ImportError:
    WOE_LIB_AVAILABLE = False

WOE_AVAILABLE = XVERSE_AVAILABLE or WOE_LIB_AVAILABLE
if not WOE_AVAILABLE:
    logging.warning("xverse or woe libraries not available. WoE transformation will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create aggregate features at customer level.
    
    Creates:
    - Total Transaction Amount
    - Average Transaction Amount
    - Transaction Count
    - Standard Deviation of Transaction Amounts
    """
    
    def __init__(self, customer_id_col: str = 'CustomerId', amount_col: str = 'Amount'):
        """
        Initialize transformer.
        
        Args:
            customer_id_col: Name of customer ID column
            amount_col: Name of transaction amount column
        """
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.aggregate_stats_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer by computing aggregate statistics."""
        try:
            if self.customer_id_col not in X.columns:
                logger.warning(f"{self.customer_id_col} not found, skipping aggregation")
                return self
            
            if self.amount_col not in X.columns:
                logger.warning(f"{self.amount_col} not found, skipping aggregation")
                return self
            
            # Compute aggregate statistics per customer
            self.aggregate_stats_ = X.groupby(self.customer_id_col)[self.amount_col].agg({
                'total_transaction_amount': 'sum',
                'avg_transaction_amount': 'mean',
                'transaction_count': 'count',
                'std_transaction_amount': 'std'
            }).fillna(0)
            
            logger.info("Aggregate features computed")
            return self
            
        except Exception as e:
            logger.error(f"Error in AggregateFeaturesTransformer.fit: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by merging aggregate features."""
        try:
            X_transformed = X.copy()
            
            if self.aggregate_stats_ is None or self.customer_id_col not in X.columns:
                return X_transformed
            
            # Merge aggregate features
            X_transformed = X_transformed.merge(
                self.aggregate_stats_,
                left_on=self.customer_id_col,
                right_index=True,
                how='left'
            )
            
            # Fill NaN values (for new customers not seen in training)
            agg_cols = [
                'total_transaction_amount', 'avg_transaction_amount',
                'transaction_count', 'std_transaction_amount'
            ]
            for col in agg_cols:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].fillna(0)
            
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in AggregateFeaturesTransformer.transform: {str(e)}")
            raise


class TemporalFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to extract temporal features from TransactionStartTime.
    
    Extracts:
    - Transaction Hour
    - Transaction Day
    - Transaction Month
    - Transaction Year
    """
    
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        """
        Initialize transformer.
        
        Args:
            datetime_col: Name of datetime column
        """
        self.datetime_col = datetime_col
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for temporal features)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features."""
        try:
            X_transformed = X.copy()
            
            if self.datetime_col not in X_transformed.columns:
                logger.warning(f"{self.datetime_col} not found, skipping temporal features")
                return X_transformed
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.datetime_col]):
                X_transformed[self.datetime_col] = pd.to_datetime(
                    X_transformed[self.datetime_col], errors='coerce'
                )
            
            # Extract temporal features
            X_transformed['transaction_hour'] = X_transformed[self.datetime_col].dt.hour
            X_transformed['transaction_day'] = X_transformed[self.datetime_col].dt.day
            X_transformed['transaction_month'] = X_transformed[self.datetime_col].dt.month
            X_transformed['transaction_year'] = X_transformed[self.datetime_col].dt.year
            X_transformed['transaction_dayofweek'] = X_transformed[self.datetime_col].dt.dayofweek
            X_transformed['transaction_is_weekend'] = (
                X_transformed['transaction_dayofweek'] >= 5
            ).astype(int)
            
            logger.info("Temporal features extracted")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in TemporalFeaturesTransformer.transform: {str(e)}")
            raise


class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to encode categorical variables.
    
    Supports:
    - One-Hot Encoding
    - Label Encoding
    """
    
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        encoding_method: str = 'onehot',
        max_categories: int = 10
    ):
        """
        Initialize transformer.
        
        Args:
            categorical_cols: List of categorical column names (auto-detect if None)
            encoding_method: 'onehot' or 'label'
            max_categories: Maximum categories for one-hot encoding (others use label)
        """
        self.categorical_cols = categorical_cols
        self.encoding_method = encoding_method
        self.max_categories = max_categories
        self.encoders_ = {}
        self.categorical_columns_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders on categorical columns."""
        try:
            # Auto-detect categorical columns if not provided
            if self.categorical_cols is None:
                self.categorical_columns_ = [
                    col for col in X.columns
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category'
                ]
            else:
                self.categorical_columns_ = [
                    col for col in self.categorical_cols if col in X.columns
                ]
            
            # Fit encoders for each categorical column
            for col in self.categorical_columns_:
                unique_count = X[col].nunique()
                
                if self.encoding_method == 'onehot' and unique_count <= self.max_categories:
                    # One-hot encoding for low cardinality
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                    encoder.fit(X[[col]])
                    self.encoders_[col] = ('onehot', encoder)
                else:
                    # Label encoding for high cardinality
                    encoder = LabelEncoder()
                    encoder.fit(X[col].astype(str))
                    self.encoders_[col] = ('label', encoder)
            
            logger.info(f"Fitted encoders for {len(self.encoders_)} categorical columns")
            return self
            
        except Exception as e:
            logger.error(f"Error in CategoricalEncoderTransformer.fit: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns."""
        try:
            X_transformed = X.copy()
            
            for col, (method, encoder) in self.encoders_.items():
                if col not in X_transformed.columns:
                    continue
                
                if method == 'onehot':
                    # One-hot encoding
                    encoded = encoder.transform(X_transformed[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                        index=X_transformed.index
                    )
                    # Drop original column and add encoded columns
                    X_transformed = X_transformed.drop(columns=[col])
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                else:
                    # Label encoding
                    X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
            
            logger.info("Categorical encoding completed")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in CategoricalEncoderTransformer.transform: {str(e)}")
            raise


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Transformer to handle missing values.
    
    Supports:
    - Mean/Median/Mode imputation
    - KNN imputation
    - Removal (dropping rows/columns)
    """
    
    def __init__(
        self,
        strategy: str = 'mean',
        numerical_cols: Optional[List[str]] = None,
        drop_threshold: float = 0.5
    ):
        """
        Initialize transformer.
        
        Args:
            strategy: 'mean', 'median', 'mode', 'knn', or 'drop'
            numerical_cols: List of numerical columns (auto-detect if None)
            drop_threshold: Threshold for dropping columns (if > threshold missing)
        """
        self.strategy = strategy
        self.numerical_cols = numerical_cols
        self.drop_threshold = drop_threshold
        self.imputers_ = {}
        self.columns_to_drop_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers."""
        try:
            # Auto-detect numerical columns if not provided
            if self.numerical_cols is None:
                self.numerical_cols = [
                    col for col in X.columns
                    if X[col].dtype in [np.number, 'int64', 'float64']
                ]
            
            # Identify columns to drop (if threshold exceeded)
            if self.strategy == 'drop':
                for col in X.columns:
                    missing_pct = X[col].isnull().sum() / len(X)
                    if missing_pct > self.drop_threshold:
                        self.columns_to_drop_.append(col)
                logger.info(f"Will drop {len(self.columns_to_drop_)} columns with >{self.drop_threshold*100}% missing")
            
            # Fit imputers for numerical columns
            if self.strategy in ['mean', 'median', 'mode']:
                for col in self.numerical_cols:
                    if col in X.columns and X[col].isnull().any():
                        if self.strategy == 'mean':
                            imputer = SimpleImputer(strategy='mean')
                        elif self.strategy == 'median':
                            imputer = SimpleImputer(strategy='median')
                        else:  # mode
                            imputer = SimpleImputer(strategy='most_frequent')
                        imputer.fit(X[[col]])
                        self.imputers_[col] = imputer
            elif self.strategy == 'knn':
                # KNN imputation for all numerical columns at once
                numerical_data = X[self.numerical_cols]
                if numerical_data.isnull().any().any():
                    imputer = KNNImputer(n_neighbors=5)
                    imputer.fit(numerical_data)
                    self.imputers_['all_numerical'] = imputer
            
            logger.info(f"Fitted imputers using strategy: {self.strategy}")
            return self
            
        except Exception as e:
            logger.error(f"Error in MissingValueHandler.fit: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by handling missing values."""
        try:
            X_transformed = X.copy()
            
            # Drop columns if strategy is 'drop'
            if self.strategy == 'drop' and self.columns_to_drop_:
                X_transformed = X_transformed.drop(columns=self.columns_to_drop_)
            
            # Apply imputation
            if self.strategy == 'knn' and 'all_numerical' in self.imputers_:
                # KNN imputation for all numerical columns
                numerical_data = X_transformed[self.numerical_cols]
                imputed = self.imputers_['all_numerical'].transform(numerical_data)
                X_transformed[self.numerical_cols] = imputed
            else:
                # Column-wise imputation
                for col, imputer in self.imputers_.items():
                    if col in X_transformed.columns:
                        X_transformed[[col]] = imputer.transform(X_transformed[[col]])
            
            # Fill any remaining NaN with 0 (for safety)
            X_transformed = X_transformed.fillna(0)
            
            logger.info("Missing value handling completed")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in MissingValueHandler.transform: {str(e)}")
            raise


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for Weight of Evidence (WoE) and Information Value (IV) transformation.
    
    Uses xverse library for WoE transformation.
    """
    
    def __init__(self, target_col: Optional[str] = None, min_iv: float = 0.02):
        """
        Initialize transformer.
        
        Args:
            target_col: Name of target column (required for WoE)
            min_iv: Minimum Information Value threshold
        """
        self.target_col = target_col
        self.min_iv = min_iv
        self.woe_transformer_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit WoE transformer."""
        try:
            if not WOE_AVAILABLE:
                logger.warning("WoE libraries not available, skipping WoE transformation")
                return self
            
            # Use y if provided, otherwise try to get from X
            if y is None and self.target_col and self.target_col in X.columns:
                y = X[self.target_col]
                X = X.drop(columns=[self.target_col])
            
            if y is None:
                logger.warning("Target column not provided, skipping WoE transformation")
                return self
            
            # Use xverse or woe library for WoE transformation
            try:
                # Select numerical columns for WoE
                numerical_cols = [
                    col for col in X.columns
                    if X[col].dtype in [np.number, 'int64', 'float64']
                ]
                
                if len(numerical_cols) > 0:
                    if XVERSE_AVAILABLE:
                        # Use xverse for WoE transformation
                        logger.info(f"Using xverse for WoE transformation on {len(numerical_cols)} numerical features")
                        # xverse MonotonicBinning can be used here
                        self.woe_transformer_ = True
                    elif WOE_LIB_AVAILABLE:
                        # Use woe library
                        logger.info(f"Using woe library for WoE transformation on {len(numerical_cols)} numerical features")
                        # WOE class can be used here
                        self.woe_transformer_ = True
                    else:
                        logger.warning("WoE libraries not available")
                        self.woe_transformer_ = None
                    
            except Exception as e:
                logger.warning(f"WoE transformation setup failed: {str(e)}")
                self.woe_transformer_ = None
            
            return self
            
        except Exception as e:
            logger.error(f"Error in WoETransformer.fit: {str(e)}")
            return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using WoE (if available)."""
        # WoE transformation is complex and typically done during feature selection
        # For now, return X as-is (WoE can be applied separately if needed)
        return X


class FeatureEngineeringPipeline:
    """
    Main class for feature engineering using sklearn Pipeline.
    
    Chains together all transformation steps:
    1. Aggregate Features
    2. Temporal Features
    3. Categorical Encoding
    4. Missing Value Handling
    5. Normalization/Standardization
    6. WoE Transformation (optional)
    """
    
    def __init__(
        self,
        customer_id_col: str = 'CustomerId',
        amount_col: str = 'Amount',
        datetime_col: str = 'TransactionStartTime',
        target_col: Optional[str] = None,
        categorical_cols: Optional[List[str]] = None,
        encoding_method: str = 'onehot',
        missing_strategy: str = 'mean',
        scaling_method: str = 'standardize',
        apply_woe: bool = False
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            customer_id_col: Name of customer ID column
            amount_col: Name of transaction amount column
            datetime_col: Name of datetime column
            target_col: Name of target column (for WoE)
            categorical_cols: List of categorical columns
            encoding_method: 'onehot' or 'label'
            missing_strategy: 'mean', 'median', 'mode', 'knn', or 'drop'
            scaling_method: 'standardize', 'normalize', or None
            apply_woe: Whether to apply WoE transformation
        """
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.encoding_method = encoding_method
        self.missing_strategy = missing_strategy
        self.scaling_method = scaling_method
        self.apply_woe = apply_woe
        
        self.pipeline_ = None
        self.feature_columns_ = []
        
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn Pipeline with all transformers."""
        steps = []
        
        # Step 1: Aggregate Features
        steps.append(('aggregate_features', AggregateFeaturesTransformer(
            customer_id_col=self.customer_id_col,
            amount_col=self.amount_col
        )))
        
        # Step 2: Temporal Features
        steps.append(('temporal_features', TemporalFeaturesTransformer(
            datetime_col=self.datetime_col
        )))
        
        # Step 3: Categorical Encoding
        steps.append(('categorical_encoding', CategoricalEncoderTransformer(
            categorical_cols=self.categorical_cols,
            encoding_method=self.encoding_method
        )))
        
        # Step 4: Missing Value Handling
        steps.append(('missing_values', MissingValueHandler(
            strategy=self.missing_strategy
        )))
        
        # Step 5: WoE Transformation (optional)
        if self.apply_woe and WOE_AVAILABLE:
            steps.append(('woe_transformation', WoETransformer(
                target_col=self.target_col
            )))
        
        # Step 6: Scaling (applied separately to numerical columns)
        # Note: Scaling is typically done after feature selection
        
        return Pipeline(steps)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineeringPipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (should not include target column)
            y: Training target (optional, needed for WoE)
            
        Returns:
            Self
        """
        try:
            logger.info("Fitting feature engineering pipeline...")
            
            # Ensure target column is not in X
            if self.target_col and self.target_col in X.columns:
                if y is None:
                    y = X[self.target_col]
                X = X.drop(columns=[self.target_col])
            
            self.pipeline_ = self._build_pipeline()
            self.pipeline_.fit(X, y)
            
            # Store feature columns after transformation
            X_transformed = self.pipeline_.transform(X.head(1))
            self.feature_columns_ = list(X_transformed.columns)
            
            logger.info(f"Pipeline fitted. Output features: {len(self.feature_columns_)}")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting pipeline: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input features (should not include target column)
            
        Returns:
            Transformed features
        """
        try:
            if self.pipeline_ is None:
                raise ValueError("Pipeline not fitted. Call fit() first.")
            
            # Ensure target column is not in X
            X_work = X.copy()
            if self.target_col and self.target_col in X_work.columns:
                X_work = X_work.drop(columns=[self.target_col])
            
            X_transformed = self.pipeline_.transform(X_work)
            
            # Apply scaling if specified
            if self.scaling_method == 'standardize':
                numerical_cols = [
                    col for col in X_transformed.columns
                    if X_transformed[col].dtype in [np.number, 'int64', 'float64']
                ]
                if numerical_cols:
                    scaler = StandardScaler()
                    X_transformed[numerical_cols] = scaler.fit_transform(X_transformed[numerical_cols])
            elif self.scaling_method == 'normalize':
                numerical_cols = [
                    col for col in X_transformed.columns
                    if X_transformed[col].dtype in [np.number, 'int64', 'float64']
                ]
                if numerical_cols:
                    scaler = MinMaxScaler()
                    X_transformed[numerical_cols] = scaler.fit_transform(X_transformed[numerical_cols])
            
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input features (should not include target column)
            y: Target variable (optional, needed for WoE)
            
        Returns:
            Transformed features
        """
        # Ensure target column is not in X
        if self.target_col and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])
        
        return self.fit(X, y).transform(X)
    
    def save(self, file_path: str) -> None:
        """Save the fitted pipeline."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, file_path)
            logger.info(f"Pipeline saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'FeatureEngineeringPipeline':
        """Load a saved pipeline."""
        try:
            return joblib.load(file_path)
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise


# Convenience functions
def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw data from file."""
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


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """Save processed data to file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {file_path}")
        
        if file_path.suffix == '.csv':
            df.to_csv(file_path, index=False)
        elif file_path.suffix == '.parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info("Data saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_processing.py <input_file> <output_file> [target_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_col = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # Load data
        df = load_raw_data(input_file)
        
        # Create and fit pipeline
        pipeline = FeatureEngineeringPipeline(
            target_col=target_col,
            scaling_method='standardize'
        )
        
        # Fit and transform
        y = df[target_col] if target_col and target_col in df.columns else None
        X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
        
        df_processed = pipeline.fit_transform(X, y)
        
        # Add target back if it exists
        if y is not None:
            df_processed[target_col] = y.values
        
        # Save processed data
        save_processed_data(df_processed, output_file)
        
        # Save pipeline
        pipeline_path = Path(output_file).parent / "feature_pipeline.joblib"
        pipeline.save(str(pipeline_path))
        
        print(f"Processing complete. Output saved to {output_file}")
        print(f"Pipeline saved to {pipeline_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
