"""
Data processing and feature engineering module.

This module handles loading raw data, performing feature engineering,
and preparing data for model training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Class for processing and engineering features from transaction data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor.
        
        Args:
            config: Optional configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.feature_columns: List[str] = []
        self.scaler = None
        self.feature_encoders: Dict = {}
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            DataFrame containing raw data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
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
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        
        Args:
            df: Raw DataFrame with transaction data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Starting feature engineering")
            df_processed = df.copy()
            
            # Convert TransactionStartTime to datetime if it exists
            if 'TransactionStartTime' in df_processed.columns:
                df_processed['TransactionStartTime'] = pd.to_datetime(
                    df_processed['TransactionStartTime'], errors='coerce'
                )
                df_processed = self._create_temporal_features(df_processed)
            
            # Create RFM features (Recency, Frequency, Monetary)
            df_processed = self._create_rfm_features(df_processed)
            
            # Create customer-level aggregations
            df_processed = self._create_customer_aggregations(df_processed)
            
            # Create spending pattern features
            df_processed = self._create_spending_pattern_features(df_processed)
            
            # Handle outliers in Amount
            df_processed = self._handle_outliers(df_processed)
            
            # Create interaction features
            df_processed = self._create_interaction_features(df_processed)
            
            logger.info(f"Feature engineering complete. Shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from TransactionStartTime."""
        try:
            if 'TransactionStartTime' not in df.columns:
                return df
            
            df['transaction_year'] = df['TransactionStartTime'].dt.year
            df['transaction_month'] = df['TransactionStartTime'].dt.month
            df['transaction_day'] = df['TransactionStartTime'].dt.day
            df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
            df['transaction_hour'] = df['TransactionStartTime'].dt.hour
            df['transaction_is_weekend'] = (df['transaction_dayofweek'] >= 5).astype(int)
            
            return df
        except Exception as e:
            logger.warning(f"Error creating temporal features: {str(e)}")
            return df
    
    def _create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Recency, Frequency, Monetary features at customer level."""
        try:
            if 'CustomerId' not in df.columns:
                logger.warning("CustomerId not found, skipping RFM features")
                return df
            
            # Calculate customer-level aggregations
            customer_stats = df.groupby('CustomerId').agg({
                'TransactionStartTime': ['max', 'count'],
                'Amount': ['sum', 'mean', 'std'],
                'Value': ['sum', 'mean']
            }).reset_index()
            
            customer_stats.columns = [
                'CustomerId', 'last_transaction_date', 'transaction_frequency',
                'total_amount', 'avg_amount', 'std_amount',
                'total_value', 'avg_value'
            ]
            
            # Calculate recency (days since last transaction)
            if 'TransactionStartTime' in df.columns:
                max_date = df['TransactionStartTime'].max()
                customer_stats['recency_days'] = (
                    max_date - customer_stats['last_transaction_date']
                ).dt.days
            
            # Merge back to original dataframe
            df = df.merge(customer_stats, on='CustomerId', how='left')
            
            return df
        except Exception as e:
            logger.warning(f"Error creating RFM features: {str(e)}")
            return df
    
    def _create_customer_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregation features."""
        try:
            if 'CustomerId' not in df.columns:
                return df
            
            # Count unique values per customer
            customer_diversity = df.groupby('CustomerId').agg({
                'ProductCategory': 'nunique',
                'ProviderId': 'nunique',
                'ChannelId': 'nunique',
                'ProductId': 'nunique'
            }).reset_index()
            
            customer_diversity.columns = [
                'CustomerId', 'unique_categories', 'unique_providers',
                'unique_channels', 'unique_products'
            ]
            
            # Fraud-related aggregations
            if 'FraudResult' in df.columns:
                fraud_stats = df.groupby('CustomerId').agg({
                    'FraudResult': ['sum', 'mean']
                }).reset_index()
                fraud_stats.columns = ['CustomerId', 'fraud_count', 'fraud_rate']
                customer_diversity = customer_diversity.merge(fraud_stats, on='CustomerId', how='left')
            
            # Merge back
            df = df.merge(customer_diversity, on='CustomerId', how='left')
            
            return df
        except Exception as e:
            logger.warning(f"Error creating customer aggregations: {str(e)}")
            return df
    
    def _create_spending_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spending pattern features."""
        try:
            # Refund/credit ratio
            if 'Amount' in df.columns and 'CustomerId' in df.columns:
                refund_stats = df.groupby('CustomerId').apply(
                    lambda x: (x['Amount'] < 0).sum() / len(x) if len(x) > 0 else 0
                ).reset_index(name='refund_ratio')
                refund_stats.columns = ['CustomerId', 'refund_ratio']
                df = df.merge(refund_stats, on='CustomerId', how='left')
            
            # Spending volatility (coefficient of variation)
            if 'CustomerId' in df.columns and 'Amount' in df.columns:
                volatility = df.groupby('CustomerId')['Amount'].apply(
                    lambda x: x.std() / x.mean() if x.mean() != 0 else 0
                ).reset_index(name='spending_volatility')
                volatility.columns = ['CustomerId', 'spending_volatility']
                df = df.merge(volatility, on='CustomerId', how='left')
            
            # Transaction amount bins
            if 'Amount' in df.columns:
                df['amount_bin'] = pd.cut(
                    df['Amount'],
                    bins=[-np.inf, 0, 1000, 5000, 50000, np.inf],
                    labels=['negative', 'small', 'medium', 'large', 'very_large']
                )
            
            return df
        except Exception as e:
            logger.warning(f"Error creating spending pattern features: {str(e)}")
            return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers by capping at percentiles."""
        try:
            if 'Amount' in df.columns:
                # Cap outliers at 1st and 99th percentiles
                lower_bound = df['Amount'].quantile(0.01)
                upper_bound = df['Amount'].quantile(0.99)
                df['Amount_capped'] = df['Amount'].clip(lower=lower_bound, upper=upper_bound)
            
            if 'Value' in df.columns:
                lower_bound = df['Value'].quantile(0.01)
                upper_bound = df['Value'].quantile(0.99)
                df['Value_capped'] = df['Value'].clip(lower=lower_bound, upper=upper_bound)
            
            return df
        except Exception as e:
            logger.warning(f"Error handling outliers: {str(e)}")
            return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        try:
            # Amount × Channel interaction
            if 'Amount' in df.columns and 'ChannelId' in df.columns:
                df['amount_channel_interaction'] = df['Amount'] * df['ChannelId'].astype('category').cat.codes
            
            # ProductCategory × Amount (if both exist)
            if 'ProductCategory' in df.columns and 'Amount' in df.columns:
                category_codes = df['ProductCategory'].astype('category').cat.codes
                df['category_amount_interaction'] = df['Amount'] * category_codes
            
            return df
        except Exception as e:
            logger.warning(f"Error creating interaction features: {str(e)}")
            return df
    
    def preprocess_for_training(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            feature_columns: List of feature column names (if None, auto-select)
            
        Returns:
            Tuple of (X, y) where X is features and y is target
            
        Raises:
            ValueError: If target column is missing
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            if feature_columns is None:
                # Auto-select features (exclude IDs, target, and datetime columns)
                exclude_cols = [
                    target_column, 'TransactionId', 'BatchId', 'AccountId',
                    'SubscriptionId', 'CustomerId', 'TransactionStartTime',
                    'last_transaction_date'
                ]
                feature_columns = [
                    col for col in df.columns
                    if col not in exclude_cols and df[col].dtype in [np.number, 'int64', 'float64']
                ]
            
            # Remove any feature columns that don't exist
            feature_columns = [col for col in feature_columns if col in df.columns]
            
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            self.feature_columns = feature_columns
            logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            file_path: Path to save the processed data
            
        Raises:
            ValueError: If file format is unsupported
        """
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


# Convenience functions for backward compatibility
def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw data from file."""
    processor = DataProcessor()
    return processor.load_raw_data(file_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset."""
    processor = DataProcessor()
    return processor.engineer_features(df)


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """Save processed data to file."""
    processor = DataProcessor()
    processor.save_processed_data(df, file_path)


def preprocess_for_training(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data for model training."""
    processor = DataProcessor()
    return processor.preprocess_for_training(df, target_column, feature_columns)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_processing.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        processor = DataProcessor()
        df_raw = processor.load_raw_data(input_file)
        df_processed = processor.engineer_features(df_raw)
        processor.save_processed_data(df_processed, output_file)
        print(f"Processing complete. Output saved to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
