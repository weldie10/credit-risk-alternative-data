"""
Unit tests for data processing and feature engineering.

Tests helper functions and transformations in the data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    FeatureEngineeringPipeline,
    AggregateFeaturesTransformer,
    TemporalFeaturesTransformer,
    CategoricalEncoderTransformer,
    MissingValueHandler
)
from src.target_engineering import ProxyTargetEngineer


class TestAggregateFeaturesTransformer:
    """Test cases for AggregateFeaturesTransformer."""
    
    def test_aggregate_features_creates_expected_columns(self):
        """Test that aggregate features transformer creates expected columns."""
        # Create sample data
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 150, 250, 300],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        
        # Initialize and fit transformer
        transformer = AggregateFeaturesTransformer(
            customer_id_col='CustomerId',
            amount_col='Amount'
        )
        transformer.fit(df)
        
        # Transform
        df_transformed = transformer.transform(df)
        
        # Check expected columns exist
        expected_cols = [
            'total_transaction_amount',
            'avg_transaction_amount',
            'transaction_count',
            'std_transaction_amount'
        ]
        
        for col in expected_cols:
            assert col in df_transformed.columns, f"Expected column {col} not found"
    
    def test_aggregate_features_calculates_correct_values(self):
        """Test that aggregate features calculate correct values."""
        # Create sample data with known values
        df = pd.DataFrame({
            'CustomerId': [1, 1, 1, 2, 2],
            'Amount': [100, 200, 300, 150, 250],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        
        transformer = AggregateFeaturesTransformer(
            customer_id_col='CustomerId',
            amount_col='Amount'
        )
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Check customer 1 values
        customer_1_data = df_transformed[df_transformed['CustomerId'] == 1].iloc[0]
        assert customer_1_data['total_transaction_amount'] == 600, "Total amount incorrect"
        assert customer_1_data['avg_transaction_amount'] == 200, "Average amount incorrect"
        assert customer_1_data['transaction_count'] == 3, "Transaction count incorrect"


class TestTemporalFeaturesTransformer:
    """Test cases for TemporalFeaturesTransformer."""
    
    def test_temporal_features_creates_expected_columns(self):
        """Test that temporal features transformer creates expected columns."""
        # Create sample data
        df = pd.DataFrame({
            'TransactionStartTime': pd.date_range('2024-01-15 10:30:00', periods=5, freq='6H'),
            'Amount': [100, 200, 150, 250, 300]
        })
        
        transformer = TemporalFeaturesTransformer(datetime_col='TransactionStartTime')
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Check expected columns exist
        expected_cols = [
            'transaction_hour',
            'transaction_day',
            'transaction_month',
            'transaction_year',
            'transaction_dayofweek',
            'transaction_is_weekend'
        ]
        
        for col in expected_cols:
            assert col in df_transformed.columns, f"Expected column {col} not found"
    
    def test_temporal_features_hour_extraction(self):
        """Test that hour is correctly extracted."""
        df = pd.DataFrame({
            'TransactionStartTime': pd.to_datetime(['2024-01-15 14:30:00', '2024-01-15 22:45:00']),
            'Amount': [100, 200]
        })
        
        transformer = TemporalFeaturesTransformer(datetime_col='TransactionStartTime')
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        assert df_transformed.iloc[0]['transaction_hour'] == 14, "Hour extraction incorrect"
        assert df_transformed.iloc[1]['transaction_hour'] == 22, "Hour extraction incorrect"


class TestCategoricalEncoderTransformer:
    """Test cases for CategoricalEncoderTransformer."""
    
    def test_onehot_encoding_creates_binary_columns(self):
        """Test that one-hot encoding creates binary columns."""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Amount': [100, 200, 150, 250, 300]
        })
        
        transformer = CategoricalEncoderTransformer(
            categorical_cols=['Category'],
            encoding_method='onehot',
            max_categories=10
        )
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Check that original column is removed
        assert 'Category' not in df_transformed.columns, "Original categorical column should be removed"
        
        # Check that encoded columns exist (one-hot creates n-1 columns for n categories)
        encoded_cols = [col for col in df_transformed.columns if col.startswith('Category_')]
        assert len(encoded_cols) > 0, "One-hot encoded columns should be created"
    
    def test_label_encoding_assigns_integers(self):
        """Test that label encoding assigns unique integers."""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Amount': [100, 200, 150, 250, 300]
        })
        
        transformer = CategoricalEncoderTransformer(
            categorical_cols=['Category'],
            encoding_method='label',
            max_categories=2  # Force label encoding
        )
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Check that column still exists but with integer values
        assert 'Category' in df_transformed.columns, "Label encoded column should exist"
        assert df_transformed['Category'].dtype in [np.int64, np.int32, int], "Category should be integer"


class TestMissingValueHandler:
    """Test cases for MissingValueHandler."""
    
    def test_mean_imputation_fills_missing_values(self):
        """Test that mean imputation fills missing values."""
        df = pd.DataFrame({
            'Amount': [100, 200, np.nan, 400, 500],
            'Value': [100, 200, 300, 400, 500]
        })
        
        transformer = MissingValueHandler(strategy='mean')
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Check that NaN values are filled
        assert df_transformed['Amount'].isna().sum() == 0, "Missing values should be filled"
        assert df_transformed['Amount'].mean() > 0, "Mean imputation should produce valid values"
    
    def test_median_imputation_fills_missing_values(self):
        """Test that median imputation fills missing values."""
        df = pd.DataFrame({
            'Amount': [100, 200, np.nan, 400, 500],
            'Value': [100, 200, 300, 400, 500]
        })
        
        transformer = MissingValueHandler(strategy='median')
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        assert df_transformed['Amount'].isna().sum() == 0, "Missing values should be filled"


class TestFeatureEngineeringPipeline:
    """Test cases for FeatureEngineeringPipeline."""
    
    def test_pipeline_returns_dataframe_with_features(self):
        """Test that pipeline returns DataFrame with engineered features."""
        # Create sample data
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 150, 250, 300],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D'),
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Value': [100, 200, 150, 250, 300]
        })
        
        pipeline = FeatureEngineeringPipeline(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            scaling_method=None  # Skip scaling for faster test
        )
        
        df_transformed = pipeline.fit_transform(df)
        
        # Check that output is DataFrame
        assert isinstance(df_transformed, pd.DataFrame), "Output should be DataFrame"
        
        # Check that features are added
        assert len(df_transformed.columns) > len(df.columns), "Features should be added"
    
    def test_pipeline_handles_missing_values(self):
        """Test that pipeline handles missing values correctly."""
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, np.nan, 150, 250, 300],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D'),
            'Value': [100, 200, 150, 250, 300]
        })
        
        pipeline = FeatureEngineeringPipeline(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            missing_strategy='mean',
            scaling_method=None
        )
        
        df_transformed = pipeline.fit_transform(df)
        
        # Check that missing values are handled
        assert df_transformed['Amount'].isna().sum() == 0, "Missing values should be handled"


class TestProxyTargetEngineer:
    """Test cases for ProxyTargetEngineer."""
    
    def test_rfm_calculation_creates_expected_columns(self):
        """Test that RFM calculation creates expected columns."""
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 150, 250, 300],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        
        engineer = ProxyTargetEngineer(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime'
        )
        
        rfm_data = engineer.calculate_rfm_metrics(df)
        
        # Check expected columns
        expected_cols = ['Recency', 'Frequency', 'Monetary']
        for col in expected_cols:
            assert col in rfm_data.columns, f"Expected column {col} not found"
    
    def test_target_variable_is_binary(self):
        """Test that target variable is binary (0 or 1)."""
        df = pd.DataFrame({
            'CustomerId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Amount': [100, 200, 300, 150, 250, 350, 50, 100, 150],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=9, freq='D')
        })
        
        engineer = ProxyTargetEngineer(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            n_clusters=3,
            random_state=42
        )
        
        df_with_target = engineer.fit_transform(df)
        
        # Check that is_high_risk is binary
        assert 'is_high_risk' in df_with_target.columns, "is_high_risk column should exist"
        assert set(df_with_target['is_high_risk'].unique()).issubset({0, 1}), "Target should be binary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
