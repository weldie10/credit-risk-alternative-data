"""
Example script demonstrating how to use the FeatureEngineeringPipeline.

This script shows how to:
1. Load raw data
2. Create and fit the feature engineering pipeline
3. Transform data
4. Save processed data and pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    FeatureEngineeringPipeline,
    load_raw_data,
    save_processed_data
)
import pandas as pd


def main():
    """Example usage of FeatureEngineeringPipeline."""
    
    # Example 1: Basic usage
    print("=" * 80)
    print("Example 1: Basic Feature Engineering Pipeline")
    print("=" * 80)
    
    # Load data (replace with your data path)
    data_path = "../data/raw/data.csv"
    
    try:
        # Load raw data
        df = load_raw_data(data_path)
        print(f"\nLoaded data: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        # Create pipeline
        pipeline = FeatureEngineeringPipeline(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            target_col='FraudResult',  # Example target column
            encoding_method='onehot',
            missing_strategy='mean',
            scaling_method='standardize',
            apply_woe=False  # Set to True if WoE libraries are available
        )
        
        # Separate target if it exists
        target_col = 'FraudResult'
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
        
        # Fit and transform
        print("\nFitting and transforming data...")
        X_transformed = pipeline.fit_transform(X, y)
        
        print(f"\nTransformed data shape: {X_transformed.shape}")
        print(f"Number of features: {len(pipeline.feature_columns_)}")
        print(f"\nSample feature columns: {pipeline.feature_columns_[:10]}")
        
        # Add target back if it exists
        if y is not None:
            X_transformed[target_col] = y.values
        
        # Save processed data
        output_path = "../data/processed/processed_data.csv"
        save_processed_data(X_transformed, output_path)
        print(f"\nProcessed data saved to: {output_path}")
        
        # Save pipeline
        pipeline_path = "../models/feature_pipeline.joblib"
        pipeline.save(pipeline_path)
        print(f"Pipeline saved to: {pipeline_path}")
        
        print("\n✅ Feature engineering complete!")
        
    except FileNotFoundError:
        print(f"\n⚠️  Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file location.")
        
        # Show example with sample data
        print("\n" + "=" * 80)
        print("Example 2: Using Sample Data")
        print("=" * 80)
        
        # Create sample data
        sample_data = {
            'CustomerId': [1, 1, 2, 2, 3, 3],
            'Amount': [100, 200, 150, 250, 300, 400],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=6, freq='D'),
            'ProductCategory': ['A', 'B', 'A', 'C', 'B', 'A'],
            'ChannelId': ['web', 'mobile', 'web', 'mobile', 'web', 'mobile'],
            'FraudResult': [0, 0, 1, 0, 0, 1]
        }
        df_sample = pd.DataFrame(sample_data)
        
        print(f"\nSample data shape: {df_sample.shape}")
        print(df_sample.head())
        
        # Create and fit pipeline
        pipeline = FeatureEngineeringPipeline(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            target_col='FraudResult',
            encoding_method='onehot',
            missing_strategy='mean',
            scaling_method='standardize'
        )
        
        y_sample = df_sample['FraudResult']
        X_sample = df_sample.drop(columns=['FraudResult'])
        
        X_transformed = pipeline.fit_transform(X_sample, y_sample)
        
        print(f"\n✅ Transformed data shape: {X_transformed.shape}")
        print(f"✅ Number of features: {len(pipeline.feature_columns_)}")
        print(f"\nTransformed columns: {list(X_transformed.columns)}")
        print("\n" + "=" * 80)
        print("Pipeline Summary:")
        print("=" * 80)
        print("1. ✅ Aggregate Features (Total, Average, Count, Std Dev)")
        print("2. ✅ Temporal Features (Hour, Day, Month, Year)")
        print("3. ✅ Categorical Encoding (One-Hot/Label)")
        print("4. ✅ Missing Value Handling (Mean/Median/Mode/KNN)")
        print("5. ✅ Normalization/Standardization")
        print("6. ✅ WoE Transformation (optional)")
        print("\nAll transformations are chained using sklearn Pipeline!")


if __name__ == "__main__":
    main()

