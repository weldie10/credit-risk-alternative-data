"""
Example script demonstrating proxy target variable engineering.

This script shows how to:
1. Load raw transaction data
2. Calculate RFM metrics
3. Cluster customers using K-Means
4. Identify high-risk customers
5. Create is_high_risk target variable
6. Merge target back into dataset
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.target_engineering import ProxyTargetEngineer, create_proxy_target
from src.data_processing import load_raw_data, save_processed_data
import pandas as pd


def main():
    """Example usage of ProxyTargetEngineer."""
    
    print("=" * 80)
    print("Proxy Target Variable Engineering Example")
    print("=" * 80)
    
    # Example 1: Using convenience function
    print("\nExample 1: Using Convenience Function")
    print("-" * 80)
    
    data_path = "../data/raw/data.csv"
    
    try:
        # Load data
        df = load_raw_data(data_path)
        print(f"\n✅ Loaded data: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")
        
        # Create proxy target using convenience function
        print("\n📊 Creating proxy target variable...")
        df_with_target = create_proxy_target(
            df,
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            snapshot_date=None,  # Use max date
            n_clusters=3,
            random_state=42
        )
        
        print(f"\n✅ Target variable created!")
        print(f"Dataset shape: {df_with_target.shape}")
        print(f"\nTarget distribution:")
        print(df_with_target['is_high_risk'].value_counts())
        print(f"\nTarget percentage:")
        print(df_with_target['is_high_risk'].value_counts(normalize=True) * 100)
        
        # Save
        output_path = "../data/processed/data_with_target.csv"
        save_processed_data(df_with_target, output_path)
        print(f"\n✅ Saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"\n⚠️  Data file not found: {data_path}")
        print("Creating example with sample data...")
        
        # Example 2: Step-by-step with sample data
        print("\n" + "=" * 80)
        print("Example 2: Step-by-Step with Sample Data")
        print("=" * 80)
        
        # Create sample transaction data
        np.random.seed(42)
        n_transactions = 1000
        n_customers = 50
        
        sample_data = {
            'CustomerId': np.random.randint(1, n_customers + 1, n_transactions),
            'Amount': np.random.normal(1000, 500, n_transactions).clip(10, 10000),
            'TransactionStartTime': pd.date_range('2024-01-01', periods=n_transactions, freq='6H'),
            'ProductCategory': np.random.choice(['A', 'B', 'C'], n_transactions),
            'ChannelId': np.random.choice(['web', 'mobile'], n_transactions)
        }
        df_sample = pd.DataFrame(sample_data)
        
        print(f"\nSample data shape: {df_sample.shape}")
        print(f"Unique customers: {df_sample['CustomerId'].nunique()}")
        print(f"\nFirst few rows:")
        print(df_sample.head())
        
        # Initialize engineer
        engineer = ProxyTargetEngineer(
            customer_id_col='CustomerId',
            amount_col='Amount',
            datetime_col='TransactionStartTime',
            snapshot_date=None,
            n_clusters=3,
            random_state=42
        )
        
        # Step 1: Calculate RFM
        print("\n" + "-" * 80)
        print("Step 1: Calculating RFM Metrics")
        print("-" * 80)
        rfm_data = engineer.calculate_rfm_metrics(df_sample)
        print(f"\nRFM data shape: {rfm_data.shape}")
        print(rfm_data.head(10))
        
        # Step 2: Cluster customers
        print("\n" + "-" * 80)
        print("Step 2: Clustering Customers (K-Means)")
        print("-" * 80)
        rfm_data = engineer.cluster_customers(rfm_data)
        print(f"\nClusters assigned. Cluster distribution:")
        print(rfm_data['cluster'].value_counts().sort_index())
        
        # Step 3: Identify high-risk cluster
        print("\n" + "-" * 80)
        print("Step 3: Identifying High-Risk Cluster")
        print("-" * 80)
        high_risk_cluster = engineer.identify_high_risk_cluster(rfm_data)
        print(f"\n✅ High-risk cluster: {high_risk_cluster}")
        
        # Step 4: Create target variable
        print("\n" + "-" * 80)
        print("Step 4: Creating Target Variable")
        print("-" * 80)
        rfm_data = engineer.create_target_variable(rfm_data)
        print(f"\nTarget variable created!")
        print(rfm_data[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'cluster', 'is_high_risk']].head(10))
        
        # Step 5: Merge back
        print("\n" + "-" * 80)
        print("Step 5: Merging Target to Original Dataset")
        print("-" * 80)
        df_with_target = engineer.merge_target_to_dataset(df_sample, rfm_data)
        print(f"\n✅ Target merged! Final dataset shape: {df_with_target.shape}")
        print(f"\nFinal target distribution:")
        print(df_with_target['is_high_risk'].value_counts())
        print(f"\nSample of final dataset:")
        print(df_with_target[['CustomerId', 'Amount', 'TransactionStartTime', 'is_high_risk']].head(10))
        
        print("\n" + "=" * 80)
        print("✅ Proxy Target Variable Engineering Complete!")
        print("=" * 80)
        print("\nSummary:")
        print("1. ✅ RFM metrics calculated (Recency, Frequency, Monetary)")
        print("2. ✅ Customers clustered into 3 groups using K-Means")
        print("3. ✅ High-risk cluster identified (least engaged customers)")
        print("4. ✅ Binary target variable 'is_high_risk' created")
        print("5. ✅ Target merged back into transaction dataset")
        print("\nThe dataset is now ready for model training!")


if __name__ == "__main__":
    main()

