"""
Proxy Target Variable Engineering Module.

This module creates a credit risk target variable by:
1. Calculating RFM (Recency, Frequency, Monetary) metrics
2. Clustering customers using K-Means
3. Identifying high-risk (disengaged) customers
4. Creating binary target variable is_high_risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProxyTargetEngineer:
    """
    Class for engineering proxy target variable using RFM analysis and clustering.
    """
    
    def __init__(
        self,
        customer_id_col: str = 'CustomerId',
        amount_col: str = 'Amount',
        datetime_col: str = 'TransactionStartTime',
        snapshot_date: Optional[str] = None,
        n_clusters: int = 3,
        random_state: int = 42
    ):
        """
        Initialize ProxyTargetEngineer.
        
        Args:
            customer_id_col: Name of customer ID column
            amount_col: Name of transaction amount column
            datetime_col: Name of datetime column
            snapshot_date: Snapshot date for calculating recency (YYYY-MM-DD format, or None for max date)
            n_clusters: Number of clusters for K-Means
            random_state: Random state for reproducibility
        """
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.rfm_data_ = None
        self.scaler_ = None
        self.kmeans_model_ = None
        self.cluster_labels_ = None
        self.high_risk_cluster_ = None
        self.customer_risk_labels_ = None
        
    def calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        try:
            logger.info("Calculating RFM metrics...")
            
            # Validate required columns
            required_cols = [self.customer_id_col, self.amount_col, self.datetime_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert datetime column if needed
            if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
                df[self.datetime_col] = pd.to_datetime(
                    df[self.datetime_col], errors='coerce'
                )
            
            # Determine snapshot date
            if self.snapshot_date is None:
                snapshot_date = df[self.datetime_col].max()
                logger.info(f"Using max transaction date as snapshot: {snapshot_date}")
            else:
                snapshot_date = pd.to_datetime(self.snapshot_date)
                logger.info(f"Using provided snapshot date: {snapshot_date}")
            
            # Calculate RFM metrics per customer
            rfm_data = df.groupby(self.customer_id_col).agg({
                self.datetime_col: [
                    lambda x: (snapshot_date - x.max()).days,  # Recency
                    'count'  # Frequency
                ],
                self.amount_col: 'sum'  # Monetary
            }).reset_index()
            
            # Flatten column names
            rfm_data.columns = [
                self.customer_id_col,
                'Recency',
                'Frequency',
                'Monetary'
            ]
            
            # Handle edge cases
            # Recency: if negative (future dates), set to 0
            rfm_data['Recency'] = rfm_data['Recency'].clip(lower=0)
            
            # Frequency: ensure at least 1
            rfm_data['Frequency'] = rfm_data['Frequency'].clip(lower=1)
            
            # Monetary: use absolute value (sum of amounts)
            rfm_data['Monetary'] = rfm_data['Monetary'].abs()
            
            # Handle any remaining NaN values
            rfm_data = rfm_data.fillna(0)
            
            logger.info(f"RFM metrics calculated for {len(rfm_data)} customers")
            logger.info(f"RFM Statistics:\n{rfm_data[['Recency', 'Frequency', 'Monetary']].describe()}")
            
            self.rfm_data_ = rfm_data
            return rfm_data
            
        except Exception as e:
            logger.error(f"Error calculating RFM metrics: {str(e)}")
            raise
    
    def preprocess_rfm(self, rfm_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Preprocess RFM data for clustering (scaling).
        
        Args:
            rfm_data: RFM DataFrame (uses self.rfm_data_ if None)
            
        Returns:
            Scaled RFM features as numpy array
        """
        try:
            if rfm_data is None:
                if self.rfm_data_ is None:
                    raise ValueError("RFM data not calculated. Call calculate_rfm_metrics() first.")
                rfm_data = self.rfm_data_
            
            # Extract RFM features
            rfm_features = rfm_data[['Recency', 'Frequency', 'Monetary']].values
            
            # Scale features (important for K-Means)
            self.scaler_ = StandardScaler()
            rfm_scaled = self.scaler_.fit_transform(rfm_features)
            
            logger.info("RFM features scaled using StandardScaler")
            return rfm_scaled
            
        except Exception as e:
            logger.error(f"Error preprocessing RFM: {str(e)}")
            raise
    
    def cluster_customers(
        self,
        rfm_data: Optional[pd.DataFrame] = None,
        rfm_scaled: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Cluster customers using K-Means based on RFM metrics.
        
        Args:
            rfm_data: RFM DataFrame (uses self.rfm_data_ if None)
            rfm_scaled: Scaled RFM features (will compute if None)
            
        Returns:
            RFM DataFrame with cluster labels added
        """
        try:
            logger.info(f"Clustering customers into {self.n_clusters} groups...")
            
            if rfm_data is None:
                if self.rfm_data_ is None:
                    raise ValueError("RFM data not calculated. Call calculate_rfm_metrics() first.")
                rfm_data = self.rfm_data_.copy()
            else:
                rfm_data = rfm_data.copy()
            
            if rfm_scaled is None:
                rfm_scaled = self.preprocess_rfm(rfm_data)
            
            # Fit K-Means
            self.kmeans_model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = self.kmeans_model_.fit_predict(rfm_scaled)
            rfm_data['cluster'] = cluster_labels
            
            self.cluster_labels_ = cluster_labels
            
            # Calculate silhouette score for validation
            silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
            logger.info(f"K-Means clustering completed. Silhouette score: {silhouette_avg:.4f}")
            
            # Analyze clusters
            cluster_summary = rfm_data.groupby('cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                self.customer_id_col: 'count'
            }).round(2)
            cluster_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
            logger.info(f"\nCluster Summary:\n{cluster_summary}")
            
            return rfm_data
            
        except Exception as e:
            logger.error(f"Error clustering customers: {str(e)}")
            raise
    
    def identify_high_risk_cluster(self, rfm_data: Optional[pd.DataFrame] = None) -> int:
        """
        Identify the high-risk cluster (least engaged customers).
        
        High-risk customers typically have:
        - High Recency (not active recently)
        - Low Frequency (few transactions)
        - Low Monetary (low spending)
        
        Args:
            rfm_data: RFM DataFrame with cluster labels (uses self.rfm_data_ if None)
            
        Returns:
            Cluster number identified as high-risk
        """
        try:
            if rfm_data is None:
                if self.rfm_data_ is None:
                    raise ValueError("RFM data not calculated. Call calculate_rfm_metrics() first.")
                rfm_data = self.rfm_data_
            
            if 'cluster' not in rfm_data.columns:
                raise ValueError("Clusters not assigned. Call cluster_customers() first.")
            
            logger.info("Identifying high-risk cluster...")
            
            # Calculate cluster centroids (mean RFM values per cluster)
            cluster_centroids = rfm_data.groupby('cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            })
            
            # Normalize metrics for comparison (higher is better for Frequency and Monetary)
            # Lower is better for Recency (more recent = lower recency)
            # Create a risk score: High Recency + Low Frequency + Low Monetary = High Risk
            cluster_centroids['risk_score'] = (
                cluster_centroids['Recency'] / cluster_centroids['Recency'].max() -  # Higher recency = higher risk
                cluster_centroids['Frequency'] / cluster_centroids['Frequency'].max() -  # Lower frequency = higher risk
                cluster_centroids['Monetary'] / cluster_centroids['Monetary'].max()  # Lower monetary = higher risk
            )
            
            # Cluster with highest risk score is the high-risk cluster
            high_risk_cluster = cluster_centroids['risk_score'].idxmax()
            
            logger.info(f"\nCluster Risk Scores:\n{cluster_centroids[['Recency', 'Frequency', 'Monetary', 'risk_score']]}")
            logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")
            
            # Show characteristics of high-risk cluster
            high_risk_stats = cluster_centroids.loc[high_risk_cluster]
            logger.info(f"\nHigh-Risk Cluster Characteristics:")
            logger.info(f"  Average Recency: {high_risk_stats['Recency']:.2f} days")
            logger.info(f"  Average Frequency: {high_risk_stats['Frequency']:.2f} transactions")
            logger.info(f"  Average Monetary: {high_risk_stats['Monetary']:.2f}")
            
            self.high_risk_cluster_ = high_risk_cluster
            return high_risk_cluster
            
        except Exception as e:
            logger.error(f"Error identifying high-risk cluster: {str(e)}")
            raise
    
    def create_target_variable(
        self,
        rfm_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create binary target variable is_high_risk.
        
        Args:
            rfm_data: RFM DataFrame with cluster labels (uses self.rfm_data_ if None)
            
        Returns:
            DataFrame with is_high_risk column
        """
        try:
            if rfm_data is None:
                if self.rfm_data_ is None:
                    raise ValueError("RFM data not calculated. Call calculate_rfm_metrics() first.")
                rfm_data = self.rfm_data_.copy()
            else:
                rfm_data = rfm_data.copy()
            
            if 'cluster' not in rfm_data.columns:
                raise ValueError("Clusters not assigned. Call cluster_customers() first.")
            
            if self.high_risk_cluster_ is None:
                self.identify_high_risk_cluster(rfm_data)
            
            # Create binary target: 1 for high-risk cluster, 0 for others
            rfm_data['is_high_risk'] = (rfm_data['cluster'] == self.high_risk_cluster_).astype(int)
            
            # Store customer risk labels
            self.customer_risk_labels_ = rfm_data[[self.customer_id_col, 'is_high_risk']].copy()
            
            # Log target distribution
            target_dist = rfm_data['is_high_risk'].value_counts()
            logger.info(f"\nTarget Variable Distribution:")
            logger.info(f"  Low Risk (0): {target_dist.get(0, 0)} customers ({target_dist.get(0, 0)/len(rfm_data)*100:.2f}%)")
            logger.info(f"  High Risk (1): {target_dist.get(1, 0)} customers ({target_dist.get(1, 0)/len(rfm_data)*100:.2f}%)")
            
            return rfm_data
            
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            raise
    
    def merge_target_to_dataset(
        self,
        df: pd.DataFrame,
        rfm_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge is_high_risk target variable back into main dataset.
        
        Args:
            df: Main transaction dataset
            rfm_data: RFM DataFrame with is_high_risk (uses self.rfm_data_ if None)
            
        Returns:
            DataFrame with is_high_risk column merged
        """
        try:
            if rfm_data is None:
                if self.customer_risk_labels_ is None:
                    raise ValueError("Target variable not created. Call create_target_variable() first.")
                risk_labels = self.customer_risk_labels_
            else:
                if 'is_high_risk' not in rfm_data.columns:
                    rfm_data = self.create_target_variable(rfm_data)
                risk_labels = rfm_data[[self.customer_id_col, 'is_high_risk']]
            
            # Merge risk labels
            df_merged = df.merge(
                risk_labels,
                on=self.customer_id_col,
                how='left'
            )
            
            # Fill any missing values (shouldn't happen, but safety check)
            df_merged['is_high_risk'] = df_merged['is_high_risk'].fillna(0).astype(int)
            
            logger.info(f"Target variable merged. Dataset shape: {df_merged.shape}")
            logger.info(f"High-risk transactions: {df_merged['is_high_risk'].sum()} ({df_merged['is_high_risk'].sum()/len(df_merged)*100:.2f}%)")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"Error merging target variable: {str(e)}")
            raise
    
    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Complete pipeline: Calculate RFM, cluster, identify high-risk, create target, and merge.
        
        Args:
            df: Raw transaction dataset
            
        Returns:
            Dataset with is_high_risk target variable
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Proxy Target Variable Engineering Pipeline")
            logger.info("=" * 80)
            
            # Step 1: Calculate RFM metrics
            rfm_data = self.calculate_rfm_metrics(df)
            
            # Step 2: Cluster customers
            rfm_data = self.cluster_customers(rfm_data)
            
            # Step 3: Identify high-risk cluster
            self.identify_high_risk_cluster(rfm_data)
            
            # Step 4: Create target variable
            rfm_data = self.create_target_variable(rfm_data)
            
            # Step 5: Merge back to original dataset
            df_with_target = self.merge_target_to_dataset(df, rfm_data)
            
            logger.info("=" * 80)
            logger.info("Proxy Target Variable Engineering Complete!")
            logger.info("=" * 80)
            
            return df_with_target
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise


# Convenience function
def create_proxy_target(
    df: pd.DataFrame,
    customer_id_col: str = 'CustomerId',
    amount_col: str = 'Amount',
    datetime_col: str = 'TransactionStartTime',
    snapshot_date: Optional[str] = None,
    n_clusters: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Convenience function to create proxy target variable.
    
    Args:
        df: Transaction dataset
        customer_id_col: Name of customer ID column
        amount_col: Name of transaction amount column
        datetime_col: Name of datetime column
        snapshot_date: Snapshot date for recency calculation
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Dataset with is_high_risk column
    """
    engineer = ProxyTargetEngineer(
        customer_id_col=customer_id_col,
        amount_col=amount_col,
        datetime_col=datetime_col,
        snapshot_date=snapshot_date,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    return engineer.fit_transform(df)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python target_engineering.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "../data/processed/data_with_target.csv"
    
    try:
        from src.data_processing import load_raw_data, save_processed_data
        
        # Load data
        df = load_raw_data(input_file)
        
        # Create proxy target
        df_with_target = create_proxy_target(df)
        
        # Save
        save_processed_data(df_with_target, output_file)
        
        print(f"\n✅ Proxy target variable created successfully!")
        print(f"✅ Output saved to: {output_file}")
        print(f"\nTarget distribution:")
        print(df_with_target['is_high_risk'].value_counts())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

