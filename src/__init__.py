"""
Credit Risk Alternative Data Project

This package provides modules for:
- Data processing and feature engineering
- Proxy target variable engineering
- Model training and evaluation
- Prediction and inference
- API endpoints
"""

from .data_processing import (
    FeatureEngineeringPipeline,
    load_raw_data,
    save_processed_data
)

from .target_engineering import (
    ProxyTargetEngineer,
    create_proxy_target
)

__version__ = "0.1.0"
__all__ = [
    'FeatureEngineeringPipeline',
    'ProxyTargetEngineer',
    'load_raw_data',
    'save_processed_data',
    'create_proxy_target'
]

