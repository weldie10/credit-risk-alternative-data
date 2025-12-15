# Credit Risk Prediction Model

An end-to-end machine learning system for predicting credit risk using alternative data from eCommerce transactions. This project implements a complete MLOps pipeline including feature engineering, model training, MLflow tracking, containerized deployment, and CI/CD automation.

## Features

- **Feature Engineering**: Automated pipeline with RFM analysis, temporal features, categorical encoding, and WoE transformation
- **Proxy Target Creation**: RFM-based customer segmentation using K-Means clustering to identify high-risk customers
- **Model Training**: Multiple algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for model versioning, metrics tracking, and model registry
- **REST API**: FastAPI-based prediction service with automatic model loading from MLflow registry
- **Containerization**: Docker and docker-compose setup for easy deployment
- **CI/CD Pipeline**: Automated testing, linting, and Docker image building on every push

## Project Structure

```
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                       # Data directory (gitignored)
│   ├── raw/                   # Raw transaction data
│   └── processed/             # Processed features and targets
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── data_processing.py     # Feature engineering pipeline
│   ├── target_engineering.py  # Proxy target variable creation
│   ├── train.py               # Model training with MLflow
│   ├── predict.py             # Inference utilities
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # API request/response models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/weldie10/credit-risk-alternative-data.git
cd credit-risk-alternative-data
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download data and place in `data/raw/` directory

## Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Feature Engineering

```python
from src.data_processing import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline(
    customer_id_col='CustomerId',
    amount_col='Amount',
    datetime_col='TransactionStartTime',
    scaling_method='standardize'
)

df_processed = pipeline.fit_transform(df)
```

### 3. Create Proxy Target Variable

```python
from src.target_engineering import create_proxy_target

df_with_target = create_proxy_target(
    df,
    customer_id_col='CustomerId',
    amount_col='Amount',
    datetime_col='TransactionStartTime',
    n_clusters=3,
    random_state=42
)
```

### 4. Train Models

```bash
python src/train.py \
    --data-path data/processed/data_with_target.csv \
    --target-column is_high_risk \
    --models XGBoost LightGBM RandomForest \
    --register-model credit-risk-model
```

### 5. Make Predictions

```python
from src.predict import Predictor

predictor = Predictor("models/model_xgboost.joblib")
predictions, probabilities = predictor.predict(X, return_proba=True)
```

## API Deployment

### Using Docker Compose

```bash
# Set environment variables
export MLFLOW_MODEL_NAME=credit-risk-model
export MLFLOW_MODEL_STAGE=Production

# Start API service
docker-compose up api
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /load_model` - Load model from MLflow registry
- `GET /model/info` - Model information
- `GET /docs` - Interactive API documentation

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Amount": 1000.0,
      "Value": 1000.0,
      "transaction_frequency": 5.0,
      "avg_transaction_amount": 200.0
    }
  }'
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
- Runs code linting (flake8, black)
- Executes unit tests
- Builds and tests Docker image

The pipeline triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Build fails if linting or tests fail.

## MLflow Tracking

View experiment results:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

Access MLflow UI at `http://localhost:5000`

## Business Context

This project addresses credit risk assessment for a buy-now-pay-later service using alternative data from eCommerce transactions. Since traditional credit history is unavailable, we:

1. **Create Proxy Targets**: Use RFM (Recency, Frequency, Monetary) analysis and clustering to identify high-risk customer segments
2. **Engineer Features**: Transform transactional data into predictive features
3. **Train Models**: Build and compare multiple algorithms to find the best performer
4. **Deploy**: Containerize the best model as a production-ready API

The approach balances regulatory compliance (Basel II requirements) with predictive performance, using interpretable models where possible while leveraging advanced algorithms for accuracy.

## License

This project is part of an educational challenge and is provided as-is.
