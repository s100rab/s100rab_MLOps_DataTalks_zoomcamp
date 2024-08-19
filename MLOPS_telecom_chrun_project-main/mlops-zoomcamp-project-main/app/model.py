""" Model for predicting user churn. """
import os
import mlflow
import pickle
import pandas as pd

from config import settings

MODEL_REGISRTY_PATH = f"models:/{settings.model_name}/{settings.stage}"
MLFLOW_URI = f"http://{settings.mlflow_host}:5000"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret.json"


mlflow.set_tracking_uri(MLFLOW_URI)

with open("app/models/min_max_scaler.bin", "rb") as f:
    scaler = pickle.load(f)

with open("app/models/ohe.bin", "rb") as f:
    ohe = pickle.load(f)


def prepare_features(users: list) -> pd.DataFrame:
    """
    Prepare features for model.

    Args:
        users (list): List of dictionaries with user data.

    Returns:
        pd.DataFrame: DataFrame with prepared features.

    """
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    df = pd.DataFrame(users)
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
    df = df.iloc[:, 1:]
    X = ohe.transform(df[cat_features])
    X = pd.DataFrame(X, columns=ohe.get_feature_names_out())
    X = pd.concat([X, df[num_features]], axis=1)
    features = X.columns.values
    X = pd.DataFrame(scaler.transform(X))
    X.columns = features
    return X


model = mlflow.pyfunc.load_model(MODEL_REGISRTY_PATH)
