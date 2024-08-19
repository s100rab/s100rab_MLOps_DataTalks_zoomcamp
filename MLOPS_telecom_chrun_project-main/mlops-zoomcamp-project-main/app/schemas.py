from pydantic import BaseModel


class UserInfo(BaseModel):
    """
    User information
    """
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ChurnPredictionResponse(BaseModel):
    """
    Response for churn prediction
    """
    predictions: list[int]  # 0 or 1
