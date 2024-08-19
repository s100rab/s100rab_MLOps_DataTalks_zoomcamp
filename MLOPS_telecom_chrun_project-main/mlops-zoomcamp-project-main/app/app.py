from typing import List
from fastapi import FastAPI, status, Request
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from schemas import UserInfo, ChurnPredictionResponse
from model import model, prepare_features

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    """
    Root endpoint
    """
    return templates.TemplateResponse("client_churn.html", {"request": request})


@app.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    response_model=ChurnPredictionResponse,
)
def get_user_churn(users: List[UserInfo]):  #
    """
    Predict churn for a list of users
    """
    users = [user.model_dump() for user in users]
    try:
        features = prepare_features(users)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid features: {error}")

    predictions = model.predict(features).tolist()
    return {"predictions": predictions}
