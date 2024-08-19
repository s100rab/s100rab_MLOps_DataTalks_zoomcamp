""" Settings module """
import os
from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    stage: str
    model_name: str
    mlflow_host: str

    class Config:
        env_file = ".env"


# class Settings(BaseSettings):
#     stage: str = os.environ.get("STAGE")
#     model_name: str = os.environ.get("MODEL_NAME")
#     mlflow_host: str = os.environ.get("MLFLOW_HOST")


settings = Settings()
