from prefect import flow, task
from src.data.data_processing import load_data, preprocess_data
from src.models.train_model import train_model

@task
def load_and_preprocess():
    data = load_data()
    return preprocess_data(data)

@task
def train_and_evaluate(X, y):
    model = train_model(X, y)
    # Add evaluation code here
    return model

@flow
def training_workflow():
    X, y = load_and_preprocess()
    model = train_and_evaluate(X, y)
    # Add model saving code here

if __name__ == "__main__":
    training_workflow()