# Telco Customer Churn Prediction App

Study project for [MlOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp). This project is focuses on building a predictive model for customer churn in a telecommunications company using machine learning techniques. The goal of the project is to create and deploy an application that can predict whether a customer is likely to churn (cancel their subscription) based on historical data.


## Dataset
The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index.

Dataset Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure

```
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   ├── models
│   │   ├── min_max_scaler.bin
│   │   └── ohe.bin
│   └── schemas.py
├── data
│   ├── telco-customers.csv
│   ├── test.csv
│   └── train.csv
├── notebooks
│   ├── EDA.ipynb
│   ├── Load production model.ipynb
│   └── Register the best model.ipynb
├── requirements.txt
├── templates
│   └── client_churn.html
└── tests
    ├── model_test.py
    └── test_user.py
```

## Model Training

The model training process is orchestrated using MLflow, a platform for managing end-to-end machine learning lifecycles. In this project, we use MLflow to track experiments, log parameters and metrics, and manage model artifacts.

- good tutorial about [setting Up MLFlow on GCP](https://medium.com/aiguys/mlflow-on-gcp-for-experiment-tracking-151ac5ccebc7)

## Web Application
The model web application is built using FastAPI. The integration of FastAPI with MLflow enables seamless loading of the trained model from GCP artifact storage for accurate customer churn predictions.

To showcase the functionality of the Telco Customer Churn Prediction App, I have deployed a sample version of the app on Google Cloud Run. Google Cloud Run is a serverless platform that allows you to deploy and manage containerized applications effortlessly. The sample app provides a simple interface for users to input customer information and receive churn predictions based on the model.

- Sample App Link: [Telco Churn Prediction App on Google Cloud Run](https://churn-app-image-zaw4qoyacq-lz.a.run.app)
- Or FastApi [docs page](https://churn-app-image-zaw4qoyacq-lz.a.run.app/docs)