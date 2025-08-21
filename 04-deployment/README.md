
# End-to-End Machine Learning Model Deployment

This project provides a comprehensive, hands-on exploration of deploying machine learning models using modern MLOps tools and practices. It covers three fundamental deployment patterns—**batch**, **online (web service)**, and **streaming**—to demonstrate how to serve model predictions for a variety of real-world use cases.

---

## 🚀 Project Overview

The journey from a trained model to a production-ready application is a critical step in the machine learning lifecycle. This project demystifies that process by implementing several deployment strategies, each tailored for different operational requirements.

The core focus is on building robust, scalable, and maintainable deployment pipelines. From creating a containerized web service with Flask and Docker to orchestrating batch jobs with Prefect, this repository serves as a practical guide and showcase of my skills in productionizing ML models.

---

## 🛠️ Technologies & Tools

* **Languages & Frameworks**: Python, Scikit-learn, Flask
* **Orchestration**: Prefect
* **ML Lifecycle**: MLflow
* **Containerization**: Docker
* **Cloud Platforms**: Amazon Web Services (AWS)
    * **Streaming**: Kinesis, Lambda
    * **Storage**: S3

---

## 🎯 Deployment Strategies Implemented

This project is structured around three distinct deployment methods, with dedicated code and implementation for each.

### 1. Online Deployment: Real-time Predictions via Web Service

For scenarios requiring immediate, on-demand predictions, I developed a RESTful API.

* **Flask API**: Built a lightweight web server using **Flask** to expose a prediction endpoint (`/predict`). This endpoint receives input data in real-time and returns the model's prediction.
* **Docker Containerization**: Containerized the Flask application using **Docker**. This ensures that the model and its dependencies are packaged into a portable, reproducible, and scalable unit, ready for deployment on any environment.
* **MLflow Model Registry**: Integrated the service with the **MLflow Model Registry**. Instead of hardcoding a model file, the application dynamically fetches the latest production-ready model from the registry. This decouples the model from the application code, enabling seamless model updates without service downtime.

### 2. Batch Deployment: Offline Scoring with Prefect

This approach is designed for processing large volumes of data offline, where predictions are not required in real-time.

* **Scoring Script (`score.py`)**: Developed a Python script to load a trained model, read a batch of data (e.g., from a CSV file or a database), generate predictions, and save the results.
* **Orchestration with Prefect**: Used **Prefect** to orchestrate the batch scoring script as a reliable, observable, and schedulable workflow. I created a Prefect `flow` that encapsulates the entire scoring logic.
* **Scheduled Deployments**: Configured a **Prefect deployment** to run the scoring flow on a recurring schedule (e.g., daily or hourly). This demonstrates how to automate routine prediction tasks, with capabilities for backfilling, monitoring, and alerting.

### 3. (Optional) Streaming Deployment: Serverless Real-time Inference

This section explores a more advanced, event-driven architecture for processing continuous data streams.

* **AWS Kinesis**: Implemented a data pipeline using **Amazon Kinesis** to ingest streaming data in real-time.
* **AWS Lambda**: Deployed the prediction logic as a serverless function using **AWS Lambda**. This function is automatically triggered by new data arriving in the Kinesis stream, allowing for highly scalable and cost-effective real-time inference without managing servers.
