FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY parametrized_s100rab.py process_taxi_data.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "process_taxi_data.py"]
