FROM python:3.9

# Create and activate virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the files
COPY secret.json /app/secret.json
COPY app /app/app
COPY templates /app/templates
COPY .env /app/.env

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "./app/main.py"]
