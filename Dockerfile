# Base Image: official Python runtime as base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# COPY project files
COPY . .

# Expose API port 
EXPOSE 8000

# Run Fastapi application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]