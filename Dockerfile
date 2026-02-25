FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY api-gateway-lambda/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api-gateway-lambda/ /app/

# Create directories for ML models
RUN mkdir -p /app/ml_models/models

# Expose port
EXPOSE 5000

# Run local Flask server
CMD ["python", "local_dashboard.py"]