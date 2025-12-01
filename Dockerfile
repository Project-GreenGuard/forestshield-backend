# Dockerfile for local backend development
# This simulates the Lambda environment locally

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY api-gateway-lambda/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt flask flask-cors requests

# Copy application code
COPY api-gateway-lambda/ /app/

# Expose port
EXPOSE 5000

# Run local Flask server
CMD ["python", "local_dashboard.py"]

