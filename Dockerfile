# Dockerfile for local backend development (Flask simulates API Gateway + Lambda)

FROM python:3.11-slim

WORKDIR /app

COPY api-gateway-lambda/requirements.txt /app/requirements.txt
COPY lambda-processing/requirements.txt /app/lambda-processing-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r lambda-processing-requirements.txt flask flask-cors requests

COPY api-gateway-lambda/ /app/
COPY lambda-processing/ /app/lambda-processing/

EXPOSE 5000

CMD ["python", "local_dashboard.py"]
