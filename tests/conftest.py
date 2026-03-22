import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# lambda-processing first (handler tests); api-gateway-lambda for sensor_enrichment
sys.path.insert(0, str(_ROOT / "lambda-processing"))
sys.path.insert(0, str(_ROOT / "api-gateway-lambda"))
