# Save as: C:\Users\ankit\transformer-perf-model\web\app_config.py

"""
Configuration for different environments
(local development vs Railway production)
"""

import os

# Detect if running on Railway
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") is not None

# Base directory of the project
if IS_RAILWAY:
    BASE_DIR = "/app"
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)