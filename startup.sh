#!/bin/bash

python -m venv customgptapp2
source customgptapp2/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install project requirements
echo "Installing Requirements"
python -m pip install --upgrade -r requirements.txt

# Install FastAPI and Uvicorn
echo "Installing uvicorn separately"
pip install fastapi uvicorn python-multipart

# === Guardrails Setup ===
echo "Setting Guardrails environment variables"
export GUARDRAILS_API_KEY   # Make sure this is actually set before running!

echo "Guardrails API Key is set: ${GUARDRAILS_API_KEY:0:4}****"

# === Guardrails Configure (Non-Interactive) ===
echo "Configuring Guardrails with API Key"
echo "${GUARDRAILS_API_KEY}" | guardrails configure

echo "Installing Guardrails Hub validators"
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/secrets_present
guardrails hub install hub://guardrails/guardrails_pii

# Confirm Uvicorn version
echo "The uvicorn version installed is:"
uvicorn --version

# Start the FastAPI app with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind=0.0.0.0:8000

echo "Startup Completed for customgptapp2 again"
