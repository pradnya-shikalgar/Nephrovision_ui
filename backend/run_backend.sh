#!/bin/bash
# Script to launch NephroVision FastAPI Backend
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================="
echo "   Starting NephroVision AI Backend Server      "
echo "================================================="

if [ -f "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
