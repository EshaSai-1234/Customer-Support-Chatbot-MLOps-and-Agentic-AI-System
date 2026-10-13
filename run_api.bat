@echo off
title Customer Insight Platform - FastAPI Server
echo Starting FastAPI Backend Server on http://127.0.0.1:8000...
set PYTHONIOENCODING=utf-8
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
pause
