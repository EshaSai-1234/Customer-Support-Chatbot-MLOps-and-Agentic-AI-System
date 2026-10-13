@echo off
title Customer Insight Platform - Streamlit Dashboard
echo Starting Streamlit SaaS Dashboard on http://localhost:8501...
set PYTHONIOENCODING=utf-8
python -m streamlit run dashboard/app.py --server.port 8501
pause
