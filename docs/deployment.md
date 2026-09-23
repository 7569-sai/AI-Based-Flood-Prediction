# Deployment Guide: AI-Based Flood Prediction & Resilience Planning

This guide provides step-by-step instructions for running, testing, and deploying the complete system.

---

## 1. Local Environment Setup

### Prerequisites
* Python 3.10+ (Verified on Python 3.10.11)
* Git
* SQLite 3

### Clone & Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd flood_prediction_resilience

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Pipeline Execution Commands

Execute the pipeline stages in sequence:

```bash
# 1. Acquire real dataset from GEE and Open-Meteo
python src/data/acquire_data.py

# 2. Run preprocessing, feature engineering & 7-vector leakage audit
python src/data/preprocess.py
python src/data/audit_leakage.py

# 3. Train all ML/DL models & run multimodal ablation study
python src/models/train.py

# 4. Generate geospatial risk maps & GeoJSON
python src/geospatial/risk_mapper.py

# 5. Generate global SHAP explanations & feature importance
python src/explainability/shap_explainer.py

# 6. Initialize SQLite database & seed metadata
python src/utils/database.py

# 7. Run automated test suite
pytest tests -v

# 8. Execute complete end-to-end demonstration
python src/demo_end_to_end.py
```

---

## 3. Starting the Backend API (FastAPI)

```bash
# Launch FastAPI server with Uvicorn
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
* **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
* **Alternative API Documentation (ReDoc)**: `http://127.0.0.1:8000/redoc`

---

## 4. Starting the Interactive Dashboard (Streamlit)

```bash
# Launch Streamlit web dashboard
streamlit run dashboard/app.py --server.port 8501
```
* Access dashboard in browser: `http://localhost:8501`

---

## 5. Docker Deployment (P1 Implementation)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
```

Build and run container:
```bash
docker build -t floodai:v2.0 .
docker run -p 8000:8000 -p 8501:8501 floodai:v2.0
```

---

## 6. Continuous Integration (CI) Workflow

Create `.github/workflows/ci.yml`:
```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run Leakage Audit & Pytest
        run: |
          python src/data/audit_leakage.py
          pytest tests -v
```
