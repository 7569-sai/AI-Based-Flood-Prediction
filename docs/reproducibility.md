# Reproducibility Guide — FloodAI Engine

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Environment & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10.11 Windows 64-bit)
- Git
- 8 GB+ RAM (16 GB recommended)
- Optional: CUDA-capable NVIDIA GPU (NVIDIA RTX 3050 supported, fallback to multi-core CPU)

### Installation Commands
```powershell
# 1. Clone repository
git clone <repo-url> flood_prediction_resilience
cd flood_prediction_resilience

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## 2. End-to-End Pipeline Execution Commands

### Step 1: Pan-India Dataset Construction
```powershell
python src/data/build_pan_india_dataset.py
```
*Generates 78,876 multi-year records across 36 national stations in `data/processed/india/pan_india_features.parquet` and partitions by basin.*

### Step 2: Data Leakage Verification
```powershell
python src/data/audit_pan_india_leakage.py
```
*Verifies all 7 leakage vectors; updates `docs/data_leakage_audit.md`.*

### Step 3: Model Training & Evaluation
```powershell
python src/models/train_pan_india.py
```
*Trains 6 baselines (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, Bi-LSTM) and the proposed ST-FloodNet architecture. Runs cross-basin holdout tests and 6-tier ablation study.*

### Step 4: TreeSHAP Global & Local Explainability
```powershell
python src/explainability/shap_explainer.py
```
*Generates beeswarm plots and local risk driver attribution in `outputs/figures/`.*

### Step 5: Database Seeding & Verification
```powershell
python src/utils/database.py
```
*Initializes SQLite database `data/flood_system.db` and synchronizes 36 stations and benchmark metrics.*

### Step 6: Automated Test Suite (Pytest)
```powershell
python -m pytest tests/ -v
```
*Executes all unit and integration tests (100% pass rate required).*

### Step 7: Launch Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
*Launches web interface at `http://localhost:8501`.*

### Step 8: Launch FastAPI Backend
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.*
