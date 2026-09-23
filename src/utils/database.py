"""
SQLite Database Layer using SQLAlchemy.
Defines ORM models for:
- predictions: stores real-time inference history and risk outputs
- locations: stores monitoring stations and geographic attributes
- model_metrics: stores validated empirical benchmark metrics across trained models
"""

import os
import sys
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "data", "flood_system.db")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.json")
STATIONS_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "kerala_static_layers.json")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(50), nullable=False)
    hazard_level = Column(Integer, nullable=False)
    model_version = Column(String(50), default="RandomForest_v1.0")
    input_payload = Column(Text, nullable=True)


class LocationRecord(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    station_id = Column(String(20), unique=True, index=True)
    location_name = Column(String(100), nullable=False)
    basin = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, nullable=False)
    slope_deg = Column(Float, nullable=False)
    population_density = Column(Float, nullable=False)


class ModelMetricRecord(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model = Column(String(50), unique=True, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)


def init_db():
    """Create all tables and seed static station locations and benchmark metrics."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    # 1. Seed locations if empty or augment with Pan-India stations
    from src.geospatial.location_service import INDIA_STATIONS
    existing_ids = {r.station_id for r in session.query(LocationRecord.station_id).all()}
    
    new_count = 0
    for st in INDIA_STATIONS:
        if st["id"] not in existing_ids:
            loc = LocationRecord(
                station_id=st["id"],
                location_name=st["name"],
                basin=st["basin"],
                district=st.get("state", "India"),
                latitude=st["lat"],
                longitude=st["lon"],
                elevation_m=st["elevation_m"],
                slope_deg=st["slope_deg"],
                population_density=st["population_density"]
            )
            session.add(loc)
            new_count += 1
    session.commit()
    if new_count > 0:
        print(f"Seeded {new_count} national monitoring stations into 'locations' table.")

    # 2. Seed model metrics
    pan_india_metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "pan_india_model_comparison.json")
    active_metrics_path = pan_india_metrics_path if os.path.exists(pan_india_metrics_path) else METRICS_PATH
    if os.path.exists(active_metrics_path):
        with open(active_metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        for m_name, m_vals in metrics_data.items():
            existing = session.query(ModelMetricRecord).filter_by(model=m_name).first()
            if existing:
                existing.accuracy = m_vals.get("accuracy", 0.0)
                existing.precision = m_vals.get("precision", 0.0)
                existing.recall = m_vals.get("recall", 0.0)
                existing.f1 = m_vals.get("f1", 0.0)
                existing.auc = m_vals.get("roc_auc", 0.0)
                existing.pr_auc = m_vals.get("pr_auc", 0.0)
                existing.brier_score = m_vals.get("brier_score", 0.0)
            else:
                rec = ModelMetricRecord(
                    model=m_name,
                    accuracy=m_vals.get("accuracy", 0.0),
                    precision=m_vals.get("precision", 0.0),
                    recall=m_vals.get("recall", 0.0),
                    f1=m_vals.get("f1", 0.0),
                    auc=m_vals.get("roc_auc", 0.0),
                    pr_auc=m_vals.get("pr_auc", 0.0),
                    brier_score=m_vals.get("brier_score", 0.0)
                )
                session.add(rec)
        session.commit()
        print(f"Synchronized {len(metrics_data)} models into 'model_metrics' table.")

    session.close()
    print(f"Database initialized successfully at: {DB_PATH}")


def log_prediction(latitude, longitude, probability, risk_score, risk_category, hazard_level, input_dict=None):
    """Log an inference transaction into the SQLite database."""
    session = SessionLocal()
    try:
        rec = PredictionRecord(
            latitude=float(latitude),
            longitude=float(longitude),
            timestamp=datetime.datetime.utcnow(),
            probability=float(probability),
            risk_score=float(risk_score),
            risk_category=str(risk_category),
            hazard_level=int(hazard_level),
            model_version="RandomForest_v1.0",
            input_payload=json.dumps(input_dict) if input_dict else None
        )
        session.add(rec)
        session.commit()
        rec_id = rec.id
        return rec_id
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
