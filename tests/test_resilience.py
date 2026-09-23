"""
Unit Tests for Resilience Recommendation Engine.
"""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.resilience.recommendation_engine import ResilienceEngine, DISCLAIMER_TEXT


def test_mandatory_disclaimer_presence():
    """Verify that every generated recommendation contains the mandatory disclaimer."""
    engine = ResilienceEngine()
    recs = engine.generate_recommendations(0.85, 0.7, 0.8, 10.0, 0.3, 150.0, "Periyar", "Aluva")
    assert recs["disclaimer"] == DISCLAIMER_TEXT
    assert "not official government guidance" in recs["disclaimer"]


def test_urgency_escalation_logic():
    """Verify urgency levels escalate proportionally to hazard probability."""
    engine = ResilienceEngine()
    
    # Low hazard
    r_low = engine.generate_recommendations(0.10, 0.5, 0.5)
    assert r_low["urgency_level"] == "ROUTINE MONITORING"
    
    # Moderate hazard
    r_mod = engine.generate_recommendations(0.35, 0.5, 0.5)
    assert r_mod["urgency_level"] == "ADVISORY / WATCH"
    
    # High hazard
    r_high = engine.generate_recommendations(0.60, 0.5, 0.5)
    assert r_high["urgency_level"] == "ELEVATED / HIGH ALERT"
    
    # Critical hazard
    r_crit = engine.generate_recommendations(0.85, 0.5, 0.5)
    assert r_crit["urgency_level"] == "CRITICAL / EMERGENCY"


def test_emergency_actions_contain_target_agency():
    """Verify all emergency actions have designated lead response agencies."""
    engine = ResilienceEngine()
    recs = engine.generate_recommendations(0.90, 0.8, 0.8, 8.0, 0.2, 200.0)
    assert len(recs["emergency_actions_immediate"]) > 0
    for a in recs["emergency_actions_immediate"]:
        assert "target_agency" in a
        assert len(a["target_agency"]) > 0
        assert "action" in a
