"""
Dedicated Resilience & Emergency Response Planning Service for FloodAI Engine.
Synthesizes project-generated response recommendations tailored to the selected
location and its 1 km local impact area across 7 operational pillars.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.resilience.recommendation_engine import ResilienceEngine


class ResilienceService:
    """
    Synthesizes actionable, multi-tier emergency response and resilience directives
    tailored to the 1 km local analysis area.
    """
    def __init__(self):
        self.engine = ResilienceEngine()

    def generate_plan(self, local_risk_res, prediction_res, location_meta, feature_dict):
        """
        Generates structured response planning across all 7 operational categories.
        """
        hazard_prob = prediction_res["hazard_probability"]
        risk_score = prediction_res["risk_score"]
        pop_exposed = local_risk_res["exposure_details"]["population_exposed"]
        elevation = float(feature_dict.get("elevation_m", 25.0))
        dist_river = float(feature_dict.get("dist_river_km", 1.0))
        r3d = float(feature_dict.get("rainfall_3d", 0.0))

        # 1. Determine Response Priority (Critical / Urgent / Guarded / Routine)
        if risk_score >= 80 or (hazard_prob >= 0.75 and pop_exposed >= 2000):
            priority = "CRITICAL (Immediate Multi-Agency Mobilization)"
            priority_color = "#dc2626"
            planning_header = "Response planning urgently recommended due to severe flood probability and high local population exposure."
        elif risk_score >= 60 or hazard_prob >= 0.50:
            priority = "URGENT (Heightened Alert & Resource Staging)"
            priority_color = "#ea580c"
            planning_header = "Response preparedness recommended: potential for significant localized inundation."
        elif risk_score >= 35:
            priority = "GUARDED (Active Hydrological Monitoring)"
            priority_color = "#d97706"
            planning_header = "Precautionary watch recommended: monitor rainfall anomalies and river gauge levels."
        else:
            priority = "ROUTINE (Standard Seasonal Baseline)"
            priority_color = "#16a34a"
            planning_header = "Low flood probability detected; routine maintenance and drainage clearing recommended."

        # 2. Synthesize 7 Operational Pillars
        # Pillar 1: Early Warning
        early_warning = [
            f"Activate emergency broadcast channels and SMS cell-broadcast across {location_meta.get('district', 'the area')}.",
            f"Set up 3-hour telemetry monitoring on upstream gauge stations in the {location_meta.get('basin', 'regional')} basin.",
            "Issue localized advisories for low-lying sectors within the 1 km analysis perimeter."
        ]

        # Pillar 2: Evacuation Preparedness
        evac_prep = [
            f"Designate high-ground assembly nodes outside the 1 km inundation risk buffer (> {elevation + 6.0:.1f} m AMSL).",
            "Pre-identify vulnerable elderly, non-ambulatory, and hospital populations in the local impact zone.",
            "Detailed evacuation routing requires verified road-network and emergency-access data — rely strictly on local administration demarcations."
        ]

        # Pillar 3: Safe Area / Shelter Consideration
        shelter = [
            f"Audit structural suitability of multi-story educational and community shelters in {location_meta.get('district', 'local district')}.",
            "Verify emergency water filtration, auxiliary solar/diesel power generators, and medical stockpiles at designated relief centers.",
            "Ensure sanitary partitions and dry sanitation arrangements for temporary evacuees."
        ]

        # Pillar 4: Emergency Resource Prioritization
        resources = [
            f"Pre-deploy National Disaster Response Force (NDRF) / SDRF inflatable rescue boat units to {location_meta.get('district', 'district')} staging depot.",
            "Stage heavy de-watering diesel pump sets (minimum 500 GPM) at known drainage choke points.",
            "Pre-position emergency ration kits, dry rations, and clean drinking water tankers."
        ]

        # Pillar 5: Critical Infrastructure Protection
        infrastructure = [
            "Construct temporary sandbag levees and flood barriers around electric distribution substations and transformer yards.",
            "Verify automated shut-off valving for municipal sewage outfalls to prevent reverse flow contamination.",
            "Ensure hospital ICU and emergency ward backup generators are elevated above ground level."
        ]

        # Pillar 6: Drainage / Water Management
        drainage = [
            f"Clear accumulated silt and solid waste blockages along drainage channels feeding into the {location_meta.get('basin', 'main')} river.",
            "Regulate upstream dam and barrage sluice gates according to rule curves to maintain flood cushion storage.",
            "Inspect culvert and stormwater bypass conduits within the 1 km core zone."
        ]

        # Pillar 7: Agricultural & Ecological Preparedness
        agriculture = [
            f"Advise farmers in the {local_risk_res['exposure_details']['agricultural_hectares']} ha local agricultural zone to harvest mature crops or clear field bund drains.",
            "Relocate livestock and farm machinery to elevated dry corrals.",
            "Store harvested grain, fertilizers, and animal feed on elevated pallets."
        ]

        return {
            "response_priority": priority,
            "priority_color": priority_color,
            "planning_header": planning_header,
            "pillars": {
                "🚨 Early Warning (0-6h)": early_warning,
                "🚶 Evacuation Preparedness (6-24h)": evac_prep,
                "🏠 Safe Area & Shelter Consideration": shelter,
                "🚒 Emergency Resource Prioritization": resources,
                "⚡ Critical Infrastructure Protection": infrastructure,
                "🌊 Drainage & Water Management": drainage,
                "🌾 Agricultural & Livestock Protection": agriculture
            },
            "disclaimer": "Project-generated response planning — not official government instructions. All operational actions must adhere to District Disaster Management Authority (DDMA) protocols."
        }
