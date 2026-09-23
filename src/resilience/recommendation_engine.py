"""
Rule-Based Resilience Planning Engine.
Synthesizes Hazard Probability, Population Exposure, and Physical Vulnerability
to generate prioritized, actionable resilience recommendations across emergency,
structural, non-structural, and agricultural domains.

DISCLAIMER: Every recommendation is explicitly tagged as a
'Project-generated planning recommendation — not official government guidance.'
"""

import os
import json

DISCLAIMER_TEXT = "Project-generated planning recommendation — not official government guidance."


class ResilienceEngine:
    def __init__(self):
        self.disclaimer = DISCLAIMER_TEXT

    def generate_recommendations(self, hazard_prob, exposure_index, vulnerability_index,
                                 elevation_m=25.0, dist_river_km=1.0, rainfall_3d=50.0,
                                 basin_name="Periyar", location_name="Aluva"):
        """
        Synthesize multi-dimensional risk components into context-specific resilience protocols.
        """
        recommendations = {
            "disclaimer": self.disclaimer,
            "location": location_name,
            "basin": basin_name,
            "risk_profile": {
                "hazard_probability": round(float(hazard_prob), 4),
                "exposure_index": round(float(exposure_index), 4),
                "vulnerability_index": round(float(vulnerability_index), 4),
                "elevation_m": round(float(elevation_m), 1),
                "dist_river_km": round(float(dist_river_km), 2),
                "rainfall_3d": round(float(rainfall_3d), 1)
            },
            "urgency_level": "Routine",
            "emergency_actions_immediate": [],
            "structural_mitigation": [],
            "policy_and_planning": [],
            "agricultural_and_ecological": []
        }

        # -------------------------------------------------------------
        # Determine Urgency Level
        # -------------------------------------------------------------
        if hazard_prob >= 0.75:
            recommendations["urgency_level"] = "CRITICAL / EMERGENCY"
        elif hazard_prob >= 0.50:
            recommendations["urgency_level"] = "ELEVATED / HIGH ALERT"
        elif hazard_prob >= 0.25:
            recommendations["urgency_level"] = "ADVISORY / WATCH"
        else:
            recommendations["urgency_level"] = "ROUTINE MONITORING"

        # -------------------------------------------------------------
        # 1. Emergency Response Actions (0 - 48 Hours)
        # -------------------------------------------------------------
        if hazard_prob >= 0.70:
            recommendations["emergency_actions_immediate"].extend([
                {
                    "action": "Immediate Public Alert & Siren Broadcast",
                    "target_agency": "District Disaster Management Authority (DDMA)",
                    "detail": f"Issue red-tier alert for {basin_name} floodplains; trigger CAP-based SMS alerts to all cell towers in {location_name}."
                },
                {
                    "action": "Mandatory Lowland Evacuation",
                    "target_agency": "Police & Civil Defense Volunteers",
                    "detail": f"Evacuate populations residing within 500m of river channel or below {max(elevation_m + 3, 5):.0f}m elevation to designated cyclone/flood relief camps."
                },
                {
                    "action": "Staging of High-Capacity Mobile De-watering Pumps",
                    "target_agency": "Fire & Rescue Services / Irrigation Dept",
                    "detail": "Deploy diesel suction pumps at critical drainage bottleneck points and road subways."
                },
                {
                    "action": "Hospital & Critical Facility Power Isolation Protocols",
                    "target_agency": "Health Services & Electricity Board (KSEB)",
                    "detail": "Verify elevated backup generator fuel reserves; prepare temporary medical evacuation routes."
                }
            ])
        elif hazard_prob >= 0.40:
            recommendations["emergency_actions_immediate"].extend([
                {
                    "action": "Yellow / Orange Watch Advisory",
                    "target_agency": "DDMA / Taluk Control Rooms",
                    "detail": f"Activate 24/7 taluk emergency response cells; monitor river gauge rates of rise in {basin_name} basin."
                },
                {
                    "action": "Shelter Inspection & Readiness Check",
                    "target_agency": "Revenue Department",
                    "detail": "Inspect designated school and community hall relief shelters; verify potable water and dry ration stocks."
                },
                {
                    "action": "Clear Stormwater Outfalls and Culverts",
                    "target_agency": "Municipal Corporation / PWD",
                    "detail": "Remove vegetative blockages and plastic waste from primary stormwater channels."
                }
            ])
        else:
            recommendations["emergency_actions_immediate"].append({
                "action": "Standard Hydrological Surveillance",
                "target_agency": "Irrigation & Water Resources Dept",
                "detail": "Continue routine telemetry polling of CWC gauge stations and dam reservoir levels."
            })

        # -------------------------------------------------------------
        # 2. Structural & Infrastructure Interventions
        # -------------------------------------------------------------
        if elevation_m <= 15.0 or dist_river_km <= 0.8:
            recommendations["structural_mitigation"].append({
                "measure": "Riverine Bund Elevation and Rip-Rap Armoring",
                "priority": "High",
                "detail": f"Construct reinforced earthen embankments or geotextile-lined revetments along low riverbanks in {location_name} to withstand a 50-year flood crest."
            })
        if vulnerability_index >= 0.60:
            recommendations["structural_mitigation"].extend([
                {
                    "measure": "Subsurface Surcharge Drainage Network Upgrades",
                    "priority": "High",
                    "detail": "Replace undersized box culverts with high-capacity arched culverts along arterial transit corridors."
                },
                {
                    "measure": "Electrical Transformer Plinth Elevation",
                    "priority": "Medium",
                    "detail": "Elevate local distribution transformers to minimum 2.5 meters above road grade to avert protracted blackout during standing inundation."
                }
            ])
        else:
            recommendations["structural_mitigation"].append({
                "measure": "Routine Embankment Maintenance",
                "priority": "Low",
                "detail": "Periodic vegetation clearing and silt removal along secondary canal banks."
            })

        # -------------------------------------------------------------
        # 3. Non-Structural & Policy Planning
        # -------------------------------------------------------------
        if exposure_index >= 0.65:
            recommendations["policy_and_planning"].append({
                "initiative": "High-Density Floodplain Zoning Enforcement",
                "timeframe": "12-24 Months",
                "detail": f"Enact strict development setbacks prohibiting new basement parking and critical infrastructure within the active 1-in-25-year return period inundation zone of {location_name}."
            })
        recommendations["policy_and_planning"].extend([
            {
                "initiative": "Community-Based Early Warning & Volunteer Network (Aapda Mitra)",
                "timeframe": "Ongoing",
                "detail": "Conduct bi-annual ward-level mock drills, register local boat owners and fishermen as first-responder rescue flotillas."
            },
            {
                "initiative": "Geospatial Cadastral Flood Inundation Indexing",
                "timeframe": "6-12 Months",
                "detail": "Integrate AI model hazard probabilities into municipal property tax and spatial GIS planning databases."
            }
        ])

        # -------------------------------------------------------------
        # 4. Agricultural & Ecological Resilience
        # -------------------------------------------------------------
        if elevation_m <= 5.0 or "delta" in basin_name.lower() or "kuttanad" in location_name.lower():
            recommendations["agricultural_and_ecological"].extend([
                {
                    "strategy": "Controlled Sluice Gate Operation and Polder De-silting",
                    "focus": "Lowland/Polder Agriculture",
                    "detail": "Operate Thanneermukkom barrage and local polder spillways to balance freshwater outflow while preventing tidal backwater surges."
                },
                {
                    "strategy": "Submergence-Tolerant Paddy Cultivation",
                    "focus": "Agronomic Adaptation",
                    "detail": "Promote flood-tolerant indigenous paddy cultivars (e.g. Pokkali, Swarna-Sub1) capable of surviving 14 days under complete water submergence."
                }
            ])
        else:
            recommendations["agricultural_and_ecological"].append({
                "strategy": "Catchment Afforestation & Soil Stabilization",
                "focus": "Upper Catchment Soil Water Retention",
                "detail": "Plant deep-rooted vetiver grass and endemic bamboo species along steep slopes to retard surface runoff velocities and reduce siltation."
            })

        return recommendations


if __name__ == "__main__":
    engine = ResilienceEngine()
    recs = engine.generate_recommendations(
        hazard_prob=0.88,
        exposure_index=0.85,
        vulnerability_index=0.82,
        elevation_m=10.0,
        dist_river_km=0.3,
        rainfall_3d=185.0,
        basin_name="Periyar",
        location_name="Aluva"
    )
    print("=" * 70)
    print(f"RESILIENCE REPORT FOR: {recs['location']} ({recs['basin']} Basin)")
    print(f"URGENCY: {recs['urgency_level']}")
    print(f"DISCLAIMER: {recs['disclaimer']}")
    print("=" * 70)
    print("\nImmediate Emergency Actions:")
    for a in recs["emergency_actions_immediate"]:
        print(f"  * [{a['target_agency']}] {a['action']}: {a['detail']}")
    print("\nStructural Interventions:")
    for s in recs["structural_mitigation"]:
        print(f"  * [{s['priority']} Priority] {s['measure']}: {s['detail']}")
