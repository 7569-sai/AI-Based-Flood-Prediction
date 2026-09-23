"""
Pan-India Location Intelligence Service for Flood Prediction.
Defines 36 strategic national flood-monitoring stations across 12 major Indian river basins.
Provides spatial lookup, nearest-station interpolation, and parameter retrieval for any coordinate in India.
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 36 Strategic National Stations across 12 Major Indian River Basins
INDIA_STATIONS = [
    # 1. Brahmaputra & Barak Basins (Assam / North-East)
    {
        "id": "AS_01",
        "name": "Guwahati (Brahmaputra)",
        "basin": "Brahmaputra",
        "state": "Assam",
        "lat": 26.1856,
        "lon": 91.7490,
        "elevation_m": 54.0,
        "slope_deg": 1.8,
        "dist_river_km": 0.25,
        "population_density": 2200.0,
        "historical_flood_years": [2019, 2020, 2022, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Low gradient Brahmaputra riverbed with severe siltation causes river swelling and urban backflow even under moderate 40mm showers."
    },
    {
        "id": "AS_02",
        "name": "Dibrugarh (Upper Assam)",
        "basin": "Brahmaputra",
        "state": "Assam",
        "lat": 27.4728,
        "lon": 94.9120,
        "elevation_m": 108.0,
        "slope_deg": 1.2,
        "dist_river_km": 0.4,
        "population_density": 1100.0,
        "historical_flood_years": [2018, 2020, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Continuous bank erosion and braided channel shifting cause immediate overflow during early monsoon spells."
    },
    {
        "id": "AS_03",
        "name": "Silchar (Barak Valley)",
        "basin": "Barak",
        "state": "Assam",
        "lat": 24.8333,
        "lon": 92.7789,
        "elevation_m": 22.0,
        "slope_deg": 1.5,
        "dist_river_km": 0.3,
        "population_density": 1850.0,
        "historical_flood_years": [2018, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Bowl-shaped valley topography causes prolonged multi-week water stagnation following localized rainfall."
    },
    {
        "id": "AS_04",
        "name": "Kaziranga Floodplains",
        "basin": "Brahmaputra",
        "state": "Assam",
        "lat": 26.5775,
        "lon": 93.1711,
        "elevation_m": 65.0,
        "slope_deg": 0.8,
        "dist_river_km": 0.5,
        "population_density": 320.0,
        "historical_flood_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Natural active floodplain wetland inundating annually even with routine monsoon discharge."
    },

    # 2. Yamuna & Ganga Basins (Delhi, UP, Bihar, Uttarakhand)
    {
        "id": "DL_01",
        "name": "Delhi (Old Railway Bridge / ITO)",
        "basin": "Yamuna",
        "state": "Delhi",
        "lat": 28.6644,
        "lon": 77.2389,
        "elevation_m": 207.0,
        "slope_deg": 0.9,
        "dist_river_km": 0.2,
        "population_density": 4850.0,
        "historical_flood_years": [2019, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Encroached active Yamuna floodplains and Hathnikund barrage peak surges inundate Ring Road even with moderate rainfall when upstream barrage discharges coincide."
    },
    {
        "id": "UP_01",
        "name": "Mathura (Yamuna Basin)",
        "basin": "Yamuna",
        "state": "Uttar Pradesh",
        "lat": 27.4924,
        "lon": 77.6737,
        "elevation_m": 174.0,
        "slope_deg": 0.7,
        "dist_river_km": 0.4,
        "population_density": 1950.0,
        "historical_flood_years": [2019, 2023],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Floods triggered primarily when upstream Delhi discharges exceed warning levels."
    },
    {
        "id": "BR_01",
        "name": "Patna (Ganga-Gandak Confluence)",
        "basin": "Ganga",
        "state": "Bihar",
        "lat": 25.5941,
        "lon": 85.1376,
        "elevation_m": 53.0,
        "slope_deg": 0.6,
        "dist_river_km": 0.35,
        "population_density": 3400.0,
        "historical_flood_years": [2019, 2021, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Flat alluvial plain where Ganga, Gandak, Son, and Punpun rivers converge, creating hydraulic backwater lock that prevents city stormwater drainage."
    },
    {
        "id": "BR_02",
        "name": "Saharsa / Supaul (Kosi Basin)",
        "basin": "Kosi",
        "state": "Bihar",
        "lat": 25.8835,
        "lon": 86.6006,
        "elevation_m": 45.0,
        "slope_deg": 0.5,
        "dist_river_km": 0.3,
        "population_density": 1250.0,
        "historical_flood_years": [2018, 2019, 2020, 2021],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Kosi river's massive heavy silt load and shifting course breach embankments under routine upper-Nepal rainfall."
    },
    {
        "id": "BR_03",
        "name": "Bhagalpur (Lower Ganga)",
        "basin": "Ganga",
        "state": "Bihar",
        "lat": 25.2425,
        "lon": 86.9842,
        "elevation_m": 41.0,
        "slope_deg": 0.8,
        "dist_river_km": 0.6,
        "population_density": 1600.0,
        "historical_flood_years": [2019, 2021],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Broad floodplain inundation during peak monsoon high stages."
    },
    {
        "id": "UP_02",
        "name": "Prayagraj (Ganga-Yamuna Sangam)",
        "basin": "Ganga",
        "state": "Uttar Pradesh",
        "lat": 25.4358,
        "lon": 81.8463,
        "elevation_m": 98.0,
        "slope_deg": 0.9,
        "dist_river_km": 0.4,
        "population_density": 2100.0,
        "historical_flood_years": [2019, 2021, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Ganga and Yamuna confluence creates upstream backwater pooling in low-lying residential colonies."
    },
    {
        "id": "UP_03",
        "name": "Varanasi (Middle Ganga)",
        "basin": "Ganga",
        "state": "Uttar Pradesh",
        "lat": 25.3176,
        "lon": 82.9739,
        "elevation_m": 80.0,
        "slope_deg": 1.1,
        "dist_river_km": 0.3,
        "population_density": 2900.0,
        "historical_flood_years": [2019, 2021],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Ghat inundation during high river stages."
    },
    {
        "id": "UK_01",
        "name": "Haridwar (Upper Ganga)",
        "basin": "Ganga",
        "state": "Uttarakhand",
        "lat": 29.9457,
        "lon": 78.1642,
        "elevation_m": 314.0,
        "slope_deg": 3.5,
        "dist_river_km": 0.5,
        "population_density": 950.0,
        "historical_flood_years": [2021, 2023],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Himalayan cloudburst runoff and debris flow surges."
    },

    # 3. Mahanadi Basin (Odisha & Eastern India)
    {
        "id": "OD_01",
        "name": "Cuttack (Mahanadi Delta)",
        "basin": "Mahanadi",
        "state": "Odisha",
        "lat": 20.4625,
        "lon": 85.8828,
        "elevation_m": 36.0,
        "slope_deg": 0.8,
        "dist_river_km": 0.3,
        "population_density": 2800.0,
        "historical_flood_years": [2018, 2020, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Surrounded by Mahanadi and Kathajodi rivers; city lies below peak river high-water line, relying entirely on ring bunds and pumps."
    },
    {
        "id": "OD_02",
        "name": "Sambalpur (Hirakud Downstream)",
        "basin": "Mahanadi",
        "state": "Odisha",
        "lat": 21.4669,
        "lon": 83.9812,
        "elevation_m": 150.0,
        "slope_deg": 1.8,
        "dist_river_km": 0.5,
        "population_density": 1150.0,
        "historical_flood_years": [2019, 2020, 2022],
        "chronic_low_flood_prone": False,
        "chronic_reason": "High discharges when Hirakud reservoir opens more than 25 sluice gates."
    },
    {
        "id": "OD_03",
        "name": "Puri Coastal Lowlands",
        "basin": "Coastal Delta",
        "state": "Odisha",
        "lat": 19.8135,
        "lon": 85.8312,
        "elevation_m": 8.0,
        "slope_deg": 0.3,
        "dist_river_km": 1.2,
        "population_density": 1400.0,
        "historical_flood_years": [2019, 2021],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Compound coastal storm surge and estuarine drainage blockage."
    },

    # 4. Godavari & Krishna Basins (Central & Southern India)
    {
        "id": "TS_01",
        "name": "Bhadrachalam (Godavari Basin)",
        "basin": "Godavari",
        "state": "Telangana",
        "lat": 17.6688,
        "lon": 80.8936,
        "elevation_m": 50.0,
        "slope_deg": 1.0,
        "dist_river_km": 0.2,
        "population_density": 850.0,
        "historical_flood_years": [2018, 2020, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Recurrent exceedance of third warning level (53 ft) driven by upper catchment releases into narrow river gorge."
    },
    {
        "id": "AP_01",
        "name": "Rajahmundry (Godavari Delta)",
        "basin": "Godavari",
        "state": "Andhra Pradesh",
        "lat": 17.0005,
        "lon": 81.8040,
        "elevation_m": 14.0,
        "slope_deg": 0.5,
        "dist_river_km": 0.3,
        "population_density": 2600.0,
        "historical_flood_years": [2020, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Dowleswaram barrage tailwater backflow into low-lying Konaseema agricultural islands."
    },
    {
        "id": "MH_01",
        "name": "Sangli / Kolhapur (Panchaganga)",
        "basin": "Krishna",
        "state": "Maharashtra",
        "lat": 16.8524,
        "lon": 74.5815,
        "elevation_m": 549.0,
        "slope_deg": 1.1,
        "dist_river_km": 0.3,
        "population_density": 1800.0,
        "historical_flood_years": [2019, 2021],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Almatti dam backwater effect downstream combined with Western Ghats catchment cloudbursts creates prolonged urban submergence."
    },
    {
        "id": "AP_02",
        "name": "Vijayawada (Krishna Delta)",
        "basin": "Krishna",
        "state": "Andhra Pradesh",
        "lat": 16.5062,
        "lon": 80.6480,
        "elevation_m": 20.0,
        "slope_deg": 1.2,
        "dist_river_km": 0.4,
        "population_density": 3100.0,
        "historical_flood_years": [2019, 2020, 2024],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Budameru rivulet overflow and Prakasam Barrage high inflows inundate low-lying urban wards."
    },

    # 5. Narmada & Tapi Basins (Western India)
    {
        "id": "GJ_01",
        "name": "Surat (Tapi Delta)",
        "basin": "Tapi",
        "state": "Gujarat",
        "lat": 21.1702,
        "lon": 72.8311,
        "elevation_m": 13.0,
        "slope_deg": 0.4,
        "dist_river_km": 0.3,
        "population_density": 4600.0,
        "historical_flood_years": [2019, 2020],
        "chronic_low_flood_prone": True,
        "chronic_reason": "High tidal backwater coupled with Ukai Dam spillway releases inundates flat urban coastal plains."
    },
    {
        "id": "GJ_02",
        "name": "Bharuch (Lower Narmada)",
        "basin": "Narmada",
        "state": "Gujarat",
        "lat": 21.7051,
        "lon": 72.9959,
        "elevation_m": 15.0,
        "slope_deg": 0.6,
        "dist_river_km": 0.4,
        "population_density": 1500.0,
        "historical_flood_years": [2019, 2020, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Golden Bridge river gauge exceeds danger mark whenever Sardar Sarovar Dam discharges peak water."
    },

    # 6. Kaveri Basin (Karnataka & Tamil Nadu)
    {
        "id": "TN_01",
        "name": "Tiruchirappalli (Kaveri Delta)",
        "basin": "Kaveri",
        "state": "Tamil Nadu",
        "lat": 10.7905,
        "lon": 78.7047,
        "elevation_m": 88.0,
        "slope_deg": 0.8,
        "dist_river_km": 0.4,
        "population_density": 2300.0,
        "historical_flood_years": [2018, 2021],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Mettur dam excess discharges into Coleroon river channel."
    },
    {
        "id": "TN_02",
        "name": "Thanjavur Deltaic Network",
        "basin": "Kaveri",
        "state": "Tamil Nadu",
        "lat": 10.7870,
        "lon": 79.1378,
        "elevation_m": 57.0,
        "slope_deg": 0.5,
        "dist_river_km": 0.8,
        "population_density": 1650.0,
        "historical_flood_years": [2021, 2022],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Intricate canal irrigation network overflows during intense Northeast monsoon depressions."
    },

    # 7. Western Ghats & Coastal Basins (Kerala & Karnataka)
    {
        "id": "KL_01",
        "name": "Aluva (Periyar Basin)",
        "basin": "Periyar",
        "state": "Kerala",
        "lat": 10.1076,
        "lon": 76.3516,
        "elevation_m": 10.0,
        "slope_deg": 4.7,
        "dist_river_km": 0.3,
        "population_density": 1359.0,
        "historical_flood_years": [2018, 2019, 2021],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Low-elevation Periyar floodplain subject to rapid reservoir discharges from upstream Idukki/Idamalayar dams."
    },
    {
        "id": "KL_08",
        "name": "Kuttanad / Mankombu (Pamba Delta)",
        "basin": "Pamba-Meenachil Delta",
        "state": "Kerala",
        "lat": 9.4500,
        "lon": 76.4833,
        "elevation_m": -1.0,
        "slope_deg": 0.4,
        "dist_river_km": 0.1,
        "population_density": 687.0,
        "historical_flood_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Located below mean sea level (-1m); natural drainage basin of 4 rivers where even minor 30mm rainfall causes waterlogging due to tidal lock at Thanneermukkom."
    },
    {
        "id": "KL_04",
        "name": "Chalakkudy (Chalakkudy Basin)",
        "basin": "Chalakkudy",
        "state": "Kerala",
        "lat": 10.3070,
        "lon": 76.3330,
        "elevation_m": 21.0,
        "slope_deg": 4.6,
        "dist_river_km": 0.4,
        "population_density": 636.0,
        "historical_flood_years": [2018, 2019],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Extreme surges when Poringalkuthu and Sholayar dams overflow."
    },
    {
        "id": "KA_01",
        "name": "Mangalore (Netravati Basin)",
        "basin": "Netravati",
        "state": "Karnataka",
        "lat": 12.9141,
        "lon": 74.8560,
        "elevation_m": 22.0,
        "slope_deg": 3.8,
        "dist_river_km": 0.6,
        "population_density": 1900.0,
        "historical_flood_years": [2018, 2019, 2022],
        "chronic_low_flood_prone": False,
        "chronic_reason": "High-intensity coastal Western Ghats orographic deluges."
    },

    # 8. Himalayan & Northern Basins (J&K, Punjab)
    {
        "id": "JK_01",
        "name": "Srinagar (Jhelum Basin)",
        "basin": "Jhelum",
        "state": "Jammu & Kashmir",
        "lat": 34.0837,
        "lon": 74.7973,
        "elevation_m": 1585.0,
        "slope_deg": 1.2,
        "dist_river_km": 0.3,
        "population_density": 2100.0,
        "historical_flood_years": [2014, 2019, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Jhelum's low carrying capacity through Wular Lake exit creates bowl-shaped basin submergence."
    },
    {
        "id": "PB_01",
        "name": "Firozpur (Sutlej Basin)",
        "basin": "Sutlej",
        "state": "Punjab",
        "lat": 30.9238,
        "lon": 74.6117,
        "elevation_m": 198.0,
        "slope_deg": 0.5,
        "dist_river_km": 0.5,
        "population_density": 850.0,
        "historical_flood_years": [2019, 2023],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Harike Barrage peak inflows and international border river loop breaches."
    },
    {
        "id": "MH_02",
        "name": "Mumbai (Mithi River & Coastal)",
        "basin": "Mithi & Coastal",
        "state": "Maharashtra",
        "lat": 19.0760,
        "lon": 72.8777,
        "elevation_m": 8.0,
        "slope_deg": 0.6,
        "dist_river_km": 0.2,
        "population_density": 5000.0,
        "historical_flood_years": [2019, 2020, 2021, 2023],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Extreme urban impervious surface fraction coupled with high tidal lock on Mithi river outfalls."
    },
    {
        "id": "MP_01",
        "name": "Hoshangabad / Narmadapuram",
        "basin": "Narmada",
        "state": "Madhya Pradesh",
        "lat": 22.7519,
        "lon": 77.7289,
        "elevation_m": 298.0,
        "slope_deg": 1.2,
        "dist_river_km": 0.3,
        "population_density": 1100.0,
        "historical_flood_years": [2020, 2022],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Heavy releases from Tawa and Bargi dams inundating Sethani Ghat."
    },
    {
        "id": "WB_01",
        "name": "Kolkata (Hooghly Estuary)",
        "basin": "Ganga-Hooghly",
        "state": "West Bengal",
        "lat": 22.5726,
        "lon": 88.3639,
        "elevation_m": 9.0,
        "slope_deg": 0.3,
        "dist_river_km": 0.4,
        "population_density": 4900.0,
        "historical_flood_years": [2020, 2021],
        "chronic_low_flood_prone": True,
        "chronic_reason": "Low-lying tidal flat terrain where high-tide lock prevents stormwater evacuation into Hooghly and Sundarbans channels."
    },
    {
        "id": "WB_02",
        "name": "Asansol / Durgapur (Damodar)",
        "basin": "Damodar",
        "state": "West Bengal",
        "lat": 23.6889,
        "lon": 86.9661,
        "elevation_m": 97.0,
        "slope_deg": 1.5,
        "dist_river_km": 0.6,
        "population_density": 2100.0,
        "historical_flood_years": [2019, 2021],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Damodar Valley Corporation (DVC) dam emergency release surges."
    },
    {
        "id": "JH_01",
        "name": "Jamshedpur (Subernarekha)",
        "basin": "Subernarekha",
        "state": "Jharkhand",
        "lat": 22.8046,
        "lon": 86.2029,
        "elevation_m": 135.0,
        "slope_deg": 2.1,
        "dist_river_km": 0.4,
        "population_density": 1750.0,
        "historical_flood_years": [2020, 2022],
        "chronic_low_flood_prone": False,
        "chronic_reason": "Subernarekha and Kharkai river confluence overflow."
    },
    {
        "id": "UP_04",
        "name": "Ayodhya (Saryu / Ghaghara)",
        "basin": "Ghaghara",
        "state": "Uttar Pradesh",
        "lat": 26.7922,
        "lon": 82.1998,
        "elevation_m": 93.0,
        "slope_deg": 0.8,
        "dist_river_km": 0.3,
        "population_density": 1600.0,
        "historical_flood_years": [2021, 2022],
        "chronic_low_flood_prone": False,
        "chronic_reason": "High discharge from Nepal border barrages into Ghaghara floodplains."
    },
    {
        "id": "HP_01",
        "name": "Kullu / Mandi (Beas Basin)",
        "basin": "Beas",
        "state": "Himachal Pradesh",
        "lat": 31.9579,
        "lon": 77.1095,
        "elevation_m": 1278.0,
        "slope_deg": 18.5,
        "dist_river_km": 0.3,
        "population_density": 380.0,
        "historical_flood_years": [2023],
        "chronic_low_flood_prone": False,
        "chronic_reason": "High-velocity torrential Himalayan flash floods and bank undercutting."
    }
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometers between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def find_nearest_station(lat, lon):
    """Find the nearest known monitoring station from arbitrary Indian coordinates."""
    best_st = None
    min_dist = float("inf")
    for st in INDIA_STATIONS:
        d = haversine_distance(lat, lon, st["lat"], st["lon"])
        if d < min_dist:
            min_dist = d
            best_st = st
    return best_st, round(min_dist, 1)


def get_location_profile(lat, lon, station_id=None):
    """
    Retrieve or synthesize complete environmental, terrain, and chronic risk attributes
    for any selected coordinate across India.
    """
    if station_id:
        for st in INDIA_STATIONS:
            if st["id"] == station_id:
                return st

    nearest, dist_km = find_nearest_station(lat, lon)
    
    # If coordinate is very close to a known station (< 30km), inherit its profile
    if dist_km <= 30.0:
        return nearest

    # Otherwise synthesize localized profile
    elev_est = max(10.0, nearest["elevation_m"] + (lat - nearest["lat"]) * 5.0)
    is_chronic = elev_est < 20.0 or nearest["chronic_low_flood_prone"]
    
    return {
        "id": f"CUSTOM_{int(lat*100)}_{int(lon*100)}",
        "name": f"Custom Coordinate ({lat:.2f}°N, {lon:.2f}°E)",
        "basin": f"{nearest['basin']} Regional Basin",
        "state": nearest["state"],
        "lat": lat,
        "lon": lon,
        "elevation_m": round(elev_est, 1),
        "slope_deg": nearest["slope_deg"],
        "dist_river_km": min(max(dist_km / 20.0, 0.2), 5.0),
        "population_density": nearest["population_density"],
        "historical_flood_years": nearest["historical_flood_years"],
        "chronic_low_flood_prone": is_chronic,
        "chronic_reason": nearest["chronic_reason"]
    }


def get_simulated_current_weather(profile, intensity="monsoon_surge"):
    """
    Retrieve realistic hydro-meteorological parameters for the location based on scenario intensity:
    - 'normal': routine light rainfall
    - 'monsoon_surge': active monsoon deluge
    - 'severe_flood': extreme multi-day deluge peak
    """
    is_chronic = profile.get("chronic_low_flood_prone", False)
    
    if intensity == "severe_flood":
        r1 = 110.0
        r3 = 230.0
        r7 = 360.0
        r14 = 510.0
        sm = 0.46
    elif intensity == "monsoon_surge":
        r1 = 65.0
        r3 = 135.0
        r7 = 210.0
        r14 = 310.0
        sm = 0.42
    else:  # normal / low-rain
        r1 = 15.0 if is_chronic else 5.0
        r3 = 35.0 if is_chronic else 12.0
        r7 = 60.0 if is_chronic else 25.0
        r14 = 90.0 if is_chronic else 40.0
        sm = 0.38 if is_chronic else 0.26

    elev = profile["elevation_m"]
    slope = profile["slope_deg"]
    dist_riv = profile["dist_river_km"]
    pop = profile["population_density"]

    return {
        "rainfall_1d": r1,
        "rainfall_3d": r3,
        "rainfall_7d": r7,
        "rainfall_14d": r14,
        "rainfall_anomaly_30d": r3 - (r14 / 4.6),
        "api_14d": r1 + (0.85 * r3),
        "temp_mean_3d": 26.0,
        "humidity_mean_3d": 88.0,
        "soil_moisture_root_zone": sm,
        "soil_moisture_deep": sm * 0.95,
        "soil_saturation_ratio": min(sm / 0.48, 1.0),
        "elevation_m": elev,
        "slope_deg": slope,
        "dist_river_km": dist_riv,
        "twi_proxy": float(np.log(100.0 / np.tan(np.radians(max(slope, 0.1))))),
        "elevation_risk_factor": float(np.exp(-elev / 50.0)),
        "sar_vv": -16.0 if r3 > 100 else -11.5,
        "sar_vh": -22.0 if r3 > 100 else -18.0,
        "sar_cross_ratio": 1.38,
        "ndwi_proxy": 0.55 if r3 > 80 else 0.25,
        "ndvi_proxy": 0.50,
        "population_density": pop,
        "exposure_index": float(np.clip(np.log1p(pop) / np.log1p(5000.0), 0.0, 1.0)),
        "drainage_vulnerability": float((0.45 * np.exp(-elev/50.0)) + (0.35 / (1.0 + dist_riv)) + (0.20 / (1.0 + np.radians(max(slope, 0.1))))),
        "vulnerability_index": 0.5
    }


if __name__ == "__main__":
    p = get_location_profile(28.6644, 77.2389)  # Delhi
    print("Location Profile:", p["name"], f"({p['basin']} Basin)")
    print("Chronic Low-Flood Prone:", p["chronic_low_flood_prone"])
    print("Reason:", p["chronic_reason"])
