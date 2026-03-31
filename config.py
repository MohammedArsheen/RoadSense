import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, SNAPSHOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Configuration
CONFIG = {
    "roads": {
        "Guindy Road": {
            "id": "road_1",
            "lat": 13.0067,
            "lng": 80.2206,
            "video": "data/Traffic.mp4"
        }
    },
    "thresholds": {
        "low": 5,
        "medium": 15
    },
    "google_maps_api_key": "YOUR_API_KEY_HERE", # User should replace this
    "poll_interval_ms": 1000
}
