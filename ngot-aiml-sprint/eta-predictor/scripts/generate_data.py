import random
import csv
from pathlib import Path
import math

OUT = Path(__file__).resolve().parents[1] / "data" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

N = 6000

# Simple Ghana bounding box roughly
min_lat, max_lat = 4.5, 11.2
min_lon, max_lon = -3.3, 1.2

vehicle_types = ["truck", "van", "motorcycle"]

with open(OUT / "logistics_synthetic.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = [
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "cargo_weight_kg",
        "hour_of_day",
        "vehicle_type",
        "num_stops",
        "distance_km",
        "eta_min",
    ]
    writer.writerow(header)

    for _ in range(N):
        o_lat = random.uniform(min_lat, max_lat)
        o_lon = random.uniform(min_lon, max_lon)
        d_lat = random.uniform(min_lat, max_lat)
        d_lon = random.uniform(min_lon, max_lon)
        weight = random.uniform(1, 2000)
        hour = random.randint(0, 23)
        vt = random.choices(vehicle_types, weights=(0.6, 0.3, 0.1))[0]
        stops = random.randint(0, 5)
        # Haversine
        r = 6371.0
        lat1 = math.radians(o_lat)
        lon1 = math.radians(o_lon)
        lat2 = math.radians(d_lat)
        lon2 = math.radians(d_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = r * c
        # ETA heuristic
        base_speed = 50 if vt == "truck" else (40 if vt == "van" else 30)
        eta = dist / base_speed * 60 + weight * 0.02 + stops * 10 + random.uniform(-5, 5)
        writer.writerow([o_lat, o_lon, d_lat, d_lon, weight, hour, vt, stops, dist, max(1, eta)])

print("Wrote:", OUT / "logistics_synthetic.csv")
