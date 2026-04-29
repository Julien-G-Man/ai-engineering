"""
EXERCISE 2 — ETAPredictor class 

Create a class SimpleETAPredictor with: __init__(self, speed_kmh=60), predict(self, distance_km, 
num_stops=1) method that returns (distance_km / speed_kmh) * 60 + num_stops * 20 
Add a property called version that returns '1.0.0' 
Add a __repr__ that prints nicely: SimpleETAPredictor(speed_kmh=60) 
Write 3 assertions to test: predict(120) ≈ 140, predict(0, 0) == 0, version == '1.0.0' 
"""

import math

class SimpleETAPredictor:
    def __init__(self, speed_kmh=60):
        self.speed_kmh = speed_kmh
        
    def predict(self, distance_km, num_stops=1):
        return (distance_km / self.speed_kmh) * 60 + num_stops * 20

    @property
    def version(self):
        return "1.0.0"
    
    def __repr__(self):
        return f"SimpleETAPredictor(speed_kmh={self.speed_kmh})"


predictor = SimpleETAPredictor()

assert math.isclose(predictor.predict(120), 140.0)
assert predictor.predict(0, 0) == 0
assert predictor.version == "1.0.0"