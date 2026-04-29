import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ETARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lon: float = Field(..., ge=-180, le=180)
    dest_lat: float = Field(..., ge=-90, le=90)
    dest_lon: float = Field(..., ge=-180, le=180)
    cargo_weight_kg: float = Field(..., gt=0, le=20000)
    hour_of_day: int = Field(..., ge=0, le=23)
    vehicle_type: Literal["truck", "van", "motorcycle"] = "truck"

    @model_validator(mode="after")
    def check_motorcycle_weight(self):
        if self.vehicle_type == "motorcycle" and self.cargo_weight_kg > 100:
            raise ValueError("Motorcycle weight limit: 100kg")
        return self

    @computed_field
    @property
    def distance_km(self) -> float:
        # Haversine formula
        r = 6371.0
        lat1 = math.radians(self.origin_lat)
        lon1 = math.radians(self.origin_lon)
        lat2 = math.radians(self.dest_lat)
        lon2 = math.radians(self.dest_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def to_feature_vector(self) -> list[float]:
        # Simple feature vector
        return [
            self.distance_km,
            self.cargo_weight_kg,
            float(self.hour_of_day),
            1.0 if self.vehicle_type == "truck" else 0.0,
        ]

class ETAResponse(BaseModel):
    eta_minutes: float
    eta_human_readable: str
    distance_km: float
    confidence_low: float
    confidence_high: float
    is_rush_hour: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
