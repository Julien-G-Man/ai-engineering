from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

class SimpleETA(BaseModel): 
    """A simple ETA model — just distance and speed.""" 
    distance_km: float
    speed_kmh: float
    driver_name: str
    
    
class ETARequest(BaseModel):
    """Validated input for ETA prediction.""" 
    # Field(...) means the field is REQUIRED — no default value 
    # ge = greater than or equal to (>=) 
    # le = less than or equal to (<=) 
    # gt = greater than (>) 
    # lt = less than (<) 
  
    origin_lat: float = Field(..., ge=-90,   le=90,    description='Origin latitude') 
    origin_lon: float = Field(..., ge=-180,  le=180,   description='Origin longitude') 
    
    dest_lat:   float = Field(..., ge=-90,   le=90,    description='Destination latitude') 
    dest_lon:   float = Field(..., ge=-180,  le=180,   description='Destination longitude') 
    
    cargo_weight_kg: float = Field(..., gt=0, le=20000, description='Cargo weight in kg') 
    hour_of_day: int = Field(..., ge=0, le=23, description='Hour of departure (0=midnight, 23=11pm)') 
    num_stops:   int = Field(1, ge=1, le=20, description='Number of delivery stops (default: 1)') 
    vehicle_type: Literal['truck', 'van', 'motorcycle'] = 'truck'
    
    # @field_validator runs AFTER the type check 
    # It validates a single field 
    @field_validator('cargo_weight_kg') 
    @classmethod 
    def weight_must_make_sense_for_vehicle(cls, v: float, info) -> float: 
        """Motorcycles can carry at most 100kg.""" 
        # Note: at this point we can't access other fields easily in field_validator 
        # For cross-field validation, use model_validator (see below) 
        if v <= 0: 
            raise ValueError('Cargo weight must be positive') 
        return round(v, 2)   # Round to 2 decimal places 
  
    # @model_validator runs AFTER all fields are validated 
    # It can check relationships BETWEEN fields 
    @model_validator(mode='after') 
    def check_origin_and_dest_differ(self) -> 'ETARequest': 
        """Origin and destination cannot be the same point.""" 
        if (abs(self.origin_lat - self.dest_lat) < 0.001 and 
                abs(self.origin_lon - self.dest_lon) < 0.001): 
            raise ValueError('Origin and destination appear to be the same location') 
        return self 
  
    @model_validator(mode='after') 
    def motorcycle_weight_limit(self) -> 'ETARequest': 
        """A motorcycle cannot carry more than 100kg.""" 
        if self.vehicle_type == 'motorcycle' and self.cargo_weight_kg > 100: 
            raise ValueError(f'Motorcycle cannot carry {self.cargo_weight_kg}kg — limit is 100kg') 
        return self 