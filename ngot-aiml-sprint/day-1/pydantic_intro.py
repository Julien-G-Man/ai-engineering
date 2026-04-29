from pydantic import BaseModel, Field

class ETARequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    
# pydantic validates automatically

"""
1. Basic class

class ETAReq(BaseModel):
distance_km: float
driver: str


2. Field constraints

origin_lat: float = Field(
..., ge=-90, le=90)


3. field_validator + model_valivdator

@field_validator("cargo_weight")
model_validator(mode="after")
def check_origin_not_dest(self): ...


4. computed_field + to_feature_vector()

@computed_field
@property
def distance_km(self) -> float: ...
def to_feature_vector(self) -> list[float]: ...
"""

