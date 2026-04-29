from pydantic_schemas import ETARequest
from pydantic import ValidationError

# Valid Accra to Kumasi trip 
req = ETARequest( 
    origin_lat=5.6037, origin_lon=-0.1870, 
    dest_lat=6.6885,   dest_lon=-1.6244, 
    cargo_weight_kg=500, 
    hour_of_day=9, 
    num_stops=3, 
) 
print(req)

# Test constraint violation 
try: 
    bad = ETARequest(
        origin_lat=999,  # INVALID — greater than 90
        origin_lon=-0.1870, 
        dest_lat=6.6885, dest_lon=-1.6244,
        cargo_weight_kg=500, hour_of_day=9
    )
except ValidationError as e:
    print(e)