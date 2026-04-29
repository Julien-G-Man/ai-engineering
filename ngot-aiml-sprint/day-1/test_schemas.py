from pydantic_schemas import SimpleETA

trip = SimpleETA(distance_km=253.5, speed_kmh=80, driver_name="Kwame")

print(trip.distance_km)