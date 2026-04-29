from eta_predictor.schemas import ETARequest
import pytest

def test_eta_request_valid():
    req = ETARequest(
        origin_lat=5.6,
        origin_lon=-0.2,
        dest_lat=6.4,
        dest_lon=-0.1,
        cargo_weight_kg=50,
        hour_of_day=8,
        vehicle_type="truck",
    )
    assert req.distance_km >= 0

def test_motorcycle_weight_limit():
    with pytest.raises(ValueError):
        ETARequest(
            origin_lat=5.6,
            origin_lon=-0.2,
            dest_lat=6.4,
            dest_lon=-0.1,
            cargo_weight_kg=150,
            hour_of_day=8,
            vehicle_type="motorcycle",
        )
