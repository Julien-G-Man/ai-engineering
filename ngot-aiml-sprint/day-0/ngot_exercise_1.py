"""
EXERCISE 1 — Feature Engineering Function 

Write a function called engineer_features(df) that takes a pandas DataFrame with columns: 
distance_km, hour_of_day, num_stops 
Add a column: is_rush_hour (True if hour_of_day is 7–9 or 17–19) 
Add a column: log_distance (natural log of distance_km using math.log) 
Add a column: distance_x_stops (multiply distance_km and num_stops) 
Return the modified DataFrame 
Bonus: write a pytest test that verifies each new column is correct 
"""

import pandas as pd
import math

def engineer_features(df):
    df["is_rush_hour"] = df["hour_of_day"].apply(is_rush_hour)
    df["log_distance"] = df["distance_km"].apply(math.log)
    df["distance_x_stops"] = df["distance_km"] * df["num_stops"]
    
    return df

def is_rush_hour(hour):
    if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19):
        return True
    return False


def test_engineer_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "distance_km": [10, 5],
            "hour_of_day": [8, 14],
            "num_stops": [2, 3],
        }
    )

    result = engineer_features(df.copy())

    assert result["is_rush_hour"].tolist() == [True, False]
    assert result["log_distance"].tolist() == [math.log(10), math.log(5)]
    assert result["distance_x_stops"].tolist() == [20, 15]