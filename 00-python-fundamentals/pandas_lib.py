"""
What Pandas does

Pandas gives you the DataFrame — a spreadsheet-like table in Python. You will use it every day for:
loading CSV datasets, cleaning data, engineering features, exploring statistics.
A DataFrame has named columns (like a SQL table) and row indices. You can think of it as a dict of
NumPy arrays.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_dataframe() -> pd.DataFrame:
    """Create a DataFrame from dict and optionally replace it with a compatible CSV."""
    df = pd.DataFrame(
        {
            "origin": ["Accra", "Kumasi", "Tamale", "Cape Coast"],
            "dist_km": [0, 253, 590, 145],
            "eta_min": [0, 285, 600, 150],
            "is_delayed": [False, True, False, True],
        }
    )

    project_root = Path(__file__).resolve().parents[1]
    csv_candidates = [
        project_root / "data" / "raw" / "logistics_eta.csv",
        project_root / "data" / "logistics_eta.csv",
        project_root / "data" / "processed" / "clean.csv",
    ]

    required_columns = {"dist_km", "eta_min", "is_delayed"}

    for csv_path in csv_candidates:
        if csv_path.exists():
            csv_df = pd.read_csv(csv_path)
            if required_columns.issubset(csv_df.columns):
                print(f"Loaded CSV dataset: {csv_path}")
                return csv_df
            print(f"Skipping incompatible CSV (missing required columns): {csv_path}")

    print("Using in-memory sample DataFrame.")
    return df


def explore_dataframe(df: pd.DataFrame) -> None:
    """Print basic exploration outputs."""
    print("\n--- Exploring DataFrame ---")
    print(df.shape)
    print(df.dtypes)
    print(df.head(5))
    print(df.tail(3))
    print(df.describe(include="all"))
    df.info()
    print(df.isnull().sum())


def select_data(df: pd.DataFrame) -> None:
    """Run common selection and filtering examples."""
    print("\n--- Selecting Data ---")

    distances = df["dist_km"]
    features = df[["dist_km", "eta_min", "is_delayed"]]

    long_trips = df[df["dist_km"] > 200]
    delayed = df[df["is_delayed"] == True]
    complex_filter = df[(df["dist_km"] > 100) & (df["eta_min"] < 300)]

    print(distances.head())
    print(features.head())
    print(long_trips.head())
    print(delayed.head())
    print(complex_filter.head())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns and one-hot encoded buckets."""
    print("\n--- Feature Engineering ---")

    engineered = df.copy()
    valid_eta = engineered["eta_min"].replace(0, np.nan)
    engineered["speed_kmh"] = engineered["dist_km"] / (valid_eta / 60)

    engineered["dist_bucket"] = engineered["dist_km"].apply(
        lambda d: "short" if d < 100 else ("medium" if d < 300 else "long")
    )

    dummies = pd.get_dummies(engineered["dist_bucket"], prefix="dist")
    engineered = pd.concat([engineered, dummies], axis=1)

    print(engineered.head())
    return engineered


def run_aggregations(df: pd.DataFrame) -> None:
    """Print aggregation examples."""
    print("\n--- Aggregations ---")
    print(df["eta_min"].mean())
    print(df.groupby("dist_bucket")["eta_min"].mean())


def save_dataframe(df: pd.DataFrame) -> Path:
    """Save processed DataFrame to data/processed/logistics_clean.csv."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "logistics_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved DataFrame to: {output_path}")
    return output_path


def main() -> None:
    df = create_dataframe()
    explore_dataframe(df)
    select_data(df)
    engineered_df = engineer_features(df)
    run_aggregations(engineered_df)
    save_dataframe(engineered_df)


if __name__ == "__main__":
    main()
