"""
EXERCISE 4 — Data Cleaning Pipeline 

Download any CSV dataset (or use the Day 1 generated logistics_eta.csv) 
Load with pandas. Print shape, dtypes, and number of null values per column 
Drop any rows with null values 
Clip any numeric column to remove values beyond the 99th percentile 
Save the cleaned DataFrame to data/processed/clean.csv 
"""

from pathlib import Path

import pandas as pd


def load_dataset(project_root: Path) -> pd.DataFrame:
	candidate_paths = [
		project_root / "data" / "logistics_eta.csv",
		project_root / "data" / "data.csv",
	]

	for path in candidate_paths:
		if path.exists():
			print(f"Loading dataset from: {path}")
			return pd.read_csv(path)

	raise FileNotFoundError("No CSV dataset found in data/.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	print(f"Shape: {df.shape}")
	print("Dtypes:")
	print(df.dtypes)
	print("Null values per column:")
	print(df.isnull().sum())

	cleaned = df.dropna().copy()

	numeric_columns = cleaned.select_dtypes(include="number").columns
	for col in numeric_columns:
		upper_bound = cleaned[col].quantile(0.99)
		cleaned[col] = cleaned[col].clip(upper=upper_bound)

	return cleaned


def main() -> None:
	project_root = Path(__file__).resolve().parents[0]
	df = load_dataset(project_root)
	cleaned_df = clean_dataframe(df)

	output_dir = project_root / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "clean.csv"
	cleaned_df.to_csv(output_path, index=False)
	print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
	main()
