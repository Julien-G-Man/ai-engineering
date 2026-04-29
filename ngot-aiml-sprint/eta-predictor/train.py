import mlflow
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

DATA = Path("data/generated/logistics_synthetic.csv")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA)
    X = df[["distance_km", "cargo_weight_kg", "hour_of_day"]]
    y = df["eta_min"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    experiments = [
        ("gbr", GradientBoostingRegressor(n_estimators=100)),
        ("rf", RandomForestRegressor(n_estimators=100)),
        ("ridge", Ridge()),
    ]

    mlflow.set_experiment("eta_experiments")

    for name, model in experiments:
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            mlflow.log_metric("val_mae", mae)
            # save model
            out = MODELS / f"{name}.joblib"
            joblib.dump(model, out)
            mlflow.log_artifact(str(out))
            print(name, mae)
