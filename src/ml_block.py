from __future__ import annotations

import glob
import os
import warnings
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
MODEL_PATH = f"{MODELS_DIR}/best_model.joblib"
SCALER_PATH = f"{MODELS_DIR}/scaler.joblib"
ENCODER_PATH = f"{MODELS_DIR}/encoder.joblib"
FEATURE_COLS_PATH = f"{MODELS_DIR}/feature_cols.joblib"
TRAIN_DATA_PATH = f"{MODELS_DIR}/train_data.joblib"

MAKE_MAP = {
    "audi": "Audi",
    "bmw": "BMW",
    "cclass": "Mercedes-Benz",
    "focus": "Ford",
    "ford": "Ford",
    "hyundi": "Hyundai",
    "merc": "Mercedes-Benz",
    "skoda": "Skoda",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "vw": "Volkswagen",
}

LUXURY_BRANDS = {
    "bmw", "mercedes-benz", "audi", "lexus", "porsche",
    "ferrari", "bentley", "tesla", "volvo", "jaguar", "land rover",
}
SPORT_KEYWORDS = ["sport", "gti", "rs", "turbo", "amg", "s-line", "coupe", "gt", "r-line"]


def load_data(data_dir: str = "data") -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Download the UK Used Cars dataset from Kaggle and place the CSV files there."
        )

    dfs = []
    for path in csv_files:
        basename = os.path.basename(path).replace(".csv", "").lower().strip()
        if basename.startswith("unclean"):
            continue
        make = MAKE_MAP.get(basename, basename.capitalize())
        df = pd.read_csv(path)
        df["make"] = make
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined.columns = [c.strip().lower() for c in combined.columns]
    rename = {"fueltype": "fuel_type", "enginesize": "engine_size"}
    combined = combined.rename(columns=rename)

    if "mileage" in combined.columns:
        combined["mileage"] = combined["mileage"] * 1.60934

    keep = ["make", "model", "year", "price", "mileage", "fuel_type", "transmission"]
    combined = combined[[c for c in keep if c in combined.columns]]

    combined = combined[(combined["price"] >= 500) & (combined["price"] <= 200_000)]
    combined = combined.dropna(subset=["price", "year", "mileage"])

    return combined.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    current_year = datetime.now().year

    df["car_age"] = current_year - df["year"].astype(int)
    df["km_per_year"] = df["mileage"] / (df["car_age"] + 0.1)

    make_lower = df.get("make", pd.Series([""] * len(df))).str.lower().fillna("")
    model_lower = df.get("model", pd.Series([""] * len(df))).str.lower().fillna("")

    df["is_luxury"] = make_lower.isin(LUXURY_BRANDS).astype(int)
    df["is_sport"] = model_lower.apply(
        lambda x: int(any(kw in str(x) for kw in SPORT_KEYWORDS))
    )

    return df


CAT_COLS_BASE = ["make", "fuel_type", "transmission"]
NUM_COLS_BASE = ["year", "mileage", "car_age", "km_per_year", "is_luxury", "is_sport"]


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: Optional[OrdinalEncoder] = None,
    fit: bool = True,
) -> tuple[np.ndarray, OrdinalEncoder, list[str]]:

    cat_cols = [c for c in CAT_COLS_BASE if c in df.columns]
    num_cols = NUM_COLS_BASE.copy()
    if "condition_score" in df.columns:
        num_cols.append("condition_score")
    num_cols = [c for c in num_cols if c in df.columns]

    df_feat = df.copy()
    df_feat[cat_cols] = df_feat[cat_cols].astype(str)

    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_feat[cat_cols] = encoder.fit_transform(df_feat[cat_cols])
    else:
        df_feat[cat_cols] = encoder.transform(df_feat[cat_cols])

    feature_cols = num_cols + cat_cols
    X = df_feat[feature_cols].values.astype(float)
    return X, encoder, feature_cols


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def train_and_save(data_dir: str = "data") -> dict:
    print("-- Loading data...")
    df = load_data(data_dir)
    print(f"   {len(df):,} rows after cleaning")

    df = engineer_features(df)
    X, encoder, feature_cols = build_feature_matrix(df, fit=True)
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

    results: dict[str, dict] = {}

    print("\n-- LinearRegression (baseline)")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["LinearRegression"] = evaluate_model(y_test, lr.predict(X_test))
    _print_metrics(results["LinearRegression"])

    print("\n-- RandomForestRegressor")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["RandomForest"] = evaluate_model(y_test, rf.predict(X_test))
    _print_metrics(results["RandomForest"])

    print("\n-- GradientBoostingRegressor")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    results["GradientBoosting"] = evaluate_model(y_test, gb.predict(X_test))
    _print_metrics(results["GradientBoosting"])

    print("\n-- MLPRegressor")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    mlp = MLPRegressor(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        verbose=True,
    )
    mlp.fit(X_tr_s, y_train)
    results["MLP"] = evaluate_model(y_test, mlp.predict(X_te_s))
    _print_metrics(results["MLP"])

    print("\n-- 5-fold CV (GradientBoosting)")
    cv_scores = cross_val_score(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    print(f"   CV RMSE: {-cv_scores.mean():,.0f} +/- {cv_scores.std():,.0f}")

    print("\n-- RandomizedSearchCV (GradientBoosting, 10 iter, cv=3)")
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    rscv = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=0, n_jobs=-1, verbose=0,
    )
    rscv.fit(X_train, y_train)
    best_gb = rscv.best_estimator_
    results["GradientBoosting_tuned"] = evaluate_model(y_test, best_gb.predict(X_test))
    print(f"   Best params: {rscv.best_params_}")
    _print_metrics(results["GradientBoosting_tuned"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_gb, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    joblib.dump(df, TRAIN_DATA_PATH)

    print(f"\n-- Models saved to {MODELS_DIR}/")
    print(f"-- Best test RMSE: {results['GradientBoosting_tuned']['RMSE']:,.0f}")
    return results


def _print_metrics(m: dict) -> None:
    print(f"   RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  R2={m['R2']:.3f}  MAPE={m['MAPE']:.1f}%")


_cache: dict = {}


def _download_artefacts_if_missing() -> None:
    if os.path.exists(MODEL_PATH):
        try:
            joblib.load(MODEL_PATH)
            return
        except Exception:
            print("Existing model artefacts are incompatible — retraining...")
            os.remove(MODEL_PATH)

    # Try downloading pre-trained artefacts first
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading model artefacts from HuggingFace...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        for fname in ["best_model.joblib", "encoder.joblib", "feature_cols.joblib", "scaler.joblib", "train_data.joblib"]:
            hf_hub_download(
                repo_id="Viiko10/autovalue-ai-models",
                filename=fname,
                repo_type="model",
                local_dir=MODELS_DIR,
            )
        # Verify the downloaded model loads correctly
        joblib.load(MODEL_PATH)
        print("Model artefacts downloaded and verified.")
        return
    except Exception as e:
        print(f"Download failed or incompatible: {e}")

    # Fallback: retrain from train_data.joblib if available
    train_data_path = TRAIN_DATA_PATH
    if os.path.exists(train_data_path):
        print("Retraining model from training data (first startup, ~2 min)...")
        df = joblib.load(train_data_path)
        X, encoder, feature_cols = build_feature_matrix(df, fit=True)
        y = df["price"].values
        X_train, X_test, y_train, y_test = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
        model.fit(X_train, y_train)
        scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
        scaler.fit(X_train)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(feature_cols, FEATURE_COLS_PATH)
        print("Retraining complete.")
    else:
        print("No training data available. Run training locally first.")


def _load_artefacts() -> None:
    if not _cache:
        _download_artefacts_if_missing()
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["scaler"] = joblib.load(SCALER_PATH)
        _cache["encoder"] = joblib.load(ENCODER_PATH)
        _cache["feature_cols"] = joblib.load(FEATURE_COLS_PATH)
        _cache["train_df"] = joblib.load(TRAIN_DATA_PATH)


def predict_price(
    make: str,
    model_name: str,
    year: int,
    mileage_km: float,
    fuel_type: str,
    transmission: str,
    condition_score: float = 0.75,
) -> float:
    _load_artefacts()

    current_year = datetime.now().year
    car_age = current_year - int(year)
    km_per_year = mileage_km / (car_age + 0.1)
    is_luxury = int(make.lower() in LUXURY_BRANDS)
    is_sport = int(any(kw in model_name.lower() for kw in SPORT_KEYWORDS))

    row = pd.DataFrame([{
        "make": make,
        "model": model_name,
        "year": year,
        "mileage": mileage_km,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "car_age": car_age,
        "km_per_year": km_per_year,
        "is_luxury": is_luxury,
        "is_sport": is_sport,
        "condition_score": condition_score,
    }])

    encoder = _cache["encoder"]
    feature_cols: list[str] = _cache["feature_cols"]

    cat_cols = [c for c in CAT_COLS_BASE if c in feature_cols]
    row[cat_cols] = encoder.transform(row[cat_cols].astype(str))

    X = row[feature_cols].values.astype(float)
    price = float(_cache["model"].predict(X)[0])
    return max(500.0, price)


def get_similar_cars(
    make: str,
    model_name: str,
    year: int,
    mileage_km: float,
    n: int = 3,
) -> pd.DataFrame:
    _load_artefacts()
    df = _cache["train_df"]

    mask = (
        (df["make"].str.lower() == make.lower())
        & (df["year"].between(year - 2, year + 2))
        & (df["mileage"].between(mileage_km * 0.6, mileage_km * 1.4))
    )
    result = df[mask].head(n)

    if len(result) < n:
        result = df[df["make"].str.lower() == make.lower()].head(n)

    if len(result) == 0:
        result = df.sample(min(n, len(df)), random_state=42)

    return result[["make", "model", "year", "mileage", "price"]].head(n)
