from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_data_with_time_loss_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_live_sector_model_full_data.json"
)

JOBLIB_PATH = (
    MODEL_DIR
    / "xgboost_live_sector_model_full_data.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "live_sector_model_full_data_features.json"
)

PREDICTIONS_PATH = (
    MODEL_DIR
    / "live_sector_model_full_data_predictions.csv"
)


# =========================================================
# SETTINGS
# =========================================================

TARGET = "sector_time_loss_capped"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# These historical features can be created from live ACC.
LIVE_NUMERIC_FEATURES = [
    # Brake: ACC physics.brake * 100
    "BRAKE _Min",
    "BRAKE _Max",
    "BRAKE _Avg",
    "BRAKE _Start",
    "BRAKE _End",
    "BRAKE _Std Dev",

    # Speed: ACC physics.speed_kmh
    "Corr Speed kmh_Min",
    "Corr Speed kmh_Max",
    "Corr Speed kmh_Avg",
    "Corr Speed kmh_Start",
    "Corr Speed kmh_End",
    "Corr Speed kmh_Std Dev",

    # Throttle: ACC physics.gas * 100
    "THROTTLE _Min",
    "THROTTLE _Max",
    "THROTTLE _Avg",
    "THROTTLE _Start",
    "THROTTLE _End",
    "THROTTLE _Std Dev",

    # Section information
    "validlap",
    "Sector_Start",
    "Sector_Length",
]


# =========================================================
# LOAD FULL DATASET
# =========================================================

def load_data() -> pd.DataFrame:
    """
    Load the complete sector dataset.

    No range-based filtering is applied.
    Rows are removed only when required values are missing
    or cannot be converted to numeric values.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print("\nOriginal dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    required_columns = {
        "PID",
        "Lap",
        "Sector",
        TARGET,
        *LIVE_NUMERIC_FEATURES,
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Convert model inputs and target to numeric.
    for column in LIVE_NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    rows_before = len(df)

    # No unit/range filtering.
    # Only rows with missing required values are removed.
    df = df.dropna(
        subset=[
            "PID",
            "Lap",
            "Sector",
            TARGET,
            *LIVE_NUMERIC_FEATURES,
        ]
    ).copy()

    removed_missing = rows_before - len(df)

    print("\nFull-data preparation:")
    print(
        "Rows removed only because required "
        f"values were missing: {removed_missing}"
    )
    print(f"Rows remaining: {len(df)}")
    print(
        "Participants remaining: "
        f"{df['PID'].nunique()}"
    )
    print(
        "Laps remaining: "
        f"{df[['PID', 'Lap']].drop_duplicates().shape[0]}"
    )
    print(
        "Track sections remaining: "
        f"{df['Sector'].nunique()}"
    )

    return df


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_features(
    df: pd.DataFrame,
):
    """
    Prepare XGBoost input features, target, groups,
    metadata, and median values.
    """

    y = df[TARGET].copy()

    # Keep all sections from the same lap together.
    # Group by participant only.
# Every participant will appear entirely in either
# training or testing, never both.
    groups = df["PID"].astype(str)

    metadata = df[
        [
            "PID",
            "Lap",
            "Sector",
        ]
    ].copy()

    X = df[
        LIVE_NUMERIC_FEATURES
        + ["Sector"]
    ].copy()

    # Convert section names into one-hot columns.
    X = pd.get_dummies(
        X,
        columns=["Sector"],
        dtype=int,
    )

    # Replace infinity with missing values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove completely empty columns.
    X = X.dropna(
        axis=1,
        how="all",
    )

    # Remove constant columns.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:
        print(
            f"\nRemoving {len(constant_columns)} "
            "constant columns."
        )

        X = X.drop(
            columns=constant_columns
        )

    # Save medians for future live inference.
    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    )

    print(
        f"\nNumber of model features: {X.shape[1]}"
    )

    print("\nModel features:")

    for feature in X.columns:
        print(f" - {feature}")

    return (
        X,
        y,
        groups,
        metadata,
        medians,
    )


# =========================================================
# GROUPED TRAIN-TEST SPLIT
# =========================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
):
    """
    Split by PID + Lap.

    All sections from one lap remain completely inside
    either the training set or the testing set.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_index, test_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_index
    ].copy()

    X_test = X.iloc[
        test_index
    ].copy()

    y_train = y.iloc[
        train_index
    ].copy()

    y_test = y.iloc[
        test_index
    ].copy()

    metadata_train = metadata.iloc[
        train_index
    ].copy()

    metadata_test = metadata.iloc[
        test_index
    ].copy()

    train_groups = groups.iloc[
        train_index
    ]

    test_groups = groups.iloc[
        test_index
    ]

    lap_overlap = set(
        train_groups
    ).intersection(
        set(test_groups)
    )

    train_participants = set(
        metadata_train["PID"].astype(str)
    )

    test_participants = set(
        metadata_test["PID"].astype(str)
    )

    participant_overlap = (
        train_participants
        .intersection(test_participants)
    )

    print("\nData split:")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(
    f"Training participants: "
    f"{train_groups.nunique()}"
)

    print(
        f"Testing participants: "
        f"{test_groups.nunique()}"
    )

    print(
        f"Participant-group overlap: "
        f"{len(lap_overlap)}"
    )
    print(
        f"Participant overlap: "
        f"{len(participant_overlap)}"
    )

    if lap_overlap:
        raise RuntimeError(
            "Data leakage detected: complete laps "
            "appear in both sets."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    )


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:
    """
    Train the full-data live-compatible sector model.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    print(
        "\nTraining full-data "
        "live-compatible model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    return model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Evaluate XGBoost predictions.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 65)
    print("FULL-DATA LIVE-COMPATIBLE MODEL")
    print("=" * 65)

    print(f"\nXGBoost MAE:  {mae:.4f} seconds")
    print(f"XGBoost RMSE: {rmse:.4f} seconds")
    print(f"XGBoost R²:   {r2:.4f}")

    return {
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================================================
# BASELINE
# =========================================================

def evaluate_baseline(
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Predict the average training time loss for each sector.
    """

    baseline_train = metadata_train.copy()

    baseline_train["target"] = (
        y_train.to_numpy()
    )

    sector_averages = (
        baseline_train
        .groupby("Sector")["target"]
        .mean()
    )

    overall_average = float(
        y_train.mean()
    )

    baseline_predictions = (
        metadata_test["Sector"]
        .map(sector_averages)
        .fillna(overall_average)
        .to_numpy()
    )

    mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    r2 = r2_score(
        y_test,
        baseline_predictions,
    )

    print("\n" + "=" * 65)
    print("PER-SECTOR AVERAGE BASELINE")
    print("=" * 65)

    print(f"\nBaseline MAE:  {mae:.4f} seconds")
    print(f"Baseline RMSE: {rmse:.4f} seconds")
    print(f"Baseline R²:   {r2:.4f}")

    return {
        "predictions": baseline_predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "sector_averages": sector_averages,
    }


# =========================================================
# COMPARE MODEL WITH BASELINE
# =========================================================

def compare_results(
    xgboost_results: dict,
    baseline_results: dict,
) -> None:
    """
    Compare XGBoost MAE against baseline MAE.
    """

    xgb_mae = xgboost_results["mae"]
    baseline_mae = baseline_results["mae"]

    improvement_seconds = (
        baseline_mae - xgb_mae
    )

    if baseline_mae > 0:
        improvement_percentage = (
            improvement_seconds
            / baseline_mae
            * 100
        )
    else:
        improvement_percentage = 0.0

    print("\n" + "=" * 65)
    print("XGBOOST VS BASELINE")
    print("=" * 65)

    print(
        f"\nBaseline MAE: "
        f"{baseline_mae:.4f} seconds"
    )

    print(
        f"XGBoost MAE:  "
        f"{xgb_mae:.4f} seconds"
    )

    print(
        f"Error reduced by: "
        f"{improvement_seconds:.4f} seconds"
    )

    print(
        f"Percentage improvement: "
        f"{improvement_percentage:.2f}%"
    )

    if xgb_mae < baseline_mae:
        print(
            "\nResult: XGBoost is better "
            "than the per-sector average baseline."
        )
    else:
        print(
            "\nResult: XGBoost did not beat "
            "the baseline."
        )


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

def save_predictions(
    metadata_test: pd.DataFrame,
    y_test: pd.Series,
    xgboost_predictions,
    baseline_predictions,
) -> None:
    """
    Save test predictions for later analysis.
    """

    results = metadata_test.copy()

    results["actual_time_loss"] = (
        y_test.to_numpy()
    )

    results["xgboost_prediction"] = (
        xgboost_predictions
    )

    results["baseline_prediction"] = (
        baseline_predictions
    )

    results["xgboost_absolute_error"] = (
        np.abs(
            results["actual_time_loss"]
            - results["xgboost_prediction"]
        )
    )

    results["baseline_absolute_error"] = (
        np.abs(
            results["actual_time_loss"]
            - results["baseline_prediction"]
        )
    )

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"\nTest predictions saved to:\n"
        f"{PREDICTIONS_PATH}"
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def print_feature_importance(
    model: XGBRegressor,
    feature_names,
) -> None:
    """
    Print the top 20 XGBoost features.
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print("\n" + "=" * 65)
    print("TOP 20 FEATURES")
    print("=" * 65)

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model: XGBRegressor,
    feature_columns,
    medians: pd.Series,
) -> None:
    """
    Save model and feature metadata.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    joblib.dump(
        model,
        JOBLIB_PATH,
    )

    metadata = {
        "target": TARGET,
        "training_mode": (
            "full historical dataset; "
            "no range-based row filtering"
        ),
        "feature_columns": list(
            feature_columns
        ),
        "feature_medians": {
            column: float(value)
            for column, value
            in medians.items()
            if pd.notna(value)
        },
        "live_conversions": {
            "Corr Speed kmh": (
                "physics.speed_kmh"
            ),
            "THROTTLE": (
                "physics.gas * 100"
            ),
            "BRAKE": (
                "physics.brake * 100"
            ),
            "validlap": (
                "int(graphics.is_valid_lap)"
            ),
        },
    }

    with FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("FILES SAVED")
    print("=" * 65)

    print(f"\nModel JSON:\n{MODEL_PATH}")
    print(f"\nModel joblib:\n{JOBLIB_PATH}")
    print(f"\nFeature file:\n{FEATURES_PATH}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    df = load_data()

    (
        X,
        y,
        groups,
        metadata,
        medians,
    ) = prepare_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    ) = split_data(
        X,
        y,
        groups,
        metadata,
    )

    model = train_model(
        X_train,
        y_train,
    )

    xgboost_results = evaluate_model(
        model,
        X_test,
        y_test,
    )

    baseline_results = evaluate_baseline(
        metadata_train,
        metadata_test,
        y_train,
        y_test,
    )

    compare_results(
        xgboost_results,
        baseline_results,
    )

    save_predictions(
        metadata_test=metadata_test,
        y_test=y_test,
        xgboost_predictions=(
            xgboost_results["predictions"]
        ),
        baseline_predictions=(
            baseline_results["predictions"]
        ),
    )

    print_feature_importance(
        model=model,
        feature_names=X.columns,
    )

    save_model(
        model=model,
        feature_columns=X.columns,
        medians=medians,
    )


if __name__ == "__main__":
    main()