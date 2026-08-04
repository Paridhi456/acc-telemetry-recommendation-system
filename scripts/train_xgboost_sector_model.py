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
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_data_with_time_loss_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_JSON_PATH = (
    MODEL_DIR
    / "xgboost_sector_time_loss_model.json"
)

MODEL_JOBLIB_PATH = (
    MODEL_DIR
    / "xgboost_sector_time_loss_model.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "sector_model_features.json"
)

PREDICTIONS_PATH = (
    MODEL_DIR
    / "sector_model_test_predictions.csv"
)


# =========================================================
# SETTINGS
# =========================================================

TARGET_COLUMN = "sector_time_loss_capped"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

def load_data() -> pd.DataFrame:
    """
    Load the prepared sector-level dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(
    df: pd.DataFrame,
):
    """
    Prepare:
    - model features
    - target
    - lap groups
    - metadata
    - feature medians
    """

    df = df.copy()

    # Convert target to numeric.
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    # Remove rows without target.
    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    # PID and Lap are required for grouped splitting.
    if "PID" not in df.columns:
        raise ValueError("PID column is missing.")

    if "Lap" not in df.columns:
        raise ValueError("Lap column is missing.")

    # -----------------------------------------------------
    # GROUP ID
    # -----------------------------------------------------
    # Every row belonging to the same participant-lap
    # stays entirely in training or entirely in testing.
    # -----------------------------------------------------

    groups = (
        df["PID"].astype(str)
        + "_"
        + df["Lap"].astype(str)
    )

    # Target predicted by XGBoost.
    y = df[TARGET_COLUMN].copy()

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------
    # Metadata is not used directly by the model.
    # It is kept for evaluation and baseline comparison.
    # -----------------------------------------------------

    metadata_columns = [
        column
        for column in [
            "PID",
            "Lap",
            "Sector",
            "Type",
            "SectorTime",
            "reference_sector_time",
            "sector_time_loss",
            "sector_time_loss_capped",
        ]
        if column in df.columns
    ]

    metadata = df[metadata_columns].copy()

    # -----------------------------------------------------
    # REMOVE TARGET LEAKAGE AND IDENTIFIERS
    # -----------------------------------------------------

    columns_to_remove = [
        # Target
        "sector_time_loss_capped",

        # Columns used to calculate the target
        "SectorTime",
        "reference_sector_time",
        "sector_time_loss",
        "time_loss_was_capped",

        # Lap-level leakage or labels
        "LapTime",
        "Lap Time",
        "lap_time",
        "cluster",
        "cluster_label",
        "z",

        # Identifiers
        "PID",
        "Lap",
    ]

    existing_columns_to_remove = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    X = df.drop(
        columns=existing_columns_to_remove
    )

    # -----------------------------------------------------
    # ENCODE TEXT COLUMNS
    # -----------------------------------------------------
    # Sector and Type are text.
    # One-hot encoding turns them into 0/1 columns.
    # -----------------------------------------------------

    categorical_columns = [
        column
        for column in [
            "Sector",
            "Type",
        ]
        if column in X.columns
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dummy_na=True,
        dtype=int,
    )

    # Keep numerical columns only.
    X = X.select_dtypes(
        include=["number"]
    ).copy()

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

    # Store medians for future live prediction.
    feature_medians = X.median(
        numeric_only=True
    )

    # Fill missing values.
    X = X.fillna(
        feature_medians
    )

    if X.empty:
        raise ValueError(
            "No usable model features remain."
        )

    print(
        f"\nNumber of model features: {X.shape[1]}"
    )

    print("\nFirst 20 model features:")

    for feature in X.columns[:20]:
        print(f" - {feature}")

    return (
        X,
        y,
        groups,
        metadata,
        feature_medians,
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
    Split the data while keeping every complete lap together.

    Example:
    P004 Lap 3 cannot appear partly in training
    and partly in testing.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_indices
    ].copy()

    X_test = X.iloc[
        test_indices
    ].copy()

    y_train = y.iloc[
        train_indices
    ].copy()

    y_test = y.iloc[
        test_indices
    ].copy()

    metadata_train = metadata.iloc[
        train_indices
    ].copy()

    metadata_test = metadata.iloc[
        test_indices
    ].copy()

    train_groups = groups.iloc[
        train_indices
    ]

    test_groups = groups.iloc[
        test_indices
    ]

    overlap = set(
        train_groups
    ).intersection(
        set(test_groups)
    )

    if overlap:
        raise RuntimeError(
            "Data leakage detected: "
            "some laps appear in both sets."
        )

    print("\nData split completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(
        f"Training laps: "
        f"{train_groups.nunique()}"
    )
    print(
        f"Testing laps: "
        f"{test_groups.nunique()}"
    )
    print("Lap overlap between sets: 0")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    )


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:
    """
    Train the sector time-loss regression model.
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
        "\nTraining sector XGBoost model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    return model


# =========================================================
# EVALUATE XGBOOST
# =========================================================

def evaluate_xgboost(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Evaluate the XGBoost model.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 65)
    print("XGBOOST SECTOR MODEL RESULTS")
    print("=" * 65)

    print(f"\nMAE:  {mae:.4f} seconds")
    print(f"RMSE: {rmse:.4f} seconds")
    print(f"R²:   {r2:.4f}")

    print(
        "\nMeaning:"
        f"\nThe XGBoost prediction differs from the "
        f"true capped time loss by approximately "
        f"{mae:.3f} seconds on average."
    )

    return {
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =========================================================
# PER-SECTOR AVERAGE BASELINE
# =========================================================

def evaluate_sector_average_baseline(
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Create a simple baseline.

    For every sector, calculate the average time loss
    from training data.

    Example:
    Average Turn 5 loss in training = 1.8 seconds.

    The baseline predicts 1.8 seconds for every
    Turn 5 row in the test set.
    """

    if "Sector" not in metadata_train.columns:
        raise ValueError(
            "Sector column is required "
            "for baseline evaluation."
        )

    train_baseline_df = (
        metadata_train.copy()
    )

    train_baseline_df[
        "training_target"
    ] = y_train.to_numpy()

    # Average time loss for every sector.
    sector_averages = (
        train_baseline_df
        .groupby("Sector")[
            "training_target"
        ]
        .mean()
    )

    # Backup value if a sector is missing.
    overall_average = y_train.mean()

    # Map each test sector to its training average.
    baseline_predictions = (
        metadata_test["Sector"]
        .map(sector_averages)
        .fillna(overall_average)
        .to_numpy()
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    baseline_mse = mean_squared_error(
        y_test,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        baseline_mse
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions,
    )

    print("\n" + "=" * 65)
    print("PER-SECTOR AVERAGE BASELINE")
    print("=" * 65)

    print(
        f"\nBaseline MAE:  "
        f"{baseline_mae:.4f} seconds"
    )

    print(
        f"Baseline RMSE: "
        f"{baseline_rmse:.4f} seconds"
    )

    print(
        f"Baseline R²:   "
        f"{baseline_r2:.4f}"
    )

    print(
        "\nAverage training target "
        "for every sector:"
    )

    print(
        sector_averages
        .sort_index()
        .to_string()
    )

    return {
        "predictions": baseline_predictions,
        "mae": baseline_mae,
        "rmse": baseline_rmse,
        "r2": baseline_r2,
        "sector_averages": sector_averages,
    }


# =========================================================
# COMPARE XGBOOST WITH BASELINE
# =========================================================

def compare_models(
    xgboost_results: dict,
    baseline_results: dict,
) -> None:
    """
    Compare XGBoost with the simple sector-average guess.
    """

    xgboost_mae = xgboost_results["mae"]
    baseline_mae = baseline_results["mae"]

    improvement_seconds = (
        baseline_mae
        - xgboost_mae
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
        f"{xgboost_mae:.4f} seconds"
    )

    print(
        f"Error reduced by: "
        f"{improvement_seconds:.4f} seconds"
    )

    print(
        f"Percentage improvement: "
        f"{improvement_percentage:.2f}%"
    )

    if xgboost_mae < baseline_mae:
        print(
            "\nResult: XGBoost is better."
        )

        print(
            "This means the telemetry and physics "
            "features provide useful information "
            "beyond simply knowing the sector."
        )

    elif xgboost_mae == baseline_mae:
        print(
            "\nResult: XGBoost and the baseline "
            "perform equally."
        )

    else:
        print(
            "\nResult: The simple baseline is better."
        )

        print(
            "This would mean XGBoost is not yet "
            "learning enough useful information "
            "from the telemetry features."
        )


# =========================================================
# PERFORMANCE BY SECTOR
# =========================================================

def evaluate_by_sector(
    metadata_test: pd.DataFrame,
    y_test: pd.Series,
    xgboost_predictions,
    baseline_predictions,
) -> pd.DataFrame:
    """
    Compare XGBoost and baseline error for every sector.
    """

    results = metadata_test.copy()

    results[
        "actual_time_loss_target"
    ] = y_test.to_numpy()

    results[
        "xgboost_prediction"
    ] = xgboost_predictions

    results[
        "baseline_prediction"
    ] = baseline_predictions

    results[
        "xgboost_absolute_error"
    ] = np.abs(
        results[
            "actual_time_loss_target"
        ]
        - results[
            "xgboost_prediction"
        ]
    )

    results[
        "baseline_absolute_error"
    ] = np.abs(
        results[
            "actual_time_loss_target"
        ]
        - results[
            "baseline_prediction"
        ]
    )

    results[
        "xgboost_better"
    ] = (
        results[
            "xgboost_absolute_error"
        ]
        <
        results[
            "baseline_absolute_error"
        ]
    )

    if "Sector" in results.columns:
        sector_results = (
            results
            .groupby("Sector")
            .agg(
                rows=(
                    "actual_time_loss_target",
                    "size",
                ),
                actual_mean=(
                    "actual_time_loss_target",
                    "mean",
                ),
                xgboost_mean=(
                    "xgboost_prediction",
                    "mean",
                ),
                baseline_mean=(
                    "baseline_prediction",
                    "mean",
                ),
                xgboost_mae=(
                    "xgboost_absolute_error",
                    "mean",
                ),
                baseline_mae=(
                    "baseline_absolute_error",
                    "mean",
                ),
                xgboost_win_rate=(
                    "xgboost_better",
                    "mean",
                ),
            )
        )

        sector_results[
            "mae_improvement"
        ] = (
            sector_results[
                "baseline_mae"
            ]
            - sector_results[
                "xgboost_mae"
            ]
        )

        sector_results = (
            sector_results
            .sort_values(
                "mae_improvement",
                ascending=False,
            )
        )

        print("\n" + "=" * 65)
        print("PERFORMANCE BY TRACK SECTION")
        print("=" * 65)

        print(
            sector_results.to_string()
        )

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"\nDetailed test predictions saved to:\n"
        f"{PREDICTIONS_PATH}"
    )

    return results


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def print_feature_importance(
    model: XGBRegressor,
    feature_names,
) -> None:
    """
    Print the 25 most important model features.
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
    print("TOP 25 IMPORTANT FEATURES")
    print("=" * 65)

    print(
        importance_df
        .head(25)
        .to_string(index=False)
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model_outputs(
    model: XGBRegressor,
    feature_names,
    feature_medians: pd.Series,
) -> None:
    """
    Save model and feature information for live prediction.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_JSON_PATH
    )

    joblib.dump(
        model,
        MODEL_JOBLIB_PATH,
    )

    feature_information = {
        "target": TARGET_COLUMN,
        "feature_columns": list(
            feature_names
        ),
        "feature_medians": {
            column: float(value)
            for column, value
            in feature_medians.items()
            if column in feature_names
            and pd.notna(value)
        },
    }

    with open(
        FEATURES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_information,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("FILES SAVED")
    print("=" * 65)

    print(
        f"\nModel JSON:\n"
        f"{MODEL_JSON_PATH}"
    )

    print(
        f"\nModel joblib:\n"
        f"{MODEL_JOBLIB_PATH}"
    )

    print(
        f"\nFeature information:\n"
        f"{FEATURES_PATH}"
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> None:
    df = load_data()

    (
        X,
        y,
        groups,
        metadata,
        feature_medians,
    ) = prepare_data(df)

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

    xgboost_results = evaluate_xgboost(
        model,
        X_test,
        y_test,
    )

    baseline_results = (
        evaluate_sector_average_baseline(
            metadata_train=metadata_train,
            metadata_test=metadata_test,
            y_train=y_train,
            y_test=y_test,
        )
    )

    compare_models(
        xgboost_results=xgboost_results,
        baseline_results=baseline_results,
    )

    evaluate_by_sector(
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

    save_model_outputs(
        model=model,
        feature_names=X.columns,
        feature_medians=feature_medians,
    )


if __name__ == "__main__":
    main()