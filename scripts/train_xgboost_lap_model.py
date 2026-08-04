from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "lap_data_with_labels.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "xgboost_lap_classifier.json"
ENCODER_PATH = MODEL_DIR / "lap_label_mapping.json"
FEATURES_PATH = MODEL_DIR / "lap_feature_columns.json"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TARGET_COLUMN = "cluster_label"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data() -> pd.DataFrame:
    """
    Load the labelled lap-level dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def prepare_data(df: pd.DataFrame):
    """
    Prepare features and labels for model training.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    # Remove rows where Fast/Slow label is missing
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Clean label text
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only expected labels
    df = df[df[TARGET_COLUMN].isin(["fast", "slow"])].copy()

    print("\nLabel distribution:")
    print(df[TARGET_COLUMN].value_counts())

    # Convert labels to numbers
    # XGBoost requires numerical targets
    label_mapping = {
        "slow": 0,
        "fast": 1,
    }

    y = df[TARGET_COLUMN].map(label_mapping)

    # Columns that should not be used as model inputs
    #
    # PID and Lap are identifiers.
    # cluster_label is the target.
    # cluster may directly reveal the target.
    # LapTime may cause target leakage if labels were created from lap time.
    columns_to_remove = [
        TARGET_COLUMN,
        "cluster",
        "LapTime",
        "Lap Time",
        "lap_time",
        "z",
        "PID",
        "Lap",
    ]

    existing_columns_to_remove = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    X = df.drop(columns=existing_columns_to_remove)

    # Keep numeric columns only for the first model
    X = X.select_dtypes(include=["number"]).copy()

    # Replace infinite values with missing values
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    # Remove columns that are completely empty
    X = X.dropna(axis=1, how="all")

    # Fill remaining missing values with the median
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    # Remove constant columns because they provide no useful information
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        print(
            f"\nRemoving {len(constant_columns)} constant columns."
        )
        X = X.drop(columns=constant_columns)

    if X.empty:
        raise ValueError(
            "No numeric feature columns remain after preparation."
        )

    print(f"\nNumber of model features: {X.shape[1]}")
    print("\nFirst 10 model features:")

    for feature in X.columns[:10]:
        print(f" - {feature}")

    return X, y, label_mapping


def train_model(X_train, y_train) -> XGBClassifier:
    """
    Create and train the XGBoost classifier.
    """

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nTraining XGBoost model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    return model


def evaluate_model(model, X_test, y_test) -> None:
    """
    Evaluate the trained model using unseen test data.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Slow", "Fast"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


def print_feature_importance(model, feature_names) -> None:
    """
    Print the 20 most important features.
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nTop 20 important features:")
    print(importance_df.head(20).to_string(index=False))


def save_outputs(
    model,
    feature_names,
    label_mapping,
    feature_medians,
) -> None:
    """
    Save the model and everything required for later prediction.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model in its native format
    model.save_model(MODEL_PATH)

    # Save the exact feature names and order
    feature_information = {
        "features": list(feature_names),
        "medians": {
            key: float(value)
            for key, value in feature_medians.items()
        },
    }

    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_information, file, indent=4)

    # Save label meaning
    inverse_mapping = {
        str(number): label.capitalize()
        for label, number in label_mapping.items()
    }

    with open(ENCODER_PATH, "w", encoding="utf-8") as file:
        json.dump(inverse_mapping, file, indent=4)

    # Optional: save complete Python model object too
    joblib.dump(
        model,
        MODEL_DIR / "xgboost_lap_classifier.joblib",
    )

    print("\nSaved files:")

    print(f"Model:       {MODEL_PATH}")
    print(f"Features:    {FEATURES_PATH}")
    print(f"Label map:   {ENCODER_PATH}")


def main() -> None:
    """
    Run the complete training pipeline.
    """

    df = load_data()

    X, y, label_mapping = prepare_data(df)

    # Store medians for future live-data missing-value handling
    feature_medians = X.median(numeric_only=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_feature_importance(
        model,
        X.columns,
    )

    save_outputs(
        model=model,
        feature_names=X.columns,
        label_mapping=label_mapping,
        feature_medians=feature_medians,
    )


if __name__ == "__main__":
    main()