from __future__ import annotations

import shap
import csv
import json
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from pyaccsharedmemory import accSharedMemory


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECTION_MAP_PATH = (
    PROJECT_ROOT
    / "data"
    / "laguna_seca_section_map.json"
)

HISTORICAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_data_with_time_loss_cleaned.csv"
)

FEATURE_FILE_PATH = (
    PROJECT_ROOT
    / "models"
    / "live_sector_model_full_data_features.json"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "live_section_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_live_sector_model_full_data.joblib"
)
# ============================================================
# MODEL FEATURES
# ============================================================

BRAKE_FEATURES = [
    "BRAKE _Min",
    "BRAKE _Max",
    "BRAKE _Avg",
    "BRAKE _Start",
    "BRAKE _End",
    "BRAKE _Std Dev",
]

SPEED_FEATURES = [
    "Corr Speed kmh_Min",
    "Corr Speed kmh_Max",
    "Corr Speed kmh_Avg",
    "Corr Speed kmh_Start",
    "Corr Speed kmh_End",
    "Corr Speed kmh_Std Dev",
]

THROTTLE_FEATURES = [
    "THROTTLE _Min",
    "THROTTLE _Max",
    "THROTTLE _Avg",
    "THROTTLE _Start",
    "THROTTLE _End",
    "THROTTLE _Std Dev",
]

SECTION_ONE_HOT_FEATURES = [
    "Sector_Str 0-1 (End)",
    "Sector_Str 0-1 (Start)",
    "Sector_Str 1-2",
    "Sector_Str 2-3",
    "Sector_Str 3-4",
    "Sector_Str 4-5",
    "Sector_Str 5-6",
    "Sector_Str 7-8",
    "Sector_Str 8-9",
    "Sector_Str 9-10",
    "Sector_Turn 1",
    "Sector_Turn 10",
    "Sector_Turn 2",
    "Sector_Turn 3",
    "Sector_Turn 4",
    "Sector_Turn 5",
    "Sector_Turn 6",
    "Sector_Turn 7",
    "Sector_Turn 8",
    "Sector_Turn 9",
]

MODEL_FEATURES = (
    BRAKE_FEATURES
    + SPEED_FEATURES
    + THROTTLE_FEATURES
    + [
        "validlap",
        "Sector_Start",
        "Sector_Length",
    ]
    + SECTION_ONE_HOT_FEATURES
)


# Extra columns saved for debugging.
OUTPUT_COLUMNS = [
    "recorded_at",
    "completed_lap",
    "section_name",
    "sample_count",
    "predicted_sector_time_loss",
    "shap_base_value",
    "top_shap_feature_1",
    "top_shap_feature_value_1",
    "top_shap_value_1",
    "top_shap_feature_2",
    "top_shap_feature_value_2",
    "top_shap_value_2",
    "top_shap_feature_3",
    "top_shap_feature_value_3",
    "top_shap_value_3",
] + MODEL_FEATURES


# ============================================================
# FILE LOADING
# ============================================================

def load_section_map() -> dict[str, Any]:
    if not SECTION_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Section map not found:\n{SECTION_MAP_PATH}"
        )

    with SECTION_MAP_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_historical_section_metadata() -> dict[str, dict[str, float]]:
    """
        Load the original historical Sector_Start and Sector_Length.

        Important:
        The detector map uses non-overlapping ranges, but the model
        must receive the original historical median values.

    """

    if not HISTORICAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found:\n"
            f"{HISTORICAL_DATA_PATH}"
        )

    df = pd.read_csv(
        HISTORICAL_DATA_PATH,
        low_memory=False,
        usecols=[
            "Sector",
            "Sector_Start",
            "Sector_Length",
        ],
    )

    df["Sector_Start"] = pd.to_numeric(
        df["Sector_Start"],
        errors="coerce",
    )

    df["Sector_Length"] = pd.to_numeric(
        df["Sector_Length"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Sector",
            "Sector_Start",
            "Sector_Length",
        ]
    )

    metadata_df = (
        df.groupby("Sector", as_index=False)
        .agg(
            Sector_Start=("Sector_Start", "median"),
            Sector_Length=("Sector_Length", "median"),
        )
    )

    metadata: dict[str, dict[str, float]] = {}

    for _, row in metadata_df.iterrows():
        section_name = str(row["Sector"])

        metadata[section_name] = {
            "Sector_Start": float(row["Sector_Start"]),
            "Sector_Length": float(row["Sector_Length"]),
        }

    return metadata


def verify_feature_file() -> None:
    """
    Check that the collector's feature names agree with the
    feature names saved during model training.
    """

    if not FEATURE_FILE_PATH.exists():
        print(
            "\nWarning: model feature file was not found:"
            f"\n{FEATURE_FILE_PATH}"
        )
        print(
            "The collector will continue using the hard-coded "
            "41-feature list."
        )
        return

    with FEATURE_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    saved_features = None

    if isinstance(config, list):
        saved_features = config

    elif isinstance(config, dict):
        for key in [
            "features",
            "feature_names",
            "model_features",
            "feature_columns",
            "columns",
        ]:
            value = config.get(key)

            if isinstance(value, list):
                saved_features = value
                break

    if saved_features is None:
        print(
            "\nWarning: feature names could not be extracted "
            "from the JSON file."
        )
        return

    saved_features = [
        str(feature)
        for feature in saved_features
    ]

    if saved_features == MODEL_FEATURES:
        print(
            "\nFeature validation passed: "
            "all 41 features match the model."
        )
        return

    missing_from_collector = [
        feature
        for feature in saved_features
        if feature not in MODEL_FEATURES
    ]

    extra_in_collector = [
        feature
        for feature in MODEL_FEATURES
        if feature not in saved_features
    ]

    print("\nFeature validation warning!")

    if missing_from_collector:
        print(
            "Missing from collector:",
            missing_from_collector,
        )

    if extra_in_collector:
        print(
            "Extra in collector:",
            extra_in_collector,
        )

    if not missing_from_collector and not extra_in_collector:
        print(
            "Names match, but their order is different."
        )

def load_prediction_model():
    """
    Load the trained live-compatible XGBoost model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(
        "\nXGBoost model loaded successfully:"
        f"\n{MODEL_PATH}"
    )

    return model

def create_shap_explainer(model):
    """
    Create a SHAP TreeExplainer for the trained
    XGBoost regression model.

    The explainer is created once and reused for
    every completed live section.
    """

    explainer = shap.TreeExplainer(model)

    print("\nSHAP explainer created successfully.")

    return explainer
# ============================================================
# SECTION DETECTION
# ============================================================

def find_current_section(
    normalized_position: float,
    track_length: float,
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if normalized_position is None:
        return None

    normalized_position = float(
        normalized_position
    ) % 1.0

    distance = (
        normalized_position
        * track_length
    )

    first_start = float(
        sections[0]["start"]
    )

    search_distance = distance

    # The last section crosses the start/finish line.
    if distance < first_start:
        search_distance = (
            distance + track_length
        )

    for section in sections:
        start = float(section["start"])
        end = float(section["end"])

        if start <= search_distance < end:
            return {
                "name": str(section["Sector"]),
                "distance": distance,
                "detector_start": start,
                "detector_end": end,
            }

    return None


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_summary(
    values: list[float],
    prefix: str,
) -> dict[str, float]:
    """
    Calculate Min, Max, Avg, Start, End and Std Dev.
    """

    if not values:
        raise ValueError(
            f"No values available for {prefix}"
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        f"{prefix}_Min": float(np.min(array)),
        f"{prefix}_Max": float(np.max(array)),
        f"{prefix}_Avg": float(np.mean(array)),
        f"{prefix}_Start": float(array[0]),
        f"{prefix}_End": float(array[-1]),
        f"{prefix}_Std Dev": float(
            np.std(array, ddof=0)
        ),
    }


def create_feature_row(
    section_name: str,
    speed_samples: list[float],
    brake_samples: list[float],
    throttle_samples: list[float],
    valid_lap_samples: list[int],
    completed_lap: int,
    historical_metadata: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if section_name not in historical_metadata:
        raise KeyError(
            f"No historical metadata found for "
            f"section: {section_name}"
        )

    row: dict[str, Any] = {}

    row["recorded_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row["completed_lap"] = completed_lap
    row["section_name"] = section_name
    row["sample_count"] = len(speed_samples)

    # Brake and throttle are converted from ACC's 0–1
    # values into percentage-like 0–100 values.
    row.update(
        calculate_summary(
            brake_samples,
            "BRAKE ",
        )
    )

    row.update(
        calculate_summary(
            speed_samples,
            "Corr Speed kmh",
        )
    )

    row.update(
        calculate_summary(
            throttle_samples,
            "THROTTLE ",
        )
    )

    # Mark the section invalid if any sample was invalid.
    row["validlap"] = int(
        all(valid_lap_samples)
    )

    historical_values = (
        historical_metadata[section_name]
    )

    row["Sector_Start"] = historical_values[
        "Sector_Start"
    ]

    row["Sector_Length"] = historical_values[
        "Sector_Length"
    ]

    # Create all 20 one-hot values.
    for feature_name in SECTION_ONE_HOT_FEATURES:
        row[feature_name] = 0

    active_feature = (
        f"Sector_{section_name}"
    )

    if active_feature not in SECTION_ONE_HOT_FEATURES:
        raise KeyError(
            f"One-hot feature not found for "
            f"section '{section_name}': "
            f"{active_feature}"
        )

    row[active_feature] = 1

    # Ensure all 41 required features exist.
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in row
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: "
            f"{missing_features}"
        )

    return row

def predict_sector_time_loss(
    model,
    row: dict[str, Any],
) -> float:
    """
    Send one completed 41-feature section row
    into the trained XGBoost model.
    """

    model_input = pd.DataFrame(
        [
            {
                feature: row[feature]
                for feature in MODEL_FEATURES
            }
        ],
        columns=MODEL_FEATURES,
    )

    prediction = model.predict(
        model_input
    )

    return float(prediction[0])

def explain_prediction(
    explainer,
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate SHAP values for one completed live section.

    Returns:
    - base value
    - all feature contributions
    - top positive contributions
    - top negative contributions
    """

    # Build one row containing only the exact
    # 41 features expected by the model.
    model_input = pd.DataFrame(
        [
            {
                feature: row[feature]
                for feature in MODEL_FEATURES
            }
        ],
        columns=MODEL_FEATURES,
    )

    # Modern SHAP API.
    explanation = explainer(model_input)

    # One row was passed, so use row index 0.
    shap_values = np.asarray(
        explanation.values[0],
        dtype=float,
    )

    feature_values = np.asarray(
        model_input.iloc[0].values,
        dtype=float,
    )

    # SHAP base value may be scalar or a one-element array.
    base_value = float(
        np.asarray(
            explanation.base_values[0]
        ).reshape(-1)[0]
    )

    contributions = []

    for feature_name, feature_value, shap_value in zip(
        MODEL_FEATURES,
        feature_values,
        shap_values,
    ):
        contributions.append(
            {
                "feature": feature_name,
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
                "absolute_shap_value": float(
                    abs(shap_value)
                ),
            }
        )

    # Features pushing the prediction toward more time loss.
    positive_contributions = sorted(
        [
            item
            for item in contributions
            if item["shap_value"] > 0
        ],
        key=lambda item: item["shap_value"],
        reverse=True,
    )

    # Features pushing the prediction toward less time loss.
    negative_contributions = sorted(
        [
            item
            for item in contributions
            if item["shap_value"] < 0
        ],
        key=lambda item: item["shap_value"],
    )

    # Features with the largest effect in either direction.
    strongest_contributions = sorted(
        contributions,
        key=lambda item: item[
            "absolute_shap_value"
        ],
        reverse=True,
    )

    return {
        "base_value": base_value,
        "all_contributions": contributions,
        "top_positive": positive_contributions[:3],
        "top_negative": negative_contributions[:3],
        "strongest": strongest_contributions[:5],
    }
# ============================================================
# CSV SAVING
# ============================================================

def append_row_to_csv(
    row: dict[str, Any],
) -> None:
    OUTPUT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = OUTPUT_CSV_PATH.exists()

    with OUTPUT_CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                column: row.get(column)
                for column in OUTPUT_COLUMNS
            }
        )


# ============================================================
# MAIN LIVE LOOP
# ============================================================

def main() -> None:
    section_data = load_section_map()

    track_length = float(
        section_data["track_length"]
    )

    sections = section_data["sections"]

    historical_metadata = (
        load_historical_section_metadata()
    )

    verify_feature_file()
    model = load_prediction_model()
 
    shap_explainer = create_shap_explainer(
        model
    )
    acc = accSharedMemory()

    current_section_name: str | None = None

    speed_samples: list[float] = []
    brake_samples: list[float] = []
    throttle_samples: list[float] = []
    valid_lap_samples: list[int] = []

    # The script may begin halfway through a section.
    # Therefore, the first observed section is skipped.
    first_section_is_partial = True

    print("\nLive section feature collector")
    print("=" * 65)
    print("Drive on Laguna Seca.")
    print("Press Ctrl+C to stop.")
    print(
        f"\nOutput file:\n{OUTPUT_CSV_PATH}"
    )

    try:
        while True:
            shared = acc.read_shared_memory()

            if shared is None:
                print(
                    "Waiting for ACC...",
                    end="\r",
                )
                time.sleep(0.5)
                continue

            physics = shared.Physics
            graphics = shared.Graphics

            current = find_current_section(
                normalized_position=(
                    graphics.normalized_car_position
                ),
                track_length=track_length,
                sections=sections,
            )

            if current is None:
                time.sleep(0.02)
                continue

            detected_section = current["name"]

            # First reading after script startup.
            if current_section_name is None:
                current_section_name = (
                    detected_section
                )

                print(
                    f"\nStarted inside: "
                    f"{current_section_name}"
                )

                print(
                    "The first section will be skipped "
                    "because it may be incomplete."
                )

            # A section change means the previous section
            # has just finished.
            elif detected_section != current_section_name:
                if speed_samples:
                    if first_section_is_partial:
                        print(
                            f"\nSkipped initial partial "
                            f"section: "
                            f"{current_section_name}"
                        )

                        first_section_is_partial = False

                    else:
                        completed_lap = int(
                            getattr(
                                graphics,
                                "completed_lap",
                                0,
                            )
                        )

                        row = create_feature_row(
                        section_name=current_section_name,
                        speed_samples=speed_samples,
                        brake_samples=brake_samples,
                        throttle_samples=throttle_samples,
                        valid_lap_samples=valid_lap_samples,
                        completed_lap=completed_lap,
                        historical_metadata=historical_metadata,
                        )

                        predicted_time_loss = predict_sector_time_loss(
                        model=model,
                        row=row,
                        )

                        row["predicted_sector_time_loss"] = (
                        predicted_time_loss
                        )

                        shap_result = explain_prediction(
                        explainer=shap_explainer,
                        row=row,
                        )

                        append_row_to_csv(row)

                        print(
                            f"\nSaved completed section: "
                            f"{current_section_name}"
                        )

                        print(
                            f"Samples: "
                            f"{row['sample_count']}"
                        )

                        print(
                            f"Speed average: "
                            f"{row['Corr Speed kmh_Avg']:.2f} "
                            f"km/h"
                        )

                        print(
                            f"Brake average: "
                            f"{row['BRAKE _Avg']:.2f}"
                        )

                        print(
                            f"Throttle average: "
                            f"{row['THROTTLE _Avg']:.2f}"
                        )

                        print(
                            f"Valid lap: "
                            f"{row['validlap']}"
                        )

                        print(
                            f"Predicted sector time loss: "
                            f"{predicted_time_loss:+.3f} seconds"
                        )

                        print(
                            f"SHAP base value: "
                            f"{shap_result['base_value']:+.3f} seconds"
                        )

                        print("Top features increasing predicted loss:")

                        for item in shap_result["top_positive"]:
                            print(
                                f" - {item['feature']}: "
                                f"value={item['feature_value']:.3f}, "
                                f"SHAP={item['shap_value']:+.3f} s"
                            )

                        print("Top features reducing predicted loss:")

                        for item in shap_result["top_negative"]:
                            print(
                                f" - {item['feature']}: "
                                f"value={item['feature_value']:.3f}, "
                                f"SHAP={item['shap_value']:+.3f} s"
                            )

                        # Start collecting the newly entered section.
                        current_section_name = detected_section

                        speed_samples = []
                        brake_samples = []
                        throttle_samples = []
                        valid_lap_samples = []

                        print(
                            f"\nEntered: {current_section_name}"
                        )
            # Read current live values.
            speed_kmh = float(
                physics.speed_kmh
            )

            brake_percent = float(
                physics.brake
            ) * 100.0

            throttle_percent = float(
                physics.gas
            ) * 100.0

            valid_lap = int(
                bool(graphics.is_valid_lap)
            )

            speed_samples.append(
                speed_kmh
            )

            brake_samples.append(
                brake_percent
            )

            throttle_samples.append(
                throttle_percent
            )

            valid_lap_samples.append(
                valid_lap
            )

            print(
                f"Section: "
                f"{current_section_name:<18} | "
                f"Samples: "
                f"{len(speed_samples):<5} | "
                f"Speed: "
                f"{speed_kmh:6.1f} km/h | "
                f"Brake: "
                f"{brake_percent:5.1f} | "
                f"Throttle: "
                f"{throttle_percent:5.1f}",
                end="\r",
            )

            # Approximately 50 samples per second.
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nCollector stopped.")

    finally:
        try:
            acc.close()
        except Exception:
            pass

        print(
            f"\nCollected rows are stored in:"
            f"\n{OUTPUT_CSV_PATH}"
        )


if __name__ == "__main__":
    main()