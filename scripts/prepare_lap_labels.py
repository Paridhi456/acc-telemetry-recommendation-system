from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAP_FILE = PROJECT_ROOT / "data" / "_Channel Report_Only.csv"
DETAIL_FILE = PROJECT_ROOT / "data" / "_Channel Report_Detail.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "lap_data_with_labels.csv"


def prepare_lap_labels() -> None:
    if not LAP_FILE.exists():
        raise FileNotFoundError(f"Lap report not found: {LAP_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"Detail report not found: {DETAIL_FILE}")

    lap_df = pd.read_csv(LAP_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    print("Lap-level file shape:", lap_df.shape)
    print("Detail file shape:", detail_df.shape)

    required_columns = ["PID", "Lap", "cluster_label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in detail_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in detail report: {missing_columns}"
        )

    labels_df = (
        detail_df[["PID", "Lap", "cluster_label"]]
        .dropna(subset=["cluster_label"])
        .drop_duplicates()
    )

    conflicts = (
        labels_df.groupby(["PID", "Lap"])["cluster_label"]
        .nunique()
    )

    conflicting_laps = conflicts[conflicts > 1]

    if not conflicting_laps.empty:
        raise ValueError(
            "Conflicting labels were found for some laps:\n"
            f"{conflicting_laps}"
        )

    labels_df = labels_df.drop_duplicates(
        subset=["PID", "Lap"]
    )

    merged_df = lap_df.merge(
        labels_df,
        on=["PID", "Lap"],
        how="left",
        validate="one_to_one",
    )

    print("\nMerged file shape:", merged_df.shape)
    print(
        "Laps with labels:",
        merged_df["cluster_label"].notna().sum(),
    )
    print(
        "Laps without labels:",
        merged_df["cluster_label"].isna().sum(),
    )

    print("\nLabel distribution:")
    print(
        merged_df["cluster_label"]
        .value_counts(dropna=False)
    )

    merged_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved merged dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_lap_labels()