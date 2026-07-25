from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "_Channel Report_Only.csv"

TARGET_COLUMN = "cluster_label"
GROUP_COLUMN = "PID"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def validate_data(df: pd.DataFrame) -> None:
    required_columns = [TARGET_COLUMN, GROUP_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"\nParticipants: {df[GROUP_COLUMN].nunique()}")

    print("\nTarget counts:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))

    print("\nTarget percentages:")
    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )
    print(percentages)

    print("\nMissing target values:")
    print(df[TARGET_COLUMN].isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    dataframe = load_data()
    validate_data(dataframe)