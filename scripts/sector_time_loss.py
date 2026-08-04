from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "_Channel Report_Detail.csv"
)

CLEANED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_data_with_time_loss_cleaned.csv"
)

OUTLIERS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_time_outliers.csv"
)

REFERENCE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_reference_times.csv"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

FAST_REFERENCE_PERCENTAGE = 0.20
IQR_MULTIPLIER = 1.5

# Capping limits for the training target.
# Rows are not deleted.
LOWER_CAP_PERCENTILE = 0.01
UPPER_CAP_PERCENTILE = 0.99


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    required_columns = {
        "Sector",
        "SectorTime",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def prepare_basic_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SectorTime"] = pd.to_numeric(
        df["SectorTime"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Sector", "SectorTime"]
    ).copy()

    df = df[df["SectorTime"] > 0].copy()

    print("\nAfter basic cleaning:")
    print(f"Rows remaining: {len(df)}")
    print(f"Unique sections: {df['Sector'].nunique()}")

    return df


def remove_sector_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove only clear section-specific timing outliers
    using the IQR method.
    """

    cleaned_groups = []
    outlier_groups = []

    print("\n" + "=" * 70)
    print("OUTLIER REMOVAL BY TRACK SECTION")
    print("=" * 70)

    for sector_name, group in df.groupby(
        "Sector",
        sort=True,
    ):
        group = group.copy()

        q1 = group["SectorTime"].quantile(0.25)
        q3 = group["SectorTime"].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - IQR_MULTIPLIER * iqr
        upper_limit = q3 + IQR_MULTIPLIER * iqr

        valid_mask = group["SectorTime"].between(
            lower_limit,
            upper_limit,
            inclusive="both",
        )

        valid_group = group[valid_mask].copy()
        outlier_group = group[~valid_mask].copy()

        cleaned_groups.append(valid_group)

        if not outlier_group.empty:
            outlier_group["outlier_lower_limit"] = lower_limit
            outlier_group["outlier_upper_limit"] = upper_limit
            outlier_group["outlier_reason"] = (
                "SectorTime outside section-specific IQR limits"
            )

            outlier_groups.append(outlier_group)

        print(
            f"{sector_name:<18} | "
            f"original: {len(group):>4} | "
            f"kept: {len(valid_group):>4} | "
            f"removed: {len(outlier_group):>3} | "
            f"limits: {lower_limit:.3f} to {upper_limit:.3f}"
        )

    cleaned_df = pd.concat(
        cleaned_groups,
        ignore_index=True,
    )

    if outlier_groups:
        outliers_df = pd.concat(
            outlier_groups,
            ignore_index=True,
        )
    else:
        outliers_df = pd.DataFrame()

    print("\nOutlier-removal summary:")
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Removed rows: {len(outliers_df)}")

    return cleaned_df, outliers_df


def calculate_reference_times(
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the fast reference time for each section
    using the median of its fastest 20% of cleaned rows.
    """

    reference_rows = []

    for sector_name, group in cleaned_df.groupby(
        "Sector",
        sort=True,
    ):
        group = group.sort_values(
            "SectorTime",
            ascending=True,
        )

        reference_count = max(
            1,
            int(
                len(group)
                * FAST_REFERENCE_PERCENTAGE
            ),
        )

        fastest_group = group.head(reference_count)

        reference_time = fastest_group[
            "SectorTime"
        ].median()

        reference_rows.append(
            {
                "Sector": sector_name,
                "reference_sector_time": reference_time,
                "total_clean_records": len(group),
                "reference_records": reference_count,
            }
        )

    return pd.DataFrame(reference_rows)


def create_sector_time_loss(
    cleaned_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create both the original and capped time-loss targets.
    """

    result_df = cleaned_df.merge(
        reference_df[
            [
                "Sector",
                "reference_sector_time",
            ]
        ],
        on="Sector",
        how="left",
        validate="many_to_one",
    )

    result_df["sector_time_loss"] = (
        result_df["SectorTime"]
        - result_df["reference_sector_time"]
    )

    # Find global capping limits.
    lower_cap = result_df[
        "sector_time_loss"
    ].quantile(LOWER_CAP_PERCENTILE)

    upper_cap = result_df[
        "sector_time_loss"
    ].quantile(UPPER_CAP_PERCENTILE)

    # Keep all rows, but limit extreme target values.
    result_df["sector_time_loss_capped"] = (
        result_df["sector_time_loss"]
        .clip(
            lower=lower_cap,
            upper=upper_cap,
        )
    )

    result_df["time_loss_was_capped"] = (
        result_df["sector_time_loss"]
        != result_df["sector_time_loss_capped"]
    )

    print("\nTarget capping limits:")
    print(f"Lower 1% cap: {lower_cap:.6f}")
    print(f"Upper 99% cap: {upper_cap:.6f}")

    print(
        "Rows with capped target:",
        int(result_df["time_loss_was_capped"].sum()),
    )

    return result_df


def print_results(
    final_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print("REFERENCE TIMES")
    print("=" * 70)

    print(reference_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("ORIGINAL SECTOR TIME-LOSS SUMMARY")
    print("=" * 70)

    print(
        final_df["sector_time_loss"]
        .describe()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("CAPPED TRAINING TARGET SUMMARY")
    print("=" * 70)

    print(
        final_df["sector_time_loss_capped"]
        .describe()
        .to_string()
    )

    display_columns = [
        column
        for column in [
            "PID",
            "Lap",
            "Sector",
            "SectorTime",
            "reference_sector_time",
            "sector_time_loss",
            "sector_time_loss_capped",
            "time_loss_was_capped",
        ]
        if column in final_df.columns
    ]

    print("\nLargest original sector time losses:")

    print(
        final_df[display_columns]
        .sort_values(
            "sector_time_loss",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nFastest relative section records:")

    print(
        final_df[display_columns]
        .sort_values(
            "sector_time_loss",
            ascending=True,
        )
        .head(20)
        .to_string(index=False)
    )

    if not outliers_df.empty:
        print("\nLargest removed IQR outliers:")

        outlier_columns = [
            column
            for column in [
                "PID",
                "Lap",
                "Sector",
                "SectorTime",
                "outlier_lower_limit",
                "outlier_upper_limit",
            ]
            if column in outliers_df.columns
        ]

        print(
            outliers_df[outlier_columns]
            .sort_values(
                "SectorTime",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )


def save_outputs(
    final_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> None:
    CLEANED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        CLEANED_OUTPUT_PATH,
        index=False,
    )

    reference_df.to_csv(
        REFERENCE_OUTPUT_PATH,
        index=False,
    )

    outliers_df.to_csv(
        OUTLIERS_OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\nCleaned sector dataset:\n"
        f"{CLEANED_OUTPUT_PATH}"
    )

    print(
        f"\nRemoved IQR outliers:\n"
        f"{OUTLIERS_OUTPUT_PATH}"
    )

    print(
        f"\nSection reference times:\n"
        f"{REFERENCE_OUTPUT_PATH}"
    )


def main() -> None:
    df = load_data()

    prepared_df = prepare_basic_data(df)

    cleaned_df, outliers_df = remove_sector_outliers(
        prepared_df
    )

    reference_df = calculate_reference_times(
        cleaned_df
    )

    final_df = create_sector_time_loss(
        cleaned_df,
        reference_df,
    )

    print_results(
        final_df=final_df,
        reference_df=reference_df,
        outliers_df=outliers_df,
    )

    save_outputs(
        final_df=final_df,
        outliers_df=outliers_df,
        reference_df=reference_df,
    )


if __name__ == "__main__":
    main()