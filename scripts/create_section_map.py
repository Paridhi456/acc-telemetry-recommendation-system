from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sector_data_with_time_loss_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "laguna_seca_section_map.json"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    required_columns = {
        "Sector",
        "Sector_Start",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["Sector_Start"] = pd.to_numeric(
        df["Sector_Start"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Sector",
            "Sector_Start",
        ]
    ).copy()

    # One stable start value for every custom section.
    section_map = (
        df.groupby(
            "Sector",
            as_index=False,
        )
        .agg(
            start=(
                "Sector_Start",
                "median",
            )
        )
        .sort_values("start")
        .reset_index(drop=True)
    )

    # Historical track length.
    # Your current section data gives 3442 as the lap end.
    track_length = 3442.0

    # Each section finishes when the next section begins.
    section_map["end"] = (
        section_map["start"].shift(-1)
    )

    first_section_start = float(
        section_map.iloc[0]["start"]
    )

    # Turn 10 crosses the start/finish line and continues
    # until the first section starts at distance 282.
    section_map.loc[
        section_map.index[-1],
        "end",
    ] = track_length + first_section_start

    section_map["length"] = (
        section_map["end"]
        - section_map["start"]
    )

    output_data = {
        "track": "Laguna Seca",
        "track_length": track_length,
        "first_section_start": first_section_start,
        "sections": section_map.to_dict(
            orient="records"
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=4,
        )

    print("\nCorrected Laguna Seca section map")
    print("=" * 75)

    print(
        section_map[
            [
                "Sector",
                "start",
                "end",
                "length",
            ]
        ].to_string(index=False)
    )

    print(f"\nTrack length: {track_length:.1f}")
    print(f"First section starts at: {first_section_start:.1f}")

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()