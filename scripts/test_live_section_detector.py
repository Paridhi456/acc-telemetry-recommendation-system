
from pathlib import Path
import json
import time

from pyaccsharedmemory import accSharedMemory


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECTION_MAP_PATH = (
    PROJECT_ROOT
    / "data"
    / "laguna_seca_section_map.json"
)


def load_section_map():
    if not SECTION_MAP_PATH.exists():
        raise FileNotFoundError(
            "Section map not found. Run "
            "scripts/create_section_map.py first."
        )

    with SECTION_MAP_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_current_section(
    normalized_position: float,
    track_length: float,
    sections: list[dict],
):
    """
    Convert normalized lap position into the matching
    historical custom section.
    """

def find_current_section(
    normalized_position: float,
    track_length: float,
    sections: list[dict],
):
    """
    Find the current custom section.

    The final section may wrap across the
    start/finish line.
    """

    if normalized_position is None:
        return None

    normalized_position = float(
        normalized_position
    )

    normalized_position = (
        normalized_position % 1.0
    )

    distance = (
        normalized_position
        * track_length
    )

    first_start = float(
        sections[0]["start"]
    )

    # A position before the first section start belongs
    # to the final section because it wraps across the
    # start/finish line.
    search_distance = distance

    if distance < first_start:
        search_distance = (
            distance + track_length
        )

    for section in sections:
        start = float(section["start"])
        end = float(section["end"])

        if start <= search_distance < end:
            return {
                "name": section["Sector"],
                "distance": distance,
                "start": start,
                "end": end,
                "length": float(
                    section["length"]
                ),
            }

    return None


def main() -> None:
    section_data = load_section_map()

    track_length = float(
        section_data["track_length"]
    )

    sections = section_data["sections"]

    acc = accSharedMemory()

    previous_section = None

    print("\nLive custom-section detector")
    print("=" * 60)
    print("Drive in ACC. Press Ctrl+C to stop.")

    try:
        while True:
            shared = acc.read_shared_memory()

            if shared is None:
                print(
                    "Waiting for ACC...",
                    end="\r",
                )
                time.sleep(1)
                continue

            graphics = shared.Graphics

            current = find_current_section(
                normalized_position=(
                    graphics.normalized_car_position
                ),
                track_length=track_length,
                sections=sections,
            )

            if current is None:
                time.sleep(0.1)
                continue

            current_name = current["name"]

            # Print only when the section changes.
            if current_name != previous_section:
                print(
                    f"\nEntered: {current_name}"
                    f"\nTrack distance: "
                    f"{current['distance']:.1f}"
                    f"\nSection range: "
                    f"{current['start']:.1f} "
                    f"to {current['end']:.1f}"
                )

                previous_section = current_name

            print(
                f"Current section: {current_name:<18} | "
                f"Position: "
                f"{graphics.normalized_car_position:.4f}",
                end="\r",
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped.")

    finally:
        acc.close()


if __name__ == "__main__":
    main()