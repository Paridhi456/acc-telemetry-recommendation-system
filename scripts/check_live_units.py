import csv
import time
from pathlib import Path

from pyaccsharedmemory import accSharedMemory


OUTPUT_PATH = Path("data/live_unit_test.csv")

SAMPLE_INTERVAL_SECONDS = 0.10


def vector_component(value, component: str):
    """
    Read x, y or z from an ACC vector object.
    """
    try:
        return getattr(value, component)
    except (AttributeError, TypeError):
        return None


def collect_live_units() -> None:
    acc = accSharedMemory()
    rows = []

    print("\nACC live unit test")
    print("=" * 60)
    print("Open ACC and enter a driving session.")
    print("The script records one sample every 0.1 seconds.")
    print("Press Ctrl+C when finished.")
    print("=" * 60)

    try:
        while True:
            shared = acc.read_shared_memory()

            if shared is None:
                print(
                    "Waiting for ACC telemetry...",
                    end="\r",
                )
                time.sleep(1)
                continue

            physics = shared.Physics
            graphics = shared.Graphics

            g_force = physics.g_force
            angular_velocity = physics.local_angular_vel
            local_velocity = physics.local_velocity

            row = {
                "timestamp": time.time(),

                # Main controls
                "speed_kmh": physics.speed_kmh,
                "gas": physics.gas,
                "brake": physics.brake,
                "steer_angle": physics.steer_angle,
                "rpm": physics.rpm,
                "gear": physics.gear,

                # G-force vector
                "g_force_x": vector_component(g_force, "x"),
                "g_force_y": vector_component(g_force, "y"),
                "g_force_z": vector_component(g_force, "z"),

                "angular_velocity_x": vector_component(
                    angular_velocity,
                    "x",
                ),
                "angular_velocity_y": vector_component(
                    angular_velocity,
                    "y",
                ),
                "angular_velocity_z": vector_component(
                    angular_velocity,
                    "z",
                ),

                "local_velocity_x": vector_component(
                    local_velocity,
                    "x",
                ),
                "local_velocity_y": vector_component(
                    local_velocity,
                    "y",
                ),
                "local_velocity_z": vector_component(
                    local_velocity,
                    "z",
                ),

                # Position and lap information
                "normalized_car_position": (
                    graphics.normalized_car_position
                ),
                "current_sector_index": (
                    graphics.current_sector_index
                ),
                "completed_lap": graphics.completed_lap,
                "is_valid_lap": graphics.is_valid_lap,
            }

            rows.append(row)

            print(
                f"Speed={physics.speed_kmh:7.2f} | "
                f"Gas={physics.gas:7.3f} | "
                f"Brake={physics.brake:7.3f} | "
                f"Steer={physics.steer_angle:8.3f} | "
                f"RPM={physics.rpm:6.0f}",
                end="\r",
            )

            time.sleep(SAMPLE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nRecording stopped.")

    finally:
        acc.close()

    if not rows:
        print("No telemetry rows were recorded.")
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRows recorded: {len(rows)}")
    print(f"Saved to:\n{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    collect_live_units()