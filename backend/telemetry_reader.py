from pyaccsharedmemory import accSharedMemory


def show_fields(name: str, obj) -> None:
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    attributes = [
        attribute
        for attribute in dir(obj)
        if not attribute.startswith("_")
        and not callable(getattr(obj, attribute))
    ]

    for attribute in sorted(attributes):
        try:
            value = getattr(obj, attribute)
            print(f"{attribute}: {value}")
        except Exception as error:
            print(f"{attribute}: <unable to read: {error}>")


def inspect_acc_fields() -> None:
    acc = accSharedMemory()

    try:
        shared = acc.read_shared_memory()

        if shared is None:
            print("No ACC telemetry available. Open ACC and start a session.")
            return

        show_fields("PHYSICS FIELDS", shared.Physics)
        show_fields("GRAPHICS FIELDS", shared.Graphics)
        show_fields("STATIC FIELDS", shared.Static)

    finally:
        acc.close()


if __name__ == "__main__":
    inspect_acc_fields()