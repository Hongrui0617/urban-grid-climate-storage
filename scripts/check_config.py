import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gridclimate.config import load_config, resolve_project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Config loaded successfully.")
    print("-" * 60)

    print(f"Project name: {config['project']['name']}")
    print(f"City: {config['project']['city']}")
    print(f"Timezone: {config['project']['timezone']}")

    print("-" * 60)

    for key, path in config["paths"].items():
        absolute_path = resolve_project_path(path)
        print(f"{key}: {absolute_path}")
        print(f"  Exists: {absolute_path.exists()}")

    print("-" * 60)

    print("Target columns:")
    for key, value in config["columns"].items():
        print(f"  {key}: {value}")

    print("-" * 60)
    print("Config check completed.")


if __name__ == "__main__":
    main()
