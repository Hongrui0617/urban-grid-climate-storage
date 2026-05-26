from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    "data_raw",
    "data_processed",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/metrics",
    "outputs/models",
    "logs",
    "src",
]

def main():
    for folder in FOLDERS:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"Project folders created under:\n{PROJECT_ROOT}")

if __name__ == "__main__":
    main()
