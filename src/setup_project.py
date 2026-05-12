from pathlib import Path

PROJECT_ROOT = Path("/Users/aaron/Desktop/2025 Fall Tokyo/研究/step 3")

FOLDERS = [
    "data_raw",
    "data_processed",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/models",
    "logs",
    "src"
]

def main():
    for folder in FOLDERS:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"Project folders created under:\n{PROJECT_ROOT}")

if __name__ == "__main__":
    main()