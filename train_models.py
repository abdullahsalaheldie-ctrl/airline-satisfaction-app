"""Train both models and write all artifacts to ./models.

Usage:  python train_models.py --data data/train.csv   (a .zip containing the csv also works)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model_evaluation import train_and_save  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()
    if not Path(args.data).exists():
        sys.exit(f"Training data not found: {args.data}. See data/README.md.")
    train_and_save(args.data, args.models_dir)
    print("Artifacts saved to", args.models_dir)
