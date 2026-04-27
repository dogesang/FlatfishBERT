"""
Configuration for sequence statistics analysis (supports Figure 3B-C).
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DATA_PATH = Path(os.getenv(
    "FLATFISH_EXP4_TRAIN",
    str(PROJECT_ROOT / "data" / "benchmark" / "Exp4_cross_dedup" / "train.json")
))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

TARGET_LABELS = ["CDS", "intron"]

SPECIES_NAMES = {
    "GCF_000523025.1": "C. semilaevis",
    "GCF_022379125.1": "S. maximus",
    "GCF_024713975.1": "P. olivaceus",
}
