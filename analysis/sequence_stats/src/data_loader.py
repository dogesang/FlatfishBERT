"""
数据加载模块：为 Figure3 生成序列统计输入
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from config import TRAIN_DATA_PATH, TARGET_LABELS, SPECIES_NAMES


def load_data_by_species_and_label(
    data_path: Optional[Path] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    加载序列数据，按物种和标签分组

    Returns:
        Dict[species][label] = [{'sequence': str, ...}, ...]
    """
    if data_path is None:
        data_path = TRAIN_DATA_PATH
    if labels is None:
        labels = TARGET_LABELS

    print(f"Loading data from: {data_path}")

    data = defaultdict(lambda: defaultdict(list))
    with open(data_path, 'r') as f:
        samples = json.load(f)

    print(f"Total samples in file: {len(samples):,}")

    for sample in samples:
        label = sample.get('label_name')
        species = sample.get('species')
        seq = sample.get('sequence', '')

        if label in labels and species and seq:
            data[species][label].append({
                'sequence': seq,
                'length': len(seq),
                'gene_name': sample.get('gene_name', ''),
            })

    data = {sp: dict(labels_dict) for sp, labels_dict in data.items()}

    print("\nLoaded samples by species and label:")
    for species in sorted(data.keys()):
        sp_name = SPECIES_NAMES.get(species, species)
        print(f"\n  {sp_name} ({species}):")
        for label in labels:
            count = len(data[species].get(label, []))
            print(f"    {label}: {count:,}")

    return data
