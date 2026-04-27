"""
基础统计模块：生成 Figure3 Panel B 所需的 region_stats.csv
"""
import pandas as pd
from typing import Dict, List

from config import OUTPUT_DIR, SPECIES_NAMES


def calculate_gc(sequence: str) -> float:
    """计算单条序列的GC含量"""
    seq_upper = sequence.upper()
    gc_count = seq_upper.count('G') + seq_upper.count('C')
    valid_bases = sum(seq_upper.count(b) for b in 'ACGT')
    if valid_bases == 0:
        return 0.0
    return gc_count / valid_bases



def compute_basic_stats_by_species(data: Dict[str, Dict[str, List[dict]]]) -> pd.DataFrame:
    """
    计算所有序列的基础统计信息（保留物种信息）

    Returns:
        DataFrame with columns: [species, species_name, label, length, gc]
    """
    records = []

    for species, labels_dict in data.items():
        sp_name = SPECIES_NAMES.get(species, species)
        for label, samples in labels_dict.items():
            print(f"Computing stats for {sp_name} - {label}: {len(samples):,} sequences")
            for sample in samples:
                seq = sample['sequence']
                records.append({
                    'species': species,
                    'species_name': sp_name,
                    'label': label,
                    'length': len(seq),
                    'gc': calculate_gc(seq),
                })

    return pd.DataFrame(records)



def save_region_stats(df: pd.DataFrame, output_path=None) -> str:
    """保存 region_stats.csv"""
    if output_path is None:
        output_path = OUTPUT_DIR / "region_stats.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved region stats to: {output_path}")
    return str(output_path)
