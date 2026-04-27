#!/usr/bin/env python3
"""
只生成 Figure3 所需的最小序列统计输出：
- outputs/region_stats.csv
- outputs/periodicity_spectrum.csv
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from basic_stats import compute_basic_stats_by_species, save_region_stats
from data_loader import load_data_by_species_and_label
from periodicity import (
    compute_macro_average_spectrum,
    compute_periodicity_by_species,
    save_periodicity_data,
)



def main():
    data = load_data_by_species_and_label()

    region_stats = compute_basic_stats_by_species(data)
    save_region_stats(region_stats)

    spectra_by_species = compute_periodicity_by_species(data)
    spectra_macro = compute_macro_average_spectrum(spectra_by_species)
    save_periodicity_data(spectra_macro, spectra_by_species)

    print("\nDone: generated minimal Figure3 support files.")


if __name__ == "__main__":
    main()
