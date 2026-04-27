#!/usr/bin/env python3
"""
Exp3: CDS不拼接 + 单物种序列级去重 + 基因级划分
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data_base import BaseDataPreparer, sequence_dedup_single_species
from config import get_experiment_config


class Exp3DataPreparer(BaseDataPreparer):
    """Exp3: CDS不拼接 + 单物种序列级去重 + 基因级划分"""

    def process_cds(self, samples):
        """不拼接，直接返回"""
        return samples

    def deduplicate(self, samples):
        """单物种序列级去重"""
        return sequence_dedup_single_species(samples)

    def split_data(self, samples):
        """使用基因级划分"""
        return self.gene_level_split(samples)


def main():
    config = get_experiment_config('Exp3_seq_dedup')
    preparer = Exp3DataPreparer(
        exp_name=config.exp_name,
        output_dir=config.data_dir
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
