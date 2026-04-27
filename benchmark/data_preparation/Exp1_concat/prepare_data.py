#!/usr/bin/env python3
"""
Exp1: CDS拼接 + 位置级去重 + 基因级划分
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data_base import (
    BaseDataPreparer, concatenate_cds_by_gene, position_dedup
)
from config import get_experiment_config


class Exp1DataPreparer(BaseDataPreparer):
    """Exp1: CDS拼接 + 位置级去重 + 基因级划分"""

    def process_cds(self, samples):
        """CDS拼接"""
        return concatenate_cds_by_gene(samples)

    def deduplicate(self, samples):
        """位置级去重"""
        return position_dedup(samples)

    def split_data(self, samples):
        """使用基因级划分（intron等非CDS样本仍需基因级划分）"""
        return self.gene_level_split(samples)


def main():
    config = get_experiment_config('Exp1_concat')
    preparer = Exp1DataPreparer(
        exp_name=config.exp_name,
        output_dir=config.data_dir
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
