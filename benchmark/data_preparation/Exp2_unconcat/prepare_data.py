#!/usr/bin/env python3
"""
Exp2: CDS不拼接 + 位置级去重 + 随机分层划分（不考虑基因级）
对应原Task2A设计
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data_base import BaseDataPreparer, position_dedup
from config import get_experiment_config


class Exp2DataPreparer(BaseDataPreparer):
    """Exp2: CDS不拼接 + 位置级去重 + 随机分层划分"""

    def process_cds(self, samples):
        """不拼接，直接返回"""
        return samples

    def deduplicate(self, samples):
        """位置级去重"""
        return position_dedup(samples)

    def split_data(self, samples):
        """使用随机分层划分，不考虑基因级"""
        return self.random_stratified_split(samples)


def main():
    config = get_experiment_config('Exp2_unconcat')
    preparer = Exp2DataPreparer(
        exp_name=config.exp_name,
        output_dir=config.data_dir
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
