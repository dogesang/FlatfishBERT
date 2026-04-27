#!/usr/bin/env python3
"""
Exp5: CDS不拼接 + 位置级去重 + 基因级划分

对照关系：
- Exp1 vs Exp5: 仅CDS处理不同 → 验证CDS不拼接效果
- Exp2 vs Exp5: 仅划分策略不同 → 验证随机划分是否有数据泄露
- Exp3 vs Exp5: 仅去重策略不同 → 比较去重策略效果
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data_base import BaseDataPreparer, position_dedup
from config import get_experiment_config


class Exp5DataPreparer(BaseDataPreparer):
    """Exp5: CDS不拼接 + 位置级去重 + 基因级划分"""

    def process_cds(self, samples):
        """不拼接，直接返回"""
        return samples

    def deduplicate(self, samples):
        """位置级去重"""
        return position_dedup(samples)

    def split_data(self, samples):
        """使用基因级划分"""
        return self.gene_level_split(samples)


def main():
    config = get_experiment_config('Exp5_unconcat_gene')
    preparer = Exp5DataPreparer(
        exp_name=config.exp_name,
        output_dir=config.data_dir
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
