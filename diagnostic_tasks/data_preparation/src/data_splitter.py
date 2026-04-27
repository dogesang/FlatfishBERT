"""
数据划分模块 - 基因级和Block级数据划分

核心功能:
- 基因级划分: 确保同一基因的所有样本在同一数据集
- Block级划分: 用于intergenic区域，按染色体block划分
- Gene cluster处理: 处理重叠的gene zone
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_RATIO, EVAL_RATIO, TEST_RATIO, RANDOM_SEED, BLOCK_SIZE
from src.window_sampler import WindowSample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """数据划分结果"""
    train: List[WindowSample]
    eval: List[WindowSample]
    test: List[WindowSample]

    def get_statistics(self) -> Dict:
        """获取划分统计信息"""
        stats = {
            'train_count': len(self.train),
            'eval_count': len(self.eval),
            'test_count': len(self.test),
            'total_count': len(self.train) + len(self.eval) + len(self.test),
        }

        # 计算各集合的标签分布
        for name, samples in [('train', self.train), ('eval', self.eval), ('test', self.test)]:
            label_counts = defaultdict(int)
            for s in samples:
                label_counts[s.label] += 1
            stats[f'{name}_labels'] = dict(label_counts)

        return stats


class DataSplitter:
    """数据划分器"""

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        eval_ratio: float = EVAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = RANDOM_SEED
    ):
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        random.seed(random_seed)

        # 验证比例
        total = train_ratio + eval_ratio + test_ratio
        assert abs(total - 1.0) < 1e-6, f"划分比例之和必须为1，当前为{total}"

    def gene_level_split(
        self,
        samples: List[WindowSample],
        gene_clusters: Optional[Dict[str, List[Set[str]]]] = None
    ) -> SplitResult:
        """基因级划分

        确保同一基因（或基因簇）的所有样本在同一数据集中。

        Args:
            samples: 样本列表
            gene_clusters: 基因簇信息（可选，用于处理重叠的gene zone）

        Returns:
            划分结果
        """
        logger.info("执行基因级划分...")

        # 按 (species, gene_id) 分组
        gene_groups: Dict[Tuple[str, str], List[WindowSample]] = defaultdict(list)
        for sample in samples:
            key = (sample.species, sample.gene_id if sample.gene_id else sample.gene_name)
            gene_groups[key].append(sample)

        # 如果提供了基因簇信息，合并同一簇的基因
        if gene_clusters:
            gene_groups = self._merge_gene_clusters(gene_groups, gene_clusters)

        total_genes = len(gene_groups)
        logger.info(f"  总基因/基因簇数: {total_genes:,}")

        # 随机打乱基因列表
        all_genes = list(gene_groups.keys())
        random.shuffle(all_genes)

        # 计算划分点
        train_end = int(self.train_ratio * total_genes)
        eval_end = int((self.train_ratio + self.eval_ratio) * total_genes)

        train_genes = set(all_genes[:train_end])
        eval_genes = set(all_genes[train_end:eval_end])
        test_genes = set(all_genes[eval_end:])

        # 分配样本
        train_samples, eval_samples, test_samples = [], [], []

        for gene_key, gene_samples in gene_groups.items():
            if gene_key in train_genes:
                train_samples.extend(gene_samples)
            elif gene_key in eval_genes:
                eval_samples.extend(gene_samples)
            else:
                test_samples.extend(gene_samples)

        # 打乱样本顺序
        random.shuffle(train_samples)
        random.shuffle(eval_samples)
        random.shuffle(test_samples)

        result = SplitResult(train_samples, eval_samples, test_samples)

        logger.info(f"✓ 基因级划分完成:")
        logger.info(f"  Train: {len(train_samples):,} 样本 ({len(train_genes):,} 基因)")
        logger.info(f"  Eval: {len(eval_samples):,} 样本 ({len(eval_genes):,} 基因)")
        logger.info(f"  Test: {len(test_samples):,} 样本 ({len(test_genes):,} 基因)")

        return result

    def _merge_gene_clusters(
        self,
        gene_groups: Dict[Tuple[str, str], List[WindowSample]],
        gene_clusters: Dict[str, List[Set[str]]]
    ) -> Dict[Tuple[str, str], List[WindowSample]]:
        """合并属于同一基因簇的基因组"""
        # 构建基因到簇的映射
        gene_to_cluster: Dict[Tuple[str, str], Tuple[str, int]] = {}

        for seqname, clusters in gene_clusters.items():
            for cluster_idx, cluster_genes in enumerate(clusters):
                for gene_id in cluster_genes:
                    # 需要遍历所有物种
                    for (species, gid), _ in gene_groups.items():
                        if gid == gene_id:
                            gene_to_cluster[(species, gid)] = (seqname, cluster_idx)

        # 按簇重新分组
        cluster_groups: Dict[Tuple[str, int], List[WindowSample]] = defaultdict(list)
        ungrouped: Dict[Tuple[str, str], List[WindowSample]] = {}

        for gene_key, samples in gene_groups.items():
            if gene_key in gene_to_cluster:
                cluster_key = gene_to_cluster[gene_key]
                cluster_groups[cluster_key].extend(samples)
            else:
                ungrouped[gene_key] = samples

        # 合并结果
        merged = {}
        for cluster_key, samples in cluster_groups.items():
            merged[cluster_key] = samples
        merged.update(ungrouped)

        logger.info(f"  基因簇合并: {len(gene_groups)} 基因 → {len(merged)} 组")

        return merged

    def block_level_split(
        self,
        samples: List[WindowSample],
        block_size: int = BLOCK_SIZE
    ) -> SplitResult:
        """Block级划分

        用于intergenic区域，按染色体block划分。

        Args:
            samples: 样本列表
            block_size: Block大小

        Returns:
            划分结果
        """
        logger.info(f"执行Block级划分 (block_size={block_size:,})...")

        # 按block_id分组
        block_groups: Dict[int, List[WindowSample]] = defaultdict(list)
        for sample in samples:
            block_groups[sample.block_id].append(sample)

        total_blocks = len(block_groups)
        logger.info(f"  总Block数: {total_blocks:,}")

        # 随机打乱block列表
        all_blocks = list(block_groups.keys())
        random.shuffle(all_blocks)

        # 计算划分点
        train_end = int(self.train_ratio * total_blocks)
        eval_end = int((self.train_ratio + self.eval_ratio) * total_blocks)

        train_blocks = set(all_blocks[:train_end])
        eval_blocks = set(all_blocks[train_end:eval_end])
        test_blocks = set(all_blocks[eval_end:])

        # 分配样本
        train_samples, eval_samples, test_samples = [], [], []

        for block_id, block_samples in block_groups.items():
            if block_id in train_blocks:
                train_samples.extend(block_samples)
            elif block_id in eval_blocks:
                eval_samples.extend(block_samples)
            else:
                test_samples.extend(block_samples)

        # 打乱样本顺序
        random.shuffle(train_samples)
        random.shuffle(eval_samples)
        random.shuffle(test_samples)

        result = SplitResult(train_samples, eval_samples, test_samples)

        logger.info(f"✓ Block级划分完成:")
        logger.info(f"  Train: {len(train_samples):,} 样本 ({len(train_blocks):,} blocks)")
        logger.info(f"  Eval: {len(eval_samples):,} 样本 ({len(eval_blocks):,} blocks)")
        logger.info(f"  Test: {len(test_samples):,} 样本 ({len(test_blocks):,} blocks)")

        return result

    def random_stratified_split(
        self,
        samples: List[WindowSample]
    ) -> SplitResult:
        """随机分层划分

        按标签分层，确保各数据集的标签分布一致。

        Args:
            samples: 样本列表

        Returns:
            划分结果
        """
        logger.info("执行随机分层划分...")

        # 按标签分组
        label_groups: Dict[str, List[WindowSample]] = defaultdict(list)
        for sample in samples:
            label_groups[sample.label].append(sample)

        train_samples, eval_samples, test_samples = [], [], []

        for label, label_samples in label_groups.items():
            random.shuffle(label_samples)

            n = len(label_samples)
            train_end = int(self.train_ratio * n)
            eval_end = int((self.train_ratio + self.eval_ratio) * n)

            train_samples.extend(label_samples[:train_end])
            eval_samples.extend(label_samples[train_end:eval_end])
            test_samples.extend(label_samples[eval_end:])

        # 打乱样本顺序
        random.shuffle(train_samples)
        random.shuffle(eval_samples)
        random.shuffle(test_samples)

        result = SplitResult(train_samples, eval_samples, test_samples)

        logger.info(f"✓ 随机分层划分完成:")
        logger.info(f"  Train: {len(train_samples):,}")
        logger.info(f"  Eval: {len(eval_samples):,}")
        logger.info(f"  Test: {len(test_samples):,}")

        return result

    def combined_split(
        self,
        positive_samples: List[WindowSample],
        negative_samples: List[WindowSample],
        positive_strategy: str = 'gene_level',
        negative_strategy: str = 'block_level',
        gene_clusters: Optional[Dict[str, List[Set[str]]]] = None
    ) -> SplitResult:
        """组合划分策略

        用于Task G，正类使用基因级划分，负类使用block级划分。

        Args:
            positive_samples: 正类样本
            negative_samples: 负类样本
            positive_strategy: 正类划分策略
            negative_strategy: 负类划分策略
            gene_clusters: 基因簇信息

        Returns:
            划分结果
        """
        logger.info("执行组合划分策略...")
        logger.info(f"  正类策略: {positive_strategy}")
        logger.info(f"  负类策略: {negative_strategy}")

        # 划分正类
        if positive_strategy == 'gene_level':
            pos_result = self.gene_level_split(positive_samples, gene_clusters)
        else:
            pos_result = self.random_stratified_split(positive_samples)

        # 划分负类
        if negative_strategy == 'block_level':
            neg_result = self.block_level_split(negative_samples)
        else:
            neg_result = self.random_stratified_split(negative_samples)

        # 合并结果
        train_samples = pos_result.train + neg_result.train
        eval_samples = pos_result.eval + neg_result.eval
        test_samples = pos_result.test + neg_result.test

        # 打乱
        random.shuffle(train_samples)
        random.shuffle(eval_samples)
        random.shuffle(test_samples)

        result = SplitResult(train_samples, eval_samples, test_samples)

        logger.info(f"✓ 组合划分完成:")
        logger.info(f"  Train: {len(train_samples):,} (正类 {len(pos_result.train):,}, 负类 {len(neg_result.train):,})")
        logger.info(f"  Eval: {len(eval_samples):,} (正类 {len(pos_result.eval):,}, 负类 {len(neg_result.eval):,})")
        logger.info(f"  Test: {len(test_samples):,} (正类 {len(pos_result.test):,}, 负类 {len(neg_result.test):,})")

        return result

    def verify_no_leakage(
        self,
        result: SplitResult,
        check_gene: bool = True,
        check_block: bool = False
    ) -> bool:
        """验证数据划分没有泄露

        Args:
            result: 划分结果
            check_gene: 是否检查基因级泄露
            check_block: 是否检查block级泄露

        Returns:
            是否通过验证
        """
        logger.info("验证数据划分...")

        passed = True

        if check_gene:
            # 检查基因级泄露
            train_genes = set((s.species, s.gene_id) for s in result.train if s.gene_id)
            eval_genes = set((s.species, s.gene_id) for s in result.eval if s.gene_id)
            test_genes = set((s.species, s.gene_id) for s in result.test if s.gene_id)

            train_eval_overlap = train_genes & eval_genes
            train_test_overlap = train_genes & test_genes
            eval_test_overlap = eval_genes & test_genes

            if train_eval_overlap:
                logger.warning(f"  ⚠️ Train-Eval基因重叠: {len(train_eval_overlap)}")
                passed = False
            if train_test_overlap:
                logger.warning(f"  ⚠️ Train-Test基因重叠: {len(train_test_overlap)}")
                passed = False
            if eval_test_overlap:
                logger.warning(f"  ⚠️ Eval-Test基因重叠: {len(eval_test_overlap)}")
                passed = False

            if passed:
                logger.info("  ✓ 基因级无泄露")

        if check_block:
            # 检查block级泄露
            train_blocks = set(s.block_id for s in result.train if s.block_id >= 0)
            eval_blocks = set(s.block_id for s in result.eval if s.block_id >= 0)
            test_blocks = set(s.block_id for s in result.test if s.block_id >= 0)

            if train_blocks & eval_blocks:
                logger.warning(f"  ⚠️ Train-Eval Block重叠")
                passed = False
            if train_blocks & test_blocks:
                logger.warning(f"  ⚠️ Train-Test Block重叠")
                passed = False
            if eval_blocks & test_blocks:
                logger.warning(f"  ⚠️ Eval-Test Block重叠")
                passed = False

            if passed:
                logger.info("  ✓ Block级无泄露")

        return passed


if __name__ == "__main__":
    # 测试代码
    print("测试DataSplitter...")

    # 创建模拟数据
    samples = []
    for i in range(1000):
        gene_id = f"gene_{i % 100}"  # 100个基因
        block_id = i // 50  # 20个block
        sample = WindowSample(
            sequence="A" * 512,
            seqname="chr1",
            start=i * 512,
            end=(i + 1) * 512 - 1,
            strand='+',
            label='positive' if i % 2 == 0 else 'negative',
            region_type='test',
            gene_id=gene_id,
            block_id=block_id,
            species='test_species'
        )
        samples.append(sample)

    splitter = DataSplitter()

    # 测试基因级划分
    print("\n1. 基因级划分:")
    result = splitter.gene_level_split(samples)
    stats = result.get_statistics()
    print(f"   统计: {stats}")
    splitter.verify_no_leakage(result, check_gene=True)

    # 测试Block级划分
    print("\n2. Block级划分:")
    result = splitter.block_level_split(samples)
    splitter.verify_no_leakage(result, check_block=True)

    # 测试随机分层划分
    print("\n3. 随机分层划分:")
    result = splitter.random_stratified_split(samples)

    print("\n测试完成!")
