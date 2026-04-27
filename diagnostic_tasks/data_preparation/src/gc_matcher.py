"""
GC含量统计与匹配模块

核心功能:
- 计算样本的GC含量分布
- 对负样本进行GC匹配采样
- 生成GC分布统计报告
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED
from src.window_sampler import WindowSample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GCMatcher:
    """GC含量匹配器"""

    def __init__(
        self,
        num_bins: int = 10,
        random_seed: int = RANDOM_SEED
    ):
        """
        Args:
            num_bins: GC含量分箱数量
            random_seed: 随机种子
        """
        self.num_bins = num_bins
        self.random_seed = random_seed
        random.seed(random_seed)

        # GC分箱边界 (0-100%)
        self.bin_edges = np.linspace(0, 1, num_bins + 1)

    def calculate_gc(self, sequence: str) -> float:
        """计算序列的GC含量"""
        if not sequence:
            return 0.0
        seq = sequence.upper()
        gc_count = seq.count('G') + seq.count('C')
        valid_count = len(seq) - seq.count('N')
        if valid_count == 0:
            return 0.0
        return gc_count / valid_count

    def get_gc_bin(self, gc_content: float) -> int:
        """获取GC含量所属的分箱索引"""
        for i in range(self.num_bins):
            if gc_content <= self.bin_edges[i + 1]:
                return i
        return self.num_bins - 1

    def compute_gc_distribution(self, samples: List[WindowSample]) -> Dict[int, int]:
        """计算样本集的GC分布

        Args:
            samples: 样本列表

        Returns:
            每个分箱的样本数量
        """
        distribution = defaultdict(int)

        for sample in samples:
            gc = sample.gc_content if sample.gc_content > 0 else self.calculate_gc(sample.sequence)
            bin_idx = self.get_gc_bin(gc)
            distribution[bin_idx] += 1

        return dict(distribution)

    def match_gc_distribution(
        self,
        target_samples: List[WindowSample],
        source_samples: List[WindowSample],
        tolerance: float = 0.1
    ) -> List[WindowSample]:
        """根据目标分布从源样本中匹配采样

        Args:
            target_samples: 目标样本（用于确定GC分布）
            source_samples: 源样本（从中采样）
            tolerance: 允许的分布偏差

        Returns:
            匹配后的样本列表
        """
        logger.info("执行GC分布匹配...")

        # 计算目标分布
        target_dist = self.compute_gc_distribution(target_samples)
        total_target = sum(target_dist.values())

        # 计算目标比例
        target_ratios = {
            bin_idx: count / total_target
            for bin_idx, count in target_dist.items()
        }

        # 按GC分箱组织源样本
        source_by_bin: Dict[int, List[WindowSample]] = defaultdict(list)
        for sample in source_samples:
            gc = sample.gc_content if sample.gc_content > 0 else self.calculate_gc(sample.sequence)
            bin_idx = self.get_gc_bin(gc)
            source_by_bin[bin_idx].append(sample)

        # 计算每个分箱需要采样的数量
        total_source = len(source_samples)
        matched_samples = []

        for bin_idx in range(self.num_bins):
            target_ratio = target_ratios.get(bin_idx, 0)
            target_count = int(target_ratio * total_source)

            available = source_by_bin.get(bin_idx, [])

            if len(available) >= target_count:
                # 有足够的样本，随机采样
                selected = random.sample(available, target_count)
            else:
                # 样本不足，全部使用
                selected = available
                logger.warning(f"  分箱 {bin_idx}: 需要 {target_count}, 仅有 {len(available)}")

            matched_samples.extend(selected)

        logger.info(f"✓ GC匹配完成: {len(source_samples):,} → {len(matched_samples):,}")

        # 验证匹配效果
        self._verify_matching(target_samples, matched_samples)

        return matched_samples

    def _verify_matching(
        self,
        target_samples: List[WindowSample],
        matched_samples: List[WindowSample]
    ) -> None:
        """验证匹配效果"""
        target_dist = self.compute_gc_distribution(target_samples)
        matched_dist = self.compute_gc_distribution(matched_samples)

        total_target = sum(target_dist.values())
        total_matched = sum(matched_dist.values())

        logger.info("  GC分布对比:")
        logger.info(f"  {'Bin':<6} {'Target%':<10} {'Matched%':<10} {'Diff':<10}")

        for bin_idx in range(self.num_bins):
            target_pct = target_dist.get(bin_idx, 0) / total_target * 100 if total_target > 0 else 0
            matched_pct = matched_dist.get(bin_idx, 0) / total_matched * 100 if total_matched > 0 else 0
            diff = matched_pct - target_pct

            gc_range = f"{self.bin_edges[bin_idx]*100:.0f}-{self.bin_edges[bin_idx+1]*100:.0f}%"
            logger.info(f"  {gc_range:<6} {target_pct:>8.2f}% {matched_pct:>8.2f}% {diff:>+8.2f}%")

    def stratified_sample_by_gc(
        self,
        samples: List[WindowSample],
        target_count: int,
        target_distribution: Optional[Dict[int, float]] = None
    ) -> List[WindowSample]:
        """按GC分布分层采样

        Args:
            samples: 源样本列表
            target_count: 目标采样数量
            target_distribution: 目标分布（可选，默认均匀分布）

        Returns:
            采样后的样本列表
        """
        logger.info(f"执行GC分层采样: {len(samples):,} → {target_count:,}...")

        # 按GC分箱组织样本
        samples_by_bin: Dict[int, List[WindowSample]] = defaultdict(list)
        for sample in samples:
            gc = sample.gc_content if sample.gc_content > 0 else self.calculate_gc(sample.sequence)
            bin_idx = self.get_gc_bin(gc)
            samples_by_bin[bin_idx].append(sample)

        # 确定每个分箱的采样数量
        if target_distribution is None:
            # 均匀分布
            per_bin = target_count // self.num_bins
            bin_counts = {i: per_bin for i in range(self.num_bins)}
        else:
            bin_counts = {
                bin_idx: int(ratio * target_count)
                for bin_idx, ratio in target_distribution.items()
            }

        # 采样
        result = []
        for bin_idx, count in bin_counts.items():
            available = samples_by_bin.get(bin_idx, [])
            if len(available) >= count:
                selected = random.sample(available, count)
            else:
                selected = available
            result.extend(selected)

        random.shuffle(result)
        logger.info(f"✓ 分层采样完成: {len(result):,} 样本")

        return result

    def get_statistics(self, samples: List[WindowSample]) -> Dict:
        """获取GC统计信息

        Args:
            samples: 样本列表

        Returns:
            统计信息字典
        """
        gc_values = []
        for sample in samples:
            gc = sample.gc_content if sample.gc_content > 0 else self.calculate_gc(sample.sequence)
            gc_values.append(gc)

        gc_array = np.array(gc_values)

        stats = {
            'count': len(gc_values),
            'mean': float(np.mean(gc_array)),
            'std': float(np.std(gc_array)),
            'min': float(np.min(gc_array)),
            'max': float(np.max(gc_array)),
            'median': float(np.median(gc_array)),
            'q25': float(np.percentile(gc_array, 25)),
            'q75': float(np.percentile(gc_array, 75)),
            'distribution': self.compute_gc_distribution(samples),
        }

        return stats

    def print_statistics(self, samples: List[WindowSample], name: str = "Samples") -> None:
        """打印GC统计信息"""
        stats = self.get_statistics(samples)

        print(f"\n{name} GC统计:")
        print(f"  样本数: {stats['count']:,}")
        print(f"  均值: {stats['mean']*100:.2f}%")
        print(f"  标准差: {stats['std']*100:.2f}%")
        print(f"  范围: {stats['min']*100:.2f}% - {stats['max']*100:.2f}%")
        print(f"  中位数: {stats['median']*100:.2f}%")
        print(f"  四分位: {stats['q25']*100:.2f}% - {stats['q75']*100:.2f}%")

        print(f"\n  GC分布:")
        total = stats['count']
        for bin_idx in range(self.num_bins):
            count = stats['distribution'].get(bin_idx, 0)
            pct = count / total * 100 if total > 0 else 0
            gc_range = f"{self.bin_edges[bin_idx]*100:.0f}-{self.bin_edges[bin_idx+1]*100:.0f}%"
            bar = '█' * int(pct / 2)
            print(f"    {gc_range:<10} {count:>6} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    # 测试代码
    print("测试GCMatcher...")

    # 创建模拟数据
    def generate_sequence(gc_target: float, length: int = 512) -> str:
        """生成指定GC含量的序列"""
        gc_count = int(length * gc_target)
        at_count = length - gc_count

        bases = ['G'] * (gc_count // 2) + ['C'] * (gc_count - gc_count // 2)
        bases += ['A'] * (at_count // 2) + ['T'] * (at_count - at_count // 2)
        random.shuffle(bases)
        return ''.join(bases)

    # 生成两组不同GC分布的样本
    target_samples = []
    for i in range(500):
        # 目标分布：偏高GC (40-60%)
        gc = random.uniform(0.4, 0.6)
        seq = generate_sequence(gc)
        sample = WindowSample(
            sequence=seq, seqname='chr1', start=i*512, end=(i+1)*512-1,
            strand='+', label='target', region_type='test',
            gc_content=gc
        )
        target_samples.append(sample)

    source_samples = []
    for i in range(1000):
        # 源分布：均匀分布 (20-80%)
        gc = random.uniform(0.2, 0.8)
        seq = generate_sequence(gc)
        sample = WindowSample(
            sequence=seq, seqname='chr1', start=i*512, end=(i+1)*512-1,
            strand='+', label='source', region_type='test',
            gc_content=gc
        )
        source_samples.append(sample)

    matcher = GCMatcher(num_bins=10)

    # 打印原始统计
    matcher.print_statistics(target_samples, "Target")
    matcher.print_statistics(source_samples, "Source (原始)")

    # 执行匹配
    matched = matcher.match_gc_distribution(target_samples, source_samples)
    matcher.print_statistics(matched, "Source (匹配后)")

    print("\n测试完成!")
