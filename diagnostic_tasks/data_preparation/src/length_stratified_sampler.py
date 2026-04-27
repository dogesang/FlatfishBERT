"""
长度分层采样模块 - 根据正类长度分布对负类进行分层采样

核心功能:
- 统计正类样本的长度分布
- 根据长度分布定义分层区间
- 从负类区域按分层比例采样
"""

import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED
from src.region_definition import GenomicRegion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LengthBin:
    """长度区间"""
    min_length: int
    max_length: int
    count: int = 0
    ratio: float = 0.0

    @property
    def name(self) -> str:
        if self.max_length == float('inf'):
            return f"{self.min_length}+"
        return f"{self.min_length}-{self.max_length}"

    def contains(self, length: int) -> bool:
        return self.min_length <= length < self.max_length


class LengthStratifiedSampler:
    """长度分层采样器"""

    def __init__(
        self,
        bin_edges: List[int] = None,
        random_seed: int = RANDOM_SEED
    ):
        """
        Args:
            bin_edges: 长度区间边界，例如 [0, 100, 200, 500, 1000, 2000, 5000, inf]
                      如果为None，将根据正类长度分布自动计算
            random_seed: 随机种子
        """
        self.bin_edges = bin_edges
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.bins: List[LengthBin] = []
        self.positive_lengths: List[int] = []

    def analyze_positive_lengths(self, lengths: List[int]) -> Dict:
        """分析正类样本的长度分布

        Args:
            lengths: 正类样本长度列表

        Returns:
            长度统计信息
        """
        self.positive_lengths = lengths

        stats = {
            'count': len(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'percentiles': {
                '10%': np.percentile(lengths, 10),
                '25%': np.percentile(lengths, 25),
                '50%': np.percentile(lengths, 50),
                '75%': np.percentile(lengths, 75),
                '90%': np.percentile(lengths, 90),
                '95%': np.percentile(lengths, 95),
                '99%': np.percentile(lengths, 99),
            }
        }

        logger.info("正类长度分布统计:")
        logger.info(f"  样本数: {stats['count']:,}")
        logger.info(f"  最小值: {stats['min']:,} bp")
        logger.info(f"  最大值: {stats['max']:,} bp")
        logger.info(f"  平均值: {stats['mean']:.0f} bp")
        logger.info(f"  中位数: {stats['median']:.0f} bp")
        logger.info(f"  标准差: {stats['std']:.0f} bp")

        return stats

    def create_bins_from_percentiles(
        self,
        percentiles: List[int] = [0, 10, 25, 50, 75, 90, 100]
    ) -> List[LengthBin]:
        """根据百分位数创建长度区间

        Args:
            percentiles: 百分位数列表

        Returns:
            长度区间列表
        """
        if not self.positive_lengths:
            raise ValueError("请先调用 analyze_positive_lengths()")

        # 计算百分位数对应的长度值
        edges = [int(np.percentile(self.positive_lengths, p)) for p in percentiles]

        # 去重并排序
        edges = sorted(set(edges))

        # 确保最后一个边界足够大
        if edges[-1] < max(self.positive_lengths):
            edges.append(max(self.positive_lengths) + 1)

        self.bin_edges = edges
        return self._create_bins()

    def create_bins_from_edges(self, edges: List[int]) -> List[LengthBin]:
        """根据指定边界创建长度区间

        Args:
            edges: 长度区间边界

        Returns:
            长度区间列表
        """
        self.bin_edges = edges
        return self._create_bins()

    def _create_bins(self) -> List[LengthBin]:
        """创建长度区间"""
        self.bins = []

        for i in range(len(self.bin_edges) - 1):
            bin_obj = LengthBin(
                min_length=self.bin_edges[i],
                max_length=self.bin_edges[i + 1]
            )
            self.bins.append(bin_obj)

        # 添加最后一个开放区间
        if self.bin_edges[-1] != float('inf'):
            self.bins.append(LengthBin(
                min_length=self.bin_edges[-1],
                max_length=float('inf')
            ))

        # 统计正类在各区间的分布
        if self.positive_lengths:
            self._compute_bin_statistics()

        return self.bins

    def _compute_bin_statistics(self):
        """计算各区间的统计信息"""
        total = len(self.positive_lengths)

        for bin_obj in self.bins:
            bin_obj.count = sum(1 for l in self.positive_lengths if bin_obj.contains(l))
            bin_obj.ratio = bin_obj.count / total if total > 0 else 0

        logger.info("\n正类长度区间分布:")
        for bin_obj in self.bins:
            logger.info(f"  {bin_obj.name}: {bin_obj.count:,} ({bin_obj.ratio*100:.1f}%)")

    def get_bin_for_length(self, length: int) -> Optional[LengthBin]:
        """获取指定长度所属的区间"""
        for bin_obj in self.bins:
            if bin_obj.contains(length):
                return bin_obj
        return None

    def sample_from_regions_stratified(
        self,
        regions: List[GenomicRegion],
        sequences: Dict[str, str],
        total_samples: int,
        label: str = 'background',
        species: str = '',
        block_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """从区域中按长度分层采样

        Args:
            regions: 可采样的区域列表
            sequences: 染色体序列字典
            total_samples: 总采样数量
            label: 样本标签
            species: 物种ID
            block_ids: 每个区域对应的block ID

        Returns:
            采样得到的样本列表
        """
        if not self.bins:
            raise ValueError("请先创建长度区间")

        if block_ids is None:
            block_ids = [-1] * len(regions)

        logger.info(f"\n开始长度分层采样 (目标: {total_samples:,} 样本)...")

        # 计算每个区间需要采样的数量
        bin_targets = {}
        for bin_obj in self.bins:
            target = int(total_samples * bin_obj.ratio)
            if target > 0:
                bin_targets[bin_obj.name] = {
                    'target': target,
                    'min_length': bin_obj.min_length,
                    'max_length': bin_obj.max_length,
                    'samples': []
                }

        logger.info("各区间目标采样数:")
        for name, info in bin_targets.items():
            logger.info(f"  {name}: {info['target']:,}")

        # 按区间分组可用区域
        regions_by_bin: Dict[str, List[Tuple[GenomicRegion, int]]] = defaultdict(list)

        for region, block_id in zip(regions, block_ids):
            for bin_name, info in bin_targets.items():
                min_len = info['min_length']
                max_len = info['max_length'] if info['max_length'] != float('inf') else float('inf')

                # 检查区域是否可以提供该长度范围的样本
                if region.length >= min_len:
                    regions_by_bin[bin_name].append((region, block_id))

        # 从每个区间采样
        all_samples = []

        for bin_name, info in bin_targets.items():
            target = info['target']
            min_len = info['min_length']
            max_len = info['max_length'] if info['max_length'] != float('inf') else None

            available_regions = regions_by_bin.get(bin_name, [])

            if not available_regions:
                logger.warning(f"  {bin_name}: 无可用区域")
                continue

            # 随机打乱区域
            random.shuffle(available_regions)

            sampled = 0
            attempts = 0
            max_attempts = target * 10  # 防止无限循环

            while sampled < target and attempts < max_attempts:
                # 随机选择一个区域
                region, block_id = random.choice(available_regions)

                # 确定采样长度
                if max_len is None:
                    # 开放区间，从正类长度分布中采样
                    valid_lengths = [l for l in self.positive_lengths if l >= min_len and l <= region.length]
                    if valid_lengths:
                        sample_length = random.choice(valid_lengths)
                    else:
                        sample_length = min(min_len, region.length)
                else:
                    # 闭合区间
                    actual_max = min(max_len - 1, region.length)
                    if actual_max < min_len:
                        attempts += 1
                        continue
                    sample_length = random.randint(min_len, actual_max)

                # 确定采样位置
                if region.length == sample_length:
                    start = region.start
                else:
                    max_start = region.end - sample_length + 1
                    start = random.randint(region.start, max_start)

                end = start + sample_length - 1

                # 提取序列
                if region.seqname in sequences:
                    seq = sequences[region.seqname][start - 1:end]  # 1-based to 0-based

                    if len(seq) == sample_length:
                        sample = {
                            'sequence': seq,
                            'seqname': region.seqname,
                            'start': start,
                            'end': end,
                            'strand': region.strand,
                            'label': label,
                            'region_type': 'deep_intergenic',
                            'gene_id': '',
                            'gene_name': '',
                            'block_id': block_id,
                            'species': species,
                            'sequence_length': sample_length,
                            'length_bin': bin_name
                        }
                        all_samples.append(sample)
                        sampled += 1

                attempts += 1

            logger.info(f"  {bin_name}: 采样 {sampled:,}/{target:,}")

        logger.info(f"\n总采样: {len(all_samples):,} 样本")

        return all_samples

    def print_sampling_summary(self, samples: List[Dict]):
        """打印采样结果摘要"""
        if not samples:
            logger.info("无样本")
            return

        # 按长度区间统计
        bin_counts = defaultdict(int)
        for s in samples:
            bin_name = s.get('length_bin', 'unknown')
            bin_counts[bin_name] += 1

        logger.info("\n采样结果按长度区间分布:")
        total = len(samples)
        for bin_name, count in sorted(bin_counts.items()):
            logger.info(f"  {bin_name}: {count:,} ({count/total*100:.1f}%)")

        # 长度统计
        lengths = [s['sequence_length'] for s in samples]
        logger.info(f"\n采样长度统计:")
        logger.info(f"  最小值: {min(lengths):,} bp")
        logger.info(f"  最大值: {max(lengths):,} bp")
        logger.info(f"  平均值: {np.mean(lengths):.0f} bp")
        logger.info(f"  中位数: {np.median(lengths):.0f} bp")


if __name__ == "__main__":
    # 测试代码
    print("测试 LengthStratifiedSampler...")

    # 模拟正类长度分布
    np.random.seed(42)
    positive_lengths = list(np.random.lognormal(mean=6, sigma=1, size=10000).astype(int))
    positive_lengths = [max(50, min(l, 50000)) for l in positive_lengths]  # 限制范围

    sampler = LengthStratifiedSampler()

    # 分析正类长度
    stats = sampler.analyze_positive_lengths(positive_lengths)

    # 创建区间
    bins = sampler.create_bins_from_percentiles([0, 10, 25, 50, 75, 90, 100])

    print("\n测试完成!")
