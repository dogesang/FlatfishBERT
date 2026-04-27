#!/usr/bin/env python3
"""
V3 Unified Experiments - 基础数据处理类
提供所有实验共用的数据处理功能
"""

import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import logging

from config import (
    RAW_DATA_PATH, ORIGINAL_TO_BINARY, LABEL2ID,
    TRAIN_RATIO, EVAL_RATIO, TEST_RATIO, RANDOM_SEED
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseDataPreparer(ABC):
    """基础数据准备类 - 所有实验继承此类"""

    def __init__(self, exp_name: str, output_dir: Path):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(RANDOM_SEED)

    def load_raw_data(self) -> List[Dict]:
        """加载原始数据"""
        logger.info(f"加载原始数据: {RAW_DATA_PATH}")
        with open(RAW_DATA_PATH, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ 加载完成: {len(data):,} 条样本")
        return data

    def convert_to_binary_labels(self, samples: List[Dict]) -> List[Dict]:
        """转换为二分类标签"""
        logger.info("转换为二分类标签...")

        converted = []
        skipped = 0

        for sample in samples:
            original_label = sample.get('label_name', '')

            if original_label not in ORIGINAL_TO_BINARY:
                skipped += 1
                continue

            new_sample = sample.copy()
            new_sample['original_label'] = original_label
            new_sample['label'] = ORIGINAL_TO_BINARY[original_label]
            new_sample['label_id'] = LABEL2ID[new_sample['label']]
            converted.append(new_sample)

        logger.info(f"✓ 转换完成: {len(converted):,} 条, 跳过: {skipped:,} 条")

        # 统计标签分布
        label_counts = Counter([s['label'] for s in converted])
        for label, count in sorted(label_counts.items()):
            pct = count / len(converted) * 100
            logger.info(f"  {label}: {count:,} ({pct:.2f}%)")

        return converted

    def random_stratified_split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """随机分层划分 - 按标签分层，不考虑基因级"""
        logger.info("执行随机分层划分...")

        from sklearn.model_selection import train_test_split

        # 提取标签用于分层
        labels = [s['label'] for s in samples]

        # 先分出train和temp（eval+test）
        train_samples, temp_samples, train_labels, temp_labels = train_test_split(
            samples, labels,
            test_size=(EVAL_RATIO + TEST_RATIO),
            stratify=labels,
            random_state=RANDOM_SEED
        )

        # 再从temp分出eval和test
        eval_ratio_adjusted = EVAL_RATIO / (EVAL_RATIO + TEST_RATIO)
        eval_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1 - eval_ratio_adjusted),
            stratify=temp_labels,
            random_state=RANDOM_SEED
        )

        logger.info(f"✓ 划分完成:")
        logger.info(f"  Train: {len(train_samples):,} 条")
        logger.info(f"  Eval: {len(eval_samples):,} 条")
        logger.info(f"  Test: {len(test_samples):,} 条")

        # 打印各集合的标签分布
        for name, data in [("Train", train_samples), ("Eval", eval_samples), ("Test", test_samples)]:
            label_counts = Counter([s['label'] for s in data])
            for label, count in sorted(label_counts.items()):
                pct = count / len(data) * 100
                logger.info(f"    {name} {label}: {count:,} ({pct:.2f}%)")

        return train_samples, eval_samples, test_samples

    def gene_level_split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """基因级划分 - 确保同一基因的所有样本在同一集合"""
        logger.info("执行基因级划分...")

        # 按 (species, gene_name) 分组
        gene_groups = defaultdict(list)
        for sample in samples:
            gene_key = (sample.get('species', 'unknown'), sample.get('gene_name', 'unknown'))
            gene_groups[gene_key].append(sample)

        total_genes = len(gene_groups)
        logger.info(f"✓ 总基因数: {total_genes:,}")

        # 随机打乱基因列表
        all_genes = list(gene_groups.keys())
        random.shuffle(all_genes)

        # 划分基因
        train_end = int(TRAIN_RATIO * total_genes)
        eval_end = int((TRAIN_RATIO + EVAL_RATIO) * total_genes)

        train_genes = set(all_genes[:train_end])
        eval_genes = set(all_genes[train_end:eval_end])
        test_genes = set(all_genes[eval_end:])

        # 验证无重叠
        assert len(train_genes & eval_genes) == 0
        assert len(train_genes & test_genes) == 0
        assert len(eval_genes & test_genes) == 0

        # 提取样本
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

        logger.info(f"✓ 划分完成:")
        logger.info(f"  Train: {len(train_samples):,} 条 ({len(train_genes):,} 基因)")
        logger.info(f"  Eval: {len(eval_samples):,} 条 ({len(eval_genes):,} 基因)")
        logger.info(f"  Test: {len(test_samples):,} 条 ({len(test_genes):,} 基因)")

        return train_samples, eval_samples, test_samples

    def compute_class_weights(self, samples: List[Dict]) -> Dict[str, float]:
        """计算类别权重"""
        label_counts = Counter([s['label'] for s in samples])
        total = sum(label_counts.values())

        weights = {}
        for label, count in label_counts.items():
            weights[label] = total / (len(label_counts) * count)

        logger.info("类别权重:")
        for label, weight in weights.items():
            logger.info(f"  {label}: {weight:.4f}")

        return weights

    def save_data(self, samples: List[Dict], filename: str):
        """保存数据"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ 保存: {output_path} ({len(samples):,} 条, {size_mb:.1f} MB)")

    def save_metadata(self, train_samples, eval_samples, test_samples, extra_info: Dict = None):
        """保存元数据"""
        metadata = {
            'experiment': self.exp_name,
            'total_samples': len(train_samples) + len(eval_samples) + len(test_samples),
            'train_samples': len(train_samples),
            'eval_samples': len(eval_samples),
            'test_samples': len(test_samples),
            'train_distribution': dict(Counter([s['label'] for s in train_samples])),
            'eval_distribution': dict(Counter([s['label'] for s in eval_samples])),
            'test_distribution': dict(Counter([s['label'] for s in test_samples])),
            'original_label_distribution': {
                'train': dict(Counter([s['original_label'] for s in train_samples])),
                'eval': dict(Counter([s['original_label'] for s in eval_samples])),
                'test': dict(Counter([s['original_label'] for s in test_samples])),
            },
            'random_seed': RANDOM_SEED,
        }

        if extra_info:
            metadata.update(extra_info)

        output_path = self.output_dir / "metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 元数据: {output_path}")

    @abstractmethod
    def process_cds(self, samples: List[Dict]) -> List[Dict]:
        """处理CDS - 子类实现（拼接或不拼接）"""
        pass

    @abstractmethod
    def deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """去重 - 子类实现"""
        pass

    def split_data(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """数据划分 - 默认使用基因级划分，子类可覆盖"""
        return self.gene_level_split(samples)

    def prepare(self):
        """执行完整的数据准备流程"""
        logger.info("=" * 70)
        logger.info(f"开始数据准备: {self.exp_name}")
        logger.info("=" * 70)

        # 1. 加载原始数据
        data = self.load_raw_data()

        # 2. 处理CDS（拼接或不拼接）
        data = self.process_cds(data)

        # 3. 去重
        data = self.deduplicate(data)

        # 4. 转换标签
        data = self.convert_to_binary_labels(data)

        # 5. 数据划分（子类可覆盖选择划分方式）
        train_samples, eval_samples, test_samples = self.split_data(data)

        # 6. 计算类别权重
        class_weights = self.compute_class_weights(train_samples)

        # 7. 保存数据
        logger.info("\n保存数据...")
        self.save_data(train_samples, "train.json")
        self.save_data(eval_samples, "eval.json")
        self.save_data(test_samples, "test.json")

        # 8. 保存类别权重
        weights_path = self.output_dir / "class_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f, indent=2)
        logger.info(f"✓ 类别权重: {weights_path}")

        # 9. 保存元数据
        self.save_metadata(train_samples, eval_samples, test_samples)

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ {self.exp_name} 数据准备完成!")
        logger.info("=" * 70)

        return train_samples, eval_samples, test_samples


# ============================================================
# 去重工具函数
# ============================================================

def sequence_md5(sequence: str) -> str:
    """计算序列的MD5哈希"""
    return hashlib.md5(sequence.encode()).hexdigest()


def position_dedup(samples: List[Dict]) -> List[Dict]:
    """位置级去重 - 优先保留编码类别(CDS)标签

    当同一位置存在多个标签时，优先保留CDS标签。
    这是因为CDS注释通常有更强的实验证据支持，
    而同一位置的lncRNA_exon可能是同一基因的非编码转录本。
    """
    logger.info("执行位置级去重（优先保留CDS标签）...")
    original_count = len(samples)

    # 编码类别优先级（数字越小优先级越高）
    LABEL_PRIORITY = {
        'CDS': 0,           # 最高优先级
        'intron': 1,
        'tRNA': 2,
        'rRNA': 3,
        'snRNA': 4,
        'snoRNA': 5,
        'lncRNA_exon': 6,   # 最低优先级
    }

    # 第一遍：收集每个位置的所有样本
    position_samples = defaultdict(list)
    for sample in samples:
        key = (
            sample.get('species', ''),
            sample.get('seqname', ''),
            sample.get('start', 0),
            sample.get('end', 0),
            sample.get('strand', '+')
        )
        position_samples[key].append(sample)

    # 第二遍：每个位置选择优先级最高的样本
    deduplicated = []
    label_conflicts = 0
    cds_preserved = 0

    for key, samples_at_pos in position_samples.items():
        if len(samples_at_pos) == 1:
            deduplicated.append(samples_at_pos[0])
        else:
            # 多个样本，按标签优先级排序
            labels = set(s.get('label_name', '') for s in samples_at_pos)
            if len(labels) > 1:
                label_conflicts += 1
                if 'CDS' in labels:
                    cds_preserved += 1

            # 选择优先级最高的样本
            best_sample = min(
                samples_at_pos,
                key=lambda s: LABEL_PRIORITY.get(s.get('label_name', ''), 99)
            )
            deduplicated.append(best_sample)

    removed = original_count - len(deduplicated)
    logger.info(f"✓ 去重: {original_count:,} → {len(deduplicated):,} (移除 {removed:,}, {removed/original_count*100:.2f}%)")
    logger.info(f"  标签冲突位置: {label_conflicts:,}, 其中CDS被优先保留: {cds_preserved:,}")

    return deduplicated


def sequence_dedup_single_species(samples: List[Dict]) -> List[Dict]:
    """单物种序列级去重 - 每个物种内部去重，优先保留CDS标签"""
    logger.info("执行单物种序列级去重（优先保留CDS标签）...")
    original_count = len(samples)

    # 编码类别优先级（数字越小优先级越高）
    LABEL_PRIORITY = {
        'CDS': 0, 'intron': 1, 'tRNA': 2, 'rRNA': 3,
        'snRNA': 4, 'snoRNA': 5, 'lncRNA_exon': 6,
    }

    # 按物种分组
    species_groups = defaultdict(list)
    for sample in samples:
        species_groups[sample.get('species', 'unknown')].append(sample)

    deduplicated = []
    total_removed = 0
    total_conflicts = 0

    for species, species_samples in species_groups.items():
        # 收集每个序列的所有样本
        seq_samples = defaultdict(list)
        for sample in species_samples:
            seq_hash = sequence_md5(sample['sequence'])
            seq_samples[seq_hash].append(sample)

        species_kept = 0
        species_conflicts = 0

        for seq_hash, samples_list in seq_samples.items():
            if len(samples_list) == 1:
                deduplicated.append(samples_list[0])
            else:
                # 多个样本，检查标签冲突并选择优先级最高的
                labels = set(s.get('label_name', '') for s in samples_list)
                if len(labels) > 1:
                    species_conflicts += 1
                best_sample = min(
                    samples_list,
                    key=lambda s: LABEL_PRIORITY.get(s.get('label_name', ''), 99)
                )
                deduplicated.append(best_sample)
            species_kept += 1

        removed = len(species_samples) - species_kept
        total_removed += removed
        total_conflicts += species_conflicts
        logger.info(f"  {species}: {len(species_samples):,} → {species_kept:,} (移除 {removed:,}, 标签冲突 {species_conflicts:,})")

    logger.info(f"✓ 总计: {original_count:,} → {len(deduplicated):,} (移除 {total_removed:,}, {total_removed/original_count*100:.2f}%)")
    logger.info(f"  总标签冲突: {total_conflicts:,}")

    return deduplicated


def sequence_dedup_cross_species(samples: List[Dict]) -> List[Dict]:
    """跨物种序列级去重 - 所有物种一起去重，优先保留CDS标签"""
    logger.info("执行跨物种序列级去重（优先保留CDS标签）...")
    original_count = len(samples)

    # 编码类别优先级（数字越小优先级越高）
    LABEL_PRIORITY = {
        'CDS': 0, 'intron': 1, 'tRNA': 2, 'rRNA': 3,
        'snRNA': 4, 'snoRNA': 5, 'lncRNA_exon': 6,
    }

    # 收集每个序列的所有样本
    seq_samples = defaultdict(list)
    for sample in samples:
        seq_hash = sequence_md5(sample['sequence'])
        seq_samples[seq_hash].append(sample)

    deduplicated = []
    cross_species_dups = 0
    label_conflicts = 0

    for seq_hash, samples_list in seq_samples.items():
        if len(samples_list) == 1:
            deduplicated.append(samples_list[0])
        else:
            # 检查跨物种重复
            species_set = set(s.get('species') for s in samples_list)
            if len(species_set) > 1:
                cross_species_dups += 1

            # 检查标签冲突
            labels = set(s.get('label_name', '') for s in samples_list)
            if len(labels) > 1:
                label_conflicts += 1

            # 选择优先级最高的样本
            best_sample = min(
                samples_list,
                key=lambda s: LABEL_PRIORITY.get(s.get('label_name', ''), 99)
            )
            deduplicated.append(best_sample)

    removed = original_count - len(deduplicated)
    logger.info(f"✓ 去重: {original_count:,} → {len(deduplicated):,} (移除 {removed:,}, {removed/original_count*100:.2f}%)")
    logger.info(f"  跨物种重复: {cross_species_dups:,}, 标签冲突: {label_conflicts:,}")

    return deduplicated


# ============================================================
# CDS拼接工具函数
# ============================================================

def concatenate_cds_by_gene(samples: List[Dict]) -> List[Dict]:
    """将同一基因的CDS按位置顺序拼接"""
    logger.info("执行CDS拼接...")

    # 分离CDS和非CDS
    cds_samples = [s for s in samples if s.get('label_name') == 'CDS']
    non_cds_samples = [s for s in samples if s.get('label_name') != 'CDS']

    logger.info(f"  CDS样本: {len(cds_samples):,}")
    logger.info(f"  非CDS样本: {len(non_cds_samples):,}")

    # 按 (species, gene_name) 分组CDS
    gene_cds = defaultdict(list)
    for sample in cds_samples:
        gene_key = (sample.get('species', ''), sample.get('gene_name', ''))
        gene_cds[gene_key].append(sample)

    # 拼接每个基因的CDS
    concatenated_cds = []
    for gene_key, cds_list in gene_cds.items():
        species, gene_name = gene_key

        # 按位置排序（考虑正负链）
        if cds_list[0].get('strand', '+') == '+':
            cds_list.sort(key=lambda x: x.get('start', 0))
        else:
            cds_list.sort(key=lambda x: x.get('start', 0), reverse=True)

        # 拼接序列
        concat_seq = ''.join([cds['sequence'] for cds in cds_list])

        # 创建拼接后的样本
        first_cds = cds_list[0]
        last_cds = cds_list[-1]

        concatenated_sample = {
            'sequence': concat_seq,
            'label_name': 'CDS',
            'sequence_length': len(concat_seq),
            'gene_name': gene_name,
            'species': species,
            'seqname': first_cds.get('seqname', ''),
            'start': min(cds.get('start', 0) for cds in cds_list),
            'end': max(cds.get('end', 0) for cds in cds_list),
            'strand': first_cds.get('strand', '+'),
            'feature_type': 'CDS_concatenated',
            'num_exons': len(cds_list),
        }
        concatenated_cds.append(concatenated_sample)

    logger.info(f"✓ CDS拼接: {len(cds_samples):,} exons → {len(concatenated_cds):,} genes")

    # 合并结果
    result = concatenated_cds + non_cds_samples
    logger.info(f"✓ 总样本数: {len(result):,}")

    return result
