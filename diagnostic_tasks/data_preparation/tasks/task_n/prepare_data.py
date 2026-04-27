#!/usr/bin/env python3
"""
Task N 数据准备脚本 - ncRNA亚型分类 (五分类)

任务描述:
- 从 Task G v3 的正例数据中筛选ncRNA样本
- 标签: lncRNA_exon / tRNA / rRNA / snRNA / snoRNA (五分类)
- 使用原始长度（不使用512bp窗口）

数据来源:
- 复用 Task G v3 的正例数据（筛选ncRNA类型）
- 按 original_label 直接使用

划分策略:
- 复用 Task G v3 的 train/eval/test 划分（避免数据泄露）

不平衡处理:
- WeightedRandomSampler + Class Weight Loss
- 类别极度不平衡：lncRNA_exon占91.8%，snRNA仅占0.9%
"""

import sys
import json
import random
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATA_OUTPUT_ROOT, RANDOM_SEED
)
from src.utils import save_json, compute_class_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Task N 五分类标签定义
TASK_N_LABELS = ['lncRNA_exon', 'tRNA', 'rRNA', 'snRNA', 'snoRNA']

# 标签到ID的映射
LABEL2ID = {'lncRNA_exon': 0, 'tRNA': 1, 'rRNA': 2, 'snRNA': 3, 'snoRNA': 4}
ID2LABEL = {0: 'lncRNA_exon', 1: 'tRNA', 2: 'rRNA', 3: 'snRNA', 4: 'snoRNA'}


def load_task_g_v3_data(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载Task G v3数据

    Args:
        data_dir: Task G v3数据目录

    Returns:
        train_data, eval_data, test_data
    """
    logger.info(f"加载Task G v3数据: {data_dir}")

    train_path = data_dir / "train.json"
    eval_path = data_dir / "eval.json"
    test_path = data_dir / "test.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Task G v3训练数据不存在: {train_path}")

    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    logger.info(f"✓ 加载完成:")
    logger.info(f"  Train: {len(train_data):,}")
    logger.info(f"  Eval: {len(eval_data):,}")
    logger.info(f"  Test: {len(test_data):,}")

    return train_data, eval_data, test_data


def filter_ncrna_samples(data: List[Dict]) -> List[Dict]:
    """筛选ncRNA样本

    Args:
        data: 原始数据

    Returns:
        筛选后的ncRNA样本
    """
    ncrna_samples = []

    for sample in data:
        original_label = sample.get('original_label', '')
        if original_label in TASK_N_LABELS:
            ncrna_samples.append(sample)

    return ncrna_samples


def convert_to_task_n_format(samples: List[Dict]) -> List[Dict]:
    """转换为Task N格式

    Args:
        samples: 原始样本

    Returns:
        转换后的样本
    """
    task_n_samples = []

    for sample in samples:
        original_label = sample.get('original_label', '')

        if original_label not in TASK_N_LABELS:
            continue

        label_id = LABEL2ID[original_label]

        task_n_sample = {
            'sequence': sample['sequence'],
            'label': label_id,
            'label_name': original_label,
            'seqname': sample['seqname'],
            'start': sample['start'],
            'end': sample['end'],
            'strand': sample.get('strand', '+'),
            'species': sample.get('species', ''),
            'gene_id': sample.get('gene_id', ''),
            'gene_name': sample.get('gene_name', ''),
            'sequence_length': sample.get('sequence_length', len(sample['sequence']))
        }
        task_n_samples.append(task_n_sample)

    return task_n_samples


def print_data_statistics(train_data, eval_data, test_data, task_name: str) -> Dict:
    """打印数据统计信息"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{task_name} 数据统计")
    logger.info(f"{'='*70}")

    all_data = train_data + eval_data + test_data

    if not all_data:
        logger.warning("没有数据!")
        return {}

    # 总体统计
    logger.info(f"\n总样本数: {len(all_data):,}")
    logger.info(f"  Train: {len(train_data):,} ({len(train_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Eval: {len(eval_data):,} ({len(eval_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data):,} ({len(test_data)/len(all_data)*100:.1f}%)")

    # 五分类标签分布
    logger.info(f"\n五分类标签分布:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if not data:
            continue
        label_counts = Counter([d['label_name'] for d in data])
        logger.info(f"  {name}:")
        for label in TASK_N_LABELS:
            count = label_counts.get(label, 0)
            pct = count / len(data) * 100 if data else 0
            logger.info(f"    {label}: {count:,} ({pct:.1f}%)")

    # 长度统计
    logger.info(f"\n长度统计:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if data:
            lengths = [d['sequence_length'] for d in data]
            logger.info(f"  {name}: min={min(lengths)}, max={max(lengths)}, "
                       f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # 按类别的长度统计
    logger.info(f"\n按类别的长度统计 (Train):")
    for label in TASK_N_LABELS:
        label_samples = [d for d in train_data if d['label_name'] == label]
        if label_samples:
            lengths = [d['sequence_length'] for d in label_samples]
            logger.info(f"  {label}: n={len(label_samples)}, min={min(lengths)}, max={max(lengths)}, "
                       f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # 物种分布
    logger.info(f"\n物种分布:")
    species_counts = Counter([d['species'] for d in all_data])
    for species, count in sorted(species_counts.items()):
        logger.info(f"  {species}: {count:,} ({count/len(all_data)*100:.1f}%)")

    stats = {
        'total_samples': len(all_data),
        'train_samples': len(train_data),
        'eval_samples': len(eval_data),
        'test_samples': len(test_data),
        'label_distribution': dict(Counter([d['label_name'] for d in all_data])),
        'species_distribution': dict(species_counts)
    }

    return stats


def compute_sampler_weights(train_data: List[Dict]) -> List[float]:
    """计算WeightedRandomSampler的权重

    Args:
        train_data: 训练数据

    Returns:
        每个样本的权重列表
    """
    # 统计各类别数量
    label_counts = Counter([d['label'] for d in train_data])
    total_samples = len(train_data)
    num_classes = len(label_counts)

    # 计算每个类别的权重（反比于类别频率）
    class_weights = {}
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (num_classes * count)

    # 为每个样本分配权重
    sample_weights = [class_weights[d['label']] for d in train_data]

    logger.info(f"\nWeightedRandomSampler 类别权重:")
    for label in sorted(class_weights.keys()):
        label_name = ID2LABEL[label]
        logger.info(f"  {label_name}: {class_weights[label]:.4f}")

    return sample_weights


def prepare_task_n_data(
    source_task: str = "Task_G_v3"
):
    """准备Task N数据

    Args:
        source_task: 源任务名称（默认Task_G_v3）
    """
    logger.info("=" * 70)
    logger.info("Task N: ncRNA亚型分类 (五分类) 数据准备")
    logger.info("=" * 70)
    logger.info("设计特点:")
    logger.info("  1. 从Task G v3正例数据中筛选ncRNA样本")
    logger.info("  2. 五分类: lncRNA_exon / tRNA / rRNA / snRNA / snoRNA")
    logger.info("  3. 使用原始长度（不使用512bp窗口）")
    logger.info("  4. 复用Task G v3的train/eval/test划分")
    logger.info("  5. 使用WeightedRandomSampler + Class Weight Loss处理极度不平衡")
    logger.info(f"\n五分类标签: {TASK_N_LABELS}")
    logger.info("\n⚠️ 注意: 类别极度不平衡，lncRNA_exon占~92%，snRNA仅占~1%")

    # 加载Task G v3数据
    source_dir = DATA_OUTPUT_ROOT / source_task
    train_data_raw, eval_data_raw, test_data_raw = load_task_g_v3_data(source_dir)

    # 筛选ncRNA样本
    logger.info("\n筛选ncRNA样本...")
    train_ncrna = filter_ncrna_samples(train_data_raw)
    eval_ncrna = filter_ncrna_samples(eval_data_raw)
    test_ncrna = filter_ncrna_samples(test_data_raw)

    logger.info(f"✓ 筛选完成:")
    logger.info(f"  Train ncRNA: {len(train_ncrna):,}")
    logger.info(f"  Eval ncRNA: {len(eval_ncrna):,}")
    logger.info(f"  Test ncRNA: {len(test_ncrna):,}")

    # 转换为Task N格式
    logger.info("\n转换为Task N格式...")
    train_data = convert_to_task_n_format(train_ncrna)
    eval_data = convert_to_task_n_format(eval_ncrna)
    test_data = convert_to_task_n_format(test_ncrna)

    logger.info(f"✓ 转换完成:")
    logger.info(f"  Train: {len(train_data):,}")
    logger.info(f"  Eval: {len(eval_data):,}")
    logger.info(f"  Test: {len(test_data):,}")

    # 打乱数据
    random.shuffle(train_data)
    random.shuffle(eval_data)
    random.shuffle(test_data)

    # 计算类别权重（用于Class Weight Loss）
    logger.info("\n计算类别权重...")
    all_labels = [d['label'] for d in train_data]
    class_weights = compute_class_weights(all_labels)

    # 转换为标签名称的权重
    class_weights_named = {ID2LABEL[int(k)]: v for k, v in class_weights.items()}
    logger.info(f"Class Weight Loss 权重: {class_weights_named}")

    # 计算WeightedRandomSampler权重
    sampler_weights = compute_sampler_weights(train_data)

    # 打印统计信息
    stats = print_data_statistics(train_data, eval_data, test_data, "Task N")

    # 保存数据
    logger.info("\n保存数据...")
    output_dir = DATA_OUTPUT_ROOT / "Task_N"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(train_data, output_dir / "train.json")
    save_json(eval_data, output_dir / "eval.json")
    save_json(test_data, output_dir / "test.json")
    save_json(class_weights, output_dir / "class_weights.json")
    save_json(sampler_weights, output_dir / "sampler_weights.json")

    # 保存元数据
    metadata = {
        'task': 'Task_N',
        'version': '2.0',
        'description': 'ncRNA亚型分类 (五分类: lncRNA_exon/tRNA/rRNA/snRNA/snoRNA)',
        'labels': TASK_N_LABELS,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'num_labels': 5,
        'source_task': source_task,
        'design_changes': [
            '从Task G v3正例数据中筛选ncRNA样本',
            '五分类: lncRNA_exon / tRNA / rRNA / snRNA / snoRNA',
            '使用原始长度（不使用512bp窗口）',
            '复用Task G v3的train/eval/test划分',
            'WeightedRandomSampler + Class Weight Loss处理极度不平衡'
        ],
        'imbalance_handling': {
            'weighted_random_sampler': True,
            'class_weight_loss': True
        },
        'imbalance_warning': '类别极度不平衡：lncRNA_exon占~92%，snRNA仅占~1%',
        'statistics': stats,
        'class_weights': class_weights,
        'class_weights_named': class_weights_named
    }
    save_json(metadata, output_dir / "metadata.json")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Task N 数据准备完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return train_data, eval_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Task N 数据准备 - ncRNA亚型分类 (五分类)')
    parser.add_argument('--source-task', type=str, default='Task_G_v3',
                       help='源任务名称（默认Task_G_v3）')

    args = parser.parse_args()

    prepare_task_n_data(
        source_task=args.source_task
    )


if __name__ == "__main__":
    main()
