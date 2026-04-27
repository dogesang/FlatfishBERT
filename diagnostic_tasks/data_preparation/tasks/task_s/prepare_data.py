#!/usr/bin/env python3
"""
Task S 数据准备脚本 - 基因内部结构分类 (三分类)

任务描述:
- 从 Task G v3 的正例数据中筛选样本
- 标签: CDS / intron / ncRNA_gene (三分类)
- 使用原始长度（不使用512bp窗口）

数据来源:
- 复用 Task G v3 的正例数据（gene_body部分）
- 按 original_label 重新分组为三类

划分策略:
- 复用 Task G v3 的 train/eval/test 划分（避免数据泄露）

不平衡处理:
- WeightedRandomSampler + Class Weight Loss
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


# Task S 三分类标签定义
TASK_S_LABELS = ['CDS', 'intron', 'ncRNA_gene']

# 标签到ID的映射
LABEL2ID = {'CDS': 0, 'intron': 1, 'ncRNA_gene': 2}
ID2LABEL = {0: 'CDS', 1: 'intron', 2: 'ncRNA_gene'}

# 原始细粒度标签到Task S标签的映射
ORIGINAL_TO_TASK_S = {
    'CDS': 'CDS',
    'intron': 'intron',
    'tRNA': 'ncRNA_gene',
    'rRNA': 'ncRNA_gene',
    'snRNA': 'ncRNA_gene',
    'snoRNA': 'ncRNA_gene',
    'lncRNA_exon': 'ncRNA_gene'
}

# ncRNA子类型（用于统计）
NCRNA_SUBTYPES = ['tRNA', 'rRNA', 'snRNA', 'snoRNA', 'lncRNA_exon']


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


def filter_gene_body_samples(data: List[Dict]) -> List[Dict]:
    """筛选gene_body样本（排除background）

    Args:
        data: 原始数据

    Returns:
        筛选后的gene_body样本
    """
    gene_body_samples = []

    for sample in data:
        # 只保留gene_body样本（排除background）
        if sample.get('label_name') == 'gene_body' or sample.get('original_label') in ORIGINAL_TO_TASK_S:
            original_label = sample.get('original_label', '')
            if original_label in ORIGINAL_TO_TASK_S:
                gene_body_samples.append(sample)

    return gene_body_samples


def convert_to_task_s_format(samples: List[Dict]) -> List[Dict]:
    """转换为Task S格式

    将原始细粒度标签转换为Task S三分类标签

    Args:
        samples: 原始样本

    Returns:
        转换后的样本
    """
    task_s_samples = []

    for sample in samples:
        original_label = sample.get('original_label', '')

        if original_label not in ORIGINAL_TO_TASK_S:
            continue

        task_s_label = ORIGINAL_TO_TASK_S[original_label]
        label_id = LABEL2ID[task_s_label]

        task_s_sample = {
            'sequence': sample['sequence'],
            'label': label_id,
            'label_name': task_s_label,
            'original_label': original_label,  # 保留原始细粒度标签
            'seqname': sample['seqname'],
            'start': sample['start'],
            'end': sample['end'],
            'strand': sample.get('strand', '+'),
            'species': sample.get('species', ''),
            'gene_id': sample.get('gene_id', ''),
            'gene_name': sample.get('gene_name', ''),
            'sequence_length': sample.get('sequence_length', len(sample['sequence']))
        }
        task_s_samples.append(task_s_sample)

    return task_s_samples


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

    # 三分类标签分布
    logger.info(f"\n三分类标签分布:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if not data:
            continue
        label_counts = Counter([d['label_name'] for d in data])
        logger.info(f"  {name}:")
        for label in TASK_S_LABELS:
            count = label_counts.get(label, 0)
            pct = count / len(data) * 100 if data else 0
            logger.info(f"    {label}: {count:,} ({pct:.1f}%)")

    # ncRNA子类型分布
    logger.info(f"\nncRNA子类型分布 (original_label):")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if not data:
            continue
        original_counts = Counter([d.get('original_label', '') for d in data])
        ncrna_samples = [d for d in data if d['label_name'] == 'ncRNA_gene']
        if ncrna_samples:
            logger.info(f"  {name} (ncRNA_gene内部):")
            for subtype in NCRNA_SUBTYPES:
                count = original_counts.get(subtype, 0)
                pct = count / len(ncrna_samples) * 100 if ncrna_samples else 0
                if count > 0:
                    logger.info(f"    {subtype}: {count:,} ({pct:.1f}%)")

    # 长度统计
    logger.info(f"\n长度统计:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if data:
            lengths = [d['sequence_length'] for d in data]
            logger.info(f"  {name}: min={min(lengths)}, max={max(lengths)}, "
                       f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # 按类别的长度统计
    logger.info(f"\n按类别的长度统计 (Train):")
    for label in TASK_S_LABELS:
        label_samples = [d for d in train_data if d['label_name'] == label]
        if label_samples:
            lengths = [d['sequence_length'] for d in label_samples]
            logger.info(f"  {label}: min={min(lengths)}, max={max(lengths)}, "
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
        'original_label_distribution': dict(Counter([d.get('original_label', '') for d in all_data])),
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


def prepare_task_s_data(
    source_task: str = "Task_G_v3"
):
    """准备Task S数据

    Args:
        source_task: 源任务名称（默认Task_G_v3）
    """
    logger.info("=" * 70)
    logger.info("Task S: 基因内部结构分类 (三分类) 数据准备")
    logger.info("=" * 70)
    logger.info("设计特点:")
    logger.info("  1. 从Task G v3正例数据中筛选")
    logger.info("  2. 三分类: CDS / intron / ncRNA_gene")
    logger.info("  3. 使用原始长度（不使用512bp窗口）")
    logger.info("  4. 复用Task G v3的train/eval/test划分")
    logger.info("  5. 使用WeightedRandomSampler + Class Weight Loss处理不平衡")
    logger.info(f"\n三分类标签: {TASK_S_LABELS}")
    logger.info(f"ncRNA子类型: {NCRNA_SUBTYPES}")

    # 加载Task G v3数据
    source_dir = DATA_OUTPUT_ROOT / source_task
    train_data_raw, eval_data_raw, test_data_raw = load_task_g_v3_data(source_dir)

    # 筛选gene_body样本
    logger.info("\n筛选gene_body样本...")
    train_gene_body = filter_gene_body_samples(train_data_raw)
    eval_gene_body = filter_gene_body_samples(eval_data_raw)
    test_gene_body = filter_gene_body_samples(test_data_raw)

    logger.info(f"✓ 筛选完成:")
    logger.info(f"  Train gene_body: {len(train_gene_body):,}")
    logger.info(f"  Eval gene_body: {len(eval_gene_body):,}")
    logger.info(f"  Test gene_body: {len(test_gene_body):,}")

    # 转换为Task S格式
    logger.info("\n转换为Task S格式...")
    train_data = convert_to_task_s_format(train_gene_body)
    eval_data = convert_to_task_s_format(eval_gene_body)
    test_data = convert_to_task_s_format(test_gene_body)

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
    stats = print_data_statistics(train_data, eval_data, test_data, "Task S")

    # 保存数据
    logger.info("\n保存数据...")
    output_dir = DATA_OUTPUT_ROOT / "Task_S"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(train_data, output_dir / "train.json")
    save_json(eval_data, output_dir / "eval.json")
    save_json(test_data, output_dir / "test.json")
    save_json(class_weights, output_dir / "class_weights.json")
    save_json(sampler_weights, output_dir / "sampler_weights.json")

    # 保存元数据
    metadata = {
        'task': 'Task_S',
        'version': '2.0',
        'description': '基因内部结构分类 (三分类: CDS/intron/ncRNA_gene)',
        'labels': TASK_S_LABELS,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'num_labels': 3,
        'ncrna_subtypes': NCRNA_SUBTYPES,
        'original_to_task_s_mapping': ORIGINAL_TO_TASK_S,
        'source_task': source_task,
        'design_changes': [
            '从Task G v3正例数据中筛选',
            '三分类: CDS / intron / ncRNA_gene',
            '使用原始长度（不使用512bp窗口）',
            '复用Task G v3的train/eval/test划分',
            'WeightedRandomSampler + Class Weight Loss处理不平衡'
        ],
        'imbalance_handling': {
            'weighted_random_sampler': True,
            'class_weight_loss': True
        },
        'statistics': stats,
        'class_weights': class_weights,
        'class_weights_named': class_weights_named
    }
    save_json(metadata, output_dir / "metadata.json")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Task S 数据准备完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return train_data, eval_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Task S 数据准备 - 基因内部结构分类 (三分类)')
    parser.add_argument('--source-task', type=str, default='Task_G_v3',
                       help='源任务名称（默认Task_G_v3）')

    args = parser.parse_args()

    prepare_task_s_data(
        source_task=args.source_task
    )


if __name__ == "__main__":
    main()
