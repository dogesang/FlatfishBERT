#!/usr/bin/env python3
"""
Task S 数据准备脚本 - 反向互补增强版本

任务描述:
- 基于 Task S baseline，对少数类进行反向互补数据增强
- 增强类别: ncRNA_gene (仅占2.1%，F1仅49.74%)
- 增强范围: 训练集 + 验证集 (测试集保持原样)

增强原理:
- DNA双链互补配对 (A-T, G-C)
- 反向互补序列在生物学上等价
- 可有效扩充少数类样本

数据来源:
- 复用 Task S baseline 数据
- 对 ncRNA_gene 样本生成反向互补序列

输出:
- Task_S_aug 目录下的增强数据
"""

import sys
import json
import random
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from copy import deepcopy

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

# 需要增强的少数类
AUGMENT_LABELS = ['ncRNA_gene']

# 标签到ID的映射
LABEL2ID = {'CDS': 0, 'intron': 1, 'ncRNA_gene': 2}
ID2LABEL = {0: 'CDS', 1: 'intron', 2: 'ncRNA_gene'}

# 碱基互补配对表
COMPLEMENT_MAP = {
    'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
    'a': 't', 't': 'a', 'g': 'c', 'c': 'g',
    'N': 'N', 'n': 'n',  # N保持不变
    'R': 'Y', 'Y': 'R',  # R(A/G) <-> Y(C/T)
    'S': 'S', 'W': 'W',  # S(G/C) <-> S, W(A/T) <-> W
    'K': 'M', 'M': 'K',  # K(G/T) <-> M(A/C)
    'B': 'V', 'V': 'B',  # B(C/G/T) <-> V(A/C/G)
    'D': 'H', 'H': 'D',  # D(A/G/T) <-> H(A/C/T)
}


def reverse_complement(sequence: str) -> str:
    """计算DNA序列的反向互补序列

    Args:
        sequence: 原始DNA序列

    Returns:
        反向互补序列
    """
    # 互补
    complement = ''.join(COMPLEMENT_MAP.get(base, base) for base in sequence)
    # 反向
    return complement[::-1]


def load_task_s_baseline(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载Task S baseline数据

    Args:
        data_dir: Task S数据目录

    Returns:
        train_data, eval_data, test_data
    """
    logger.info(f"加载Task S baseline数据: {data_dir}")

    train_path = data_dir / "train.json"
    eval_path = data_dir / "eval.json"
    test_path = data_dir / "test.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Task S训练数据不存在: {train_path}")

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


def augment_with_reverse_complement(
    data: List[Dict],
    augment_labels: List[str]
) -> Tuple[List[Dict], Dict[str, int]]:
    """对指定类别进行反向互补增强

    Args:
        data: 原始数据
        augment_labels: 需要增强的类别列表

    Returns:
        增强后的数据, 增强统计信息
    """
    augmented_data = []
    augment_stats = {label: 0 for label in augment_labels}

    for sample in data:
        # 原始样本保留
        augmented_data.append(sample)

        # 检查是否需要增强
        label_name = sample.get('label_name', '')
        if label_name in augment_labels:
            # 创建反向互补样本
            aug_sample = deepcopy(sample)
            aug_sample['sequence'] = reverse_complement(sample['sequence'])
            aug_sample['is_augmented'] = True
            aug_sample['augment_type'] = 'reverse_complement'
            aug_sample['original_strand'] = sample.get('strand', '+')
            # 反向互补后链方向翻转
            aug_sample['strand'] = '-' if sample.get('strand', '+') == '+' else '+'

            augmented_data.append(aug_sample)
            augment_stats[label_name] += 1

    return augmented_data, augment_stats


def print_augmentation_summary(
    original_train: List[Dict],
    augmented_train: List[Dict],
    original_eval: List[Dict],
    augmented_eval: List[Dict],
    test_data: List[Dict]
):
    """打印增强前后的数据统计对比"""

    logger.info("\n" + "=" * 70)
    logger.info("数据增强统计对比")
    logger.info("=" * 70)

    # 训练集对比
    logger.info("\n【训练集】")
    logger.info(f"{'类别':<15} {'原始':>12} {'增强后':>12} {'增加':>12}")
    logger.info("-" * 55)

    orig_train_counts = Counter([d['label_name'] for d in original_train])
    aug_train_counts = Counter([d['label_name'] for d in augmented_train])

    for label in TASK_S_LABELS:
        orig = orig_train_counts.get(label, 0)
        aug = aug_train_counts.get(label, 0)
        diff = aug - orig
        logger.info(f"{label:<15} {orig:>12,} {aug:>12,} {'+' + str(diff) if diff > 0 else str(diff):>12}")

    logger.info("-" * 55)
    logger.info(f"{'总计':<15} {len(original_train):>12,} {len(augmented_train):>12,} {'+' + str(len(augmented_train) - len(original_train)):>12}")

    # 验证集对比
    logger.info("\n【验证集】")
    logger.info(f"{'类别':<15} {'原始':>12} {'增强后':>12} {'增加':>12}")
    logger.info("-" * 55)

    orig_eval_counts = Counter([d['label_name'] for d in original_eval])
    aug_eval_counts = Counter([d['label_name'] for d in augmented_eval])

    for label in TASK_S_LABELS:
        orig = orig_eval_counts.get(label, 0)
        aug = aug_eval_counts.get(label, 0)
        diff = aug - orig
        logger.info(f"{label:<15} {orig:>12,} {aug:>12,} {'+' + str(diff) if diff > 0 else str(diff):>12}")

    logger.info("-" * 55)
    logger.info(f"{'总计':<15} {len(original_eval):>12,} {len(augmented_eval):>12,} {'+' + str(len(augmented_eval) - len(original_eval)):>12}")

    # 测试集 (不变)
    logger.info("\n【测试集】(不增强)")
    test_counts = Counter([d['label_name'] for d in test_data])
    for label in TASK_S_LABELS:
        count = test_counts.get(label, 0)
        logger.info(f"  {label}: {count:,}")
    logger.info(f"  总计: {len(test_data):,}")

    # 类别占比变化
    logger.info("\n【训练集类别占比变化】")
    logger.info(f"{'类别':<15} {'原始占比':>12} {'增强后占比':>12}")
    logger.info("-" * 45)

    for label in TASK_S_LABELS:
        orig_pct = orig_train_counts.get(label, 0) / len(original_train) * 100
        aug_pct = aug_train_counts.get(label, 0) / len(augmented_train) * 100
        logger.info(f"{label:<15} {orig_pct:>11.1f}% {aug_pct:>11.1f}%")


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


def prepare_task_s_augmented():
    """准备Task S增强数据"""

    logger.info("=" * 70)
    logger.info("Task S 反向互补数据增强")
    logger.info("=" * 70)
    logger.info("\n增强策略:")
    logger.info(f"  - 增强类别: {AUGMENT_LABELS}")
    logger.info(f"  - 不增强类别: CDS, intron (多数类)")
    logger.info(f"  - 训练集: 增强")
    logger.info(f"  - 验证集: 增强")
    logger.info(f"  - 测试集: 不增强 (保持原样用于公平评估)")

    # 加载baseline数据
    baseline_dir = DATA_OUTPUT_ROOT / "Task_S"
    train_data, eval_data, test_data = load_task_s_baseline(baseline_dir)

    # 对训练集进行增强
    logger.info("\n对训练集进行反向互补增强...")
    augmented_train, train_stats = augment_with_reverse_complement(train_data, AUGMENT_LABELS)
    logger.info(f"✓ 训练集增强完成:")
    for label, count in train_stats.items():
        logger.info(f"  {label}: +{count:,} 样本")

    # 对验证集进行增强
    logger.info("\n对验证集进行反向互补增强...")
    augmented_eval, eval_stats = augment_with_reverse_complement(eval_data, AUGMENT_LABELS)
    logger.info(f"✓ 验证集增强完成:")
    for label, count in eval_stats.items():
        logger.info(f"  {label}: +{count:,} 样本")

    # 测试集保持不变
    logger.info("\n测试集保持原样 (不增强)")

    # 打乱增强后的数据
    random.shuffle(augmented_train)
    random.shuffle(augmented_eval)

    # 打印增强统计
    print_augmentation_summary(
        train_data, augmented_train,
        eval_data, augmented_eval,
        test_data
    )

    # 计算类别权重（用于Class Weight Loss）
    logger.info("\n计算类别权重...")
    all_labels = [d['label'] for d in augmented_train]
    class_weights = compute_class_weights(all_labels)

    # 转换为标签名称的权重
    class_weights_named = {ID2LABEL[int(k)]: v for k, v in class_weights.items()}
    logger.info(f"Class Weight Loss 权重: {class_weights_named}")

    # 计算WeightedRandomSampler权重
    sampler_weights = compute_sampler_weights(augmented_train)

    # 保存数据
    logger.info("\n保存增强数据...")
    output_dir = DATA_OUTPUT_ROOT / "Task_S_aug"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(augmented_train, output_dir / "train.json")
    save_json(augmented_eval, output_dir / "eval.json")
    save_json(test_data, output_dir / "test.json")  # 测试集不变
    save_json(class_weights, output_dir / "class_weights.json")
    save_json(sampler_weights, output_dir / "sampler_weights.json")

    # 统计信息
    aug_train_counts = Counter([d['label_name'] for d in augmented_train])
    aug_eval_counts = Counter([d['label_name'] for d in augmented_eval])
    test_counts = Counter([d['label_name'] for d in test_data])

    stats = {
        'total_samples': len(augmented_train) + len(augmented_eval) + len(test_data),
        'train_samples': len(augmented_train),
        'eval_samples': len(augmented_eval),
        'test_samples': len(test_data),
        'train_label_distribution': dict(aug_train_counts),
        'eval_label_distribution': dict(aug_eval_counts),
        'test_label_distribution': dict(test_counts),
        'augmentation_stats': {
            'train': train_stats,
            'eval': eval_stats
        }
    }

    # 保存元数据
    metadata = {
        'task': 'Task_S_aug',
        'version': '1.0',
        'description': '基因内部结构分类 - 反向互补数据增强版本',
        'base_task': 'Task_S',
        'labels': TASK_S_LABELS,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'num_labels': 3,
        'augmentation': {
            'method': 'reverse_complement',
            'augmented_labels': AUGMENT_LABELS,
            'non_augmented_labels': ['CDS', 'intron'],
            'train_augmented': True,
            'eval_augmented': True,
            'test_augmented': False,
            'reason': '对少数类(ncRNA_gene)进行反向互补增强，提升类别平衡性'
        },
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
    logger.info("✅ Task S 反向互补增强数据准备完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return augmented_train, augmented_eval, test_data


def main():
    parser = argparse.ArgumentParser(
        description='Task S 数据准备 - 反向互补增强版本'
    )
    args = parser.parse_args()

    prepare_task_s_augmented()


if __name__ == "__main__":
    main()
