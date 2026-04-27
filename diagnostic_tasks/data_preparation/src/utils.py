"""
工具函数模块

提供通用的文件读写、数据处理等工具函数
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """保存JSON文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"✓ 保存: {file_path}")


def load_gff(gff_path: Path) -> List[Dict]:
    """加载GFF文件为字典列表

    Args:
        gff_path: GFF文件路径

    Returns:
        特征列表，每个特征为一个字典
    """
    logger.info(f"加载GFF: {gff_path}")
    features = []

    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            # 解析属性
            attributes = {}
            for item in parts[8].split(';'):
                item = item.strip()
                if '=' in item:
                    key, value = item.split('=', 1)
                    attributes[key] = value

            feature = {
                'seqname': parts[0],
                'source': parts[1],
                'feature_type': parts[2],
                'start': int(parts[3]),
                'end': int(parts[4]),
                'score': parts[5],
                'strand': parts[6],
                'frame': parts[7],
                'attributes': attributes
            }
            features.append(feature)

    logger.info(f"✓ 加载 {len(features):,} 个特征")
    return features


def load_fasta(fasta_path: Path) -> Dict[str, str]:
    """加载FASTA文件为字典

    Args:
        fasta_path: FASTA文件路径

    Returns:
        序列字典 {seqname: sequence}
    """
    logger.info(f"加载FASTA: {fasta_path}")
    sequences = {}
    current_name = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq).upper()
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_name:
            sequences[current_name] = ''.join(current_seq).upper()

    total_length = sum(len(seq) for seq in sequences.values())
    logger.info(f"✓ 加载 {len(sequences)} 个序列, 总长度 {total_length:,} bp")
    return sequences


def compute_class_weights(labels: List[str]) -> Dict[str, float]:
    """计算类别权重（用于处理类别不平衡）

    Args:
        labels: 标签列表

    Returns:
        类别权重字典
    """
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    weights = {}
    for label, count in label_counts.items():
        weights[label] = total / (num_classes * count)

    logger.info("类别权重:")
    for label, weight in sorted(weights.items()):
        count = label_counts[label]
        pct = count / total * 100
        logger.info(f"  {label}: {count:,} ({pct:.2f}%) -> weight={weight:.4f}")

    return weights


def samples_to_training_format(
    samples: List[Dict],
    label2id: Dict[str, int]
) -> List[Dict]:
    """将样本转换为训练格式

    Args:
        samples: 样本列表
        label2id: 标签到ID的映射

    Returns:
        训练格式的样本列表
    """
    training_data = []

    for sample in samples:
        if isinstance(sample, dict):
            label = sample.get('label', '')
            sequence = sample.get('sequence', '')
        else:
            # WindowSample对象
            label = sample.label
            sequence = sample.sequence

        if label not in label2id:
            continue

        training_sample = {
            'sequence': sequence,
            'label': label,
            'label_id': label2id[label],
        }

        # 保留其他元数据
        if isinstance(sample, dict):
            for key in ['seqname', 'start', 'end', 'strand', 'gene_id', 'gene_name',
                       'species', 'gc_content', 'region_type', 'block_id']:
                if key in sample:
                    training_sample[key] = sample[key]
        else:
            training_sample.update({
                'seqname': sample.seqname,
                'start': sample.start,
                'end': sample.end,
                'strand': sample.strand,
                'gene_id': sample.gene_id,
                'gene_name': sample.gene_name,
                'species': sample.species,
                'gc_content': sample.gc_content,
                'region_type': sample.region_type,
                'block_id': sample.block_id,
            })

        training_data.append(training_sample)

    return training_data


def print_data_statistics(
    train_data: List[Dict],
    eval_data: List[Dict],
    test_data: List[Dict],
    task_name: str = "Task"
) -> Dict:
    """打印数据集统计信息

    Args:
        train_data: 训练数据
        eval_data: 验证数据
        test_data: 测试数据
        task_name: 任务名称

    Returns:
        统计信息字典
    """
    print(f"\n{'='*60}")
    print(f"{task_name} 数据统计")
    print(f"{'='*60}")

    stats = {
        'train_count': len(train_data),
        'eval_count': len(eval_data),
        'test_count': len(test_data),
        'total_count': len(train_data) + len(eval_data) + len(test_data),
    }

    print(f"\n样本数量:")
    print(f"  Train: {stats['train_count']:,}")
    print(f"  Eval:  {stats['eval_count']:,}")
    print(f"  Test:  {stats['test_count']:,}")
    print(f"  Total: {stats['total_count']:,}")

    # 标签分布
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        labels = [d['label'] for d in data]
        label_counts = Counter(labels)
        total = len(labels)

        print(f"\n{name} 标签分布:")
        stats[f'{name.lower()}_labels'] = {}
        for label, count in sorted(label_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {label}: {count:,} ({pct:.2f}%)")
            stats[f'{name.lower()}_labels'][label] = count

    # 物种分布
    all_data = train_data + eval_data + test_data
    species_counts = Counter(d.get('species', 'unknown') for d in all_data)
    if len(species_counts) > 1:
        print(f"\n物种分布:")
        for species, count in sorted(species_counts.items()):
            pct = count / len(all_data) * 100
            print(f"  {species}: {count:,} ({pct:.2f}%)")
        stats['species_distribution'] = dict(species_counts)

    print(f"\n{'='*60}")

    return stats


def validate_samples(samples: List[Dict], window_size: int = 512) -> Tuple[List[Dict], Dict]:
    """验证样本有效性

    Args:
        samples: 样本列表
        window_size: 期望的窗口大小

    Returns:
        (有效样本列表, 验证统计)
    """
    valid_samples = []
    invalid_reasons = Counter()

    for sample in samples:
        seq = sample.get('sequence', '')

        # 检查序列长度
        if len(seq) != window_size:
            invalid_reasons['wrong_length'] += 1
            continue

        # 检查序列内容
        valid_bases = set('ATCGN')
        if not all(base in valid_bases for base in seq.upper()):
            invalid_reasons['invalid_bases'] += 1
            continue

        # 检查N含量
        n_ratio = seq.upper().count('N') / len(seq)
        if n_ratio > 0.1:  # 超过10%的N
            invalid_reasons['too_many_N'] += 1
            continue

        # 检查标签
        if not sample.get('label'):
            invalid_reasons['missing_label'] += 1
            continue

        valid_samples.append(sample)

    stats = {
        'total': len(samples),
        'valid': len(valid_samples),
        'invalid': len(samples) - len(valid_samples),
        'invalid_reasons': dict(invalid_reasons)
    }

    if stats['invalid'] > 0:
        logger.warning(f"验证: {stats['valid']:,}/{stats['total']:,} 有效")
        for reason, count in invalid_reasons.items():
            logger.warning(f"  {reason}: {count:,}")

    return valid_samples, stats


if __name__ == "__main__":
    # 测试代码
    print("测试utils模块...")

    # 测试类别权重计算
    labels = ['A'] * 100 + ['B'] * 300 + ['C'] * 600
    weights = compute_class_weights(labels)
    print(f"\n类别权重: {weights}")

    # 测试样本验证
    test_samples = [
        {'sequence': 'A' * 512, 'label': 'test'},
        {'sequence': 'A' * 100, 'label': 'test'},  # 长度错误
        {'sequence': 'X' * 512, 'label': 'test'},  # 无效碱基
        {'sequence': 'N' * 512, 'label': 'test'},  # 太多N
        {'sequence': 'A' * 512, 'label': ''},      # 缺少标签
    ]
    valid, stats = validate_samples(test_samples)
    print(f"\n验证结果: {stats}")

    print("\n测试完成!")
