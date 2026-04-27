#!/usr/bin/env python3
"""
Task G 数据准备脚本 v3 - 细粒度基因内部结构分类

改进设计 (相比v2):
1. 将gene body按GFF标注切分为细粒度类别
2. 只保留有明确标签依据的细粒度数据
3. 跨物种序列级去重
4. 基因级划分

细粒度标签 (有GFF标注依据):
- CDS: 编码序列 (直接从GFF的CDS特征提取)
- intron: 内含子 (从exon间隙计算)
- tRNA: 转运RNA
- rRNA: 核糖体RNA
- snRNA: 小核RNA
- snoRNA: 小核仁RNA
- lncRNA_exon: 长链非编码RNA的外显子

划分策略:
- 基因级划分 (同一基因的所有样本在同一数据集)
"""

import sys
import json
import random
import hashlib
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    SPECIES_CONFIG, DATA_OUTPUT_ROOT, PROJECT_ROOT,
    TRAIN_RATIO, EVAL_RATIO, TEST_RATIO, RANDOM_SEED,
    get_species_paths, BLOCK_SIZE
)
from src.utils import save_json, compute_class_weights
from src.region_definition import RegionDefinition, GenomicRegion
from src.length_stratified_sampler import LengthStratifiedSampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 二分类标签定义
BINARY_LABELS = ['gene_body', 'background']

# 细粒度正例标签（用于从GFF提取，训练时统一为gene_body）
POSITIVE_FINE_LABELS = ['CDS', 'intron', 'tRNA', 'rRNA', 'snRNA', 'snoRNA', 'lncRNA_exon']

# 标签到ID的映射（二分类）
LABEL2ID = {'gene_body': 0, 'background': 1}
ID2LABEL = {0: 'gene_body', 1: 'background'}


def sequence_md5(sequence: str) -> str:
    """计算序列的MD5哈希"""
    return hashlib.md5(sequence.encode()).hexdigest()


def load_fasta_sequences(fasta_path: Path) -> Dict[str, str]:
    """加载FASTA文件中的所有序列"""
    logger.info(f"加载FASTA文件: {fasta_path}")
    sequences = {}
    current_seqname = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_seqname:
                    sequences[current_seqname] = ''.join(current_seq).upper()
                current_seqname = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_seqname:
            sequences[current_seqname] = ''.join(current_seq).upper()

    total_length = sum(len(seq) for seq in sequences.values())
    logger.info(f"✓ 加载 {len(sequences)} 个序列, 总长度 {total_length:,} bp")

    return sequences


def parse_gff_attributes(attributes_str: str) -> Dict[str, str]:
    """解析GFF属性字段"""
    attributes = {}
    for item in attributes_str.split(';'):
        item = item.strip()
        if '=' in item:
            key, value = item.split('=', 1)
            attributes[key] = value
    return attributes


def get_gene_id_from_attributes(attributes: Dict) -> str:
    """从属性中提取基因ID"""
    for key in ['gene', 'gene_id', 'Parent', 'locus_tag']:
        if key in attributes:
            value = attributes[key]
            if key == 'Parent' and ',' in value:
                value = value.split(',')[0]
            return value
    return ''


def extract_fine_grained_features(
    gff_path: Path,
    fasta_path: Path,
    species_id: str,
    min_length: int = 10
) -> List[Dict]:
    """从GFF中提取细粒度特征

    Args:
        gff_path: GFF文件路径
        fasta_path: FASTA文件路径
        species_id: 物种ID
        min_length: 最小序列长度

    Returns:
        样本列表
    """
    logger.info(f"从GFF提取细粒度特征: {gff_path}")

    # 加载序列
    sequences = load_fasta_sequences(fasta_path)

    samples = []
    feature_counts = defaultdict(int)

    # 存储转录本信息用于计算intron和lncRNA_exon
    transcripts: Dict[str, List[Dict]] = defaultdict(list)  # transcript_id -> exons
    transcript_info: Dict[str, Dict] = {}  # transcript_id -> {type, gene_id, seqname, strand}

    # 第一遍：收集所有特征
    logger.info("第一遍：收集特征...")

    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 9:
                continue

            seqname, source, feature_type, start, end, score, strand, frame, attributes_str = parts
            start, end = int(start), int(end)

            if seqname not in sequences:
                continue

            attributes = parse_gff_attributes(attributes_str)
            gene_id = get_gene_id_from_attributes(attributes)
            gene_name = attributes.get('Name', attributes.get('gene', gene_id))

            # 直接提取的特征类型
            if feature_type == 'CDS':
                seq = sequences[seqname][start - 1:end]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'label': 'CDS',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['CDS'] += 1

            elif feature_type == 'tRNA':
                seq = sequences[seqname][start - 1:end]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'label': 'tRNA',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['tRNA'] += 1

            elif feature_type == 'rRNA':
                seq = sequences[seqname][start - 1:end]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'label': 'rRNA',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['rRNA'] += 1

            elif feature_type == 'snRNA':
                seq = sequences[seqname][start - 1:end]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'label': 'snRNA',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['snRNA'] += 1

            elif feature_type == 'snoRNA':
                seq = sequences[seqname][start - 1:end]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'label': 'snoRNA',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['snoRNA'] += 1

            # 记录转录本信息
            elif feature_type in ('mRNA', 'lnc_RNA', 'transcript', 'ncRNA', 'primary_transcript'):
                transcript_id = attributes.get('ID', '')
                if transcript_id:
                    transcript_info[transcript_id] = {
                        'type': feature_type,
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'seqname': seqname,
                        'strand': strand
                    }

            # 记录exon用于计算intron和lncRNA_exon
            elif feature_type == 'exon':
                parent = attributes.get('Parent', '')
                if parent:
                    # 处理多个parent的情况
                    for p in parent.split(','):
                        transcripts[p].append({
                            'seqname': seqname,
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'gene_id': gene_id,
                            'gene_name': gene_name
                        })

    # 第二遍：从exon计算intron和lncRNA_exon
    logger.info("第二遍：计算intron和lncRNA_exon...")

    for transcript_id, exons in transcripts.items():
        if len(exons) < 1:
            continue

        # 获取转录本信息
        t_info = transcript_info.get(transcript_id, {})
        t_type = t_info.get('type', '')

        # 按位置排序exon
        exons.sort(key=lambda x: x['start'])
        seqname = exons[0]['seqname']
        strand = exons[0]['strand']
        gene_id = t_info.get('gene_id', exons[0].get('gene_id', ''))
        gene_name = t_info.get('gene_name', exons[0].get('gene_name', ''))

        if seqname not in sequences:
            continue

        # 如果是lnc_RNA，提取exon作为lncRNA_exon
        if t_type == 'lnc_RNA':
            for exon in exons:
                seq = sequences[seqname][exon['start'] - 1:exon['end']]
                if len(seq) >= min_length:
                    samples.append({
                        'sequence': seq,
                        'seqname': seqname,
                        'start': exon['start'],
                        'end': exon['end'],
                        'strand': strand,
                        'label': 'lncRNA_exon',
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'species': species_id,
                        'sequence_length': len(seq)
                    })
                    feature_counts['lncRNA_exon'] += 1

        # 计算intron（从所有转录本类型）
        if len(exons) >= 2:
            for i in range(len(exons) - 1):
                intron_start = exons[i]['end'] + 1
                intron_end = exons[i + 1]['start'] - 1

                if intron_end >= intron_start:
                    seq = sequences[seqname][intron_start - 1:intron_end]
                    if len(seq) >= min_length:
                        samples.append({
                            'sequence': seq,
                            'seqname': seqname,
                            'start': intron_start,
                            'end': intron_end,
                            'strand': strand,
                            'label': 'intron',
                            'gene_id': gene_id,
                            'gene_name': gene_name,
                            'species': species_id,
                            'sequence_length': len(seq)
                        })
                        feature_counts['intron'] += 1

    logger.info(f"✓ 提取完成:")
    for label in POSITIVE_FINE_LABELS:
        count = feature_counts.get(label, 0)
        logger.info(f"  {label}: {count:,}")
    logger.info(f"  总计: {len(samples):,}")

    return samples


def position_level_dedup(
    samples: List[Dict],
    label_priority: Dict[str, int] = None
) -> List[Dict]:
    """位置级去重（同一物种内同一位置的重复）

    当同一位置有多个标签时，按优先级保留

    Args:
        samples: 样本列表
        label_priority: 标签优先级（数字越小优先级越高）

    Returns:
        去重后的样本列表
    """
    logger.info("位置级去重...")

    if label_priority is None:
        label_priority = {
            'CDS': 0,
            'intron': 1,
            'tRNA': 2,
            'rRNA': 3,
            'snRNA': 4,
            'snoRNA': 5,
            'lncRNA_exon': 6
        }

    original_count = len(samples)

    # 按 (species, seqname, start, end) 分组
    pos_groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for sample in samples:
        key = (sample['species'], sample['seqname'], sample['start'], sample['end'])
        pos_groups[key].append(sample)

    # 对每组选择优先级最高的样本
    deduplicated = []
    conflict_count = 0

    for pos_key, group in pos_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # 检查是否有标签冲突
            labels = set(s['label'] for s in group)
            if len(labels) > 1:
                conflict_count += 1

            # 按优先级排序，选择优先级最高的
            group.sort(key=lambda x: label_priority.get(x['label'], 999))
            deduplicated.append(group[0])

    removed = original_count - len(deduplicated)
    logger.info(f"✓ 位置级去重: {original_count:,} → {len(deduplicated):,} (去除 {removed:,})")
    if conflict_count > 0:
        logger.info(f"  标签冲突位置: {conflict_count:,}")

    return deduplicated


def cross_species_sequence_dedup(
    samples: List[Dict],
    label_priority: Dict[str, int] = None
) -> List[Dict]:
    """跨物种序列级去重

    当同一序列有多个标签时，按优先级保留

    Args:
        samples: 样本列表
        label_priority: 标签优先级（数字越小优先级越高）

    Returns:
        去重后的样本列表
    """
    logger.info("跨物种序列级去重...")

    if label_priority is None:
        # 默认优先级：CDS > intron > ncRNA类型
        label_priority = {
            'CDS': 0,
            'intron': 1,
            'tRNA': 2,
            'rRNA': 3,
            'snRNA': 4,
            'snoRNA': 5,
            'lncRNA_exon': 6
        }

    original_count = len(samples)

    # 按序列哈希分组
    seq_groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        seq_hash = sequence_md5(sample['sequence'])
        seq_groups[seq_hash].append(sample)

    # 对每组选择优先级最高的样本
    deduplicated = []
    conflict_count = 0

    for seq_hash, group in seq_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # 检查是否有标签冲突
            labels = set(s['label'] for s in group)
            if len(labels) > 1:
                conflict_count += 1

            # 按优先级排序，选择优先级最高的
            group.sort(key=lambda x: label_priority.get(x['label'], 999))
            deduplicated.append(group[0])

    removed = original_count - len(deduplicated)
    logger.info(f"✓ 跨物种序列级去重: {original_count:,} → {len(deduplicated):,} (去除 {removed:,})")
    if conflict_count > 0:
        logger.info(f"  标签冲突位置: {conflict_count:,}")

    return deduplicated


def gene_level_split(
    samples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    eval_ratio: float = EVAL_RATIO
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """基因级划分

    确保同一基因的所有样本在同一数据集

    策略：
    1. 优先使用gene_id进行分组
    2. 对于没有gene_id的样本，使用基因组位置聚类（重叠的样本归为同一组）
    """
    logger.info("执行基因级划分...")

    # 分离有gene_id和无gene_id的样本
    samples_with_gene_id = []
    samples_without_gene_id = []

    for sample in samples:
        gene_id = sample.get('gene_id', '')
        if gene_id and gene_id != 'unknown':
            samples_with_gene_id.append(sample)
        else:
            samples_without_gene_id.append(sample)

    logger.info(f"  有gene_id的样本: {len(samples_with_gene_id):,}")
    logger.info(f"  无gene_id的样本: {len(samples_without_gene_id):,}")

    # 按 (species, gene_id) 分组有gene_id的样本
    samples_by_gene = defaultdict(list)
    for sample in samples_with_gene_id:
        key = (sample['species'], sample['gene_id'])
        samples_by_gene[key].append(sample)

    # 对无gene_id的样本，按位置聚类（同一染色体上重叠的样本归为同一组）
    if samples_without_gene_id:
        # 按 (species, seqname) 分组
        by_chrom = defaultdict(list)
        for sample in samples_without_gene_id:
            key = (sample['species'], sample['seqname'])
            by_chrom[key].append(sample)

        cluster_id = 0
        for (species, seqname), chrom_samples in by_chrom.items():
            # 按起始位置排序
            chrom_samples.sort(key=lambda x: x['start'])

            # 聚类重叠的样本
            current_cluster = [chrom_samples[0]]
            current_end = chrom_samples[0]['end']

            for sample in chrom_samples[1:]:
                if sample['start'] <= current_end + 1:  # 重叠或相邻
                    current_cluster.append(sample)
                    current_end = max(current_end, sample['end'])
                else:
                    # 保存当前聚类
                    cluster_key = (species, f"_cluster_{cluster_id}")
                    samples_by_gene[cluster_key].extend(current_cluster)
                    cluster_id += 1
                    # 开始新聚类
                    current_cluster = [sample]
                    current_end = sample['end']

            # 保存最后一个聚类
            if current_cluster:
                cluster_key = (species, f"_cluster_{cluster_id}")
                samples_by_gene[cluster_key].extend(current_cluster)
                cluster_id += 1

        logger.info(f"  无gene_id样本聚类为 {cluster_id} 个组")

    genes = list(samples_by_gene.keys())
    n_genes = len(genes)
    logger.info(f"  总基因数: {n_genes:,}")

    random.shuffle(genes)

    n_train = int(n_genes * train_ratio)
    n_eval = int(n_genes * eval_ratio)

    train_genes = genes[:n_train]
    eval_genes = genes[n_train:n_train + n_eval]
    test_genes = genes[n_train + n_eval:]

    train_samples = []
    eval_samples = []
    test_samples = []

    for gene in train_genes:
        train_samples.extend(samples_by_gene[gene])
    for gene in eval_genes:
        eval_samples.extend(samples_by_gene[gene])
    for gene in test_genes:
        test_samples.extend(samples_by_gene[gene])

    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    random.shuffle(test_samples)

    split_info = {
        'total_genes': n_genes,
        'train_genes': len(train_genes),
        'eval_genes': len(eval_genes),
        'test_genes': len(test_genes)
    }

    logger.info(f"✓ 基因级划分完成:")
    logger.info(f"  Train: {len(train_samples):,} 样本 ({len(train_genes):,} 基因)")
    logger.info(f"  Eval: {len(eval_samples):,} 样本 ({len(eval_genes):,} 基因)")
    logger.info(f"  Test: {len(test_samples):,} 样本 ({len(test_genes):,} 基因)")

    return train_samples, eval_samples, test_samples, split_info


def samples_to_training_format(samples: List[Dict], label2id: Dict[str, int]) -> List[Dict]:
    """转换为训练格式

    将细粒度正例标签统一转换为gene_body，负例保持background
    """
    training_data = []

    for sample in samples:
        original_label = sample['label']

        # 将细粒度正例标签统一为gene_body
        if original_label in POSITIVE_FINE_LABELS:
            label_name = 'gene_body'
        elif original_label == 'background':
            label_name = 'background'
        else:
            continue  # 跳过未知标签

        label_id = label2id[label_name]

        training_sample = {
            'sequence': sample['sequence'],
            'label': label_id,
            'label_name': label_name,
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
        training_data.append(training_sample)

    return training_data


def print_data_statistics(train_data, eval_data, test_data, task_name: str) -> Dict:
    """打印数据统计信息"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{task_name} 数据统计")
    logger.info(f"{'='*70}")

    all_data = train_data + eval_data + test_data

    # 总体统计
    logger.info(f"\n总样本数: {len(all_data):,}")
    logger.info(f"  Train: {len(train_data):,} ({len(train_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Eval: {len(eval_data):,} ({len(eval_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data):,} ({len(test_data)/len(all_data)*100:.1f}%)")

    # 二分类标签分布
    logger.info(f"\n二分类标签分布:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        label_counts = Counter([d['label_name'] for d in data])
        logger.info(f"  {name}:")
        for label in BINARY_LABELS:
            count = label_counts.get(label, 0)
            pct = count / len(data) * 100 if data else 0
            logger.info(f"    {label}: {count:,} ({pct:.1f}%)")

    # 原始细粒度标签分布（正例内部）
    logger.info(f"\n正例细粒度标签分布 (original_label):")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        original_counts = Counter([d.get('original_label', d['label_name']) for d in data])
        logger.info(f"  {name}:")
        for label in POSITIVE_FINE_LABELS + ['background']:
            count = original_counts.get(label, 0)
            pct = count / len(data) * 100 if data else 0
            if count > 0:
                logger.info(f"    {label}: {count:,} ({pct:.1f}%)")

    # 长度统计
    logger.info(f"\n长度统计:")
    for name, data in [('Train', train_data), ('Eval', eval_data), ('Test', test_data)]:
        if data:
            lengths = [d['sequence_length'] for d in data]
            logger.info(f"  {name}: min={min(lengths)}, max={max(lengths)}, "
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
        'binary_label_distribution': dict(Counter([d['label_name'] for d in all_data])),
        'original_label_distribution': dict(Counter([d.get('original_label', d['label_name']) for d in all_data])),
        'species_distribution': dict(species_counts)
    }

    return stats


def block_level_split(
    samples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    eval_ratio: float = EVAL_RATIO
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Block级划分（用于负例）"""
    logger.info("执行Block级划分...")

    # 按block_id分组
    samples_by_block = defaultdict(list)
    for sample in samples:
        samples_by_block[sample['block_id']].append(sample)

    blocks = list(samples_by_block.keys())
    n_blocks = len(blocks)
    logger.info(f"  总Block数: {n_blocks:,}")

    random.shuffle(blocks)

    n_train = int(n_blocks * train_ratio)
    n_eval = int(n_blocks * eval_ratio)

    train_blocks = blocks[:n_train]
    eval_blocks = blocks[n_train:n_train + n_eval]
    test_blocks = blocks[n_train + n_eval:]

    train_samples = []
    eval_samples = []
    test_samples = []

    for block in train_blocks:
        train_samples.extend(samples_by_block[block])
    for block in eval_blocks:
        eval_samples.extend(samples_by_block[block])
    for block in test_blocks:
        test_samples.extend(samples_by_block[block])

    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    random.shuffle(test_samples)

    logger.info(f"✓ Block级划分完成:")
    logger.info(f"  Train: {len(train_samples):,} 样本 ({len(train_blocks):,} blocks)")
    logger.info(f"  Eval: {len(eval_samples):,} 样本 ({len(eval_blocks):,} blocks)")
    logger.info(f"  Test: {len(test_samples):,} 样本 ({len(test_blocks):,} blocks)")

    return train_samples, eval_samples, test_samples


def prepare_task_g_v3(
    species_ids: list = None,
    min_length: int = 10,
    negative_ratio: float = 1.0,
    length_percentiles: List[int] = [0, 10, 25, 50, 75, 90, 100]
):
    """准备Task G v3数据

    Args:
        species_ids: 要处理的物种ID列表（默认全部）
        min_length: 最小序列长度
        negative_ratio: 负例与正例的比例（默认1:1）
        length_percentiles: 长度分层的百分位数
    """
    logger.info("=" * 70)
    logger.info("Task G v3: 细粒度基因内部结构分类 + 背景区域 数据准备")
    logger.info("=" * 70)
    logger.info("设计特点:")
    logger.info("  1. 将gene body按GFF标注切分为细粒度类别（7类正例）")
    logger.info("  2. 从deep intergenic区域采样背景序列（1类负例）")
    logger.info("  3. 负例根据正例长度分布进行分层采样")
    logger.info("  4. 跨物种序列级去重")
    logger.info("  5. 正例基因级划分，负例Block级划分")
    logger.info(f"\n二分类标签: {BINARY_LABELS}")
    logger.info(f"正例细粒度来源: {POSITIVE_FINE_LABELS}")
    logger.info(f"负例比例: 1:{negative_ratio}")

    if species_ids is None:
        species_ids = list(SPECIES_CONFIG.keys())

    logger.info(f"\n处理物种: {species_ids}")

    # 收集所有物种的正例样本和deep intergenic区域
    all_positive_samples = []
    all_deep_intergenic_regions = []  # (region, block_id, species)
    all_sequences = {}  # species -> sequences

    for species_id in species_ids:
        species_info = SPECIES_CONFIG[species_id]
        logger.info(f"\n{'='*50}")
        logger.info(f"处理物种: {species_info['name']} ({species_info['common_name']})")
        logger.info(f"{'='*50}")

        gff_path, fasta_path = get_species_paths(species_id)

        if not gff_path.exists():
            logger.warning(f"GFF文件不存在: {gff_path}")
            continue
        if not fasta_path.exists():
            logger.warning(f"FASTA文件不存在: {fasta_path}")
            continue

        # 1. 提取细粒度特征（正例）
        samples = extract_fine_grained_features(
            gff_path, fasta_path, species_id, min_length
        )
        all_positive_samples.extend(samples)

        # 2. 解析区域定义，获取deep intergenic区域
        logger.info("\n解析基因组区域...")
        rd = RegionDefinition()
        rd.parse_gff(gff_path, fasta_path)
        rd.compute_deep_intergenic()

        stats = rd.get_statistics()
        logger.info(f"  Deep intergenic regions: {stats['total_deep_intergenic']:,}")
        logger.info(f"  Deep intergenic total length: {stats['deep_intergenic_total_length']:,} bp")

        # 3. 加载序列（用于负例采样）
        sequences = {}
        current_seqname = None
        current_seq = []
        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if current_seqname:
                        sequences[current_seqname] = ''.join(current_seq).upper()
                    current_seqname = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seqname:
                sequences[current_seqname] = ''.join(current_seq).upper()
        all_sequences[species_id] = sequences

        # 4. 收集deep intergenic区域（带block分配）
        block_assignments = rd.assign_blocks_to_intergenic()
        for seqname, block_regions in block_assignments.items():
            for region, block_id in block_regions:
                all_deep_intergenic_regions.append({
                    'region': region,
                    'block_id': block_id,
                    'species': species_id
                })

    logger.info(f"\n{'='*50}")
    logger.info("所有物种数据收集完成")
    logger.info(f"{'='*50}")
    logger.info(f"总正例样本: {len(all_positive_samples):,}")
    logger.info(f"总Deep intergenic区域: {len(all_deep_intergenic_regions):,}")

    # 位置级去重（正例）
    all_positive_samples = position_level_dedup(all_positive_samples)

    # 跨物种序列级去重（正例）
    all_positive_samples = cross_species_sequence_dedup(all_positive_samples)

    logger.info(f"\n去重后正例样本: {len(all_positive_samples):,}")

    # 分析正例长度分布
    logger.info("\n分析正例长度分布...")
    positive_lengths = [s['sequence_length'] for s in all_positive_samples]
    sampler = LengthStratifiedSampler()
    sampler.analyze_positive_lengths(positive_lengths)
    sampler.create_bins_from_percentiles(length_percentiles)

    # 从deep intergenic区域进行长度分层采样（负例）
    logger.info("\n从Deep intergenic区域进行长度分层采样...")

    target_negative_count = int(len(all_positive_samples) * negative_ratio)
    logger.info(f"目标负例样本数: {target_negative_count:,} (正负比例 1:{negative_ratio})")

    all_negative_samples = []

    for species_id in species_ids:
        if species_id not in all_sequences:
            continue

        sequences = all_sequences[species_id]

        # 获取该物种的deep intergenic区域
        species_regions = [
            (item['region'], item['block_id'])
            for item in all_deep_intergenic_regions
            if item['species'] == species_id
        ]

        if not species_regions:
            continue

        regions = [r for r, _ in species_regions]
        block_ids = [b for _, b in species_regions]

        # 按物种比例分配采样数
        species_positive_count = sum(1 for s in all_positive_samples if s['species'] == species_id)
        species_target = int(target_negative_count * species_positive_count / len(all_positive_samples))

        logger.info(f"\n{species_id}: 目标采样 {species_target:,} 个负例样本")

        species_negative = sampler.sample_from_regions_stratified(
            regions=regions,
            sequences=sequences,
            total_samples=species_target,
            label='background',
            species=species_id,
            block_ids=block_ids
        )

        all_negative_samples.extend(species_negative)

    logger.info(f"\n总负例样本: {len(all_negative_samples):,}")

    # 跨物种序列级去重（负例）
    logger.info("\n跨物种序列级去重（负例）...")
    original_neg_count = len(all_negative_samples)
    seen_hashes = set()
    deduplicated_neg = []
    for sample in all_negative_samples:
        seq_hash = sequence_md5(sample['sequence'])
        if seq_hash not in seen_hashes:
            seen_hashes.add(seq_hash)
            deduplicated_neg.append(sample)
    all_negative_samples = deduplicated_neg
    logger.info(f"✓ 负例去重: {original_neg_count:,} → {len(all_negative_samples):,}")

    # 数据划分
    logger.info("\n执行数据划分...")

    # 正例：基因级划分
    pos_train, pos_eval, pos_test, gene_split_info = gene_level_split(all_positive_samples)

    # 负例：Block级划分
    neg_train, neg_eval, neg_test = block_level_split(all_negative_samples)

    # 合并
    train_samples = pos_train + neg_train
    eval_samples = pos_eval + neg_eval
    test_samples = pos_test + neg_test

    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    random.shuffle(test_samples)

    logger.info(f"\n合并后:")
    logger.info(f"  Train: {len(train_samples):,} (正例 {len(pos_train):,}, 负例 {len(neg_train):,})")
    logger.info(f"  Eval: {len(eval_samples):,} (正例 {len(pos_eval):,}, 负例 {len(neg_eval):,})")
    logger.info(f"  Test: {len(test_samples):,} (正例 {len(pos_test):,}, 负例 {len(neg_test):,})")

    # 转换为训练格式
    logger.info("\n转换为训练格式...")
    train_data = samples_to_training_format(train_samples, LABEL2ID)
    eval_data = samples_to_training_format(eval_samples, LABEL2ID)
    test_data = samples_to_training_format(test_samples, LABEL2ID)

    # 计算类别权重
    logger.info("\n计算类别权重...")
    all_labels = [d['label'] for d in train_data]
    class_weights = compute_class_weights(all_labels)

    # 转换为标签名称的权重
    class_weights_named = {ID2LABEL[int(k)]: v for k, v in class_weights.items()}
    logger.info(f"类别权重: {class_weights_named}")

    # 打印统计信息
    stats = print_data_statistics(train_data, eval_data, test_data, "Task G v3")

    # 保存数据
    logger.info("\n保存数据...")
    output_dir = DATA_OUTPUT_ROOT / "Task_G_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(train_data, output_dir / "train.json")
    save_json(eval_data, output_dir / "eval.json")
    save_json(test_data, output_dir / "test.json")
    save_json(class_weights, output_dir / "class_weights.json")

    # 保存元数据
    metadata = {
        'task': 'Task_G_v3',
        'version': '3.0',
        'description': 'Gene body vs Background 二分类（正例来自细粒度切分）',
        'labels': BINARY_LABELS,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'num_labels': 2,
        'positive_fine_labels': POSITIVE_FINE_LABELS,
        'species': species_ids,
        'design_changes': [
            '正例：将gene body按GFF标注切分为7类细粒度特征（CDS/intron/tRNA/rRNA/snRNA/snoRNA/lncRNA_exon）',
            '训练时：7类细粒度正例统一标记为gene_body',
            '负例：从deep intergenic区域采样背景序列（background）',
            '负例根据正例长度分布进行分层采样',
            '跨物种序列级去重',
            '正例基因级划分，负例Block级划分'
        ],
        'sampling_strategy': {
            'positive': 'fine_grained_feature_extraction',
            'negative': 'length_stratified_sampling',
            'length_percentiles': length_percentiles,
            'negative_ratio': negative_ratio
        },
        'dedup_strategy': 'cross_species_sequence_level',
        'split_strategy': {
            'positive': 'gene_level',
            'negative': 'block_level'
        },
        'min_length': min_length,
        'statistics': stats,
        'gene_split_info': gene_split_info,
        'class_weights': class_weights,
        'class_weights_named': class_weights_named
    }
    save_json(metadata, output_dir / "metadata.json")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Task G v3 数据准备完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    return train_data, eval_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Task G v3 数据准备 - 细粒度基因内部结构分类 + 背景区域')
    parser.add_argument('--species', nargs='+', default=None,
                       help='要处理的物种ID（默认全部）')
    parser.add_argument('--min-length', type=int, default=10,
                       help='最小序列长度（默认10bp）')
    parser.add_argument('--negative-ratio', type=float, default=1.0,
                       help='负例与正例的比例（默认1.0，即1:1）')
    parser.add_argument('--length-percentiles', nargs='+', type=int,
                       default=[0, 10, 25, 50, 75, 90, 100],
                       help='长度分层的百分位数')

    args = parser.parse_args()

    prepare_task_g_v3(
        species_ids=args.species,
        min_length=args.min_length,
        negative_ratio=args.negative_ratio,
        length_percentiles=args.length_percentiles
    )


if __name__ == "__main__":
    main()
