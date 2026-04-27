#!/usr/bin/env python3
"""
泛化测试集处理脚本
对原始数据进行位置级去重和二分类标签映射
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SPECIES_CONFIGS, DATA_OUTPUT_DIR,
    ORIGINAL_TO_BINARY, LABEL2ID, LABEL_PRIORITY
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def position_dedup(samples):
    """位置级去重，优先保留CDS标签"""
    logger.info(f"执行位置级去重...")

    position_map = defaultdict(list)
    for sample in samples:
        pos_key = f"{sample['seqname']}:{sample['start']}-{sample['end']}:{sample['strand']}"
        position_map[pos_key].append(sample)

    deduped = []
    conflict_count = 0
    cds_priority_count = 0

    for pos_key, pos_samples in position_map.items():
        if len(pos_samples) == 1:
            deduped.append(pos_samples[0])
        else:
            # 多个样本在同一位置，选择优先级最高的
            best = min(pos_samples,
                      key=lambda s: LABEL_PRIORITY.get(s['label_name'], 99))
            deduped.append(best)

            labels = set(s['label_name'] for s in pos_samples)
            if len(labels) > 1:
                conflict_count += 1
                if best['label_name'] == 'CDS':
                    cds_priority_count += 1

    logger.info(f"  去重: {len(samples):,} → {len(deduped):,}")
    logger.info(f"  标签冲突位置: {conflict_count:,}, CDS优先保留: {cds_priority_count:,}")

    return deduped


def convert_to_binary(samples):
    """转换为二分类标签"""
    logger.info("转换为二分类标签...")

    for sample in samples:
        binary_label = ORIGINAL_TO_BINARY.get(sample['label_name'], 'Non-coding')
        sample['binary_label'] = binary_label
        sample['label_id'] = LABEL2ID[binary_label]

    pc_count = sum(1 for s in samples if s['binary_label'] == 'Protein-coding')
    nc_count = len(samples) - pc_count
    logger.info(f"  Protein-coding: {pc_count:,}, Non-coding: {nc_count:,}")

    return samples


def process_species(species_key):
    """处理单个物种的测试集"""
    raw_file = DATA_OUTPUT_DIR / f"{species_key}_raw.json"
    if not raw_file.exists():
        logger.error(f"原始数据不存在: {raw_file}")
        return None

    logger.info(f"\n{'='*50}")
    logger.info(f"处理: {species_key}")
    logger.info(f"{'='*50}")

    with open(raw_file, 'r') as f:
        samples = json.load(f)
    logger.info(f"加载原始数据: {len(samples):,} 样本")

    # 位置级去重
    samples = position_dedup(samples)

    # 二分类标签映射
    samples = convert_to_binary(samples)

    # 保存处理后的测试集
    output_file = DATA_OUTPUT_DIR / f"{species_key}_test.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, ensure_ascii=False)
    logger.info(f"✓ 保存: {output_file}")

    return {
        'species': species_key,
        'total': len(samples),
        'pc': sum(1 for s in samples if s['label_id'] == 0),
        'nc': sum(1 for s in samples if s['label_id'] == 1)
    }


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("泛化测试集处理 - 开始")
    logger.info("=" * 60)

    all_stats = []
    for species_key in SPECIES_CONFIGS.keys():
        stats = process_species(species_key)
        if stats:
            all_stats.append(stats)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("汇总统计")
    logger.info(f"{'='*60}")
    total_all = sum(s['total'] for s in all_stats)
    total_pc = sum(s['pc'] for s in all_stats)
    total_nc = sum(s['nc'] for s in all_stats)

    for s in all_stats:
        logger.info(f"{s['species']}: {s['total']:,} (PC:{s['pc']:,}, NC:{s['nc']:,})")

    logger.info(f"\n总计: {total_all:,} (PC:{total_pc:,}, NC:{total_nc:,})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
