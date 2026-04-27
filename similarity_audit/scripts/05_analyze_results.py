#!/usr/bin/env python3
"""
结果分析脚本：分析MMseqs2比对结果

功能:
1. 加载Run1结果并计算coverage
2. 统计identity分布
3. 识别高风险样本
4. 按标签/物种/长度分层统计
5. 生成高风险query FASTA（用于Run3）
6. 对比Run1和Run2结果（兜底验证）
7. 分析Run3近重复簇
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import config

def load_metadata(metadata_path):
    """加载元数据"""
    print(f"加载元数据: {metadata_path}")
    df = pd.read_csv(metadata_path, sep='\t')
    print(f"  样本数: {len(df):,}")
    return df

def load_mmseqs2_results(results_path):
    """加载MMseqs2结果"""
    print(f"\n加载MMseqs2结果: {results_path}")

    if not results_path.exists():
        print(f"  文件不存在，跳过")
        return None

    # 列名
    columns = ['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen',
               'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits', 'qlen', 'tlen']

    df = pd.read_csv(results_path, sep='\t', names=columns)
    print(f"  匹配数: {len(df):,}")

    # 计算coverage
    df['qcov'] = (df['qend'] - df['qstart'] + 1) / df['qlen']
    df['tcov'] = (df['tend'] - df['tstart'] + 1) / df['tlen']

    return df

def parse_fasta_header(header):
    """解析FASTA header"""
    # 格式: seq_id|label|gene|species|seqname|start|end|strand|length
    parts = header.split('|')
    return {
        'seq_id': parts[0],
        'label': parts[1] if len(parts) > 1 else 'unknown',
        'gene': parts[2] if len(parts) > 2 else 'unknown',
        'species': parts[3] if len(parts) > 3 else 'unknown',
    }

def analyze_run1(run1_df, test_meta):
    """分析Run1结果"""
    print("\n" + "="*80)
    print("分析Run1结果")
    print("="*80)

    results = {}

    # 1. 全局统计
    total_queries = config.TOTAL_TEST_SAMPLES
    queries_with_hit = len(run1_df)
    queries_no_hit = total_queries - queries_with_hit

    results['global_stats'] = {
        'total_queries': total_queries,
        'queries_with_hit': queries_with_hit,
        'queries_no_hit': queries_no_hit,
        'hit_rate': queries_with_hit / total_queries,
        'no_hit_definition': "在主run参数下（identity≥80%, coverage≥70%, E-value≤1e-5）未找到任何train集匹配的test样本"
    }

    print(f"\n全局统计:")
    print(f"  总query数: {total_queries:,}")
    print(f"  有hit数: {queries_with_hit:,} ({queries_with_hit/total_queries*100:.2f}%)")
    print(f"  无hit数: {queries_no_hit:,} ({queries_no_hit/total_queries*100:.2f}%)")

    # 2. Identity分布
    print(f"\nIdentity分布:")
    identity_dist = {}

    for min_val, max_val, label in config.IDENTITY_BINS:
        if min_val == max_val:  # 100%
            count = len(run1_df[run1_df['fident'] == 1.0])
        elif min_val == 0.0:  # no-hit
            count = queries_no_hit
        else:
            count = len(run1_df[(run1_df['fident'] >= min_val) & (run1_df['fident'] < max_val)])

        identity_dist[label] = {
            'count': int(count),
            'percentage': count / total_queries * 100
        }
        print(f"  {label}: {count:,} ({count/total_queries*100:.2f}%)")

    results['identity_distribution'] = identity_dist

    # 3. 高风险样本统计
    print(f"\n高风险样本统计:")
    high_risk_stats = {}

    for risk_level, thresholds in config.RISK_THRESHOLDS.items():
        mask = (run1_df['fident'] >= thresholds['identity']) & \
               (run1_df['qcov'] >= thresholds['qcov'])
        count = len(run1_df[mask])

        high_risk_stats[risk_level] = {
            'threshold': thresholds,
            'count': int(count),
            'percentage': count / total_queries * 100
        }

        print(f"  {risk_level}: identity≥{thresholds['identity']:.0%}, qcov≥{thresholds['qcov']:.0%}")
        print(f"    样本数: {count:,} ({count/total_queries*100:.4f}%)")

    results['high_risk_stats'] = high_risk_stats

    # 4. 提取query的元数据
    run1_df['query_label'] = run1_df['query'].apply(lambda x: parse_fasta_header(x)['label'])
    run1_df['query_species'] = run1_df['query'].apply(lambda x: parse_fasta_header(x)['species'])

    # 5. 按标签分层统计
    print(f"\n按标签分层的高风险比例:")
    by_label = {}

    # 从test_meta获取每个标签的总数
    label_counts = test_meta['label_name'].value_counts().to_dict()

    for label in config.LABELS:
        label_total = label_counts.get(label, 0)
        label_df = run1_df[run1_df['query_label'] == label]

        by_label[label] = {
            'total': int(label_total),
            'with_hit': len(label_df),
            'risk_levels': {}
        }

        for risk_level, thresholds in config.RISK_THRESHOLDS.items():
            mask = (label_df['fident'] >= thresholds['identity']) & \
                   (label_df['qcov'] >= thresholds['qcov'])
            count = len(label_df[mask])
            by_label[label]['risk_levels'][risk_level] = {
                'count': int(count),
                'percentage': count / label_total * 100 if label_total > 0 else 0
            }

        print(f"  {label} (总数: {label_total:,}):")
        for risk_level in ['extreme', 'high', 'medium']:
            count = by_label[label]['risk_levels'][risk_level]['count']
            pct = by_label[label]['risk_levels'][risk_level]['percentage']
            print(f"    {risk_level}: {count:,} ({pct:.4f}%)")

    results['by_label'] = by_label

    # 6. 按物种分层统计
    print(f"\n按物种分层的高风险比例:")
    by_species = {}

    species_counts = test_meta['species'].value_counts().to_dict()

    for species in config.SPECIES:
        species_total = species_counts.get(species, 0)
        species_df = run1_df[run1_df['query_species'] == species]

        by_species[species] = {
            'total': int(species_total),
            'with_hit': len(species_df),
            'risk_levels': {}
        }

        for risk_level, thresholds in config.RISK_THRESHOLDS.items():
            mask = (species_df['fident'] >= thresholds['identity']) & \
                   (species_df['qcov'] >= thresholds['qcov'])
            count = len(species_df[mask])
            by_species[species]['risk_levels'][risk_level] = {
                'count': int(count),
                'percentage': count / species_total * 100 if species_total > 0 else 0
            }

        print(f"  {species} (总数: {species_total:,}):")
        for risk_level in ['extreme', 'high', 'medium']:
            count = by_species[species]['risk_levels'][risk_level]['count']
            pct = by_species[species]['risk_levels'][risk_level]['percentage']
            print(f"    {risk_level}: {count:,} ({pct:.4f}%)")

    results['by_species'] = by_species

    # 7. 按长度区间分层统计
    print(f"\n按长度区间分层的高风险比例:")
    by_length = {}

    for min_len, max_len, label in config.LENGTH_BINS:
        if max_len == float('inf'):
            length_df = run1_df[run1_df['qlen'] >= min_len]
            length_total = len(test_meta[test_meta['sequence_length'] >= min_len])
        else:
            length_df = run1_df[(run1_df['qlen'] >= min_len) & (run1_df['qlen'] < max_len)]
            length_total = len(test_meta[(test_meta['sequence_length'] >= min_len) &
                                         (test_meta['sequence_length'] < max_len)])

        by_length[label] = {
            'total': int(length_total),
            'with_hit': len(length_df),
            'risk_levels': {}
        }

        for risk_level, thresholds in config.RISK_THRESHOLDS.items():
            mask = (length_df['fident'] >= thresholds['identity']) & \
                   (length_df['qcov'] >= thresholds['qcov'])
            count = len(length_df[mask])
            by_length[label]['risk_levels'][risk_level] = {
                'count': int(count),
                'percentage': count / length_total * 100 if length_total > 0 else 0
            }

        print(f"  {label} (总数: {length_total:,}):")
        for risk_level in ['extreme', 'high', 'medium']:
            count = by_length[label]['risk_levels'][risk_level]['count']
            pct = by_length[label]['risk_levels'][risk_level]['percentage']
            print(f"    {risk_level}: {count:,} ({pct:.4f}%)")

    results['by_length'] = by_length

    # 8. 提取高风险样本详情
    medium_risk_mask = (run1_df['fident'] >= config.RISK_THRESHOLDS['medium']['identity']) & \
                       (run1_df['qcov'] >= config.RISK_THRESHOLDS['medium']['qcov'])
    high_risk_samples = run1_df[medium_risk_mask].copy()

    print(f"\n高风险样本详情:")
    print(f"  中风险及以上样本数: {len(high_risk_samples):,}")

    # 保存高风险样本
    high_risk_samples_list = []
    for _, row in high_risk_samples.iterrows():
        query_info = parse_fasta_header(row['query'])
        target_info = parse_fasta_header(row['target'])

        high_risk_samples_list.append({
            'query_id': query_info['seq_id'],
            'query_label': query_info['label'],
            'query_gene': query_info['gene'],
            'query_species': query_info['species'],
            'target_id': target_info['seq_id'],
            'target_label': target_info['label'],
            'target_gene': target_info['gene'],
            'target_species': target_info['species'],
            'identity': float(row['fident']),
            'qcov': float(row['qcov']),
            'tcov': float(row['tcov']),
            'evalue': float(row['evalue']),
            'bits': float(row['bits']),
            'qlen': int(row['qlen']),
            'tlen': int(row['tlen']),
        })

    results['high_risk_samples'] = high_risk_samples_list

    return results, high_risk_samples

def generate_high_risk_fasta(high_risk_samples, test_fasta_path, output_path):
    """生成高风险query的FASTA文件（用于Run3）"""
    print(f"\n生成高风险query FASTA: {output_path}")

    high_risk_ids = set(high_risk_samples['query'].apply(lambda x: parse_fasta_header(x)['seq_id']))
    print(f"  高风险query数: {len(high_risk_ids):,}")

    with open(test_fasta_path, 'r') as in_f, open(output_path, 'w') as out_f:
        write_seq = False
        for line in in_f:
            if line.startswith('>'):
                seq_id = line.strip()[1:].split('|')[0]
                write_seq = seq_id in high_risk_ids

            if write_seq:
                out_f.write(line)

    print(f"  完成")

def compare_run1_run2(run1_df, run2_df, sample_ids_path):
    """对比Run1和Run2结果（兜底验证）"""
    print("\n" + "="*80)
    print("对比Run1和Run2结果（兜底验证）")
    print("="*80)

    if run2_df is None:
        print("Run2结果不存在，跳过对比")
        return None

    # 加载抽样ID列表
    with open(sample_ids_path, 'r') as f:
        sample_ids = set(line.strip() for line in f)

    # 提取Run1在抽样样本上的结果
    run1_sample = run1_df[run1_df['query'].apply(lambda x: parse_fasta_header(x)['seq_id'] in sample_ids)]

    print(f"\n抽样样本数: {len(sample_ids):,}")
    print(f"Run1在抽样中的匹配数: {len(run1_sample):,}")
    print(f"Run2在抽样中的匹配数: {len(run2_df):,}")

    # 统计高风险样本
    results = {
        'sample_size': len(sample_ids),
        'run1_hits': len(run1_sample),
        'run2_hits': len(run2_df),
        'risk_comparison': {}
    }

    for risk_level, thresholds in config.RISK_THRESHOLDS.items():
        run1_mask = (run1_sample['fident'] >= thresholds['identity']) & \
                    (run1_sample['qcov'] >= thresholds['qcov'])
        run2_mask = (run2_df['fident'] >= thresholds['identity']) & \
                    (run2_df['qcov'] >= thresholds['qcov'])

        run1_count = len(run1_sample[run1_mask])
        run2_count = len(run2_df[run2_mask])

        results['risk_comparison'][risk_level] = {
            'run1_count': int(run1_count),
            'run2_count': int(run2_count),
            'new_found': int(run2_count - run1_count)
        }

        print(f"\n{risk_level} (identity≥{thresholds['identity']:.0%}, qcov≥{thresholds['qcov']:.0%}):")
        print(f"  Run1: {run1_count:,}")
        print(f"  Run2: {run2_count:,}")
        print(f"  新发现: {run2_count - run1_count:,}")

    # 结论
    total_new_high_risk = results['risk_comparison']['high']['new_found']
    if total_new_high_risk == 0:
        conclusion = "✅ Run1参数充分，未发现漏检的高风险样本"
    else:
        conclusion = f"⚠️ Run2发现{total_new_high_risk}个Run1漏检的高风险样本，建议检查"

    results['conclusion'] = conclusion
    print(f"\n结论: {conclusion}")

    return results

def analyze_run3(run3_df):
    """分析Run3近重复簇结果"""
    print("\n" + "="*80)
    print("分析Run3近重复簇结果")
    print("="*80)

    if run3_df is None:
        print("Run3结果不存在，跳过分析")
        return None

    # 统计每个query的匹配数
    query_match_counts = run3_df.groupby('query').size()

    results = {
        'total_queries': len(query_match_counts),
        'total_matches': len(run3_df),
        'avg_matches_per_query': float(query_match_counts.mean()),
        'max_matches': int(query_match_counts.max()),
        'queries_with_multiple_matches': int((query_match_counts > 1).sum()),
    }

    print(f"\n统计:")
    print(f"  高风险query数: {results['total_queries']:,}")
    print(f"  总匹配数: {results['total_matches']:,}")
    print(f"  平均每个query的匹配数: {results['avg_matches_per_query']:.2f}")
    print(f"  最大匹配数: {results['max_matches']}")
    print(f"  有多个匹配的query数: {results['queries_with_multiple_matches']:,}")

    # 提取有多个匹配的query示例
    multi_match_queries = query_match_counts[query_match_counts > 1].head(10)

    print(f"\n有多个匹配的query示例（前10个）:")
    cluster_examples = []
    for query, count in multi_match_queries.items():
        query_matches = run3_df[run3_df['query'] == query]
        query_info = parse_fasta_header(query)

        targets = []
        for _, row in query_matches.iterrows():
            target_info = parse_fasta_header(row['target'])
            targets.append({
                'target_id': target_info['seq_id'],
                'target_gene': target_info['gene'],
                'identity': float(row['fident']),
                'qcov': float(row['qcov']),
            })

        cluster_examples.append({
            'query_id': query_info['seq_id'],
            'query_gene': query_info['gene'],
            'match_count': int(count),
            'targets': targets
        })

        print(f"  {query_info['seq_id']} ({query_info['gene']}): {count}个匹配")
        for target in targets[:3]:  # 只显示前3个
            print(f"    - {target['target_id']} ({target['target_gene']}): {target['identity']:.1%}")

    results['cluster_examples'] = cluster_examples

    return results

def main():
    print("="*80)
    print("Exp4 Test-Train序列相似性分析 - 结果分析")
    print("="*80)

    # 创建输出目录
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载元数据
    test_meta = load_metadata(config.TEST_METADATA)

    # 2. 加载Run1结果
    run1_df = load_mmseqs2_results(config.RUN1_RESULTS)

    if run1_df is None:
        print("错误: Run1结果不存在")
        return

    # 3. 分析Run1
    run1_results, high_risk_samples = analyze_run1(run1_df, test_meta)

    # 保存Run1分析结果
    with open(config.SUMMARY_STATS, 'w') as f:
        json.dump(run1_results['global_stats'], f, indent=2)

    with open(config.IDENTITY_DIST, 'w') as f:
        json.dump(run1_results['identity_distribution'], f, indent=2)

    with open(config.HIGH_RISK_SAMPLES, 'w') as f:
        json.dump({
            'stats': run1_results['high_risk_stats'],
            'samples': run1_results['high_risk_samples']
        }, f, indent=2)

    with open(config.BY_LABEL, 'w') as f:
        json.dump(run1_results['by_label'], f, indent=2)

    with open(config.BY_SPECIES, 'w') as f:
        json.dump(run1_results['by_species'], f, indent=2)

    with open(config.BY_LENGTH, 'w') as f:
        json.dump(run1_results['by_length'], f, indent=2)

    print(f"\n保存Run1分析结果:")
    print(f"  {config.SUMMARY_STATS}")
    print(f"  {config.IDENTITY_DIST}")
    print(f"  {config.HIGH_RISK_SAMPLES}")
    print(f"  {config.BY_LABEL}")
    print(f"  {config.BY_SPECIES}")
    print(f"  {config.BY_LENGTH}")

    # 4. 生成高风险query FASTA（用于Run3）
    generate_high_risk_fasta(
        high_risk_samples,
        config.DATA_DIR / "test_sequences.fasta",
        config.HIGH_RISK_FASTA
    )

    # 5. 加载并对比Run2结果
    run2_df = load_mmseqs2_results(config.RUN2_RESULTS)
    if run2_df is not None:
        run2_results = compare_run1_run2(run1_df, run2_df, config.SAMPLE_IDS)
        if run2_results:
            with open(config.RUN2_COMPARISON, 'w') as f:
                json.dump(run2_results, f, indent=2)
            print(f"\n保存Run2对比结果: {config.RUN2_COMPARISON}")

    # 6. 加载并分析Run3结果
    run3_df = load_mmseqs2_results(config.RUN3_RESULTS)
    if run3_df is not None:
        run3_results = analyze_run3(run3_df)
        if run3_results:
            with open(config.RUN3_CLUSTER_ANALYSIS, 'w') as f:
                json.dump(run3_results, f, indent=2)
            print(f"\n保存Run3分析结果: {config.RUN3_CLUSTER_ANALYSIS}")

    print("\n" + "="*80)
    print("结果分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
