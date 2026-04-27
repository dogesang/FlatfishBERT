#!/usr/bin/env python3
"""
V3 Unified Experiments - 长度控制测试脚本
分析不同序列长度区间的模型性能，评估模型对不同长度序列的泛化能力
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from tqdm import tqdm
import logging

from config import get_experiment_config, EXPERIMENTS, ID2LABEL, MODEL_OUTPUT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 长度区间定义
LENGTH_BINS = [
    (0, 100, "0-100"),
    (100, 200, "100-200"),
    (200, 300, "200-300"),
    (300, 500, "300-500"),
    (500, 1000, "500-1000"),
    (1000, 2000, "1000-2000"),
    (2000, float('inf'), "2000+")
]


def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_length_bin(length):
    """获取序列长度所属的区间"""
    for min_len, max_len, label in LENGTH_BINS:
        if min_len <= length < max_len:
            return label
    return LENGTH_BINS[-1][2]


def analyze_length_distribution(data):
    """分析数据集的长度分布"""
    length_stats = defaultdict(lambda: {'total': 0, 'class_0': 0, 'class_1': 0})

    for item in data:
        seq_len = len(item['sequence'])
        bin_label = get_length_bin(seq_len)
        length_stats[bin_label]['total'] += 1
        length_stats[bin_label][f"class_{item['label_id']}"] += 1

    return dict(length_stats)


def evaluate_by_length(exp_name: str, batch_size: int = 128):
    """按长度区间评估模型性能"""
    print(f"\n{'='*70}")
    print(f"长度控制测试: {exp_name}")
    print(f"{'='*70}")

    config = get_experiment_config(exp_name)

    # 检查模型是否存在
    if not config.final_model_path.exists():
        print(f"模型不存在: {config.final_model_path}")
        return None

    # 检查测试数据是否存在
    if not config.test_data.exists():
        print(f"测试数据不存在: {config.test_data}")
        return None

    # 加载模型和tokenizer
    print(f"加载模型: {config.final_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.final_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.final_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"使用设备: {device}")

    # 加载测试数据
    print(f"加载测试数据: {config.test_data}")
    test_data = load_data(config.test_data)
    print(f"测试样本数: {len(test_data):,}")

    # 分析长度分布
    length_dist = analyze_length_distribution(test_data)
    print(f"\n测试集长度分布:")
    for bin_label in [b[2] for b in LENGTH_BINS]:
        if bin_label in length_dist:
            stats = length_dist[bin_label]
            print(f"  {bin_label:>10}: {stats['total']:>6} 样本 "
                  f"(PC: {stats['class_0']:>5}, NC: {stats['class_1']:>5})")

    # 按长度区间分组数据
    binned_data = defaultdict(list)
    for item in test_data:
        seq_len = len(item['sequence'])
        bin_label = get_length_bin(seq_len)
        binned_data[bin_label].append(item)

    # 对每个长度区间进行评估
    results_by_length = {}

    print(f"\n开始按长度区间评估...")

    for bin_label in [b[2] for b in LENGTH_BINS]:
        if bin_label not in binned_data or len(binned_data[bin_label]) == 0:
            continue

        bin_data = binned_data[bin_label]
        sequences = [item['sequence'] for item in bin_data]
        true_labels = [item['label_id'] for item in bin_data]

        predictions = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]

                inputs = tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_preds)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)

        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

        results_by_length[bin_label] = {
            'samples': len(bin_data),
            'class_0_samples': int(np.sum(true_labels == 0)),
            'class_1_samples': int(np.sum(true_labels == 1)),
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'class_0': {
                'precision': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
                'recall': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
                'f1': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            },
            'class_1': {
                'precision': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0,
                'recall': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,
                'f1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0,
            },
            'confusion_matrix': {
                'tn': int(cm[0][0]) if cm.shape[0] > 0 else 0,
                'fp': int(cm[0][1]) if cm.shape[1] > 1 else 0,
                'fn': int(cm[1][0]) if cm.shape[0] > 1 else 0,
                'tp': int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            }
        }

    # 计算整体结果
    all_sequences = [item['sequence'] for item in test_data]
    all_true_labels = [item['label_id'] for item in test_data]
    all_predictions = []

    print(f"\n计算整体结果...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sequences), batch_size)):
            batch_seqs = all_sequences[i:i+batch_size]

            inputs = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_predictions.extend(batch_preds)

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='macro', zero_division=0
    )

    # 汇总结果
    final_results = {
        'experiment': exp_name,
        'description': config.get_description(),
        'total_samples': len(test_data),
        'overall': {
            'accuracy': float(overall_accuracy),
            'precision_macro': float(overall_precision),
            'recall_macro': float(overall_recall),
            'f1_macro': float(overall_f1)
        },
        'by_length': results_by_length
    }

    # 保存结果
    output_file = config.output_dir / "length_control_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存: {output_file}")

    # 打印结果表格
    print(f"\n{'='*70}")
    print(f"长度控制测试结果: {exp_name}")
    print(f"{'='*70}")
    print(f"\n整体结果:")
    print(f"  准确率: {overall_accuracy:.4f}")
    print(f"  F1 (macro): {overall_f1:.4f}")

    print(f"\n按长度区间结果:")
    print(f"{'长度区间':>12} | {'样本数':>8} | {'准确率':>8} | {'F1':>8} | {'PC F1':>8} | {'NC F1':>8}")
    print("-" * 70)

    for bin_label in [b[2] for b in LENGTH_BINS]:
        if bin_label in results_by_length:
            r = results_by_length[bin_label]
            print(f"{bin_label:>12} | {r['samples']:>8} | {r['accuracy']:>8.4f} | "
                  f"{r['f1_macro']:>8.4f} | {r['class_0']['f1']:>8.4f} | {r['class_1']['f1']:>8.4f}")

    return final_results


def generate_comparison_report(all_results: dict, output_path: Path):
    """生成长度控制测试对比报告"""
    print(f"\n{'='*70}")
    print("生成长度控制测试对比报告")
    print(f"{'='*70}")

    report = []
    report.append("# V3 Unified Experiments - 长度控制测试对比报告\n")

    # 整体结果对比
    report.append("## 整体结果对比\n")
    report.append("| 实验 | 准确率 | F1 (macro) | 测试样本 |")
    report.append("|------|--------|------------|---------|")

    for exp_name in ['Exp1_concat', 'Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']:
        if exp_name in all_results:
            r = all_results[exp_name]
            report.append(
                f"| {exp_name} | {r['overall']['accuracy']:.4f} | "
                f"{r['overall']['f1_macro']:.4f} | {r['total_samples']:,} |"
            )

    report.append("")

    # 按长度区间对比
    report.append("## 按长度区间F1对比\n")

    # 表头
    header = "| 长度区间 |"
    separator = "|----------|"
    for exp_name in ['Exp1_concat', 'Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']:
        if exp_name in all_results:
            header += f" {exp_name} |"
            separator += "----------|"
    report.append(header)
    report.append(separator)

    # 数据行
    for bin_label in [b[2] for b in LENGTH_BINS]:
        row = f"| {bin_label} |"
        for exp_name in ['Exp1_concat', 'Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']:
            if exp_name in all_results:
                if bin_label in all_results[exp_name]['by_length']:
                    f1 = all_results[exp_name]['by_length'][bin_label]['f1_macro']
                    row += f" {f1:.4f} |"
                else:
                    row += " - |"
        report.append(row)

    report.append("")

    # 分析
    report.append("## 分析\n")
    report.append("### 长度敏感性分析\n")

    for exp_name in ['Exp1_concat', 'Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']:
        if exp_name not in all_results:
            continue

        r = all_results[exp_name]
        by_length = r['by_length']

        if len(by_length) < 2:
            continue

        f1_values = [by_length[b]['f1_macro'] for b in by_length if by_length[b]['samples'] >= 10]
        if len(f1_values) >= 2:
            f1_std = np.std(f1_values)
            f1_range = max(f1_values) - min(f1_values)

            report.append(f"**{exp_name}**:")
            report.append(f"- F1标准差: {f1_std:.4f}")
            report.append(f"- F1范围: {f1_range:.4f}")
            report.append(f"- 最高F1区间: {max(by_length.keys(), key=lambda x: by_length[x]['f1_macro'] if by_length[x]['samples'] >= 10 else 0)}")
            report.append(f"- 最低F1区间: {min(by_length.keys(), key=lambda x: by_length[x]['f1_macro'] if by_length[x]['samples'] >= 10 else 1)}")
            report.append("")

    report_text = '\n'.join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"报告保存: {output_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='V3 Unified Length Control Test')
    parser.add_argument('--exp', type=str, default='all',
                       choices=list(EXPERIMENTS.keys()) + ['all'],
                       help='实验名称或"all"测试所有实验')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')

    args = parser.parse_args()

    if args.exp == 'all':
        all_results = {}
        for exp_name in EXPERIMENTS.keys():
            result = evaluate_by_length(exp_name, args.batch_size)
            if result:
                all_results[exp_name] = result

        # 生成对比报告
        if all_results:
            report_path = MODEL_OUTPUT_ROOT / "length_control_comparison.md"
            generate_comparison_report(all_results, report_path)

            # 保存汇总结果
            summary_path = MODEL_OUTPUT_ROOT / "all_length_control_results.json"
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n汇总结果: {summary_path}")
    else:
        evaluate_by_length(args.exp, args.batch_size)


if __name__ == "__main__":
    main()
