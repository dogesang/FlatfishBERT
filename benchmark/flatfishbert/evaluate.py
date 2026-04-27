#!/usr/bin/env python3
"""
V3 Unified Experiments - 统一评估脚本
在测试集上评估所有实验，生成对比报告
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import logging

from config import get_experiment_config, EXPERIMENTS, ID2LABEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def evaluate_on_test(exp_name: str, batch_size: int = 128):
    """在测试集上评估单个实验"""
    print(f"\n{'='*60}")
    print(f"评估实验: {exp_name}")
    print(f"{'='*60}")

    config = get_experiment_config(exp_name)

    # 检查模型是否存在
    if not config.final_model_path.exists():
        print(f"❌ 模型不存在: {config.final_model_path}")
        return None

    # 检查测试数据是否存在
    if not config.test_data.exists():
        print(f"❌ 测试数据不存在: {config.test_data}")
        return None

    # 加载模型和tokenizer
    print(f"加载模型: {config.final_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.final_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.final_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ 使用设备: {device}")

    # 加载测试数据
    print(f"加载测试数据: {config.test_data}")
    test_data = load_data(config.test_data)
    print(f"✓ 测试样本数: {len(test_data):,}")

    # 预测
    sequences = [item['sequence'] for item in test_data]
    true_labels = [item['label_id'] for item in test_data]

    predictions = []
    print(f"开始预测 (batch_size={batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
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

    cm = confusion_matrix(true_labels, predictions)

    results = {
        'experiment': exp_name,
        'description': config.get_description(),
        'test_samples': len(test_data),
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'class_0': {
            'name': 'Protein-coding',
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0]),
            'f1': float(f1_per_class[0]),
            'support': int(support[0])
        },
        'class_1': {
            'name': 'Non-coding',
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1]),
            'f1': float(f1_per_class[1]),
            'support': int(support[1])
        },
        'confusion_matrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        }
    }

    # 保存结果
    output_file = config.output_dir / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 结果保存: {output_file}")

    # 打印结果
    print(f"\n测试集结果:")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Protein-coding F1: {f1_per_class[0]:.4f}")
    print(f"  Non-coding F1: {f1_per_class[1]:.4f}")

    print(f"\n混淆矩阵:")
    print(f"              预测PC    预测NC")
    print(f"  实际PC:    {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  实际NC:    {cm[1][0]:>6}    {cm[1][1]:>6}")

    return results


def generate_comparison_report(all_results: dict, output_path: Path):
    """生成对比报告"""
    print("\n" + "=" * 70)
    print("生成对比报告")
    print("=" * 70)

    report = []
    report.append("# V3 Unified Experiments - 测试集结果对比报告\n")
    report.append("## 实验设计\n")
    report.append("| 实验 | CDS处理 | 去重策略 | 划分策略 |")
    report.append("|------|---------|---------|---------|")
    report.append("| Exp1_concat | 拼接 | 位置级 | 基因级 |")
    report.append("| Exp2_unconcat | 不拼接 | 位置级 | 随机分层 |")
    report.append("| Exp3_seq_dedup | 不拼接 | 单物种序列级 | 基因级 |")
    report.append("| Exp4_cross_dedup | 不拼接 | 跨物种序列级 | 基因级 |")
    report.append("")

    report.append("## 测试集结果\n")
    report.append("| 实验 | 准确率 | F1 (macro) | PC F1 | NC F1 | 测试样本 |")
    report.append("|------|--------|------------|-------|-------|---------|")

    for exp_name in ['Exp1_concat', 'Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']:
        if exp_name in all_results:
            r = all_results[exp_name]
            report.append(
                f"| {exp_name} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
                f"{r['class_0']['f1']:.4f} | {r['class_1']['f1']:.4f} | {r['test_samples']:,} |"
            )
        else:
            report.append(f"| {exp_name} | - | - | - | - | - |")

    report.append("")
    report.append("## 关键对比\n")

    # Exp1 vs Exp2: CDS拼接影响
    if 'Exp1_concat' in all_results and 'Exp2_unconcat' in all_results:
        diff = all_results['Exp1_concat']['f1_macro'] - all_results['Exp2_unconcat']['f1_macro']
        report.append("### CDS拼接 vs 不拼接 (Exp1 vs Exp2)\n")
        report.append(f"- Exp1 (拼接): F1 = {all_results['Exp1_concat']['f1_macro']:.4f}")
        report.append(f"- Exp2 (不拼接): F1 = {all_results['Exp2_unconcat']['f1_macro']:.4f}")
        report.append(f"- 差异: {diff:+.4f}")
        report.append("")

    # Exp2 vs Exp3 vs Exp4: 去重策略影响
    if all(exp in all_results for exp in ['Exp2_unconcat', 'Exp3_seq_dedup', 'Exp4_cross_dedup']):
        report.append("### 去重策略对比 (Exp2 vs Exp3 vs Exp4)\n")
        report.append(f"- Exp2 (位置级): F1 = {all_results['Exp2_unconcat']['f1_macro']:.4f}")
        report.append(f"- Exp3 (单物种序列级): F1 = {all_results['Exp3_seq_dedup']['f1_macro']:.4f}")
        report.append(f"- Exp4 (跨物种序列级): F1 = {all_results['Exp4_cross_dedup']['f1_macro']:.4f}")

        diff_2_3 = all_results['Exp2_unconcat']['f1_macro'] - all_results['Exp3_seq_dedup']['f1_macro']
        diff_3_4 = all_results['Exp3_seq_dedup']['f1_macro'] - all_results['Exp4_cross_dedup']['f1_macro']
        report.append(f"- Exp2 vs Exp3 差异: {diff_2_3:+.4f}")
        report.append(f"- Exp3 vs Exp4 差异: {diff_3_4:+.4f}")
        report.append("")

    report_text = '\n'.join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"✓ 报告保存: {output_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='V3 Unified Evaluation')
    parser.add_argument('--exp', type=str, default='all',
                       choices=list(EXPERIMENTS.keys()) + ['all'],
                       help='实验名称或"all"评估所有实验')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')

    args = parser.parse_args()

    if args.exp == 'all':
        all_results = {}
        for exp_name in EXPERIMENTS.keys():
            result = evaluate_on_test(exp_name, args.batch_size)
            if result:
                all_results[exp_name] = result

        # 生成对比报告
        if all_results:
            from config import MODEL_OUTPUT_ROOT
            report_path = MODEL_OUTPUT_ROOT / "comparison_report.md"
            generate_comparison_report(all_results, report_path)

            # 保存汇总结果
            summary_path = MODEL_OUTPUT_ROOT / "all_test_results.json"
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✓ 汇总结果: {summary_path}")
    else:
        evaluate_on_test(args.exp, args.batch_size)


if __name__ == "__main__":
    main()
