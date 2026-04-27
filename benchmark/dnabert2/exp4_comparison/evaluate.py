#!/usr/bin/env python3
"""
DNABERT2 横向对比实验 - 评估脚本
包含测试集评估和长度区间分析
"""

import sys
import os

# ============================================================
# 禁用triton/flash attention（必须在所有其他import之前）
# DNABERT2会自动fallback到PyTorch标准attention实现
# ============================================================
sys.modules['triton'] = None
sys.modules['triton.language'] = None

import json
import argparse
from pathlib import Path
from collections import defaultdict

# 添加DNABERT2模型目录到路径
DNABERT2_PATH = "${DNABERT2_MODEL_PATH:-./external_models/dnabert2}"
sys.path.insert(0, DNABERT2_PATH)

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import logging

from config import get_config, LABEL2ID, ID2LABEL, LENGTH_BINS, LENGTH_BIN_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """加载JSON数据"""
    logger.info(f"加载数据: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"✓ 加载完成: {len(data):,} 条")
    return data


def get_length_bin(length):
    """获取序列长度所属的区间"""
    for i, (low, high) in enumerate(LENGTH_BINS):
        if low <= length < high:
            return i
    return len(LENGTH_BINS) - 1


def evaluate_model(model, tokenizer, data, batch_size=128, max_length=512, device='cuda'):
    """评估模型"""
    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    all_lengths = []

    # 分批处理
    for i in tqdm(range(0, len(data), batch_size), desc="评估中"):
        batch = data[i:i+batch_size]

        sequences = [item['sequence'] for item in batch]
        labels = [item['label_id'] for item in batch]
        lengths = [len(item['sequence']) for item in batch]

        # Tokenize
        encodings = tokenizer(
            sequences,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # 移动到设备
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # 推理
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_lengths.extend(lengths)

    return np.array(all_predictions), np.array(all_labels), np.array(all_lengths)


def compute_metrics_with_details(predictions, labels):
    """计算详细的评估指标"""
    accuracy = accuracy_score(labels, predictions)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)

    cm = confusion_matrix(labels, predictions)

    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'class_0': {
            'name': ID2LABEL[0],
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0]),
            'f1': float(f1_per_class[0]),
            'support': int(support[0])
        },
        'class_1': {
            'name': ID2LABEL[1],
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1]),
            'f1': float(f1_per_class[1]),
            'support': int(support[1])
        },
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    }

    return results


def analyze_by_length(predictions, labels, lengths):
    """按长度区间分析性能"""
    length_results = {}

    for bin_idx, bin_name in enumerate(LENGTH_BIN_NAMES):
        # 找到属于该区间的样本
        mask = np.array([get_length_bin(l) == bin_idx for l in lengths])

        if mask.sum() == 0:
            continue

        bin_preds = predictions[mask]
        bin_labels = labels[mask]

        # 计算该区间的指标
        accuracy = accuracy_score(bin_labels, bin_preds)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            bin_labels, bin_preds, average='macro', zero_division=0
        )
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            bin_labels, bin_preds, average=None, zero_division=0
        )

        length_results[bin_name] = {
            'sample_count': int(mask.sum()),
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_class_0': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            'f1_class_1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0,
        }

    return length_results


def compare_with_flatfishbert(dnabert2_results, dnabert2_length_results):
    """与FlatfishBert Exp4结果对比"""

    # FlatfishBert Exp4 结果（从all_test_results.json）
    flatfishbert_results = {
        'accuracy': 0.9757245606918178,
        'f1_macro': 0.9756976383372525,
        'f1_class_0': 0.9748887645538594,
        'f1_class_1': 0.9765065121206457,
    }

    # FlatfishBert Exp4 长度区间结果
    flatfishbert_length = {
        '0-100': 0.9620,
        '100-200': 0.9660,
        '200-300': 0.9733,
        '300-500': 0.9751,
        '500-1000': 0.9762,
        '1000-2000': 0.9788,
        '2000+': 0.9724,
    }

    comparison = {
        'overall': {
            'metric': ['Accuracy', 'F1-macro', 'F1 (Protein-coding)', 'F1 (Non-coding)'],
            'FlatfishBert_Exp4': [
                flatfishbert_results['accuracy'],
                flatfishbert_results['f1_macro'],
                flatfishbert_results['f1_class_0'],
                flatfishbert_results['f1_class_1'],
            ],
            'DNABERT2': [
                dnabert2_results['accuracy'],
                dnabert2_results['f1_macro'],
                dnabert2_results['class_0']['f1'],
                dnabert2_results['class_1']['f1'],
            ],
            'difference': [
                dnabert2_results['accuracy'] - flatfishbert_results['accuracy'],
                dnabert2_results['f1_macro'] - flatfishbert_results['f1_macro'],
                dnabert2_results['class_0']['f1'] - flatfishbert_results['f1_class_0'],
                dnabert2_results['class_1']['f1'] - flatfishbert_results['f1_class_1'],
            ]
        },
        'by_length': {}
    }

    for bin_name in LENGTH_BIN_NAMES:
        if bin_name in dnabert2_length_results:
            fb_f1 = flatfishbert_length.get(bin_name, 0)
            db_f1 = dnabert2_length_results[bin_name]['f1_macro']
            comparison['by_length'][bin_name] = {
                'FlatfishBert_Exp4': fb_f1,
                'DNABERT2': db_f1,
                'difference': db_f1 - fb_f1
            }

    return comparison


def print_results(results, length_results, comparison):
    """打印结果"""
    print("\n" + "=" * 70)
    print("DNABERT2 测试集评估结果")
    print("=" * 70)

    print(f"\n整体性能:")
    print(f"  准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  F1-macro: {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    print(f"  F1 (Protein-coding): {results['class_0']['f1']:.4f}")
    print(f"  F1 (Non-coding): {results['class_1']['f1']:.4f}")

    print(f"\n混淆矩阵:")
    cm = results['confusion_matrix']
    print(f"  TN={cm['tn']:,}, FP={cm['fp']:,}")
    print(f"  FN={cm['fn']:,}, TP={cm['tp']:,}")

    print("\n" + "-" * 70)
    print("长度区间分析")
    print("-" * 70)
    print(f"{'长度区间':<12} {'样本数':>10} {'准确率':>10} {'F1-macro':>10}")
    print("-" * 42)
    for bin_name in LENGTH_BIN_NAMES:
        if bin_name in length_results:
            r = length_results[bin_name]
            print(f"{bin_name:<12} {r['sample_count']:>10,} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")

    print("\n" + "=" * 70)
    print("与 FlatfishBert Exp4 对比")
    print("=" * 70)

    print(f"\n整体性能对比:")
    print(f"{'指标':<25} {'FlatfishBert':>12} {'DNABERT2':>12} {'差异':>12}")
    print("-" * 61)
    for i, metric in enumerate(comparison['overall']['metric']):
        fb = comparison['overall']['FlatfishBert_Exp4'][i]
        db = comparison['overall']['DNABERT2'][i]
        diff = comparison['overall']['difference'][i]
        sign = '+' if diff >= 0 else ''
        print(f"{metric:<25} {fb:>11.4f} {db:>11.4f} {sign}{diff:>11.4f}")

    print(f"\n长度区间F1对比:")
    print(f"{'长度区间':<12} {'FlatfishBert':>12} {'DNABERT2':>12} {'差异':>12}")
    print("-" * 48)
    for bin_name in LENGTH_BIN_NAMES:
        if bin_name in comparison['by_length']:
            c = comparison['by_length'][bin_name]
            sign = '+' if c['difference'] >= 0 else ''
            print(f"{bin_name:<12} {c['FlatfishBert_Exp4']:>11.4f} {c['DNABERT2']:>11.4f} {sign}{c['difference']:>11.4f}")


def evaluate(model_path=None, output_dir=None, batch_size=None):
    """主评估函数"""
    print("=" * 70)
    print("DNABERT2 横向对比实验 - 测试集评估")
    print("=" * 70)

    config = get_config()

    # 如果提供了命令行参数，使用命令行参数；否则使用默认配置
    final_model_path = Path(model_path) if model_path is not None else config.final_model_path
    result_output_dir = Path(output_dir) if output_dir is not None else config.output_dir
    eval_batch_size = batch_size if batch_size is not None else config.training.per_device_eval_batch_size

    # 检查模型是否存在
    if not final_model_path.exists():
        print(f"\n❌ 模型不存在: {final_model_path}")
        print("请先运行训练脚本: python train.py")
        return

    # 加载tokenizer
    logger.info(f"加载Tokenizer: {final_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(final_model_path),
        trust_remote_code=True
    )

    # 加载模型
    logger.info(f"加载模型: {final_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(final_model_path),
        trust_remote_code=True
    )
    logger.info(f"✓ 模型加载完成")

    # 加载测试数据
    test_data = load_data(config.test_data)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    # 评估
    predictions, labels, lengths = evaluate_model(
        model, tokenizer, test_data,
        batch_size=eval_batch_size,
        max_length=config.training.max_seq_length,
        device=device
    )

    # 计算指标
    results = compute_metrics_with_details(predictions, labels)
    results['test_samples'] = len(test_data)
    results['experiment'] = config.exp_name
    results['model_path'] = str(final_model_path)

    # 长度区间分析
    length_results = analyze_by_length(predictions, labels, lengths)

    # 与FlatfishBert对比
    comparison = compare_with_flatfishbert(results, length_results)

    # 打印结果
    print_results(results, length_results, comparison)

    # 确保输出目录存在
    result_output_dir.mkdir(parents=True, exist_ok=True)

    # 保存结果
    with open(result_output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 测试结果已保存: {result_output_dir / 'test_results.json'}")

    with open(result_output_dir / "length_analysis.json", 'w') as f:
        json.dump(length_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 长度分析已保存: {result_output_dir / 'length_analysis.json'}")

    with open(result_output_dir / "comparison_with_flatfishbert.json", 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 对比结果已保存: {result_output_dir / 'comparison_with_flatfishbert.json'}")

    print("\n" + "=" * 70)
    print("✅ 评估完成!")
    print("=" * 70)

    return results, length_results, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNABERT2 模型测试集评估')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（final_model目录）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='评估batch size')
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
