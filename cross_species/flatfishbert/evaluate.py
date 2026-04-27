#!/usr/bin/env python3
"""
泛化评估脚本
使用Exp4模型在泛化测试集上进行推理和评估
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SPECIES_CONFIGS, DATA_OUTPUT_DIR, RESULTS_OUTPUT_DIR,
    EXP4_MODEL_PATH, ID2LABEL
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model():
    """加载Exp4模型"""
    logger.info(f"加载模型: {EXP4_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(EXP4_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(EXP4_MODEL_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"✓ 使用设备: {device}")
    return tokenizer, model, device


def predict(tokenizer, model, device, sequences, batch_size=128):
    """批量预测"""
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="预测"):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
    return np.array(predictions)


def compute_metrics(true_labels, predictions):
    """计算评估指标"""
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0)
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'class_0': {'name': 'Protein-coding', 'f1': float(f1_cls[0]),
                   'precision': float(p_cls[0]), 'recall': float(r_cls[0]),
                   'support': int(support[0])},
        'class_1': {'name': 'Non-coding', 'f1': float(f1_cls[1]),
                   'precision': float(p_cls[1]), 'recall': float(r_cls[1]),
                   'support': int(support[1])},
        'confusion_matrix': {'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
                            'fn': int(cm[1][0]), 'tp': int(cm[1][1])}
    }


def compute_length_metrics(samples, predictions):
    """按长度区间计算指标"""
    bins = [(0, 100), (100, 200), (200, 300), (300, 500),
            (500, 1000), (1000, 2000), (2000, float('inf'))]
    results = {}

    for low, high in bins:
        mask = [(low <= s['sequence_length'] < high) for s in samples]
        if sum(mask) == 0:
            continue
        true_l = [samples[i]['label_id'] for i in range(len(samples)) if mask[i]]
        pred_l = [predictions[i] for i in range(len(samples)) if mask[i]]
        _, _, f1, _ = precision_recall_fscore_support(
            true_l, pred_l, average='macro', zero_division=0)
        bin_name = f"{low}-{int(high)}" if high != float('inf') else f"{low}+"
        results[bin_name] = {'count': sum(mask), 'f1_macro': float(f1)}

    return results


def evaluate_species(species_key, tokenizer, model, device, batch_size=128):
    """评估单个物种"""
    test_file = DATA_OUTPUT_DIR / f"{species_key}_test.json"
    if not test_file.exists():
        logger.error(f"测试集不存在: {test_file}")
        return None

    logger.info(f"\n{'='*50}")
    logger.info(f"评估: {species_key}")
    logger.info(f"{'='*50}")

    with open(test_file, 'r') as f:
        samples = json.load(f)
    logger.info(f"加载测试集: {len(samples):,} 样本")

    sequences = [s['sequence'] for s in samples]
    true_labels = np.array([s['label_id'] for s in samples])

    predictions = predict(tokenizer, model, device, sequences, batch_size)
    metrics = compute_metrics(true_labels, predictions)
    length_metrics = compute_length_metrics(samples, predictions)

    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"F1-macro: {metrics['f1_macro']:.4f}")
    logger.info(f"PC F1: {metrics['class_0']['f1']:.4f}")
    logger.info(f"NC F1: {metrics['class_1']['f1']:.4f}")

    return {'species': species_key, 'samples': len(samples),
            'metrics': metrics, 'length_metrics': length_metrics}


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("跨物种泛化评估 - 开始")
    logger.info("=" * 60)

    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model, device = load_model()

    all_results = []
    for species_key in SPECIES_CONFIGS.keys():
        result = evaluate_species(species_key, tokenizer, model, device)
        if result:
            all_results.append(result)
            # 保存单物种结果
            out_file = RESULTS_OUTPUT_DIR / f"{species_key}_results.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)

    # 保存汇总结果
    summary_file = RESULTS_OUTPUT_DIR / "all_results.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ 汇总结果: {summary_file}")

    # 打印汇总
    logger.info(f"\n{'='*60}")
    logger.info("汇总")
    logger.info(f"{'='*60}")
    for r in all_results:
        m = r['metrics']
        logger.info(f"{r['species']}: Acc={m['accuracy']:.4f}, F1={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
