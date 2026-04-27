#!/usr/bin/env python3
"""
V3 Exp4 长度基线脚本。
仅保留生成 Figure 3 Panel A 所需的最小结果链路。
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP4_DATA_DIR = PROJECT_ROOT / "data/experiments/v3_unified/Exp4_cross_dedup"
OUTPUT_DIR = Path(__file__).resolve().parent

LABEL_NAMES = {0: "Protein-coding", 1: "Non-coding"}


def load_data(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


def analyze_length_distribution(data):
    lengths_by_class = defaultdict(list)
    for item in data:
        label_id = item["label_id"]
        seq_len = item.get("sequence_length", len(item["sequence"]))
        lengths_by_class[label_id].append(seq_len)

    stats = {}
    for label_id, lengths in lengths_by_class.items():
        stats[label_id] = {
            "count": int(len(lengths)),
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "median": float(np.median(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "q25": float(np.percentile(lengths, 25)),
            "q75": float(np.percentile(lengths, 75)),
        }
    return stats


def find_optimal_threshold(train_data):
    lengths = np.array([item.get("sequence_length", len(item["sequence"])) for item in train_data])
    labels = np.array([item["label_id"] for item in train_data])

    all_lengths = sorted(set(lengths))
    if len(all_lengths) > 1000:
        thresholds = np.unique(np.percentile(lengths, np.arange(1, 100, 0.5)))
    else:
        thresholds = all_lengths

    best_threshold = None
    best_accuracy = -1.0
    best_direction = None

    for threshold in thresholds:
        pred_less_is_pc = (lengths >= threshold).astype(int)
        acc_less_is_pc = accuracy_score(labels, pred_less_is_pc)

        pred_less_is_nc = (lengths < threshold).astype(int)
        acc_less_is_nc = accuracy_score(labels, pred_less_is_nc)

        if acc_less_is_pc > best_accuracy:
            best_accuracy = acc_less_is_pc
            best_threshold = threshold
            best_direction = "less_is_pc"

        if acc_less_is_nc > best_accuracy:
            best_accuracy = acc_less_is_nc
            best_threshold = threshold
            best_direction = "less_is_nc"

    return float(best_threshold), best_direction, float(best_accuracy)


def predict_by_length(lengths, threshold, direction):
    lengths = np.array(lengths)
    if direction == "less_is_pc":
        return (lengths >= threshold).astype(int)
    return (lengths < threshold).astype(int)


def evaluate_baseline(data, threshold, direction, dataset_name):
    lengths = np.array([item.get("sequence_length", len(item["sequence"])) for item in data])
    true_labels = np.array([item["label_id"] for item in data])
    predictions = predict_by_length(lengths, threshold, direction)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "direction": direction,
        "total_samples": len(true_labels),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": {
            LABEL_NAMES[0]: {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            LABEL_NAMES[1]: {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
    }


def main():
    train_data = load_data(EXP4_DATA_DIR / "train.json")
    test_data = load_data(EXP4_DATA_DIR / "test.json")

    train_stats = analyze_length_distribution(train_data)
    test_stats = analyze_length_distribution(test_data)
    threshold, direction, train_acc = find_optimal_threshold(train_data)
    train_results = evaluate_baseline(train_data, threshold, direction, "train")
    test_results = evaluate_baseline(test_data, threshold, direction, "test")

    summary = {
        "experiment": "V3_Exp4_Length_Baseline",
        "description": "仅使用序列长度进行分类的基线实验",
        "optimal_threshold": threshold,
        "optimal_direction": direction,
        "direction_explanation": (
            "less_is_pc: length < threshold -> Protein-coding"
            if direction == "less_is_pc"
            else "less_is_nc: length < threshold -> Non-coding"
        ),
        "train_threshold_accuracy": train_acc,
        "train_stats": {
            LABEL_NAMES[0]: train_stats[0],
            LABEL_NAMES[1]: train_stats[1],
        },
        "test_stats": {
            LABEL_NAMES[0]: test_stats[0],
            LABEL_NAMES[1]: test_stats[1],
        },
        "train_results": train_results,
        "test_results": test_results,
        "model_comparison": {
            "length_baseline_accuracy": test_results["accuracy"],
            "length_baseline_f1": test_results["f1_macro"],
            "bert_model_accuracy": 0.9757,
            "bert_model_f1": 0.9757,
            "improvement_accuracy": 0.9757 - test_results["accuracy"],
            "improvement_f1": 0.9757 - test_results["f1_macro"],
        },
    }

    output_file = OUTPUT_DIR / "exp4_length_baseline_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
