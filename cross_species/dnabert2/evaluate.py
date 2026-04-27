#!/usr/bin/env python3
"""
DNABERT2 跨物种泛化评估脚本
使用DNABERT2 Exp4模型在泛化测试集上进行推理和评估

特殊处理:
1. Triton禁用（必须在所有import之前）
2. 四层防护机制（防止GPU显存累积）
3. 双GPU并行推理
"""

# ============================================================
# CRITICAL: Triton禁用必须在所有import之前
# ============================================================
import sys
sys.modules['triton'] = None
sys.modules['triton.language'] = None

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
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SPECIES_CONFIGS, RESULTS_OUTPUT_DIR, LOGS_DIR,
    DNABERT2_MODEL_PATH, ID2LABEL, INFERENCE_BATCH_SIZE,
    MAX_SEQ_LENGTH, LENGTH_BINS, LENGTH_BIN_NAMES,
    FLATFISH_BASELINE, DNABERT2_BASELINE
)

# ============================================================
# 日志配置
# ============================================================

LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 模型加载
# ============================================================

def load_model(device_id=0):
    """加载DNABERT2模型到指定GPU"""
    logger.info(f"加载DNABERT2模型: {DNABERT2_MODEL_PATH}")
    logger.info(f"  目标设备: cuda:{device_id}")

    # DNABERT2需要trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        str(DNABERT2_MODEL_PATH),
        trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(DNABERT2_MODEL_PATH),
        trust_remote_code=True
    )

    # 四层防护机制（防止GPU显存累积）
    model.config.return_dict = True
    model.config.output_hidden_states = False
    model.config.output_attentions = False

    device = torch.device(f"cuda:{device_id}")
    model.to(device)
    model.eval()

    logger.info(f"✓ 模型加载完成")
    logger.info(f"  return_dict: {model.config.return_dict}")
    logger.info(f"  output_hidden_states: {model.config.output_hidden_states}")

    return tokenizer, model, device


# ============================================================
# 推理函数
# ============================================================

def predict(tokenizer, model, device, sequences, batch_size=64):
    """批量预测（单GPU）"""
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"推理 (GPU {device.index})"):
            batch = sequences[i:i+batch_size]

            # Tokenization
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits

            # 预测并立即移到CPU（防止显存累积）
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            predictions.extend(preds)

            # 清理显存
            del inputs, outputs, logits, preds
            torch.cuda.empty_cache()

    return np.array(predictions)


# ============================================================
# 评估指标计算
# ============================================================

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
        'class_0': {
            'name': 'Protein-coding',
            'f1': float(f1_cls[0]),
            'precision': float(p_cls[0]),
            'recall': float(r_cls[0]),
            'support': int(support[0])
        },
        'class_1': {
            'name': 'Non-coding',
            'f1': float(f1_cls[1]),
            'precision': float(p_cls[1]),
            'recall': float(r_cls[1]),
            'support': int(support[1])
        },
        'confusion_matrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        }
    }


def compute_length_metrics(samples, predictions):
    """按长度区间计算指标"""
    results = {}

    for i, (low, high) in enumerate(LENGTH_BINS):
        mask = [(low <= s['sequence_length'] < high) for s in samples]
        if sum(mask) == 0:
            continue

        true_l = [samples[j]['label_id'] for j in range(len(samples)) if mask[j]]
        pred_l = [predictions[j] for j in range(len(samples)) if mask[j]]

        _, _, f1, _ = precision_recall_fscore_support(
            true_l, pred_l, average='macro', zero_division=0)

        bin_name = LENGTH_BIN_NAMES[i]
        results[bin_name] = {
            'count': sum(mask),
            'f1_macro': float(f1)
        }

    return results


# ============================================================
# 单物种评估
# ============================================================

def evaluate_species(species_key, tokenizer, model, device, batch_size=64):
    """评估单个物种"""
    config = SPECIES_CONFIGS[species_key]
    test_file = config['test_file']

    if not test_file.exists():
        logger.error(f"测试集不存在: {test_file}")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"评估: {species_key}")
    logger.info(f"  中文名: {config['common_name']}")
    logger.info(f"  学名: {config['scientific_name']}")
    logger.info(f"  科: {config['family']}")
    logger.info(f"  亲缘关系: {config['relationship']}")
    logger.info(f"{'='*60}")

    # 加载测试集
    with open(test_file, 'r') as f:
        samples = json.load(f)
    logger.info(f"加载测试集: {len(samples):,} 样本")

    # 提取序列和标签
    sequences = [s['sequence'] for s in samples]
    true_labels = np.array([s['label_id'] for s in samples])

    # 推理
    predictions = predict(tokenizer, model, device, sequences, batch_size)

    # 计算指标
    metrics = compute_metrics(true_labels, predictions)
    length_metrics = compute_length_metrics(samples, predictions)

    # 打印结果
    logger.info(f"\n结果:")
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  PC F1: {metrics['class_0']['f1']:.4f}")
    logger.info(f"  NC F1: {metrics['class_1']['f1']:.4f}")

    # 与FlatfishBert对比
    flatfish_result = FLATFISH_BASELINE['generalization'][species_key]
    diff = metrics['f1_macro'] - flatfish_result['f1_macro']
    logger.info(f"\n与FlatfishBert对比:")
    logger.info(f"  FlatfishBert: {flatfish_result['f1_macro']:.4f}")
    logger.info(f"  DNABERT2: {metrics['f1_macro']:.4f}")
    logger.info(f"  差异: {diff:+.4f} ({diff*100:+.2f}%)")

    # 转换PosixPath为字符串以便JSON序列化
    species_info_serializable = {
        'name': config['name'],
        'common_name': config['common_name'],
        'scientific_name': config['scientific_name'],
        'family': config['family'],
        'relationship': config['relationship'],
        'test_file': str(config['test_file'])
    }

    return {
        'species': species_key,
        'species_info': species_info_serializable,
        'samples': len(samples),
        'metrics': metrics,
        'length_metrics': length_metrics,
        'comparison': {
            'flatfishbert': flatfish_result['f1_macro'],
            'dnabert2': metrics['f1_macro'],
            'difference': diff
        }
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("DNABERT2 跨物种泛化评估 - 开始")
    logger.info("=" * 70)
    logger.info(f"模型: {DNABERT2_MODEL_PATH}")
    logger.info(f"批大小: {INFERENCE_BATCH_SIZE}")
    logger.info(f"最大序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"输出目录: {RESULTS_OUTPUT_DIR}")

    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查GPU可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用！")
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"\n可用GPU数量: {num_gpus}")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 加载模型到GPU 0（使用单GPU推理，避免复杂性）
    tokenizer, model, device = load_model(device_id=0)

    # 评估所有物种
    all_results = []
    for species_key in SPECIES_CONFIGS.keys():
        result = evaluate_species(
            species_key,
            tokenizer,
            model,
            device,
            batch_size=INFERENCE_BATCH_SIZE
        )

        if result:
            all_results.append(result)

            # 保存单物种结果
            out_file = RESULTS_OUTPUT_DIR / f"{species_key}_results.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✓ 结果已保存: {out_file}")

    # 计算平均性能
    avg_f1 = np.mean([r['metrics']['f1_macro'] for r in all_results])
    avg_acc = np.mean([r['metrics']['accuracy'] for r in all_results])
    total_samples = sum([r['samples'] for r in all_results])

    # 保存汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': str(DNABERT2_MODEL_PATH),
        'total_samples': total_samples,
        'average_metrics': {
            'f1_macro': float(avg_f1),
            'accuracy': float(avg_acc),
        },
        'species_results': all_results,
        'comparison_with_flatfishbert': {
            'dnabert2_original_test': DNABERT2_BASELINE['original_test']['f1_macro'],
            'dnabert2_generalization': float(avg_f1),
            'dnabert2_drop': float(avg_f1 - DNABERT2_BASELINE['original_test']['f1_macro']),
            'flatfishbert_original_test': FLATFISH_BASELINE['original_test']['f1_macro'],
            'flatfishbert_generalization': FLATFISH_BASELINE['generalization']['average']['f1_macro'],
            'flatfishbert_drop': FLATFISH_BASELINE['generalization']['average']['f1_macro'] - FLATFISH_BASELINE['original_test']['f1_macro'],
            'advantage': float(avg_f1 - FLATFISH_BASELINE['generalization']['average']['f1_macro']),
        }
    }

    summary_file = RESULTS_OUTPUT_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印最终汇总
    logger.info(f"\n{'='*70}")
    logger.info("最终汇总")
    logger.info(f"{'='*70}")
    logger.info(f"总样本数: {total_samples:,}")
    logger.info(f"平均F1-macro: {avg_f1:.4f}")
    logger.info(f"平均准确率: {avg_acc:.4f}")

    logger.info(f"\n各物种结果:")
    for r in all_results:
        m = r['metrics']
        logger.info(f"  {r['species']}: F1={m['f1_macro']:.4f}, Acc={m['accuracy']:.4f}")

    logger.info(f"\n与FlatfishBert对比:")
    comp = summary['comparison_with_flatfishbert']
    logger.info(f"  DNABERT2:")
    logger.info(f"    原始测试集: {comp['dnabert2_original_test']:.4f}")
    logger.info(f"    泛化测试集: {comp['dnabert2_generalization']:.4f}")
    logger.info(f"    下降: {comp['dnabert2_drop']:+.4f}")
    logger.info(f"  FlatfishBert:")
    logger.info(f"    原始测试集: {comp['flatfishbert_original_test']:.4f}")
    logger.info(f"    泛化测试集: {comp['flatfishbert_generalization']:.4f}")
    logger.info(f"    下降: {comp['flatfishbert_drop']:+.4f}")
    logger.info(f"  DNABERT2优势: {comp['advantage']:+.4f} ({comp['advantage']*100:+.2f}%)")

    logger.info(f"\n✓ 汇总结果已保存: {summary_file}")
    logger.info(f"✓ 日志文件: {log_file}")
    logger.info(f"\n{'='*70}")
    logger.info("评估完成！")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
