#!/usr/bin/env python3
"""
V3 Unified Experiments - 统一训练脚本
所有实验使用相同的训练逻辑
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from datasets import Dataset
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境

from config import (
    get_experiment_config, TOKENIZER_PATH, PRETRAINED_MODEL_PATH,
    LABEL2ID, ID2LABEL, EXPERIMENTS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_curves(log_history, output_dir, exp_name):
    """绘制训练曲线图并保存

    Args:
        log_history: trainer.state.log_history
        output_dir: 输出目录
        exp_name: 实验名称
    """
    # 提取训练loss
    train_steps = []
    train_losses = []
    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(entry.get('step', 0))
            train_losses.append(entry['loss'])

    # 提取评估指标
    eval_steps = []
    eval_losses = []
    eval_f1s = []
    eval_accs = []
    for entry in log_history:
        if 'eval_loss' in entry:
            eval_steps.append(entry.get('step', 0))
            eval_losses.append(entry['eval_loss'])
            eval_f1s.append(entry.get('eval_f1_macro', 0))
            eval_accs.append(entry.get('eval_accuracy', 0))

    if not train_steps and not eval_steps:
        logger.warning("没有找到训练历史数据，跳过绘图")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{exp_name} Training Curves', fontsize=14, fontweight='bold')

    # 1. Training Loss
    ax1 = axes[0, 0]
    if train_steps:
        ax1.plot(train_steps, train_losses, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

    # 2. Eval Loss
    ax2 = axes[0, 1]
    if eval_steps:
        ax2.plot(eval_steps, eval_losses, 'r-o', markersize=3)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Eval Loss')
        ax2.set_title('Evaluation Loss')
        ax2.grid(True, alpha=0.3)

    # 3. F1-macro
    ax3 = axes[1, 0]
    if eval_steps:
        ax3.plot(eval_steps, [f*100 for f in eval_f1s], 'g-o', markersize=3)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('F1-macro (%)')
        ax3.set_title('F1-macro Score')
        ax3.grid(True, alpha=0.3)
        # 标注最佳点
        if eval_f1s:
            best_idx = np.argmax(eval_f1s)
            best_step = eval_steps[best_idx]
            best_f1 = eval_f1s[best_idx] * 100
            ax3.axvline(x=best_step, color='red', linestyle='--', alpha=0.5)
            ax3.annotate(f'Best: {best_f1:.2f}%\n@ step {best_step}',
                        xy=(best_step, best_f1),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, color='red')

    # 4. Accuracy
    ax4 = axes[1, 1]
    if eval_steps:
        ax4.plot(eval_steps, [a*100 for a in eval_accs], 'm-o', markersize=3)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / f'{exp_name}_training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 训练曲线图已保存: {output_path}")

    # 同时保存训练历史数据为JSON
    history_path = Path(output_dir) / f'{exp_name}_training_history.json'
    history_data = {
        'train': [{'step': s, 'loss': l} for s, l in zip(train_steps, train_losses)],
        'eval': [{'step': s, 'loss': l, 'f1_macro': f, 'accuracy': a}
                 for s, l, f, a in zip(eval_steps, eval_losses, eval_f1s, eval_accs)]
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"✓ 训练历史数据已保存: {history_path}")


def load_data(file_path):
    """加载JSON数据"""
    logger.info(f"加载数据: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"✓ 加载完成: {len(data):,} 条")
    return data


def load_class_weights(file_path):
    """加载类别权重"""
    with open(file_path, 'r') as f:
        weights = json.load(f)
    return weights


def preprocess_data(data, tokenizer, max_length=512):
    """预处理数据"""
    logger.info("预处理数据...")

    sequences = [item['sequence'] for item in data]
    labels = [item['label_id'] for item in data]

    encodings = tokenizer(
        sequences,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    logger.info(f"✓ 预处理完成: {len(dataset):,} 条")
    return dataset


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1,
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
    }


class WeightedLossTrainer(Trainer):
    """支持类别权重的Trainer"""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_experiment(exp_name: str):
    """训练单个实验"""
    print("=" * 70)
    print(f"训练实验: {exp_name}")
    print("=" * 70)

    # 获取配置
    config = get_experiment_config(exp_name)
    training_config = config.training
    set_seed(training_config.seed)

    print(f"\n实验描述: {config.get_description()}")
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {config.output_dir}")

    # 检查数据是否存在
    if not config.train_data.exists():
        print(f"\n❌ 训练数据不存在: {config.train_data}")
        print(f"请先运行数据准备脚本:")
        print(f"  python experiments/v3_unified/{exp_name}/prepare_data.py")
        return None

    # 加载tokenizer
    logger.info(f"加载Tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

    # 加载数据
    train_data = load_data(config.train_data)
    eval_data = load_data(config.eval_data)

    # 预处理
    train_dataset = preprocess_data(train_data, tokenizer, training_config.max_seq_length)
    eval_dataset = preprocess_data(eval_data, tokenizer, training_config.max_seq_length)

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset):,}")
    print(f"  验证集: {len(eval_dataset):,}")

    # 加载类别权重
    class_weights_tensor = None
    if training_config.use_class_weights and config.class_weights_path.exists():
        weights_dict = load_class_weights(config.class_weights_path)
        weights_list = [weights_dict['Protein-coding'], weights_dict['Non-coding']]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()
        print(f"\n类别权重: {weights_dict}")

    # 加载模型
    logger.info(f"加载预训练模型: {PRETRAINED_MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(PRETRAINED_MODEL_PATH),
        num_labels=training_config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type=training_config.problem_type
    )
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        evaluation_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        logging_dir=str(config.log_dir),
        logging_strategy=training_config.logging_strategy,
        logging_steps=training_config.logging_steps,
        report_to=training_config.report_to,
        seed=training_config.seed,
        fp16=training_config.fp16,
        dataloader_num_workers=training_config.dataloader_num_workers,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
    )

    # 创建Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            )
        ]
    )

    # 检查是否有checkpoint可以恢复
    checkpoints = list(config.output_dir.glob("checkpoint-*"))
    checkpoints = [c for c in checkpoints if c.is_dir() and not c.name.startswith("tmp-")]
    resume_checkpoint = None
    if checkpoints:
        # 按step数排序，取最新的
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        resume_checkpoint = str(checkpoints[-1])
        print(f"\n发现checkpoint，将从 {resume_checkpoint} 恢复训练")

    # 训练
    print("\n" + "=" * 70)
    print("开始训练..." if not resume_checkpoint else f"从 checkpoint 恢复训练...")
    print("=" * 70 + "\n")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 保存模型
    trainer.save_model(str(config.final_model_path))
    tokenizer.save_pretrained(str(config.final_model_path))
    print(f"\n✓ 模型保存: {config.final_model_path}")

    # 保存训练结果
    with open(config.output_dir / "train_results.json", 'w') as f:
        json.dump(train_result.metrics, f, indent=2)

    # 绘制训练曲线图
    plot_training_curves(trainer.state.log_history, config.output_dir, exp_name)

    # 最终评估
    print("\n" + "=" * 70)
    print("最终评估...")
    print("=" * 70)

    eval_results = trainer.evaluate()

    print(f"\n验证集结果:")
    print(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 (macro): {eval_results['eval_f1_macro']:.4f}")
    print(f"  F1 (Protein-coding): {eval_results['eval_f1_class_0']:.4f}")
    print(f"  F1 (Non-coding): {eval_results['eval_f1_class_1']:.4f}")

    # 保存评估结果
    with open(config.output_dir / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ {exp_name} 训练完成!")
    print("=" * 70)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description='V3 Unified Training')
    parser.add_argument('--exp', type=str, required=True,
                       choices=list(EXPERIMENTS.keys()) + ['all'],
                       help='实验名称或"all"运行所有实验')

    args = parser.parse_args()

    if args.exp == 'all':
        results = {}
        for exp_name in EXPERIMENTS.keys():
            result = train_experiment(exp_name)
            if result:
                results[exp_name] = result

        # 打印汇总
        print("\n" + "=" * 70)
        print("所有实验结果汇总")
        print("=" * 70)
        for exp_name, result in results.items():
            print(f"\n{exp_name}:")
            print(f"  准确率: {result['eval_accuracy']:.4f}")
            print(f"  F1 (macro): {result['eval_f1_macro']:.4f}")
    else:
        train_experiment(args.exp)


if __name__ == "__main__":
    main()
