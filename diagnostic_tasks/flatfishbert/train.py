#!/usr/bin/env python3
"""
V4 Bio-Hierarchy Experiments - 统一训练脚本
支持 Task G / Task S / Task N 三个任务的训练
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
    TrainerCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from config import (
    get_task_config, TOKENIZER_PATH, PRETRAINED_MODEL_PATH,
    TASK_LABEL2ID, TASK_ID2LABEL, TASKS
)


def setup_logging(task_name: str, log_dir: Path):
    """设置日志，同时输出到控制台和文件"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{task_name}_train_{timestamp}.log"

    # 创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除已有的handlers
    logger.handlers.clear()

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"日志文件: {log_file}")

    return logger, log_file


# 默认logger（在setup_logging之前使用）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_curves(log_history, output_dir, task_name):
    """绘制训练曲线图"""
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_f1s, eval_accs = [], [], [], []

    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(entry.get('step', 0))
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry.get('step', 0))
            eval_losses.append(entry['eval_loss'])
            eval_f1s.append(entry.get('eval_f1_macro', 0))
            eval_accs.append(entry.get('eval_accuracy', 0))

    if not train_steps and not eval_steps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{task_name} Training Curves', fontsize=14, fontweight='bold')

    # Training Loss
    if train_steps:
        axes[0, 0].plot(train_steps, train_losses, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

    # Eval Loss
    if eval_steps:
        axes[0, 1].plot(eval_steps, eval_losses, 'r-o', markersize=3)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Eval Loss')
        axes[0, 1].set_title('Evaluation Loss')
        axes[0, 1].grid(True, alpha=0.3)

    # F1-macro
    if eval_steps:
        axes[1, 0].plot(eval_steps, [f*100 for f in eval_f1s], 'g-o', markersize=3)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('F1-macro (%)')
        axes[1, 0].set_title('F1-macro Score')
        axes[1, 0].grid(True, alpha=0.3)
        # 标注最佳点
        if eval_f1s:
            best_idx = np.argmax(eval_f1s)
            best_step = eval_steps[best_idx]
            best_f1 = eval_f1s[best_idx] * 100
            axes[1, 0].axvline(x=best_step, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].annotate(f'Best: {best_f1:.2f}%\n@ step {best_step}',
                        xy=(best_step, best_f1),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, color='red')

    # Accuracy
    if eval_steps:
        axes[1, 1].plot(eval_steps, [a*100 for a in eval_accs], 'm-o', markersize=3)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{task_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 训练曲线图已保存")

    # 同时保存训练历史数据为JSON
    history_path = Path(output_dir) / f'{task_name}_training_history.json'
    history_data = {
        'train': [{'step': s, 'loss': l} for s, l in zip(train_steps, train_losses)],
        'eval': [{'step': s, 'loss': l, 'f1_macro': f, 'accuracy': a}
                 for s, l, f, a in zip(eval_steps, eval_losses, eval_f1s, eval_accs)]
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"✓ 训练历史数据已保存")


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
        return json.load(f)


def preprocess_data(data, tokenizer, max_length=512, batch_size=10000):
    """预处理数据 - 批量处理以节省内存"""
    logger.info(f"预处理数据... (共 {len(data):,} 条，批量大小 {batch_size})")

    # 兼容两种字段名: label_id (旧格式) 和 label (新格式)
    labels = [item.get('label_id', item.get('label')) for item in data]

    all_input_ids = []
    all_attention_mask = []

    # 批量处理tokenization
    total_batches = (len(data) + batch_size - 1) // batch_size
    for i in range(0, len(data), batch_size):
        batch_idx = i // batch_size + 1
        batch_data = data[i:i+batch_size]
        sequences = [item['sequence'] for item in batch_data]

        encodings = tokenizer(
            sequences,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )

        all_input_ids.extend(encodings['input_ids'])
        all_attention_mask.extend(encodings['attention_mask'])

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            logger.info(f"  Tokenization进度: {batch_idx}/{total_batches} ({i+len(batch_data):,}/{len(data):,})")

    dataset = Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
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

    # 每个类别的F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1,
    }

    # 添加每个类别的F1
    for i, f1_score in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1_score

    return metrics


class TrainingLoggingCallback(TrainerCallback):
    """自定义Callback，将训练过程中的指标记录到日志文件"""

    def __init__(self, logger_instance):
        self.logger = logger_instance

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次记录日志时调用"""
        if logs is None:
            return

        step = state.global_step
        epoch = state.epoch

        # 训练loss
        if 'loss' in logs and 'eval_loss' not in logs:
            lr = logs.get('learning_rate', 0)
            self.logger.info(
                f"Step {step} | Epoch {epoch:.2f} | "
                f"Loss: {logs['loss']:.4f} | LR: {lr:.2e}"
            )

        # 评估结果
        if 'eval_loss' in logs:
            eval_loss = logs.get('eval_loss', 0)
            eval_acc = logs.get('eval_accuracy', 0)
            eval_f1 = logs.get('eval_f1_macro', 0)
            self.logger.info(
                f"Step {step} | Eval | "
                f"Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | F1: {eval_f1:.4f}"
            )

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时调用"""
        self.logger.info(f"训练参数:")
        self.logger.info(f"  Epochs: {args.num_train_epochs}")
        self.logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        self.logger.info(f"  Learning rate: {args.learning_rate}")
        self.logger.info(f"  Warmup ratio: {args.warmup_ratio}")
        self.logger.info(f"  Weight decay: {args.weight_decay}")
        self.logger.info(f"  Total steps: {state.max_steps}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时调用"""
        self.logger.info(f"Epoch {int(state.epoch)} 完成")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        self.logger.info(f"训练结束，总步数: {state.global_step}")


class WeightedLossTrainer(Trainer):
    """支持类别权重和WeightedRandomSampler的Trainer"""

    def __init__(self, *args, class_weights=None, sampler_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.sampler_weights = sampler_weights

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

    def get_train_dataloader(self) -> DataLoader:
        """重写以支持WeightedRandomSampler"""
        if self.sampler_weights is None:
            return super().get_train_dataloader()

        # 使用WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=self.sampler_weights,
            num_samples=len(self.sampler_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def train_task(task_name: str):
    """训练单个任务"""
    global logger

    # 获取配置
    config = get_task_config(task_name)
    task_def = TASKS[task_name]
    label2id = TASK_LABEL2ID[task_name]
    id2label = TASK_ID2LABEL[task_name]
    training_config = config.training

    # 设置日志（输出到控制台和文件）
    logger, log_file = setup_logging(task_name, config.log_dir)

    logger.info("=" * 70)
    logger.info(f"训练任务: {task_name}")
    logger.info("=" * 70)

    set_seed(training_config.seed)

    logger.info(f"任务描述: {task_def['description']}")
    logger.info(f"标签: {task_def['labels']}")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"输出目录: {config.output_dir}")

    # 检查数据是否存在
    if not config.train_data.exists():
        logger.error(f"训练数据不存在: {config.train_data}")
        logger.error(f"请先运行数据准备脚本:")
        logger.error(f"  python experiments/v4_bio_hierarchy/tasks/{task_name.lower()}/prepare_data.py")
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

    logger.info(f"数据集大小:")
    logger.info(f"  训练集: {len(train_dataset):,}")
    logger.info(f"  验证集: {len(eval_dataset):,}")

    # 加载类别权重 (用于 Class Weight Loss)
    class_weights_tensor = None
    if training_config.use_class_weights and config.class_weights_path.exists():
        weights_dict = load_class_weights(config.class_weights_path)
        # 按label2id顺序构建权重列表
        # 兼容两种格式: 标签名称作为key (旧格式) 或 数字ID作为key (新格式)
        weights_list = []
        for i in range(config.num_labels):
            # 尝试用标签名称获取
            weight = weights_dict.get(id2label[i])
            if weight is None:
                # 尝试用数字ID获取 (字符串或整数)
                weight = weights_dict.get(str(i), weights_dict.get(i, 1.0))
            weights_list.append(weight)
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()
        logger.info(f"类别权重 (Class Weight Loss): {weights_dict}")

    # 加载采样权重 (用于 WeightedRandomSampler)
    sampler_weights = None
    sampler_weights_path = config.data_dir / "sampler_weights.json"
    if sampler_weights_path.exists():
        with open(sampler_weights_path, 'r') as f:
            sampler_weights = json.load(f)
        sampler_weights = torch.tensor(sampler_weights, dtype=torch.float64)
        logger.info(f"✓ 加载采样权重 (WeightedRandomSampler): {len(sampler_weights):,} 个样本权重")
    else:
        logger.info(f"未找到采样权重文件，不使用 WeightedRandomSampler")

    # 加载模型
    logger.info(f"加载预训练模型: {PRETRAINED_MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(PRETRAINED_MODEL_PATH),
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type=training_config.problem_type
    )
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

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
        sampler_weights=sampler_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            ),
            TrainingLoggingCallback(logger)
        ]
    )

    # 检查checkpoint
    checkpoints = list(config.output_dir.glob("checkpoint-*"))
    checkpoints = [c for c in checkpoints if c.is_dir() and not c.name.startswith("tmp-")]
    resume_checkpoint = None
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        resume_checkpoint = str(checkpoints[-1])
        logger.info(f"发现checkpoint，将从 {resume_checkpoint} 恢复训练")

    # 训练
    logger.info("=" * 70)
    logger.info("开始训练..." if not resume_checkpoint else "从 checkpoint 恢复训练...")
    logger.info("=" * 70)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 保存模型
    trainer.save_model(str(config.final_model_path))
    tokenizer.save_pretrained(str(config.final_model_path))
    logger.info(f"模型保存: {config.final_model_path}")

    # 保存训练结果
    with open(config.output_dir / "train_results.json", 'w') as f:
        json.dump(train_result.metrics, f, indent=2)

    # 绘制训练曲线
    plot_training_curves(trainer.state.log_history, config.output_dir, task_name)

    # 最终评估
    logger.info("=" * 70)
    logger.info("最终评估...")
    logger.info("=" * 70)

    eval_results = trainer.evaluate()

    logger.info(f"验证集结果:")
    logger.info(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1 (macro): {eval_results['eval_f1_macro']:.4f}")
    for i, label in enumerate(task_def['labels']):
        f1_key = f'eval_f1_class_{i}'
        if f1_key in eval_results:
            logger.info(f"  F1 ({label}): {eval_results[f1_key]:.4f}")

    # 保存评估结果
    with open(config.output_dir / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 测试集评估
    if config.test_data.exists():
        logger.info("测试集评估...")
        test_data = load_data(config.test_data)
        test_dataset = preprocess_data(test_data, tokenizer, training_config.max_seq_length)

        test_results = trainer.evaluate(test_dataset)
        test_results = {k.replace('eval_', 'test_'): v for k, v in test_results.items()}

        logger.info(f"测试集结果:")
        logger.info(f"  准确率: {test_results['test_accuracy']:.4f}")
        logger.info(f"  F1 (macro): {test_results['test_f1_macro']:.4f}")

        with open(config.output_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"✅ {task_name} 训练完成!")
    logger.info("=" * 70)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description='V4 Bio-Hierarchy Training')
    parser.add_argument('--task', type=str, required=True,
                       choices=list(TASKS.keys()) + ['all'],
                       help='任务名称或"all"运行所有任务')

    args = parser.parse_args()

    if args.task == 'all':
        results = {}
        for task_name in TASKS.keys():
            result = train_task(task_name)
            if result:
                results[task_name] = result

        # 打印汇总
        print("\n" + "=" * 70)
        print("所有任务结果汇总")
        print("=" * 70)
        for task_name, result in results.items():
            print(f"\n{task_name}:")
            print(f"  准确率: {result['eval_accuracy']:.4f}")
            print(f"  F1 (macro): {result['eval_f1_macro']:.4f}")
    else:
        train_task(args.task)


if __name__ == "__main__":
    main()
