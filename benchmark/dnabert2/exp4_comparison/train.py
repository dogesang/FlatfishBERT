#!/usr/bin/env python3
"""
DNABERT2 横向对比实验 - 训练脚本
使用与FlatfishBert Exp4完全相同的数据和超参数

特点:
1. 本地加载DNABERT2模型（处理自定义架构）
2. 详细的训练日志（时间、显存、吞吐量）
3. 与Exp4完全一致的超参数
4. 使用PyTorch标准attention（禁用triton/flash attention避免版本兼容问题）
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
import time
import argparse
from pathlib import Path
from datetime import datetime

# 添加DNABERT2模型目录到路径（必须在import transformers之前）
DNABERT2_PATH = "${DNABERT2_MODEL_PATH:-./external_models/dnabert2}"
sys.path.insert(0, DNABERT2_PATH)

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
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
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from config import get_config, LABEL2ID, ID2LABEL

# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 训练统计回调
# ============================================================

class TrainingStatsCallback(TrainerCallback):
    """记录训练统计信息的回调"""

    def __init__(self):
        self.start_time = None
        self.total_tokens = 0
        self.peak_memory = 0
        self.step_times = []
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_step_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info("=" * 70)
        logger.info("训练开始")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time

        # 更新峰值显存
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            self.peak_memory = max(self.peak_memory, current_memory)

    def on_train_end(self, args, state, control, **kwargs):
        end_time = time.time()
        total_time = end_time - self.start_time

        # 计算统计信息
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        total_steps = len(self.step_times)

        logger.info("=" * 70)
        logger.info("训练结束")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总训练时间: {total_time/3600:.2f} 小时 ({total_time:.0f} 秒)")
        logger.info(f"GPU hours: {total_time/3600:.2f}")
        logger.info(f"峰值显存: {self.peak_memory:.2f} GB")
        logger.info(f"总步数: {total_steps}")
        logger.info(f"平均每步时间: {avg_step_time:.3f} 秒")
        logger.info("=" * 70)

    def get_stats(self):
        """获取训练统计信息"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        return {
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "gpu_hours": total_time / 3600,
            "peak_memory_gb": self.peak_memory,
            "total_steps": len(self.step_times),
            "avg_step_time_seconds": np.mean(self.step_times) if self.step_times else 0,
        }


# ============================================================
# 数据加载
# ============================================================

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

    # 记录序列长度用于统计
    seq_lengths = [len(seq) for seq in sequences]
    logger.info(f"  序列长度: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.0f}")

    encodings = tokenizer(
        sequences,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

    # 统计token数量
    total_tokens = sum(sum(1 for t in ids if t != tokenizer.pad_token_id) for ids in encodings['input_ids'])
    logger.info(f"  总token数: {total_tokens:,}")

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    logger.info(f"✓ 预处理完成: {len(dataset):,} 条")
    return dataset, total_tokens


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    # DNABERT2输出可能是tuple格式，需要处理
    if isinstance(predictions, tuple):
        predictions = predictions[0]
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


# ============================================================
# 预处理logits函数（保险丝，防止显存累积）
# ============================================================

def preprocess_logits_for_metrics(logits, labels):
    """预处理logits，确保只保留分类logits并移到CPU，避免GPU显存累积

    这是HuggingFace官方推荐的方式，在缓存预测值之前对logits做预处理。
    关键操作：
    1. 提取真正的分类logits（处理tuple/3D tensor情况）
    2. 形状断言，确保是 (batch, num_classes)
    3. detach() 断开梯度计算图
    4. cpu() 移到CPU内存，避免GPU显存累积

    Args:
        logits: 模型输出的logits，可能是tuple或tensor
        labels: 标签（未使用，但API需要）

    Returns:
        只包含分类logits的CPU tensor (batch, num_classes)
    """
    # 如果是tuple，取第一个元素（通常是logits）
    if isinstance(logits, tuple):
        logits = logits[0]

    # 确保是2D tensor (batch, num_classes)
    if logits.dim() == 3:
        # 如果是 (batch, seq_len, hidden) 形状，说明拿错了，取第一个token
        logits = logits[:, 0, :]

    # 形状断言：必须是 (batch, 2) 用于二分类
    assert logits.dim() == 2, f"Expected 2D logits, got shape {logits.shape}"
    assert logits.size(1) == 2, f"Expected 2 classes, got {logits.size(1)}"

    # 关键：detach并移到CPU，避免GPU显存累积
    # 这是防止评估阶段显存周期性增长的核心操作
    return logits.detach().cpu()


# ============================================================
# 带类别权重的Trainer
# ============================================================

class WeightedLossTrainer(Trainer):
    """支持类别权重的Trainer"""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # 强制使用return_dict=True，确保返回ModelOutput对象
        # 避免tuple格式导致的位置假设错误
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# 绘图函数
# ============================================================

def plot_training_curves(log_history, output_dir, exp_name):
    """绘制训练曲线图"""
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
    fig.suptitle(f'{exp_name} Training Curves (DNABERT2)', fontsize=14, fontweight='bold')

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

    output_path = Path(output_dir) / f'{exp_name}_training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ 训练曲线图已保存: {output_path}")

    # 保存训练历史数据
    history_path = Path(output_dir) / f'{exp_name}_training_history.json'
    history_data = {
        'train': [{'step': s, 'loss': l} for s, l in zip(train_steps, train_losses)],
        'eval': [{'step': s, 'loss': l, 'f1_macro': f, 'accuracy': a}
                 for s, l, f, a in zip(eval_steps, eval_losses, eval_f1s, eval_accs)]
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"✓ 训练历史数据已保存: {history_path}")


# ============================================================
# 主训练函数
# ============================================================

def train():
    """训练DNABERT2模型"""
    print("=" * 70)
    print("DNABERT2 横向对比实验 - 训练")
    print("=" * 70)

    # 获取配置
    config = get_config()
    training_config = config.training
    set_seed(training_config.seed)

    print(f"\n实验名称: {config.exp_name}")
    print(f"DNABERT2模型路径: {config.model_path}")
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {config.output_dir}")

    # 检查数据
    if not config.train_data.exists():
        print(f"\n❌ 训练数据不存在: {config.train_data}")
        return None

    # 设置文件日志
    log_file = config.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"日志文件: {log_file}")

    # ========== 加载Tokenizer ==========
    logger.info(f"\n加载DNABERT2 Tokenizer: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path),
        trust_remote_code=True
    )
    logger.info(f"✓ Tokenizer加载完成, vocab_size={tokenizer.vocab_size}")

    # ========== 加载数据 ==========
    train_data = load_data(config.train_data)
    eval_data = load_data(config.eval_data)

    # 预处理
    train_dataset, train_tokens = preprocess_data(train_data, tokenizer, training_config.max_seq_length)
    eval_dataset, eval_tokens = preprocess_data(eval_data, tokenizer, training_config.max_seq_length)

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset):,} 条, {train_tokens:,} tokens")
    print(f"  验证集: {len(eval_dataset):,} 条, {eval_tokens:,} tokens")

    # ========== 加载类别权重 ==========
    class_weights_tensor = None
    if training_config.use_class_weights and config.class_weights_path.exists():
        weights_dict = load_class_weights(config.class_weights_path)
        weights_list = [weights_dict['Protein-coding'], weights_dict['Non-coding']]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()
        print(f"\n类别权重: {weights_dict}")

    # ========== 加载DNABERT2模型 ==========
    logger.info(f"\n加载DNABERT2模型: {config.model_path}")

    # 加载配置
    model_config = AutoConfig.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        num_labels=training_config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.model_path),
        config=model_config,
        trust_remote_code=True,
    )

    # ========== 关键修复：禁用hidden_states输出，避免评估时显存累积 ==========
    # DNABERT2的SequenceClassifierOutput默认包含hidden_states (batch, 512, 768)
    # 每样本1.5MB，151743样本 = 222GB，会导致评估时OOM
    # 三层防护：
    # 1. 强制return_dict=True，确保返回ModelOutput对象（可稳定访问.logits）
    # 2. 禁用output_hidden_states/attentions（虽然DNABERT2可能不遵守）
    # 3. 配合preprocess_logits_for_metrics + eval_accumulation_steps
    model.config.return_dict = True
    model.config.output_hidden_states = False
    model.config.output_attentions = False

    # 诊断日志：确认配置生效
    logger.info("✓ 模型配置已设置:")
    logger.info(f"  return_dict: {model.config.return_dict}")
    logger.info(f"  output_hidden_states: {model.config.output_hidden_states}")
    logger.info(f"  output_attentions: {model.config.output_attentions}")
    logger.info(f"  use_return_dict (computed): {getattr(model.config, 'use_return_dict', 'N/A')}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型加载完成")
    print(f"  总参数量: {param_count:,}")
    print(f"  可训练参数: {trainable_count:,}")

    # ========== 训练参数 ==========
    # 系统有2个GPU，per_device是每个GPU的batch size
    # DNABERT2无flash attention，单GPU显存约20GB可用，需要更小的batch
    # 设置per_device=16，总batch=16*2=32
    eval_batch_size = 16
    print(f"\n⚠️ 使用硬编码 per_device_eval_batch_size = {eval_batch_size}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   总eval_batch_size: {eval_batch_size * torch.cuda.device_count()}")
    print(f"   预期评估步数: {len(eval_dataset) // (eval_batch_size * torch.cuda.device_count()) + 1}")

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=eval_batch_size,  # 每GPU 16，总共32
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        evaluation_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        eval_accumulation_steps=10,  # 关键：每10步将预测结果从GPU搬到CPU，避免显存累积
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

    # ========== 创建Trainer ==========
    stats_callback = TrainingStatsCallback()

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 关键：预处理logits，只缓存分类结果
        class_weights=class_weights_tensor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            ),
            stats_callback
        ]
    )

    # 检查checkpoint
    checkpoints = list(config.output_dir.glob("checkpoint-*"))
    checkpoints = [c for c in checkpoints if c.is_dir() and not c.name.startswith("tmp-")]
    resume_checkpoint = None
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        resume_checkpoint = str(checkpoints[-1])
        print(f"\n发现checkpoint，将从 {resume_checkpoint} 恢复训练")

    # ========== 开始训练 ==========
    print("\n" + "=" * 70)
    print("开始训练..." if not resume_checkpoint else f"从 checkpoint 恢复训练...")
    print("=" * 70 + "\n")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ========== 保存模型 ==========
    trainer.save_model(str(config.final_model_path))
    tokenizer.save_pretrained(str(config.final_model_path))
    print(f"\n✓ 模型保存: {config.final_model_path}")

    # ========== 保存训练结果 ==========
    with open(config.output_dir / "train_results.json", 'w') as f:
        json.dump(train_result.metrics, f, indent=2)

    # 保存训练统计
    training_stats = stats_callback.get_stats()
    training_stats['train_tokens'] = train_tokens
    training_stats['eval_tokens'] = eval_tokens
    training_stats['tokens_per_second'] = train_tokens / training_stats['total_time_seconds'] if training_stats['total_time_seconds'] > 0 else 0

    with open(config.output_dir / "training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"✓ 训练统计已保存: {config.output_dir / 'training_stats.json'}")

    # 绘制训练曲线
    plot_training_curves(trainer.state.log_history, config.output_dir, config.exp_name)

    # ========== 最终评估 ==========
    print("\n" + "=" * 70)
    print("最终评估...")
    print("=" * 70)

    eval_results = trainer.evaluate()

    print(f"\n验证集结果:")
    print(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 (macro): {eval_results['eval_f1_macro']:.4f}")
    print(f"  F1 (Protein-coding): {eval_results['eval_f1_class_0']:.4f}")
    print(f"  F1 (Non-coding): {eval_results['eval_f1_class_1']:.4f}")

    with open(config.output_dir / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)

    # ========== 打印训练统计 ==========
    print("\n" + "=" * 70)
    print("训练统计")
    print("=" * 70)
    print(f"  总训练时间: {training_stats['total_time_hours']:.2f} 小时")
    print(f"  GPU hours: {training_stats['gpu_hours']:.2f}")
    print(f"  峰值显存: {training_stats['peak_memory_gb']:.2f} GB")
    print(f"  tokens/sec: {training_stats['tokens_per_second']:.0f}")

    print("\n" + "=" * 70)
    print(f"✅ DNABERT2 训练完成!")
    print("=" * 70)

    return eval_results


if __name__ == "__main__":
    train()
