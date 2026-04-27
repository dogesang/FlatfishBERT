#!/usr/bin/env python3
"""
DNABERT2 资源对比实验 - 训练脚本
与FlatfishBert使用完全相同的超参数，记录详细的资源消耗指标
"""

import sys
import os
import argparse

# 禁用triton/flash attention
sys.modules['triton'] = None
sys.modules['triton.language'] = None

import json
import time
from pathlib import Path
from datetime import datetime

# 添加DNABERT2模型目录到路径
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import logging

from config_resource import get_config, LABEL2ID, ID2LABEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ResourceMonitorCallback(TrainerCallback):
    """资源监控回调 - 记录训练和评估阶段的显存使用"""
    
    def __init__(self):
        self.start_time = None
        self.train_peak_memory = 0
        self.eval_peak_memory = 0
        self.step_times = []
        self.last_step_time = None
        self.in_evaluation = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_step_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info("="*70)
        logger.info("训练开始")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
    
    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024**3)
            if not self.in_evaluation:
                self.train_peak_memory = max(self.train_peak_memory, current_memory)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """评估开始时重置显存统计"""
        self.in_evaluation = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """每个预测步骤后记录显存"""
        if torch.cuda.is_available() and self.in_evaluation:
            current_memory = torch.cuda.max_memory_allocated() / (1024**3)
            self.eval_peak_memory = max(self.eval_peak_memory, current_memory)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """评估结束时记录显存"""
        if logs and 'eval_loss' in logs:
            self.in_evaluation = False
            if torch.cuda.is_available():
                eval_mem = torch.cuda.max_memory_allocated() / (1024**3)
                self.eval_peak_memory = max(self.eval_peak_memory, eval_mem)
                logger.info(f"评估阶段峰值显存: {self.eval_peak_memory:.2f} GB")
    
    def on_train_end(self, args, state, control, **kwargs):
        end_time = time.time()
        total_time = end_time - self.start_time
        
        logger.info("="*70)
        logger.info("训练结束")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总训练时间: {total_time/3600:.2f} 小时")
        logger.info(f"训练阶段峰值显存: {self.train_peak_memory:.2f} GB")
        logger.info(f"评估阶段峰值显存: {self.eval_peak_memory:.2f} GB")
        logger.info("="*70)
    
    def get_stats(self):
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        return {
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "gpu_hours": total_time / 3600,
            "train_peak_memory_gb": self.train_peak_memory,
            "eval_peak_memory_gb": self.eval_peak_memory,
            "total_steps": len(self.step_times),
            "avg_step_time_seconds": np.mean(self.step_times) if self.step_times else 0,
        }


def load_data(file_path):
    """加载JSON数据"""
    logger.info(f"加载数据: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"✓ 加载完成: {len(data):,} 条")
    return data


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
    
    total_tokens = sum(sum(1 for t in ids if t != tokenizer.pad_token_id) for ids in encodings['input_ids'])
    logger.info(f"  总token数: {total_tokens:,}")
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    logger.info(f"✓ 预处理完成: {len(dataset):,} 条")
    return dataset, total_tokens


def preprocess_logits_for_metrics(logits, labels):
    """预处理logits - 关键：移到CPU避免显存累积"""
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.dim() == 3:
        logits = logits[:, 0, :]
    assert logits.dim() == 2 and logits.size(1) == 2
    return logits.detach().cpu()


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
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


class WeightedLossTrainer(Trainer):
    """支持类别权重的Trainer"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(seed=2025):
    """训练DNABERT2模型"""
    print("="*70)
    print(f"DNABERT2 资源对比实验 - 种子={seed}")
    print("="*70)
    
    # 获取配置
    config = get_config(seed=seed)
    training_config = config.training
    set_seed(seed)
    
    print(f"\n实验配置:")
    print(f"  种子: {seed}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  per_device_train_batch_size: {training_config.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size: {training_config.per_device_eval_batch_size}")
    print(f"  总batch size (2 GPU): {training_config.per_device_train_batch_size * 2}")
    
    # 设置日志
    log_file = config.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 加载Tokenizer
    logger.info(f"加载DNABERT2 Tokenizer: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    
    # 加载数据
    train_data = load_data(config.train_data)
    eval_data = load_data(config.eval_data)
    train_dataset, train_tokens = preprocess_data(train_data, tokenizer, training_config.max_seq_length)
    eval_dataset, eval_tokens = preprocess_data(eval_data, tokenizer, training_config.max_seq_length)
    
    # 加载类别权重
    class_weights_tensor = None
    if training_config.use_class_weights and config.class_weights_path.exists():
        with open(config.class_weights_path, 'r') as f:
            weights_dict = json.load(f)
        weights_list = [weights_dict['Protein-coding'], weights_dict['Non-coding']]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()
    
    # 加载模型
    logger.info(f"加载DNABERT2模型: {config.model_path}")
    model_config = AutoConfig.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        num_labels=training_config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.model_path),
        config=model_config,
        trust_remote_code=True,
    )
    
    # 四层防护机制
    model.config.return_dict = True
    model.config.output_hidden_states = False
    model.config.output_attentions = False
    logger.info("✓ 已启用四层防护机制，避免评估OOM")
    
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
        eval_accumulation_steps=10,  # 关键：防止评估OOM
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
        seed=seed,
        fp16=training_config.fp16,
        dataloader_num_workers=training_config.dataloader_num_workers,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
    )
    
    # 创建Trainer
    resource_callback = ResourceMonitorCallback()
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        class_weights=class_weights_tensor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            ),
            resource_callback
        ]
    )
    
    # 开始训练
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70 + "\n")
    
    train_result = trainer.train()
    
    # 保存模型
    trainer.save_model(str(config.final_model_path))
    tokenizer.save_pretrained(str(config.final_model_path))
    
    # 保存训练结果
    with open(config.output_dir / "train_results.json", 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # 保存资源统计
    resource_stats = resource_callback.get_stats()
    resource_stats['train_tokens'] = train_tokens
    resource_stats['eval_tokens'] = eval_tokens
    resource_stats['tokens_per_second'] = train_tokens / resource_stats['total_time_seconds'] if resource_stats['total_time_seconds'] > 0 else 0
    resource_stats['samples_per_second'] = len(train_dataset) / resource_stats['total_time_seconds'] if resource_stats['total_time_seconds'] > 0 else 0
    resource_stats['steps_per_second'] = resource_stats['total_steps'] / resource_stats['total_time_seconds'] if resource_stats['total_time_seconds'] > 0 else 0
    
    with open(config.output_dir / "resource_metrics.json", 'w') as f:
        json.dump(resource_stats, f, indent=2)
    
    logger.info(f"✓ 资源统计已保存: {config.output_dir / 'resource_metrics.json'}")
    
    # 最终评估
    print("\n" + "="*70)
    print("最终评估...")
    print("="*70)
    
    eval_results = trainer.evaluate()
    
    with open(config.output_dir / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✅ 训练完成!")
    print(f"   种子: {seed}")
    print(f"   训练时间: {resource_stats['total_time_hours']:.2f} 小时")
    print(f"   训练峰值显存: {resource_stats['train_peak_memory_gb']:.2f} GB")
    print(f"   评估峰值显存: {resource_stats['eval_peak_memory_gb']:.2f} GB")
    print(f"   F1-macro: {eval_results['eval_f1_macro']:.4f}")
    
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    args = parser.parse_args()
    
    train(seed=args.seed)
