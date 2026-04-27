#!/usr/bin/env python3
"""
DNABERT2 V4 统一训练脚本。
保留审稿所需的最小训练入口。
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.modules["triton"] = None
sys.modules["triton.language"] = None

DNABERT2_CODE_PATH = os.getenv("DNABERT2_CODE_PATH", os.getenv("DNABERT2_MODEL_PATH", "${DNABERT2_MODEL_PATH:-./external_models/dnabert2}"))
sys.path.insert(0, DNABERT2_CODE_PATH)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import DNABERT2_MODEL_PATH, TASKS, get_task_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_class_weights(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


def preprocess_data(data, tokenizer, max_length=512, batch_size=10000):
    labels = [item.get("label_id", item.get("label")) for item in data]
    all_input_ids = []
    all_attention_mask = []

    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        sequences = [item["sequence"] for item in batch_data]
        encodings = tokenizer(
            sequences,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        all_input_ids.extend(encodings["input_ids"])
        all_attention_mask.extend(encodings["attention_mask"])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": labels,
    })


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
    }

    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    for i, score in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = score
    return metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.dim() == 3:
        logits = logits[:, 0, :]
    return logits.detach().cpu()


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def plot_training_curves(log_history, output_dir: Path, task_name: str):
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_f1s, eval_accs = [], [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])
            eval_f1s.append(entry.get("eval_f1_macro", 0))
            eval_accs.append(entry.get("eval_accuracy", 0))

    if not train_steps and not eval_steps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{task_name} Training Curves (DNABERT2)", fontsize=14, fontweight="bold")

    if train_steps:
        axes[0, 0].plot(train_steps, train_losses)
        axes[0, 0].set_title("Training Loss")
    if eval_steps:
        axes[0, 1].plot(eval_steps, eval_losses)
        axes[0, 1].set_title("Evaluation Loss")
        axes[1, 0].plot(eval_steps, [f * 100 for f in eval_f1s])
        axes[1, 0].set_title("F1-macro")
        axes[1, 1].plot(eval_steps, [a * 100 for a in eval_accs])
        axes[1, 1].set_title("Accuracy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Steps")

    plt.tight_layout()
    plt.savefig(output_dir / f"{task_name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(output_dir / f"{task_name}_training_history.json", "w") as f:
        json.dump({
            "train": [{"step": s, "loss": l} for s, l in zip(train_steps, train_losses)],
            "eval": [{"step": s, "loss": l, "f1_macro": f, "accuracy": a} for s, l, f, a in zip(eval_steps, eval_losses, eval_f1s, eval_accs)],
        }, f, indent=2)


def train_task(task_name: str):
    config = get_task_config(task_name)
    task_info = TASKS[task_name]
    training_config = config.training
    set_seed(training_config.seed)

    if not config.train_data.exists():
        raise FileNotFoundError(f"训练数据不存在: {config.train_data}")

    log_file = config.log_dir / f"{task_name}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"任务: {task_name}")
    logger.info(f"DNABERT2模型路径: {DNABERT2_MODEL_PATH}")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"输出目录: {config.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(DNABERT2_MODEL_PATH), trust_remote_code=True)
    train_dataset = preprocess_data(load_data(config.train_data), tokenizer, training_config.max_seq_length)
    eval_dataset = preprocess_data(load_data(config.eval_data), tokenizer, training_config.max_seq_length)

    class_weights_tensor = None
    if training_config.use_class_weights and config.class_weights_path.exists():
        weights_dict = load_class_weights(config.class_weights_path)
        weights_list = [weights_dict.get(config.id2label[i], weights_dict.get(str(i), weights_dict.get(i, 1.0))) for i in range(config.num_labels)]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()

    model_config = AutoConfig.from_pretrained(
        str(DNABERT2_MODEL_PATH),
        trust_remote_code=True,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        problem_type=training_config.problem_type,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(DNABERT2_MODEL_PATH),
        config=model_config,
        trust_remote_code=True,
    )

    args = TrainingArguments(
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

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_threshold=training_config.early_stopping_threshold,
        )],
    )

    train_result = trainer.train()
    trainer.save_model(str(config.final_model_path))
    tokenizer.save_pretrained(str(config.final_model_path))

    with open(config.output_dir / "train_results.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    with open(config.output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    plot_training_curves(trainer.state.log_history, config.output_dir, task_name)
    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    args = parser.parse_args()
    train_task(args.task)


if __name__ == "__main__":
    main()
