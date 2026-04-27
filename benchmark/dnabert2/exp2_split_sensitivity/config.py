"""
DNABERT2 Exp2实验 - 配置文件
与FlatfishBert Exp2使用完全相同的超参数
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# DNABERT2模型路径（本地）
DNABERT2_MODEL_PATH = Path(
    os.getenv("DNABERT2_MODEL_PATH", str(PROJECT_ROOT / "external_models" / "dnabert2"))
)

# Exp2数据
EXP2_DATA_DIR = PROJECT_ROOT / "data/experiments/v3_unified/Exp2_unconcat"

# 输出路径
OUTPUT_ROOT = PROJECT_ROOT / "finetune_output/dnabert2_exp2"

# ============================================================
# 标签定义（与Exp2完全一致）
# ============================================================

LABEL2ID = {'Protein-coding': 0, 'Non-coding': 1}
ID2LABEL = {0: 'Protein-coding', 1: 'Non-coding'}

# ============================================================
# 训练配置（与Exp2完全一致）
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置 - 与DNABERT2 Exp4对比实验完全一致"""

    # 模型配置
    num_labels: int = 2
    max_seq_length: int = 512
    problem_type: str = "single_label_classification"

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 梯度
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 评估和保存
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True

    # 早停
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001

    # 日志
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"

    # 其他
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # 类别权重
    use_class_weights: bool = True

    def to_dict(self) -> Dict:
        return {
            'num_labels': self.num_labels,
            'max_seq_length': self.max_seq_length,
            'num_train_epochs': self.num_train_epochs,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'eval_steps': self.eval_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'use_class_weights': self.use_class_weights,
            'seed': self.seed,
        }


@dataclass
class ExperimentConfig:
    """DNABERT2 Exp2实验配置"""

    exp_name: str = "dnabert2_exp2"
    model_path: Path = field(default_factory=lambda: DNABERT2_MODEL_PATH)
    data_dir: Path = field(default_factory=lambda: EXP2_DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_ROOT)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def train_data(self) -> Path:
        return self.data_dir / "train.json"

    @property
    def eval_data(self) -> Path:
        return self.data_dir / "eval.json"

    @property
    def test_data(self) -> Path:
        return self.data_dir / "test.json"

    @property
    def class_weights_path(self) -> Path:
        return self.data_dir / "class_weights.json"

    @property
    def final_model_path(self) -> Path:
        return self.output_dir / "final_model"


def get_config() -> ExperimentConfig:
    return ExperimentConfig()


LENGTH_BINS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 1000),
    (1000, 2000),
    (2000, float('inf'))
]

LENGTH_BIN_NAMES = [
    "0-100",
    "100-200",
    "200-300",
    "300-500",
    "500-1000",
    "1000-2000",
    "2000+"
]
