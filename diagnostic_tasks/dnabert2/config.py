"""
DNABERT2 V4 Bio-Hierarchy Experiments 配置文件。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data/experiments/v4_bio_hierarchy"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "finetune_output/dnabert2_v4_bio_hierarchy"
DNABERT2_MODEL_PATH = Path(os.getenv("DNABERT2_MODEL_PATH", "${DNABERT2_MODEL_PATH:-./external_models/dnabert2}"))

TASKS = {
    "Task_G_v3": {
        "name": "Task_G_v3",
        "description": "Gene body vs Background 二分类",
        "data_dir": DATA_ROOT / "Task_G_v3",
        "labels": ["gene_body", "background"],
        "label2id": {"gene_body": 0, "background": 1},
        "id2label": {0: "gene_body", 1: "background"},
        "num_labels": 2,
    },
    "Task_S": {
        "name": "Task_S",
        "description": "基因内部结构分类 (CDS/intron/ncRNA_gene)",
        "data_dir": DATA_ROOT / "Task_S",
        "labels": ["CDS", "intron", "ncRNA_gene"],
        "label2id": {"CDS": 0, "intron": 1, "ncRNA_gene": 2},
        "id2label": {0: "CDS", 1: "intron", 2: "ncRNA_gene"},
        "num_labels": 3,
    },
    "Task_S_aug": {
        "name": "Task_S_aug",
        "description": "基因内部结构分类 + 反向互补增强",
        "data_dir": DATA_ROOT / "Task_S_aug",
        "labels": ["CDS", "intron", "ncRNA_gene"],
        "label2id": {"CDS": 0, "intron": 1, "ncRNA_gene": 2},
        "id2label": {0: "CDS", 1: "intron", 2: "ncRNA_gene"},
        "num_labels": 3,
    },
    "Task_N": {
        "name": "Task_N",
        "description": "ncRNA亚型分类",
        "data_dir": DATA_ROOT / "Task_N",
        "labels": ["lncRNA_exon", "tRNA", "rRNA", "snRNA", "snoRNA"],
        "label2id": {"lncRNA_exon": 0, "tRNA": 1, "rRNA": 2, "snRNA": 3, "snoRNA": 4},
        "id2label": {0: "lncRNA_exon", 1: "tRNA", 2: "rRNA", 3: "snRNA", 4: "snoRNA"},
        "num_labels": 5,
    },
    "Task_N_aug": {
        "name": "Task_N_aug",
        "description": "ncRNA亚型分类 + 反向互补增强",
        "data_dir": DATA_ROOT / "Task_N_aug",
        "labels": ["lncRNA_exon", "tRNA", "rRNA", "snRNA", "snoRNA"],
        "label2id": {"lncRNA_exon": 0, "tRNA": 1, "rRNA": 2, "snRNA": 3, "snoRNA": 4},
        "id2label": {0: "lncRNA_exon", 1: "tRNA", 2: "rRNA", 3: "snRNA", 4: "snoRNA"},
        "num_labels": 5,
    },
}


@dataclass
class TrainingConfig:
    max_seq_length: int = 512
    problem_type: str = "single_label_classification"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    use_class_weights: bool = True


@dataclass
class TaskConfig:
    task_name: str
    num_labels: int
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    data_dir: Path
    output_dir: Path
    training: TrainingConfig = field(default_factory=TrainingConfig)

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


def get_task_config(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")

    task_info = TASKS[task_name]
    output_dir = MODEL_OUTPUT_ROOT / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    if task_name in {"Task_N", "Task_N_aug"}:
        training = TrainingConfig(
            num_train_epochs=10,
            per_device_train_batch_size=32,
            learning_rate=1e-5,
            warmup_ratio=0.15,
            eval_steps=100,
            save_steps=100,
            logging_steps=50,
        )
    else:
        training = TrainingConfig()

    return TaskConfig(
        task_name=task_name,
        num_labels=task_info["num_labels"],
        label2id=task_info["label2id"],
        id2label=task_info["id2label"],
        data_dir=task_info["data_dir"],
        output_dir=output_dir,
        training=training,
    )
