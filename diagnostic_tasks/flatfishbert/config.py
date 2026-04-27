"""
V4 Bio-Hierarchy Experiments - 统一配置文件
基于生物学层次的基因组区域分类任务
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent

RAW_DATA_ROOT = Path(os.getenv("FLATFISH_RAW_DATA_ROOT", str(PROJECT_ROOT / "data" / "raw")))
DATA_OUTPUT_ROOT = PROJECT_ROOT / "data/experiments/v4_bio_hierarchy"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "finetune_output/v4_bio_hierarchy"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer/flatfish_tokenizer"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "pretrain_output/pretrained_model"

SPECIES_CONFIG = {
    "GCF_000523025.1": {
        "name": "Cynoglossus_semilaevis",
        "common_name": "半滑舌鳎",
        "gff": "genomic.gff",
        "fasta": "GCF_000523025.1_Cse_v1.0_genomic.fna",
    },
    "GCF_022379125.1": {
        "name": "Scophthalmus_maximus",
        "common_name": "大菱鲆",
        "gff": "genomic.gff",
        "fasta": "GCF_022379125.1_ASM2237912v1_genomic.fna",
    },
    "GCF_024713975.1": {
        "name": "Paralichthys_olivaceus",
        "common_name": "牙鲆",
        "gff": "genomic.gff",
        "fasta": "GCF_024713975.1_ASM2471397v2_genomic.fna",
    },
}


def get_species_paths(species_id: str) -> Tuple[Path, Path]:
    if species_id not in SPECIES_CONFIG:
        raise ValueError(f"Unknown species: {species_id}")

    species_info = SPECIES_CONFIG[species_id]
    species_dir = RAW_DATA_ROOT / species_id
    return species_dir / species_info["gff"], species_dir / species_info["fasta"]


WINDOW_SIZE = 512
GENE_ZONE_EXTEND = 2000
AMBIGUOUS_BUFFER_MIN = 2000
AMBIGUOUS_BUFFER_MAX = 10000
DEEP_INTERGENIC_MIN_DIST = 10000
BLOCK_SIZE = 500000

TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

TASKS = {
    "Task_G_v3": {
        "name": "Task_G_v3",
        "description": "Gene body vs Background 二分类（正例来自细粒度切分）",
        "labels": ["gene_body", "background"],
        "num_labels": 2,
        "data_dir": DATA_OUTPUT_ROOT / "Task_G_v3",
    },
    "Task_S": {
        "name": "Task_S",
        "description": "基因内部结构分类 (三分类: CDS/intron/ncRNA_gene)",
        "labels": ["CDS", "intron", "ncRNA_gene"],
        "num_labels": 3,
        "data_dir": DATA_OUTPUT_ROOT / "Task_S",
    },
    "Task_S_aug": {
        "name": "Task_S_aug",
        "description": "基因内部结构分类 - 反向互补数据增强版本",
        "labels": ["CDS", "intron", "ncRNA_gene"],
        "num_labels": 3,
        "data_dir": DATA_OUTPUT_ROOT / "Task_S_aug",
    },
    "Task_N": {
        "name": "Task_N",
        "description": "ncRNA亚型分类 (五分类)",
        "labels": ["lncRNA_exon", "tRNA", "rRNA", "snRNA", "snoRNA"],
        "num_labels": 5,
        "data_dir": DATA_OUTPUT_ROOT / "Task_N",
    },
    "Task_N_aug": {
        "name": "Task_N_aug",
        "description": "ncRNA亚型分类 - 反向互补数据增强版本",
        "labels": ["lncRNA_exon", "tRNA", "rRNA", "snRNA", "snoRNA"],
        "num_labels": 5,
        "data_dir": DATA_OUTPUT_ROOT / "Task_N_aug",
    },
}

TASK_LABEL2ID = {
    "Task_G_v3": {"gene_body": 0, "background": 1},
    "Task_S": {"CDS": 0, "intron": 1, "ncRNA_gene": 2},
    "Task_S_aug": {"CDS": 0, "intron": 1, "ncRNA_gene": 2},
    "Task_N": {"lncRNA_exon": 0, "tRNA": 1, "rRNA": 2, "snRNA": 3, "snoRNA": 4},
    "Task_N_aug": {"lncRNA_exon": 0, "tRNA": 1, "rRNA": 2, "snRNA": 3, "snoRNA": 4},
}

TASK_ID2LABEL = {
    task_name: {v: k for k, v in label2id.items()}
    for task_name, label2id in TASK_LABEL2ID.items()
}


@dataclass
class TrainingConfig:
    max_seq_length: int = 512
    problem_type: str = "single_label_classification"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 2
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

    training = TrainingConfig()
    if task_name in {"Task_N", "Task_N_aug"}:
        training = TrainingConfig(
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=1e-5,
            warmup_ratio=0.15,
            eval_steps=100,
            save_steps=100,
            logging_steps=50,
        )

    return TaskConfig(
        task_name=task_name,
        num_labels=task_info["num_labels"],
        data_dir=task_info["data_dir"],
        output_dir=output_dir,
        training=training,
    )
