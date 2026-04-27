"""
BERT模型配置文件
用于基因序列的BERT训练
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PRETRAIN_OUTPUT_DIR = PROJECT_ROOT / "pretrain_output" / "pretrained_model"
DEFAULT_PRETRAIN_LOG_DIR = PROJECT_ROOT / "pretrain_output" / "logs"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "flatfish_tokenizer"
DEFAULT_FINETUNE_DATA_DIR = PROJECT_ROOT / "data" / "finetune"


@dataclass
class ModelConfig:
    """BERT模型架构配置"""
    tokenizer_path: str = os.getenv("FLATFISH_TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH))

    vocab_size: int = 4096
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    mlm_probability: float = 0.15


@dataclass
class PreTrainingConfig:
    """预训练配置"""
    data_dir: str = os.getenv("FLATFISH_RAW_DATA_ROOT", str(DEFAULT_RAW_DATA_DIR))
    output_dir: str = os.getenv("FLATFISH_PRETRAIN_OUTPUT", str(DEFAULT_PRETRAIN_OUTPUT_DIR))
    logging_dir: str = os.getenv("FLATFISH_PRETRAIN_LOG_DIR", str(DEFAULT_PRETRAIN_LOG_DIR))

    num_train_epochs: int = 10
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    max_seq_length: int = 512

    max_sequences: Optional[int] = None
    balanced_sampling: bool = False

    logging_steps: int = 50
    save_steps: int = 1000
    save_total_limit: int = 3
    eval_steps: int = 1000
    evaluation_strategy: str = "steps"

    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 10


@dataclass
class FineTuningConfig:
    """微调配置（用于基因序列分类任务）"""
    train_data_path: str = os.getenv("FLATFISH_FINETUNE_TRAIN", str(DEFAULT_FINETUNE_DATA_DIR / "train.json"))
    eval_data_path: str = os.getenv("FLATFISH_FINETUNE_EVAL", str(DEFAULT_FINETUNE_DATA_DIR / "eval.json"))
    test_data_path: str = os.getenv("FLATFISH_FINETUNE_TEST", str(DEFAULT_FINETUNE_DATA_DIR / "test.json"))

    pretrained_model_path: str = os.getenv("FLATFISH_PRETRAINED_MODEL", str(DEFAULT_PRETRAIN_OUTPUT_DIR))

    num_labels: int = 8
    problem_type: str = "single_label_classification"
    use_class_weights: bool = True

    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    max_seq_length: int = 512

    logging_steps: int = 100
    save_steps: int = 1000
    save_total_limit: int = 2
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"

    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
