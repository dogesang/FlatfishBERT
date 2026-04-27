"""
V3 Unified Experiments - 统一配置文件
所有实验共享的配置和常量定义
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data/processed/task2A_unconcatenated/all_samples_v3.json"

# 数据文件路径
DATA_OUTPUT_ROOT = PROJECT_ROOT / "data/experiments/v3_unified"

# 模型、日志、结果路径
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "finetune_output/v3_unified"

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer/flatfish_tokenizer"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "pretrain_output/pretrained_model"

# ============================================================
# 标签定义 (所有实验统一)
# ============================================================

# 二分类标签
PROTEIN_CODING_LABELS = ['CDS']
NON_CODING_LABELS = ['intron', 'lncRNA_exon', 'tRNA', 'rRNA', 'snRNA', 'snoRNA']

# 标签映射
LABEL2ID = {'Protein-coding': 0, 'Non-coding': 1}
ID2LABEL = {0: 'Protein-coding', 1: 'Non-coding'}

# 原始标签到二分类的映射
ORIGINAL_TO_BINARY = {}
for label in PROTEIN_CODING_LABELS:
    ORIGINAL_TO_BINARY[label] = 'Protein-coding'
for label in NON_CODING_LABELS:
    ORIGINAL_TO_BINARY[label] = 'Non-coding'

# ============================================================
# 数据划分配置 (所有实验统一)
# ============================================================

TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ============================================================
# 实验定义
# ============================================================

EXPERIMENTS = {
    'Exp1_concat': {
        'name': 'Exp1_concat',
        'description': 'CDS拼接 + 位置级去重 + 基因级划分',
        'cds_concatenate': True,
        'dedup_strategy': 'position',  # position, sequence_single, sequence_cross
        'split_strategy': 'gene_level',
    },
    'Exp2_unconcat': {
        'name': 'Exp2_unconcat',
        'description': 'CDS不拼接 + 位置级去重 + 随机分层划分',
        'cds_concatenate': False,
        'dedup_strategy': 'position',
        'split_strategy': 'random_stratified',  # 不考虑基因级
    },
    'Exp3_seq_dedup': {
        'name': 'Exp3_seq_dedup',
        'description': 'CDS不拼接 + 单物种序列级去重 + 基因级划分',
        'cds_concatenate': False,
        'dedup_strategy': 'sequence_single',
        'split_strategy': 'gene_level',
    },
    'Exp4_cross_dedup': {
        'name': 'Exp4_cross_dedup',
        'description': 'CDS不拼接 + 跨物种序列级去重 + 基因级划分',
        'cds_concatenate': False,
        'dedup_strategy': 'sequence_cross',
        'split_strategy': 'gene_level',
    },
    'Exp5_unconcat_gene': {
        'name': 'Exp5_unconcat_gene',
        'description': 'CDS不拼接 + 位置级去重 + 基因级划分',
        'cds_concatenate': False,
        'dedup_strategy': 'position',
        'split_strategy': 'gene_level',  # 与Exp2唯一区别：使用基因级划分
    },
}

# ============================================================
# 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置 - 所有实验统一

    V3.1实验分析结论 (2025-12-03):
    - V3.0配置(patience=5, eval_steps=300, threshold=0.001)导致早停过早
    - 实际只训练了0.25-0.4个epoch，F1仅达到97%
    - 需要更宽松的早停配置，确保模型充分训练

    V3.1优化策略:
    - num_train_epochs=3: 增加到3个epoch，确保充分训练
    - eval_steps=500: 减少评估频率，避免F1震荡触发早停
    - patience=10: 增加耐心值，5000步无改进才早停
    - threshold=0.0001: 收紧阈值，只有真正停滞才早停
    """

    # 模型配置
    num_labels: int = 2
    max_seq_length: int = 512
    problem_type: str = "single_label_classification"

    # 训练参数
    num_train_epochs: int = 3  # 增加到3个epoch，确保充分训练
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 梯度
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 评估和保存
    # V3.1: 减少评估频率，避免F1震荡触发早停
    evaluation_strategy: str = "steps"
    eval_steps: int = 500  # 每500步评估一次
    save_strategy: str = "steps"
    save_steps: int = 500  # 与eval_steps保持一致
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True

    # 早停
    # V3.1: 更宽松的早停配置，确保充分训练
    # patience=10: 需要10*500=5000步无改进才早停（约0.25 epoch）
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001  # 收紧阈值，只有真正停滞才早停

    # 日志
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"

    # 其他
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 16
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


# ============================================================
# 实验配置类
# ============================================================

@dataclass
class ExperimentConfig:
    """单个实验的完整配置"""

    exp_name: str
    cds_concatenate: bool
    dedup_strategy: str  # position, sequence_single, sequence_cross

    # 路径 (自动生成)
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.data_dir = DATA_OUTPUT_ROOT / self.exp_name
        self.output_dir = MODEL_OUTPUT_ROOT / self.exp_name
        self.log_dir = MODEL_OUTPUT_ROOT / self.exp_name / "logs"

        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

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
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def final_model_path(self) -> Path:
        return self.output_dir / "final_model"

    def get_description(self) -> str:
        cds_str = "CDS拼接" if self.cds_concatenate else "CDS不拼接"
        dedup_map = {
            'position': '位置级去重',
            'sequence_single': '单物种序列级去重',
            'sequence_cross': '跨物种序列级去重'
        }
        dedup_str = dedup_map.get(self.dedup_strategy, self.dedup_strategy)
        return f"{cds_str} + {dedup_str} + 基因级划分"


def get_experiment_config(exp_name: str) -> ExperimentConfig:
    """获取指定实验的配置"""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")

    exp_def = EXPERIMENTS[exp_name]
    return ExperimentConfig(
        exp_name=exp_name,
        cds_concatenate=exp_def['cds_concatenate'],
        dedup_strategy=exp_def['dedup_strategy'],
    )


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("V3 Unified Experiments - 配置信息")
    print("=" * 70)

    print(f"\n项目根目录: {PROJECT_ROOT}")
    print(f"原始数据: {RAW_DATA_PATH}")
    print(f"数据输出: {DATA_OUTPUT_ROOT}")
    print(f"模型输出: {MODEL_OUTPUT_ROOT}")

    print(f"\n标签定义:")
    print(f"  Protein-coding: {PROTEIN_CODING_LABELS}")
    print(f"  Non-coding: {NON_CODING_LABELS}")

    print(f"\n数据划分:")
    print(f"  Train: {TRAIN_RATIO*100}%")
    print(f"  Eval: {EVAL_RATIO*100}%")
    print(f"  Test: {TEST_RATIO*100}%")

    print(f"\n实验列表:")
    for exp_name, exp_def in EXPERIMENTS.items():
        config = get_experiment_config(exp_name)
        print(f"\n  {exp_name}:")
        print(f"    描述: {config.get_description()}")
        print(f"    数据目录: {config.data_dir}")
        print(f"    输出目录: {config.output_dir}")

    print("\n" + "=" * 70)
