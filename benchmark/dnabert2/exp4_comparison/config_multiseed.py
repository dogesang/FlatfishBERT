"""
DNABERT2 资源对比实验配置
与FlatfishBert使用完全相同的超参数
"""

from dataclasses import dataclass, asdict
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# DNABERT2模型路径
DNABERT2_MODEL_PATH = Path(os.getenv("DNABERT2_MODEL_PATH", str(PROJECT_ROOT / "external_models" / "dnabert2")))

# 标签映射（必须与FlatfishBert一致）
LABEL2ID = {'Protein-coding': 0, 'Non-coding': 1}
ID2LABEL = {0: 'Protein-coding', 1: 'Non-coding'}


@dataclass
class ResourceComparisonConfig:
    """资源对比实验配置"""
    
    # 实验名称
    exp_name: str = "resource_comparison"
    seed: int = 2025  # 默认种子，可通过命令行参数覆盖
    
    # 数据路径（使用Exp4数据）
    data_dir: Path = PROJECT_ROOT / "data/experiments/v3_unified/Exp4_cross_dedup"
    
    # 输出路径（根据种子动态设置）
    @property
    def output_dir(self):
        return PROJECT_ROOT / f"finetune_output/resource_comparison/dnabert2_seed{self.seed}"
    
    @property
    def log_dir(self):
        return self.output_dir / "logs"
    
    @property
    def final_model_path(self):
        return self.output_dir / "final_model"
    
    # 数据文件
    @property
    def train_data(self):
        return self.data_dir / "train.json"
    
    @property
    def eval_data(self):
        return self.data_dir / "eval.json"
    
    @property
    def test_data(self):
        return self.data_dir / "test.json"
    
    @property
    def class_weights_path(self):
        return self.data_dir / "class_weights.json"
    
    # 模型路径
    model_path: Path = DNABERT2_MODEL_PATH
    
    def __post_init__(self):
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """训练超参数配置（与FlatfishBert完全一致）"""
    
    # 基础参数
    num_labels: int = 2
    max_seq_length: int = 512
    seed: int = 2025  # 默认种子
    
    # 训练参数（与FlatfishBert Exp4完全一致）
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 64  # 从16提升到64
    per_device_eval_batch_size: int = 64   # 从16提升到64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 评估和保存策略
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    
    # 日志
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "none"
    
    # 性能优化
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # 类别权重
    use_class_weights: bool = True
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)


def get_config(seed=2025):
    """获取配置"""
    config = ResourceComparisonConfig(seed=seed)
    config.training = TrainingConfig(seed=seed)
    return config


if __name__ == "__main__":
    # 测试配置
    config = get_config(seed=2025)
    print("资源对比实验配置:")
    print(f"  实验名称: {config.exp_name}")
    print(f"  种子: {config.seed}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  模型路径: {config.model_path}")
    print(f"\n训练配置:")
    print(f"  per_device_train_batch_size: {config.training.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size: {config.training.per_device_eval_batch_size}")
    print(f"  总batch size (2 GPU): {config.training.per_device_train_batch_size * 2}")
    print(f"  learning_rate: {config.training.learning_rate}")
    print(f"  num_train_epochs: {config.training.num_train_epochs}")
