"""
DNABERT2 跨物种泛化实验配置文件
基于 FlatfishBert 泛化测试，适配 DNABERT2 特殊处理
"""

from pathlib import Path

# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERALIZATION_ROOT = PROJECT_ROOT / "generalization_test_dnabert2"

# 复用 FlatfishBert 的测试数据
FLATFISH_DATA_DIR = PROJECT_ROOT / "generalization_test/data"

# 输出路径
RESULTS_OUTPUT_DIR = GENERALIZATION_ROOT / "results"
LOGS_DIR = GENERALIZATION_ROOT / "logs"

# DNABERT2 模型路径（仅作说明，不随仓库分发）
DNABERT2_MODEL_PATH = PROJECT_ROOT / "finetune_output/dnabert2_comparison/final_model"

# ============================================================
# 物种配置（与 FlatfishBert 一致）
# ============================================================

SPECIES_CONFIGS = {
    'hippoglossus_hippoglossus': {
        'name': 'Hippoglossus_hippoglossus',
        'common_name': '大西洋庸鲽',
        'scientific_name': 'Hippoglossus hippoglossus',
        'family': '鲽科 (Pleuronectidae)',
        'relationship': '较远',
        'test_file': FLATFISH_DATA_DIR / 'hippoglossus_hippoglossus_test.json',
    },
    'solea_senegalensis': {
        'name': 'Solea_senegalensis',
        'common_name': '塞内加尔鳎',
        'scientific_name': 'Solea senegalensis',
        'family': '鳎科 (Soleidae)',
        'relationship': '较近',
        'test_file': FLATFISH_DATA_DIR / 'solea_senegalensis_test.json',
    },
    'solea_solea': {
        'name': 'Solea_solea',
        'common_name': '欧洲鳎',
        'scientific_name': 'Solea solea',
        'family': '鳎科 (Soleidae)',
        'relationship': '较近',
        'test_file': FLATFISH_DATA_DIR / 'solea_solea_test.json',
    },
}

# ============================================================
# 标签定义（与 Exp4 一致）
# ============================================================

LABEL2ID = {'Protein-coding': 0, 'Non-coding': 1}
ID2LABEL = {0: 'Protein-coding', 1: 'Non-coding'}

# ============================================================
# 推理配置
# ============================================================

INFERENCE_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512

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

# ============================================================
# 基准结果（用于对比说明）
# ============================================================

FLATFISH_BASELINE = {
    'original_test': {
        'f1_macro': 0.9757,
        'accuracy': 0.9757,
    },
    'generalization': {
        'hippoglossus_hippoglossus': {
            'f1_macro': 0.9683,
            'accuracy': 0.9683,
            'samples': 529406,
        },
        'solea_senegalensis': {
            'f1_macro': 0.9791,
            'accuracy': 0.9791,
            'samples': 519477,
        },
        'solea_solea': {
            'f1_macro': 0.9768,
            'accuracy': 0.9769,
            'samples': 519844,
        },
        'average': {
            'f1_macro': 0.9747,
            'accuracy': 0.9747,
            'samples': 1568727,
        }
    }
}

DNABERT2_BASELINE = {
    'original_test': {
        'f1_macro': 0.9872,
        'accuracy': 0.9872,
    }
}
