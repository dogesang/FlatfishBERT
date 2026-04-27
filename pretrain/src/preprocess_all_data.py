"""
预处理所有基因序列数据
将FASTA文件转换为tokenized数据集，支持数据集划分
一次性处理所有数据并自动划分train/val/test
"""
import os
import sys
import torch
import pickle
import logging
import random
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from Bio import SeqIO
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "pretrain_output" / "logs"

log_dir = Path(os.getenv("FLATFISH_PRETRAIN_LOG_DIR", str(DEFAULT_LOG_DIR)))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "preprocess.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器 - 一次性处理所有数据并自动划分数据集"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        output_dir: str,
        max_length: int = 512,
        balanced_sampling: bool = False,
        augment_reverse_complement: bool = False,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        random_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer_path = tokenizer_path
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.balanced_sampling = balanced_sampling
        self.augment_reverse_complement = augment_reverse_complement
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.chunk_size = max_length
        self.min_chunk_length = 100
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"加载分词器: {tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"✓ 分词器加载成功，词表大小: {self.tokenizer.vocab_size}")
        
        self.species_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('GCF_')]
        logger.info(f"✓ 找到 {len(self.species_dirs)} 个物种目录")
        for d in self.species_dirs:
            logger.info(f"  - {d.name}")
    
    def reverse_complement(self, seq: str) -> str:
        """生成反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement.get(base, base) for base in reversed(seq))
    
    def split_data(self, data: List[Dict], seed: int = None) -> Tuple[List, List, List]:
        """划分训练集、验证集、测试集"""
        if seed is not None:
            random.seed(seed)
        
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        total = len(shuffled_data)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        logger.info("数据集划分:")
        logger.info(f"  训练集: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
        logger.info(f"  验证集: {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
        logger.info(f"  测试集: {len(test_data):,} ({len(test_data)/total*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def process_all_sequences(self) -> Tuple[List[Dict], Dict]:
        """处理所有序列并返回所有数据（未划分）"""
        logger.info("=" * 70)
        logger.info("开始处理所有序列")
        logger.info("=" * 70)
        
        all_data = []
        stats = {
            'total_sequences': 0,
            'augmented_sequences': 0,
            'species_counts': {},
        }
        
        if self.balanced_sampling:
            logger.info("使用均衡采样模式...")
            pass
        else:
            logger.info("顺序处理所有序列...")
            for species_dir in self.species_dirs:
                species_count = 0
                logger.info(f"处理物种: {species_dir.name}")
                
                for root, dirs, files in os.walk(species_dir):
                    for file in files:
                        if file.endswith(('.fna', '.fasta')):
                            file_path = os.path.join(root, file)
                            try:
                                for record in tqdm(SeqIO.parse(file_path, "fasta"), desc=os.path.basename(file), leave=False):
                                    seq = str(record.seq).upper()
                                    if 'N' not in seq:
                                        for i in range(0, len(seq), self.chunk_size):
                                            chunk = seq[i:i+self.chunk_size]
                                            if len(chunk) >= self.min_chunk_length:
                                                clean_chunk = ''.join(c for c in chunk if c in 'ATCG')
                                                if len(clean_chunk) >= self.min_chunk_length:
                                                    encoding = self.tokenizer(
                                                        clean_chunk,
                                                        max_length=self.max_length,
                                                        padding='max_length',
                                                        truncation=True,
                                                        return_tensors='pt'
                                                    )
                                                    
                                                    all_data.append({
                                                        'input_ids': encoding['input_ids'].squeeze(0),
                                                        'attention_mask': encoding['attention_mask'].squeeze(0),
                                                    })
                                                    species_count += 1
                                                    
                                                    if self.augment_reverse_complement and random.random() < 0.5:
                                                        rc_seq = self.reverse_complement(clean_chunk)
                                                        rc_encoding = self.tokenizer(
                                                            rc_seq,
                                                            max_length=self.max_length,
                                                            padding='max_length',
                                                            truncation=True,
                                                            return_tensors='pt'
                                                        )
                                                        all_data.append({
                                                            'input_ids': rc_encoding['input_ids'].squeeze(0),
                                                            'attention_mask': rc_encoding['attention_mask'].squeeze(0),
                                                        })
                                                        stats['augmented_sequences'] += 1
                            except Exception as e:
                                logger.error(f"处理文件出错 {file_path}: {str(e)}")
                
                stats['species_counts'][species_dir.name] = species_count
                stats['total_sequences'] += species_count
                logger.info(f"✓ {species_dir.name}: {species_count:,} 序列")
        
        logger.info("=" * 70)
        logger.info(f"✓ 总序列数: {len(all_data):,}")
        
        return all_data, stats
    
    def save_datasets(self, train_data: List, val_data: List, test_data: List, stats: Dict):
        """保存划分后的数据集"""
        logger.info("=" * 70)
        logger.info("保存数据集")
        logger.info("=" * 70)
        
        train_file = self.output_dir / "train_data.pkl"
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)
        logger.info(f"✓ 训练集已保存: {train_file}")
        logger.info(f"  大小: {len(train_data):,} 序列")
        
        val_file = self.output_dir / "val_data.pkl"
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
        logger.info(f"✓ 验证集已保存: {val_file}")
        logger.info(f"  大小: {len(val_data):,} 序列")
        
        test_file = self.output_dir / "test_data.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        logger.info(f"✓ 测试集已保存: {test_file}")
        logger.info(f"  大小: {len(test_data):,} 序列")
        
        stats_file = self.output_dir / "dataset_stats.txt"
        total_size = len(train_data) + len(val_data) + len(test_data)
        with open(stats_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("数据集统计信息\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"总序列数: {stats['total_sequences']:,}\n")
            if stats['augmented_sequences'] > 0:
                f.write(f"增强序列数: {stats['augmented_sequences']:,}\n")
            f.write(f"实际序列数: {total_size:,}\n\n")
            f.write(f"训练集: {len(train_data):,} ({len(train_data)/total_size*100:.1f}%)\n")
            f.write(f"验证集: {len(val_data):,} ({len(val_data)/total_size*100:.1f}%)\n")
            f.write(f"测试集: {len(test_data):,} ({len(test_data)/total_size*100:.1f}%)\n\n")
            f.write("各物种序列数:\n")
            for species, count in stats['species_counts'].items():
                f.write(f"  {species}: {count:,}\n")
            f.write("\n" + "=" * 70 + "\n")
        
        logger.info(f"✓ 统计信息已保存到: {stats_file}")
        return train_file, val_file, test_file, stats_file
    
    def run(self):
        """运行完整预处理流程"""
        logger.info("=" * 70)
        logger.info("FlatfishBert 数据预处理 v2")
        logger.info("=" * 70)
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"序列长度: {self.max_length}")
        logger.info(f"均衡采样: {self.balanced_sampling}")
        logger.info(f"数据增强: {self.augment_reverse_complement}")
        logger.info(f"数据集划分: 训练{self.train_ratio:.1%} / 验证{self.val_ratio:.1%} / 测试{self.test_ratio:.1%}")
        logger.info(f"随机种子: {self.random_seed}")
        
        all_data, stats = self.process_all_sequences()
        train_data, val_data, test_data = self.split_data(all_data, seed=self.random_seed)
        self.save_datasets(train_data, val_data, test_data, stats)
        
        logger.info("=" * 70)
        logger.info("✓ 数据预处理完成")
        logger.info("=" * 70)


def main():
    from config import PreTrainingConfig, ModelConfig

    model_config = ModelConfig()
    pretrain_config = PreTrainingConfig()

    preprocessor = DataPreprocessor(
        data_dir=pretrain_config.data_dir,
        tokenizer_path=model_config.tokenizer_path,
        output_dir=PROJECT_ROOT / "data" / "preprocessed",
        max_length=model_config.max_position_embeddings,
        balanced_sampling=pretrain_config.balanced_sampling,
        augment_reverse_complement=False,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        random_seed=pretrain_config.seed,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
