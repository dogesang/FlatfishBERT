"""
数据加载模块
用于加载和处理基因序列数据
"""
import os
import json
import logging
from typing import List, Dict, Iterator, Optional
from pathlib import Path
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerFast
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneSequenceDataset(IterableDataset):
    """
    基因序列数据集（用于预训练）
    使用IterableDataset以支持大规模数据流式处理
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        max_sequences: Optional[int] = None,
        balanced_sampling: bool = True,
        chunk_size: int = 512,
        min_chunk_length: int = 256,
        augment_reverse_complement: bool = True,
    ):
        """
        Args:
            data_dir: NCBI数据集目录
            tokenizer: 预训练的分词器
            max_length: 最大序列长度
            max_sequences: 最大序列数量（None表示使用全部）
            balanced_sampling: 是否从各物种均衡采样
            chunk_size: 序列切分窗口大小
            min_chunk_length: 最小chunk长度
            augment_reverse_complement: 是否使用反向互补序列进行数据增强
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_sequences = max_sequences
        self.balanced_sampling = balanced_sampling
        self.chunk_size = chunk_size
        self.min_chunk_length = min_chunk_length
        self.augment_reverse_complement = augment_reverse_complement
        
        # 查找所有物种目录
        self.species_dirs = [d for d in Path(data_dir).glob("GCF_*") if d.is_dir()]
        logger.info(f"找到 {len(self.species_dirs)} 个物种目录")
        
    def reverse_complement(self, seq: str) -> str:
        """生成反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement.get(base, base) for base in reversed(seq))
    
    def read_fasta_sequences(self) -> Iterator[str]:
        """读取FASTA文件中的序列"""
        if self.balanced_sampling and self.max_sequences:
            # 均衡采样
            num_species = len(self.species_dirs)
            per_species_limit = self.max_sequences // num_species
            logger.info(f"均衡采样：每个物种 {per_species_limit:,} 序列")
            
            for species_dir in sorted(self.species_dirs):
                species_count = 0
                logger.info(f"采样物种: {species_dir.name}")
                
                for root, dirs, files in os.walk(species_dir):
                    for file in files:
                        if file.endswith(('.fna', '.fasta')):
                            file_path = os.path.join(root, file)
                            try:
                                for record in SeqIO.parse(file_path, "fasta"):
                                    if species_count >= per_species_limit:
                                        break
                                    
                                    seq = str(record.seq).upper()
                                    # 过滤包含N的序列
                                    if 'N' not in seq:
                                        # 切分成chunks
                                        for i in range(0, len(seq), self.chunk_size):
                                            chunk = seq[i:i+self.chunk_size]
                                            if len(chunk) >= self.min_chunk_length:
                                                # 只保留ATCG
                                                clean_chunk = ''.join(c for c in chunk if c in 'ATCG')
                                                if len(clean_chunk) >= self.min_chunk_length:
                                                    yield clean_chunk
                                                    species_count += 1
                                                    
                                                    # 数据增强：添加反向互补序列
                                                    if self.augment_reverse_complement and random.random() < 0.5:
                                                        yield self.reverse_complement(clean_chunk)
                                                        species_count += 1
                                                    
                                                    if species_count >= per_species_limit:
                                                        break
                            except Exception as e:
                                logger.error(f"处理文件出错 {file_path}: {str(e)}")
                
                logger.info(f"物种 {species_dir.name} 采样完成: {species_count:,} 序列")
        else:
            # 顺序读取
            count = 0
            for species_dir in sorted(self.species_dirs):
                for root, dirs, files in os.walk(species_dir):
                    for file in files:
                        if file.endswith(('.fna', '.fasta')):
                            file_path = os.path.join(root, file)
                            try:
                                for record in SeqIO.parse(file_path, "fasta"):
                                    if self.max_sequences and count >= self.max_sequences:
                                        return
                                    
                                    seq = str(record.seq).upper()
                                    if 'N' not in seq:
                                        for i in range(0, len(seq), self.chunk_size):
                                            chunk = seq[i:i+self.chunk_size]
                                            if len(chunk) >= self.min_chunk_length:
                                                clean_chunk = ''.join(c for c in chunk if c in 'ATCG')
                                                if len(clean_chunk) >= self.min_chunk_length:
                                                    yield clean_chunk
                                                    count += 1
                                                    
                                                    if self.augment_reverse_complement and random.random() < 0.5:
                                                        yield self.reverse_complement(clean_chunk)
                                                        count += 1
                                                    
                                                    if self.max_sequences and count >= self.max_sequences:
                                                        return
                            except Exception as e:
                                logger.error(f"处理文件出错 {file_path}: {str(e)}")
    
    def __iter__(self):
        """迭代器：返回tokenized序列"""
        for sequence in self.read_fasta_sequences():
            # Tokenize序列
            encoding = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            yield {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }


class FineTuneDataset(Dataset):
    """
    微调数据集（用于基因序列分类任务）
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ):
        """
        Args:
            data_path: JSON格式的数据文件路径
            tokenizer: 预训练的分词器
            max_length: 最大序列长度
            
        数据格式示例：
        [
            {
                "sequence": "ATCGATCGATCG...",
                "label": 0,
                "description": "protein coding gene"
            },
            ...
        ]
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"加载了 {len(self.data)} 条训练数据")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item['sequence']
        label = item['label']
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def create_sample_finetune_data(output_dir: str, num_samples: int = 1000, num_labels: int = 10):
    """
    创建示例微调数据
    
    这是一个示例函数，实际使用时您需要根据自己的标注数据进行调整
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def generate_random_sequence(length: int = 512) -> str:
        """生成随机DNA序列"""
        bases = ['A', 'T', 'C', 'G']
        return ''.join(random.choice(bases) for _ in range(length))
    
    # 生成训练集
    train_data = []
    for i in range(int(num_samples * 0.8)):
        train_data.append({
            'sequence': generate_random_sequence(),
            'label': random.randint(0, num_labels - 1),
            'description': f'Sample training sequence {i}'
        })
    
    # 生成验证集
    eval_data = []
    for i in range(int(num_samples * 0.1)):
        eval_data.append({
            'sequence': generate_random_sequence(),
            'label': random.randint(0, num_labels - 1),
            'description': f'Sample eval sequence {i}'
        })
    
    # 生成测试集
    test_data = []
    for i in range(int(num_samples * 0.1)):
        test_data.append({
            'sequence': generate_random_sequence(),
            'label': random.randint(0, num_labels - 1),
            'description': f'Sample test sequence {i}'
        })
    
    # 保存
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'eval.json'), 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"创建示例数据完成:")
    logger.info(f"  训练集: {len(train_data)} 样本")
    logger.info(f"  验证集: {len(eval_data)} 样本")
    logger.info(f"  测试集: {len(test_data)} 样本")
    logger.info(f"  保存位置: {output_dir}")


if __name__ == "__main__":
    # 测试数据加载器
    from transformers import PreTrainedTokenizerFast
    
    print("=" * 70)
    print("测试数据加载器")
    print("=" * 70)
    
    # 加载分词器
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.getenv("FLATFISH_TOKENIZER_PATH", "./pretrain/tokenizer/flatfish_tokenizer")
    )
    print(f"\n✓ 分词器加载成功，词表大小: {tokenizer.vocab_size}")
    
    # 测试预训练数据集
    print("\n测试预训练数据集...")
    dataset = GeneSequenceDataset(
        data_dir=os.getenv("FLATFISH_RAW_DATA_ROOT", "./data/raw/ncbi_dataset/data"),
        tokenizer=tokenizer,
        max_length=512,
        max_sequences=10,  # 只测试10条
        balanced_sampling=False,
    )
    
    count = 0
    for batch in dataset:
        count += 1
        if count <= 2:
            print(f"\n样本 {count}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  input_ids sample: {batch['input_ids'][:20]}...")
    
    print(f"\n✓ 成功读取 {count} 个样本")
    
    # 创建示例微调数据
    print("\n创建示例微调数据...")
    create_sample_finetune_data(
        output_dir="./data/finetune",
        num_samples=100,
        num_labels=10
    )
    
    print("\n" + "=" * 70)
    print("数据加载器测试完成！")
    print("=" * 70)

