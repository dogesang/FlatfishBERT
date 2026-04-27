import os
from tokenizers import Tokenizer, trainers, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit, Sequence
from tokenizers.processors import TemplateProcessing
from Bio import SeqIO
import glob
from pathlib import Path
from tqdm import tqdm
import psutil
import logging
import time
import json
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tokenizer_training.log'),
        logging.StreamHandler()
    ]
)

class ResourceMonitor:
    def __init__(self, interval: int = 30):
        self.interval = interval
        self._stop = False
    
    def start(self):
        import threading
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def _monitor(self):
        while not self._stop:
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logging.warning(f"内存告警: {memory.percent}%")
            time.sleep(self.interval)
    
    def stop(self):
        self._stop = True
        self.thread.join()

def read_fasta_files(directory: str, max_sequences: int = None, balanced_sampling: bool = True):
    """
    优化的FASTA读取函数，支持均衡采样
    
    Args:
        directory: 数据目录
        max_sequences: 最大序列数
        balanced_sampling: 是否从多个物种均衡采样（默认True）
    """
    if balanced_sampling and max_sequences:
        # 方案A：均衡采样 - 从每个物种采样相同数量的序列
        species_dirs = [d for d in Path(directory).glob("GCF_*") if d.is_dir()]
        num_species = len(species_dirs)
        
        if num_species == 0:
            logging.warning("未找到GCF_开头的物种目录")
            return
        
        per_species_limit = max_sequences // num_species
        logging.info(f"发现 {num_species} 个物种，每个物种采样 {per_species_limit:,} 序列")
        
        for species_dir in sorted(species_dirs):
            species_count = 0
            logging.info(f"开始采样物种: {species_dir.name}")
            
            for root, dirs, files in os.walk(species_dir):
                for file in files:
                    if file.endswith(('.fna', '.fasta')):
                        file_path = os.path.join(root, file)
                        try:
                            for record in SeqIO.parse(file_path, "fasta"):
                                if species_count >= per_species_limit:
                                    break
                                seq = str(record.seq).upper()
                                if 'N' not in seq:
                                    # ✅ 关键优化：无重叠切分（步长=窗口）
                                    # 原来：步长256，窗口512（50%重叠）→ 内存×2
                                    # 现在：步长512，窗口512（无重叠）→ 内存减半！
                                    for i in range(0, len(seq), 512):
                                        chunk = seq[i:i+512]
                                        if len(chunk) >= 256:
                                            yield chunk
                                            species_count += 1
                                            if species_count >= per_species_limit:
                                                break
                        except Exception as e:
                            logging.error(f"处理文件出错 {file_path}: {str(e)}")
            
            logging.info(f"物种 {species_dir.name} 采样完成: {species_count:,} 序列")
    else:
        # 方案B：顺序读取（原始逻辑）
        count = 0
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d.startswith('GCF_')]
            if any(part.startswith('GCF_') for part in Path(root).parts):
                for file in files:
                    if file.endswith(('.fna', '.fasta')):
                        file_path = os.path.join(root, file)
                        try:
                            for record in SeqIO.parse(file_path, "fasta"):
                                if max_sequences and count >= max_sequences:
                                    return
                                seq = str(record.seq).upper()
                                if 'N' not in seq:
                                    # ✅ 同样优化：无重叠切分
                                    for i in range(0, len(seq), 512):
                                        chunk = seq[i:i+512]
                                        if len(chunk) >= 256:
                                            yield chunk
                                            count += 1
                                            if max_sequences and count >= max_sequences:
                                                return
                        except Exception as e:
                            logging.error(f"处理文件出错 {file_path}: {str(e)}")

def preprocess_and_save_to_temp_files(data_dir: str, batch_size: int = 50000, max_sequences: int = None):
    """
    预处理序列并保存到临时文件
    分批保存以避免内存溢出
    """
    temp_files = []
    current_batch = []
    total_count = 0
    batch_count = 0
    
    if max_sequences:
        logging.info(f"开始预处理序列，每批 {batch_size} 条，最大处理 {max_sequences} 条")
    else:
        logging.info(f"开始预处理序列，每批 {batch_size} 条，处理所有数据")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="tokenizer_train_")
    logging.info(f"临时文件目录: {temp_dir}")
    
    # 创建README文件说明
    readme_path = os.path.join(temp_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("分词器训练中间文件\n")
        f.write("=" * 60 + "\n")
        f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据目录: {data_dir}\n")
        f.write(f"每批序列数: {batch_size}\n")
        f.write(f"最大序列数: {'不限制' if max_sequences is None else max_sequences}\n")
        f.write(f"采样策略: 均衡采样（从每个物种采样相同数量）\n")
        f.write("\n文件说明:\n")
        f.write("- batch_*.txt: 预处理后的序列数据，每行一条序列\n")
        f.write("- 所有序列仅包含ATCG碱基，长度>=256bp\n")
        f.write("- 数据来自3个物种，均衡采样\n")
        f.write("\n如需删除这些文件:\n")
        f.write(f"  rm -rf {temp_dir}\n")
        f.write("=" * 60 + "\n")
    
    try:
        sequences = read_fasta_files(data_dir, max_sequences=max_sequences)
        
        for seq in tqdm(sequences, desc="预处理序列"):
            # 清洗序列，只保留ATCG
            clean_seq = ''.join(c for c in seq if c in 'ATCG')
            
            if len(clean_seq) >= 256:
                current_batch.append(clean_seq)
                total_count += 1
                
                # 当批次满了，保存到文件
                if len(current_batch) >= batch_size:
                    batch_file = os.path.join(temp_dir, f"batch_{batch_count}.txt")
                    with open(batch_file, 'w') as f:
                        f.write('\n'.join(current_batch))
                    temp_files.append(batch_file)
                    logging.info(f"保存批次 {batch_count}: {len(current_batch)} 条序列到 {batch_file}")
                    
                    current_batch = []
                    batch_count += 1
        
        # 保存最后一批
        if current_batch:
            batch_file = os.path.join(temp_dir, f"batch_{batch_count}.txt")
            with open(batch_file, 'w') as f:
                f.write('\n'.join(current_batch))
            temp_files.append(batch_file)
            logging.info(f"保存最后批次 {batch_count}: {len(current_batch)} 条序列")
        
        logging.info(f"预处理完成！共处理 {total_count} 条序列，保存到 {len(temp_files)} 个文件")
        
        # 更新README文件，添加处理结果
        with open(readme_path, 'a') as f:
            f.write("\n处理结果:\n")
            f.write(f"- 总序列数: {total_count}\n")
            f.write(f"- 批次文件数: {len(temp_files)}\n")
            for i, batch_file in enumerate(temp_files):
                f.write(f"  {i+1}. {os.path.basename(batch_file)}\n")
        
        return temp_files, temp_dir
        
    except Exception as e:
        logging.error(f"预处理过程出错: {str(e)}")
        raise

def batch_read_from_files(file_paths: list, batch_size: int = 10000):
    """
    从多个文件批量读取序列
    """
    for file_path in file_paths:
        logging.info(f"读取文件: {file_path}")
        batch = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        batch.append(line)
                        if len(batch) >= batch_size:
                            yield from batch
                            batch = []
                
                # 返回最后一批
                if batch:
                    yield from batch
        except Exception as e:
            logging.error(f"读取文件出错 {file_path}: {str(e)}")

def train_tokenizer_from_files(
    file_paths: list,
    vocab_size: int = 4096,
    min_frequency: int = 2
) -> Tokenizer:
    """
    从文件训练BPE分词器（使用iterator流式处理，内存友好）
    参考最初版本的成功实现
    """
    logging.info(f"开始训练分词器，vocab_size={vocab_size}, min_frequency={min_frequency}")
    
    # 检查可用内存
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    logging.info(f"当前可用内存: {available_gb:.2f} GB ({memory_info.percent}% 已使用)")
    
    if available_gb < 4:
        logging.warning(f"⚠️  可用内存不足4GB，建议减少训练数据量或调整参数")
    
    tokenizer = Tokenizer(BPE())
    
    # 设置normalizer - 清洗非ATCG字符（参考最初版本）
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(r"[^ATCG]", ""),  # 移除非ATCG字符
    ])
    
    # 设置pre_tokenizer
    tokenizer.pre_tokenizer = Sequence([
        WhitespaceSplit(),
        ByteLevel(add_prefix_space=False)
    ])
    
    # 创建训练器（关键：加回truncate_long_sequences和filter_tokens）
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"],
        show_progress=True,
        initial_alphabet=list("ATCG"),
        continuing_subword_prefix="",
        end_of_word_suffix="",
        max_token_length=20,
        truncate_long_sequences=True,  # 截断过长序列，大幅减少内存
        filter_tokens=lambda x: all(c in 'ATCG' for c in x)  # 关键！过滤非法token
    )
    
    # 关键：使用iterator方式流式处理（参考最初版本的成功实现）
    # train(files=...)会一次性加载，导致OOM
    # train_from_iterator()边读边训练，内存友好
    def read_sequences_from_files(file_paths):
        """生成器：逐行读取所有文件"""
        for file_path in file_paths:
            logging.info(f"读取文件: {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 清洗：只保留ATCG
                        clean_seq = ''.join(c for c in line if c in 'ATCG')
                        if clean_seq:
                            yield clean_seq
    
    logging.info(f"使用iterator流式处理 {len(file_paths)} 个文件")
    sequences = read_sequences_from_files(file_paths)
    
    tokenizer.train_from_iterator(
        sequences,
        trainer=trainer,
        length=None  # 未知总长度，动态处理
    )
    
    logging.info("分词器训练完成")
    return tokenizer

def main():
    """
    主函数：分阶段训练分词器
    阶段1: 预处理并保存到临时文件
    阶段2: 从临时文件批量训练分词器
    阶段3: 保存分词器和配置
    """
    data_dir = os.getenv("FLATFISH_RAW_DATA_ROOT", "./data/raw/ncbi_dataset/data")
    
    # 配置参数（均衡采样 + 无重叠优化）
    # ✅ 关键优化：去除50%重叠 → chunk数量减半 → 内存大幅降低
    # 实测：80万序列（50%重叠）✅，180万序列（50%重叠）❌
    # 新策略：150万序列（无重叠），内存消耗约等于之前75万序列
    max_sequences = 1500000  # 150万chunks（每个物种50万，约40%数据）
    batch_size = 50000       # 每个临时文件的序列数
    vocab_size = 4096        # 词表大小
    min_frequency = 10       # 最小词频（无重叠后频率会降低，保持10）
    
    monitor = ResourceMonitor()
    monitor.start()
    
    temp_files = []
    temp_dir = None
    
    try:
        # ============ 阶段1: 预处理并保存到临时文件 ============
        logging.info("=" * 60)
        logging.info("阶段1: 预处理序列并保存到临时文件")
        logging.info("=" * 60)
        
        temp_files, temp_dir = preprocess_and_save_to_temp_files(
            data_dir=data_dir,
            batch_size=batch_size,
            max_sequences=max_sequences
        )
        
        memory_info = psutil.virtual_memory()
        logging.info(f"阶段1完成，当前内存使用: {memory_info.percent}%")
        
        # ============ 阶段2: 从文件训练分词器 ============
        logging.info("=" * 60)
        logging.info("阶段2: 从临时文件训练分词器")
        logging.info("=" * 60)
        
        tokenizer = train_tokenizer_from_files(
            file_paths=temp_files,
            vocab_size=vocab_size,
            min_frequency=min_frequency
        )
        
        memory_info = psutil.virtual_memory()
        logging.info(f"阶段2完成，当前内存使用: {memory_info.percent}%")
        
        # ============ 阶段3: 保存分词器 ============
        logging.info("=" * 60)
        logging.info("阶段3: 保存分词器和配置")
        logging.info("=" * 60)
        
        output_dir = "flatfish_tokenizer"
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(f"{output_dir}/tokenizer.json")
        
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "case_sensitive": False
        }
        
        with open(f"{output_dir}/tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"分词器已保存到 {output_dir}/")
        logging.info("=" * 60)
        logging.info("训练完成！")
        logging.info("=" * 60)
        
        # 保留中间文件供检查
        if temp_dir:
            logging.info("=" * 60)
            logging.info("中间文件已保留，位于:")
            logging.info(f"  临时目录: {temp_dir}")
            logging.info(f"  文件数量: {len(temp_files)}")
            for i, f in enumerate(temp_files):
                logging.info(f"    批次 {i}: {f}")
            logging.info("=" * 60)
            logging.info("如需删除中间文件，请手动执行:")
            logging.info(f"  rm -rf {temp_dir}")
            logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}", exc_info=True)
        raise
        
    finally:
        monitor.stop()

if __name__ == "__main__":
    main() 