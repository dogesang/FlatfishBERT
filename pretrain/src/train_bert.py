"""
BERT预训练脚本
支持验证集评估、测试集评估和灵活的数据量控制、从checkpoint恢复
"""
import os
import sys
import logging
import warnings
import torch
import pickle
import argparse
from datetime import datetime

# 过滤PyTorch DataParallel的无害警告
warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0')
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from torch.utils.data import Dataset, Subset
import random
import glob

from config import ModelConfig, PreTrainingConfig


log_dir = os.getenv("FLATFISH_PRETRAIN_LOG_DIR", "./pretrain_output/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_bert.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TimestampCallback(TrainerCallback):
    """添加时间戳到训练日志的回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次log时添加时间戳"""
        if logs:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 记录到logger（带时间戳）
            if state.global_step % args.logging_steps == 0:
                log_msg = f"[Step {state.global_step}/{state.max_steps}] "
                if 'loss' in logs:
                    log_msg += f"loss: {logs['loss']:.4f} "
                if 'learning_rate' in logs:
                    log_msg += f"lr: {logs['learning_rate']:.2e} "
                if 'epoch' in logs:
                    log_msg += f"epoch: {logs['epoch']:.2f}"
                logger.info(log_msg)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时添加时间戳"""
        if metrics:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[Eval at Step {state.global_step}] "
            if 'eval_loss' in metrics:
                log_msg += f"eval_loss: {metrics['eval_loss']:.4f}"
            logger.info(log_msg)
    
    def on_save(self, args, state, control, **kwargs):
        """保存checkpoint时添加时间戳"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")


class PreprocessedDataset(Dataset):
    """预处理数据集 - Map-style，支持随机访问和epoch训练"""
    
    def __init__(self, data_file: str):
        logger.info(f"加载预处理数据: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"数据文件不存在: {data_file}\n"
                f"请先运行: python src/preprocess_all_data.py"
            )
        
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        logger.info(f"✓ 数据加载完成")
        logger.info(f"  序列数: {len(self.data):,}")
        logger.info(f"  内存占用: {sys.getsizeof(self.data) / 1024 / 1024:.1f} MB")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_subset_dataset(full_dataset: Dataset, subset_size: int = None, 
                          subset_ratio: float = None, seed: int = 42) -> Dataset:
    """
    创建数据子集
    
    Args:
        full_dataset: 完整数据集
        subset_size: 子集大小（绝对值）
        subset_ratio: 子集比例（0-1之间）
        seed: 随机种子
    
    Returns:
        数据子集
    """
    total_size = len(full_dataset)
    
    if subset_size is None and subset_ratio is None:
        # 使用全部数据
        return full_dataset
    
    # 确定子集大小
    if subset_ratio is not None:
        subset_size = int(total_size * subset_ratio)
    
    subset_size = min(subset_size, total_size)
    
    # 随机采样索引
    random.seed(seed)
    indices = random.sample(range(total_size), subset_size)
    indices.sort()  # 保持顺序，有利于缓存
    
    logger.info(f"✓ 数据子集创建")
    logger.info(f"  原始大小: {total_size:,}")
    logger.info(f"  子集大小: {subset_size:,} ({subset_size/total_size*100:.1f}%)")
    
    return Subset(full_dataset, indices)


def create_model(model_config: ModelConfig) -> BertForMaskedLM:
    """创建BERT模型"""
    logger.info("创建BERT模型...")
    
    config = BertConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=model_config.attention_probs_dropout_prob,
        max_position_embeddings=model_config.max_position_embeddings,
        type_vocab_size=model_config.type_vocab_size,
        initializer_range=model_config.initializer_range,
        layer_norm_eps=model_config.layer_norm_eps,
    )
    
    model = BertForMaskedLM(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✓ 模型创建完成")
    logger.info(f"  总参数量: {num_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  模型大小: {num_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    return model


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BERT预训练")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定checkpoint恢复训练 (路径或'auto'自动查找最新)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("BERT预训练 - FlatfishBert")
    logger.info("=" * 80)
    
    # 加载配置
    model_config = ModelConfig()
    train_config = PreTrainingConfig()
    
    # 处理checkpoint恢复
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == 'auto':
            # 自动查找最新checkpoint
            checkpoints = glob.glob(os.path.join(train_config.output_dir, "checkpoint-*"))
            if checkpoints:
                resume_checkpoint = max(checkpoints, key=os.path.getmtime)
                logger.info(f"🔄 从最新checkpoint恢复: {resume_checkpoint}")
            else:
                logger.warning("⚠️  未找到checkpoint，从头开始训练")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            if os.path.exists(resume_checkpoint):
                logger.info(f"🔄 从checkpoint恢复: {resume_checkpoint}")
            else:
                logger.error(f"❌ Checkpoint不存在: {resume_checkpoint}")
                sys.exit(1)
    
    # 设置随机种子
    set_seed(train_config.seed)
    logger.info(f"随机种子: {train_config.seed}")
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info(f"计算设备: {device}")
    if torch.cuda.is_available():
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("⚠ 未检测到GPU，使用CPU训练会非常慢")
    
    # 创建输出目录
    os.makedirs(train_config.output_dir, exist_ok=True)
    os.makedirs(train_config.logging_dir, exist_ok=True)
    
    # 1. 加载分词器
    logger.info("\n" + "=" * 80)
    logger.info("步骤 1/5: 加载分词器")
    logger.info("=" * 80)
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config.tokenizer_path)
        logger.info(f"✓ 分词器加载成功")
        logger.info(f"  路径: {model_config.tokenizer_path}")
        logger.info(f"  词表大小: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"✗ 加载分词器失败: {str(e)}")
        return
    
    # 2. 加载预处理数据
    logger.info("\n" + "=" * 80)
    logger.info("步骤 2/5: 加载预处理数据")
    logger.info("=" * 80)
    
    train_file = os.getenv("FLATFISH_PRETRAIN_TRAIN", "./data/preprocessed/train_data.pkl")
    val_file = os.getenv("FLATFISH_PRETRAIN_VAL", "./data/preprocessed/val_data.pkl")
    
    try:
        train_dataset_full = PreprocessedDataset(train_file)
        
        # ⭐ 灵活控制训练数据量
        # 选项1: 使用全部数据（完整训练）
        train_dataset = train_dataset_full
        
        # 选项2: 使用固定数量（快速测试）
        # train_dataset = create_subset_dataset(train_dataset_full, subset_size=10000)
        
        # 选项3: 使用比例（快速测试，例如10%）
        # train_dataset = create_subset_dataset(train_dataset_full, subset_ratio=0.1)
        
        logger.info(f"✓ 训练数据集加载完成: {len(train_dataset):,} 序列")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"✗ 加载数据失败: {str(e)}")
        return
    
    # 加载验证集
    try:
        val_dataset = PreprocessedDataset(val_file)
        logger.info(f"✓ 验证数据集加载完成: {len(val_dataset):,} 序列")
    except Exception as e:
        logger.warning(f"⚠ 未找到验证集: {str(e)}")
        val_dataset = None
    
    # 数据收集器 - MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=model_config.mlm_probability,
    )
    logger.info(f"✓ 数据收集器配置")
    logger.info(f"  MLM遮蔽概率: {model_config.mlm_probability}")
    
    # 3. 创建模型
    logger.info("\n" + "=" * 80)
    logger.info("步骤 3/5: 创建BERT模型")
    logger.info("=" * 80)
    
    model = create_model(model_config)
    
    # 4. 配置训练
    logger.info("\n" + "=" * 80)
    logger.info("步骤 4/5: 配置训练参数")
    logger.info("=" * 80)
    
    # 计算训练步数
    num_samples = len(train_dataset)
    batch_size_per_device = train_config.per_device_train_batch_size
    gradient_accumulation = train_config.gradient_accumulation_steps
    num_gpus = max(1, gpu_count)
    
    effective_batch_size = batch_size_per_device * gradient_accumulation * num_gpus
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * train_config.num_train_epochs
    
    logger.info(f"训练配置:")
    logger.info(f"  数据集大小: {num_samples:,} 序列")
    logger.info(f"  验证集大小: {len(val_dataset):,} 序列" if val_dataset else "  验证集: 无")
    logger.info(f"  训练轮数 (epochs): {train_config.num_train_epochs}")
    logger.info(f"  GPU数量: {num_gpus}")
    logger.info(f"  每设备batch size: {batch_size_per_device}")
    logger.info(f"  梯度累积步数: {gradient_accumulation}")
    logger.info(f"  有效batch size: {effective_batch_size}")
    logger.info(f"  每epoch步数: {steps_per_epoch:,}")
    logger.info(f"  总训练步数: {total_steps:,}")
    logger.info(f"  学习率: {train_config.learning_rate}")
    logger.info(f"  Warmup步数: {train_config.warmup_steps}")
    logger.info(f"  FP16混合精度: {train_config.fp16 and torch.cuda.is_available()}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        overwrite_output_dir=True,
        
        # Epoch训练
        num_train_epochs=train_config.num_train_epochs,
        
        # Batch配置
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        
        # 优化器
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        adam_beta1=train_config.adam_beta1,
        adam_beta2=train_config.adam_beta2,
        adam_epsilon=train_config.adam_epsilon,
        max_grad_norm=train_config.max_grad_norm,
        
        # Warmup
        warmup_steps=train_config.warmup_steps,
        
        # 评估策略
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=train_config.eval_steps if val_dataset else None,
        
        # 日志和保存
        logging_dir=train_config.logging_dir,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        
        # 其他
        seed=train_config.seed,
        fp16=train_config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=train_config.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # 报告
        report_to=["tensorboard"],
        
        # 移除未使用的列
        remove_unused_columns=False,
        
        # 最佳模型
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[TimestampCallback()],  # 添加时间戳回调
    )
    
    # 预估训练时间
    if gpu_count >= 2:
        estimated_seconds_per_step = 0.4  # 双A40
    elif gpu_count == 1:
        estimated_seconds_per_step = 0.7  # 单A40
    else:
        estimated_seconds_per_step = 5.0  # CPU
    
    estimated_hours = (total_steps * estimated_seconds_per_step) / 3600
    
    logger.info(f"\n预估训练时间: {estimated_hours:.1f} 小时 ({estimated_hours/24:.1f} 天)")
    
    # 开始训练
    logger.info("\n" + "=" * 80)
    logger.info("步骤 5/5: 开始训练")
    logger.info("=" * 80)
    logger.info(f"每个epoch将遍历 {num_samples:,} 个序列")
    logger.info(f"共训练 {train_config.num_train_epochs} 个epochs")
    if val_dataset:
        logger.info(f"每 {train_config.eval_steps} 步在验证集上评估")
    logger.info("")
    logger.info("监控训练进度：")
    logger.info(f"  - 查看日志: tail -f pretrain_output/logs/train_bert.log")
    logger.info(f"  - TensorBoard: tensorboard --logdir {train_config.logging_dir}")
    logger.info(f"  - GPU监控: watch -n 1 nvidia-smi")
    logger.info("=" * 80)
    
    try:
        # 开始训练（支持从checkpoint恢复）
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # 保存最终模型
        logger.info("\n" + "=" * 80)
        logger.info("训练完成！保存模型...")
        logger.info("=" * 80)
        
        trainer.save_model(train_config.output_dir)
        tokenizer.save_pretrained(train_config.output_dir)
        
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"✓ 模型已保存到: {train_config.output_dir}")
        logger.info(f"\n训练指标:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # 在测试集上评估（如果有）
        test_file = os.getenv("FLATFISH_PRETRAIN_TEST", "./data/preprocessed/test_data.pkl")
        if os.path.exists(test_file):
            try:
                logger.info("\n" + "=" * 80)
                logger.info("在测试集上评估")
                logger.info("=" * 80)
                test_dataset = PreprocessedDataset(test_file)
                test_results = trainer.evaluate(test_dataset)
                trainer.log_metrics("test", test_results)
                trainer.save_metrics("test", test_results)
                logger.info(f"测试集结果:")
                for key, value in test_results.items():
                    logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.warning(f"测试集评估失败: {str(e)}")
        
        logger.info("\n" + "=" * 80)
        logger.info("预训练完成！")
        logger.info("=" * 80)
        logger.info(f"\n下一步：微调模型")
        logger.info(f"  1. 准备标注数据 (data/finetune/train.json)")
        logger.info(f"  2. 修改config.py中的num_labels")
        logger.info(f"  3. 运行: python src/finetune.py")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n训练被用户中断")
        logger.info("保存当前模型...")
        interrupted_dir = os.path.join(train_config.output_dir, "interrupted")
        trainer.save_model(interrupted_dir)
        tokenizer.save_pretrained(interrupted_dir)
        logger.info(f"✓ 模型已保存到: {interrupted_dir}")
        logger.info("可以稍后从此checkpoint恢复训练")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

