"""
工具函数模块
包含模型评估、可视化、推理等辅助功能
"""
import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from transformers import (
    BertForSequenceClassification,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
)
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估工具类"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # 加载模型和分词器
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        
        # 尝试加载分类模型或MLM模型
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.task_type = "classification"
        except:
            self.model = BertForMaskedLM.from_pretrained(model_path)
            self.task_type = "mlm"
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载成功: {self.task_type}")
    
    def predict_single(self, sequence: str) -> Dict:
        """
        预测单个序列
        
        Args:
            sequence: DNA序列
            
        Returns:
            预测结果字典
        """
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.task_type == "classification":
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_label].item()
            
            return {
                'predicted_label': pred_label,
                'confidence': confidence,
                'all_probabilities': probs[0].cpu().numpy().tolist(),
            }
        else:
            # MLM任务
            return {
                'logits': outputs.logits[0].cpu().numpy(),
            }
    
    def batch_predict(self, sequences: List[str], batch_size: int = 32) -> List[Dict]:
        """
        批量预测
        
        Args:
            sequences: DNA序列列表
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if self.task_type == "classification":
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
                confidences = probs.max(dim=-1).values.cpu().numpy()
                
                for j, (label, conf) in enumerate(zip(pred_labels, confidences)):
                    results.append({
                        'sequence_idx': i + j,
                        'predicted_label': int(label),
                        'confidence': float(conf),
                    })
        
        return results
    
    def evaluate_dataset(
        self, 
        data_path: str, 
        output_dir: str = None
    ) -> Dict:
        """
        评估整个数据集
        
        Args:
            data_path: 数据文件路径
            output_dir: 输出目录（保存结果和图表）
            
        Returns:
            评估指标字典
        """
        if self.task_type != "classification":
            logger.warning("只支持分类任务的数据集评估")
            return {}
        
        # 加载数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        sequences = [item['sequence'] for item in data]
        true_labels = [item['label'] for item in data]
        
        # 批量预测
        logger.info(f"评估 {len(sequences)} 个样本...")
        predictions = self.batch_predict(sequences, batch_size=32)
        pred_labels = [p['predicted_label'] for p in predictions]
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        logger.info(f"评估结果: {metrics}")
        
        # 生成混淆矩阵和报告
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存指标
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # 混淆矩阵
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 分类报告
            report = classification_report(true_labels, pred_labels)
            with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
            
            logger.info(f"评估结果已保存到: {output_dir}")
        
        return metrics


def analyze_sequence_attention(
    model_path: str,
    sequence: str,
    layer: int = -1,
    output_path: str = None
):
    """
    分析序列的注意力权重
    
    Args:
        model_path: 模型路径
        sequence: DNA序列
        layer: 要可视化的层（-1表示最后一层）
        output_path: 保存图片的路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        output_attentions=True
    )
    model.to(device)
    model.eval()
    
    # Tokenize
    inputs = tokenizer(
        sequence,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取注意力权重
    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)
    attention = attentions[layer][0].mean(dim=0).cpu().numpy()  # 平均所有注意力头
    
    # 可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention[:50, :50], cmap='viridis')  # 只显示前50个token
    plt.title(f'Attention Weights - Layer {layer}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"注意力图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(
    model_paths: List[str],
    test_data_path: str,
    output_dir: str
):
    """
    比较多个模型的性能
    
    Args:
        model_paths: 模型路径列表
        test_data_path: 测试数据路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for model_path in model_paths:
        logger.info(f"评估模型: {model_path}")
        
        evaluator = ModelEvaluator(model_path)
        metrics = evaluator.evaluate_dataset(
            test_data_path,
            output_dir=os.path.join(output_dir, os.path.basename(model_path))
        )
        
        metrics['model_name'] = os.path.basename(model_path)
        results.append(metrics)
    
    # 创建对比表格
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        df.plot(x='model_name', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"模型对比结果已保存到: {output_dir}")
    logger.info(f"\n{df.to_string()}")


def extract_embeddings(
    model_path: str,
    sequences: List[str],
    output_path: str = None
) -> np.ndarray:
    """
    提取序列的BERT embeddings
    
    Args:
        model_path: 模型路径
        sequences: 序列列表
        output_path: 保存embeddings的路径（npy格式）
        
    Returns:
        embeddings数组 (n_sequences, hidden_size)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
    except:
        model = BertForMaskedLM.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    embeddings = []
    
    logger.info(f"提取 {len(sequences)} 个序列的embeddings...")
    
    for i in range(0, len(sequences), 32):
        batch = sequences[i:i+32]
        
        inputs = tokenizer(
            batch,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.bert(**inputs, output_hidden_states=True)
            # 使用[CLS] token的hidden state作为序列表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    if output_path:
        np.save(output_path, embeddings)
        logger.info(f"Embeddings已保存到: {output_path}")
    
    return embeddings


if __name__ == "__main__":
    print("=" * 70)
    print("FlatfishBert 工具函数模块")
    print("=" * 70)
    print("\n可用功能:")
    print("1. ModelEvaluator - 模型评估")
    print("2. analyze_sequence_attention - 注意力可视化")
    print("3. compare_models - 模型对比")
    print("4. extract_embeddings - 提取序列embeddings")
    print("\n使用示例:")
    print("  from utils import ModelEvaluator")
    print("  evaluator = ModelEvaluator('path/to/model')")
    print("  result = evaluator.predict_single('ATCGATCG...')")
    print("=" * 70)

