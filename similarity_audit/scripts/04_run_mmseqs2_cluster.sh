#!/bin/bash
#
# Run3: 近重复簇检测 - 针对高风险query
#
# 目的: 对高风险query，检测是否存在多个近似匹配（近重复簇）
#
# 输入: 从Run1结果中提取的高风险query列表（Identity≥95% & Qcov≥80%）
#
# 参数:
#   -s 5.0: 标准敏感度
#   --min-seq-id 0.95: 只关注高相似度
#   -c 0.8: 更严格的覆盖度
#   --cov-mode 0: 按query覆盖度计算
#   --max-seqs 10: 返回top-10匹配
#

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seq_similarity

# 路径配置
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd $WORK_DIR

HIGH_RISK_FASTA="data/high_risk_queries.fasta"
TRAIN_FASTA="data/train_sequences.fasta"
OUTPUT_DIR="results/run3_cluster"
TMP_DIR="$OUTPUT_DIR/tmp"
RESULTS_TSV="$OUTPUT_DIR/alignment_results.tsv"
LOG_FILE="logs/run3_cluster.log"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $TMP_DIR

# 记录开始时间
echo "========================================" | tee -a $LOG_FILE
echo "Run3: 近重复簇检测 - 开始执行" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 检查输入文件
echo "检查输入文件..." | tee -a $LOG_FILE
if [ ! -f "$HIGH_RISK_FASTA" ]; then
    echo "警告: 高风险query FASTA文件不存在: $HIGH_RISK_FASTA" | tee -a $LOG_FILE
    echo "请先运行结果分析脚本生成高风险query列表" | tee -a $LOG_FILE
    echo "跳过Run3" | tee -a $LOG_FILE
    exit 0
fi
if [ ! -f "$TRAIN_FASTA" ]; then
    echo "错误: Train FASTA文件不存在: $TRAIN_FASTA" | tee -a $LOG_FILE
    exit 1
fi

HIGH_RISK_COUNT=$(wc -l < $HIGH_RISK_FASTA | awk '{print $1/2}')
echo "  高风险query FASTA: $HIGH_RISK_FASTA ($HIGH_RISK_COUNT sequences)" | tee -a $LOG_FILE
echo "  Train FASTA: $TRAIN_FASTA ($(wc -l < $TRAIN_FASTA | awk '{print $1/2}') sequences)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

if [ "$HIGH_RISK_COUNT" -eq 0 ]; then
    echo "没有高风险query，跳过Run3" | tee -a $LOG_FILE
    exit 0
fi

# 运行MMseqs2
echo "运行MMseqs2 easy-search (top-10匹配)..." | tee -a $LOG_FILE
echo "参数:" | tee -a $LOG_FILE
echo "  Sensitivity (-s): 5.0" | tee -a $LOG_FILE
echo "  Min identity (--min-seq-id): 0.95 (高相似度)" | tee -a $LOG_FILE
echo "  Coverage (-c): 0.8 (严格)" | tee -a $LOG_FILE
echo "  Coverage mode (--cov-mode): 0 (query coverage)" | tee -a $LOG_FILE
echo "  E-value (-e): 1e-5" | tee -a $LOG_FILE
echo "  Max seqs (--max-seqs): 10 (top-10)" | tee -a $LOG_FILE
echo "  Threads (--threads): 12" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

mmseqs easy-search \
  $HIGH_RISK_FASTA \
  $TRAIN_FASTA \
  $RESULTS_TSV \
  $TMP_DIR \
  --threads 12 \
  -s 5.0 \
  --min-seq-id 0.95 \
  -c 0.8 \
  --cov-mode 0 \
  -e 1e-5 \
  --max-seqs 10 \
  --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen" \
  --search-type 3 \
  2>&1 | tee -a $LOG_FILE

# 记录结束时间
echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Run3: 近重复簇检测 - 执行完成" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 统计结果
echo "结果统计:" | tee -a $LOG_FILE
TOTAL_QUERIES=$HIGH_RISK_COUNT
TOTAL_HITS=$(wc -l < $RESULTS_TSV)
UNIQUE_QUERIES=$(cut -f1 $RESULTS_TSV | sort -u | wc -l)
AVG_HITS_PER_QUERY=$(awk "BEGIN {printf \"%.2f\", $TOTAL_HITS/$UNIQUE_QUERIES}")
echo "  总query数: $TOTAL_QUERIES" | tee -a $LOG_FILE
echo "  总匹配数: $TOTAL_HITS" | tee -a $LOG_FILE
echo "  有匹配的query数: $UNIQUE_QUERIES" | tee -a $LOG_FILE
echo "  平均每个query的匹配数: $AVG_HITS_PER_QUERY" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 显示前20行结果
echo "前20行结果:" | tee -a $LOG_FILE
head -20 $RESULTS_TSV | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "输出文件: $RESULTS_TSV" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE

echo "Run3完成！" | tee -a $LOG_FILE
