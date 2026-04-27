#!/bin/bash
#
# Run2: 兜底run - 高敏感度参数（10%抽样）
#
# 目的: 证明主run不会因参数过滤而漏检高相似度序列
#
# 参数:
#   -s 7.5: 最高敏感度
#   --min-seq-id 0.7: 更宽松
#   -c 0.5: 更宽松
#   --cov-mode 0: 按query覆盖度计算
#   --max-seqs 1: 只保留best hit
#

set -e

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seq_similarity

# 路径配置
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd $WORK_DIR

TEST_FASTA="data/test_sequences_sample10pct.fasta"
TRAIN_FASTA="data/train_sequences.fasta"
OUTPUT_DIR="results/run2_sensitive"
TMP_DIR="$OUTPUT_DIR/tmp"
RESULTS_TSV="$OUTPUT_DIR/alignment_results.tsv"
LOG_FILE="logs/run2_sensitive.log"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $TMP_DIR

# 记录开始时间
echo "========================================" | tee -a $LOG_FILE
echo "Run2: 兜底run - 开始执行" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 检查输入文件
echo "检查输入文件..." | tee -a $LOG_FILE
if [ ! -f "$TEST_FASTA" ]; then
    echo "错误: Test FASTA文件不存在: $TEST_FASTA" | tee -a $LOG_FILE
    exit 1
fi
if [ ! -f "$TRAIN_FASTA" ]; then
    echo "错误: Train FASTA文件不存在: $TRAIN_FASTA" | tee -a $LOG_FILE
    exit 1
fi
echo "  Test FASTA (10%抽样): $TEST_FASTA ($(wc -l < $TEST_FASTA | awk '{print $1/2}') sequences)" | tee -a $LOG_FILE
echo "  Train FASTA: $TRAIN_FASTA ($(wc -l < $TRAIN_FASTA | awk '{print $1/2}') sequences)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 运行MMseqs2
echo "运行MMseqs2 easy-search (高敏感度)..." | tee -a $LOG_FILE
echo "参数:" | tee -a $LOG_FILE
echo "  Sensitivity (-s): 7.5 (最高)" | tee -a $LOG_FILE
echo "  Min identity (--min-seq-id): 0.7 (更宽松)" | tee -a $LOG_FILE
echo "  Coverage (-c): 0.5 (更宽松)" | tee -a $LOG_FILE
echo "  Coverage mode (--cov-mode): 0 (query coverage)" | tee -a $LOG_FILE
echo "  E-value (-e): 1e-3 (更宽松)" | tee -a $LOG_FILE
echo "  Max seqs (--max-seqs): 1" | tee -a $LOG_FILE
echo "  Threads (--threads): 12" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

mmseqs easy-search \
  $TEST_FASTA \
  $TRAIN_FASTA \
  $RESULTS_TSV \
  $TMP_DIR \
  --threads 12 \
  -s 7.5 \
  --min-seq-id 0.7 \
  -c 0.5 \
  --cov-mode 0 \
  -e 1e-3 \
  --max-seqs 1 \
  --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen" \
  --search-type 3 \
  2>&1 | tee -a $LOG_FILE

# 记录结束时间
echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Run2: 兜底run - 执行完成" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 统计结果
echo "结果统计:" | tee -a $LOG_FILE
TOTAL_QUERIES=$(wc -l < $TEST_FASTA | awk '{print $1/2}')
HITS=$(wc -l < $RESULTS_TSV)
NO_HITS=$((TOTAL_QUERIES - HITS))
echo "  总query数: $TOTAL_QUERIES" | tee -a $LOG_FILE
echo "  有hit数: $HITS" | tee -a $LOG_FILE
echo "  无hit数: $NO_HITS" | tee -a $LOG_FILE
echo "  有hit比例: $(awk "BEGIN {printf \"%.2f%%\", $HITS/$TOTAL_QUERIES*100}")" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 显示前10行结果
echo "前10行结果:" | tee -a $LOG_FILE
head -10 $RESULTS_TSV | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "输出文件: $RESULTS_TSV" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE

echo "Run2完成！" | tee -a $LOG_FILE
