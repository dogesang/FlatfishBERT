"""
Configuration for MMseqs2 similarity audit (Section 3.3).
"""

from pathlib import Path

AUDIT_ROOT = Path(__file__).resolve().parent
WORK_DIR = AUDIT_ROOT

DATA_DIR = WORK_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
TEST_METADATA = METADATA_DIR / "test_metadata.tsv"
TRAIN_METADATA = METADATA_DIR / "train_metadata.tsv"
SAMPLE_IDS = METADATA_DIR / "sample_ids.txt"

RESULTS_DIR = WORK_DIR / "results"
RUN1_DIR = RESULTS_DIR / "run1_main"
RUN2_DIR = RESULTS_DIR / "run2_sensitive"
RUN3_DIR = RESULTS_DIR / "run3_cluster"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
REPORT_DIR = RESULTS_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"

RUN1_RESULTS = RUN1_DIR / "alignment_results.tsv"
RUN2_RESULTS = RUN2_DIR / "alignment_results.tsv"
RUN3_RESULTS = RUN3_DIR / "alignment_results.tsv"

SUMMARY_STATS = ANALYSIS_DIR / "summary_stats.json"
IDENTITY_DIST = ANALYSIS_DIR / "identity_distribution.json"
HIGH_RISK_SAMPLES = ANALYSIS_DIR / "high_risk_samples.json"
BY_LABEL = ANALYSIS_DIR / "by_label.json"
BY_SPECIES = ANALYSIS_DIR / "by_species.json"
BY_LENGTH = ANALYSIS_DIR / "by_length.json"
RUN2_COMPARISON = ANALYSIS_DIR / "run2_comparison.json"
RUN3_CLUSTER_ANALYSIS = ANALYSIS_DIR / "run3_cluster_analysis.json"

HIGH_RISK_FASTA = DATA_DIR / "high_risk_queries.fasta"
LOGS_DIR = WORK_DIR / "logs"

RISK_THRESHOLDS = {
    "extreme": {"identity": 1.00, "qcov": 0.95},
    "high": {"identity": 0.99, "qcov": 0.90},
    "medium": {"identity": 0.95, "qcov": 0.80},
    "low": {"identity": 0.90, "qcov": 0.70},
}

IDENTITY_BINS = [
    (0.0, 0.8, "<80% (no-hit)"),
    (0.8, 0.85, "80-85%"),
    (0.85, 0.9, "85-90%"),
    (0.9, 0.95, "90-95%"),
    (0.95, 0.99, "95-99%"),
    (0.99, 1.0, "99-100%"),
    (1.0, 1.0, "100%"),
]

LENGTH_BINS = [
    (0, 100, "0-100bp"),
    (100, 200, "100-200bp"),
    (200, 500, "200-500bp"),
    (500, 1000, "500-1000bp"),
    (1000, 2000, "1000-2000bp"),
    (2000, float('inf'), "2000+bp"),
]

LABELS = ["CDS", "intron", "lncRNA_exon", "tRNA", "rRNA", "snRNA", "snoRNA"]
SPECIES = ["GCF_000523025.1", "GCF_022379125.1", "GCF_024713975.1"]
TOTAL_TEST_SAMPLES = 155301
