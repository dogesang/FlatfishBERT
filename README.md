# FlatfishBert

Code for: **A Governance-First Benchmark for Genomic Fragment Classification under Homology-Aware Evaluation**

## Repository Structure

```
.
├── pretrain/                       # FlatfishBert pretraining
│   ├── src/                        #   Model config, data loading, training
│   └── tokenizer/                  #   BPE tokenizer training & trained tokenizer
│
├── benchmark/                      # Core benchmark (Exp1–Exp5)
│   ├── data_preparation/           #   Data preparation for five governance regimes
│   ├── flatfishbert/               #   FlatfishBert fine-tuning & evaluation
│   └── dnabert2/                   #   DNABERT-2 baseline
│       ├── exp4_comparison/        #     Exp4 comparison (single + multi-seed)
│       └── exp2_split_sensitivity/ #     Exp2 split-sensitivity analysis
│
├── diagnostic_tasks/               # Diagnostic extension suite (Task G/S/N)
│   ├── data_preparation/           #   Data scripts (incl. reverse-complement aug)
│   ├── flatfishbert/               #   FlatfishBert training
│   └── dnabert2/                   #   DNABERT-2 training
│
├── cross_species/                  # Cross-species transfer evaluation
│   ├── flatfishbert/               #   Data extraction & evaluation
│   └── dnabert2/                   #   Evaluation on same test sets
│
├── similarity_audit/               # MMseqs2 similarity audit
│   ├── scripts/                    #   Pipeline (01–05) + prediction scripts
│   └── results/                    #   Pre-computed alignment & analysis outputs
│
├── analysis/                       # Auxiliary analyses
│   ├── length_baseline/            #   Length-only baseline
│   └── sequence_stats/             #   GC composition & 3-bp periodicity
│
├── figures/                        # Figure generation scripts & outputs (Fig. 2–4)
└── results/                        # Summary tables & metrics
```

## Paper–Code Cross-Reference

| Paper Section | Code Location |
|---|---|
| §2.3–2.5 Benchmark construction | `benchmark/data_preparation/` |
| §2.6 FlatfishBert pretraining | `pretrain/` |
| §2.6 Fine-tuning (Exp1–Exp5) | `benchmark/flatfishbert/` |
| §3.2 DNABERT-2 comparison | `benchmark/dnabert2/exp4_comparison/` |
| §3.3 MMseqs2 audit | `similarity_audit/` |
| §3.4 Length baseline & sequence stats | `analysis/` |
| §3.5 Split sensitivity | `benchmark/dnabert2/exp2_split_sensitivity/` |
| §3.6 Cross-species transfer | `cross_species/` |
| §3.6 Diagnostic tasks | `diagnostic_tasks/` |
| Fig. 2–4 | `figures/` |

## Data and Model Availability

- **Reference genomes:** NCBI RefSeq (accessions in the paper).
- **Processed data & checkpoints:** [ZENODO_DATA_DOI]
- **Archived code:** [ZENODO_CODE_DOI]

This repository does not include training data or model weights. Model checkpoints are archived separately and are not tracked in GitHub.

## Module Notes

- `benchmark/dnabert2/exp4_comparison/` contains both the representative seed-42 DNABERT-2 run and the multi-seed configuration.
- `benchmark/dnabert2/exp2_split_sensitivity/` implements the Exp2 split-reference comparison.
- `similarity_audit/scripts/` contains the sequential MMseqs2 audit pipeline: data preparation, three alignment runs, and result analysis.
- `analysis/length_baseline/` contains the Exp4 length-only baseline; the selected threshold is 236 bp.
- `cross_species/flatfishbert/` constructs and evaluates external-species test sets; `cross_species/dnabert2/` evaluates DNABERT-2 on the same test sets.
- `diagnostic_tasks/` contains Task G, Task S, Task N, and reverse-complement augmentation variants.

## Setup

```bash
conda env create -f environment.yml
conda activate flatfishbert
```

DNABERT-2 requires a separate environment; set `DNABERT2_MODEL_PATH` accordingly.

All scripts use relative paths. For custom locations:

```bash
export FLATFISH_RAW_DATA_ROOT=/path/to/raw/genomes
export DNABERT2_MODEL_PATH=/path/to/dnabert2
```

## Citation

```bibtex
@article{tang2026governance,
  title   = {A Governance-First Benchmark for Genomic Fragment Classification
             under Homology-Aware Evaluation},
  author  = {Tang, Jingwen and Zheng, Qiuhua and Zhong, Jixing
             and Cheng, Chuanhui and Fan, Yuding},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
