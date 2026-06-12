# FlatfishBERT

Code for: **A reproducible benchmark with sequence-similarity auditing for coding and non-coding genomic fragment classification across flatfish genomes**

## Authors

Jingwen Tang, Qiuhua Zheng, Jixing Zhong, Chuanhui Cheng, and Qiaomu Hu.

Chuanhui Cheng and Qiaomu Hu are co-corresponding authors.

## Repository Structure

```
.
├── pretrain/                       # FlatfishBert pretraining
│   ├── src/                        #   Model config, data loading, training
│   └── tokenizer/                  #   BPE tokenizer training & trained tokenizer
│
├── benchmark/                      # Core benchmark (Exp1–Exp5)
│   ├── data_preparation/           #   Data preparation for five benchmark regimes
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
│   └── results/                    #   Lightweight pre-computed analysis summaries
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
| §2.6 Models and fine-tuning | `pretrain/`, `benchmark/flatfishbert/` |
| §3.1 Exp1–Exp5 regime comparison | `benchmark/flatfishbert/` |
| §3.2 Principal benchmark calibration | `benchmark/dnabert2/exp4_comparison/` |
| §3.3 Sequence-similarity auditing | `similarity_audit/` |
| §3.4 Shortcut & sequence-feature analyses | `analysis/` |
| §3.5 Split-reference & cross-species | `benchmark/dnabert2/exp2_split_sensitivity/`, `cross_species/` |
| §3.6 Diagnostic extensions | `diagnostic_tasks/` |
| Fig. 2–4 | `figures/` |

## Data, Model, and Code Archives

- **Reference genomes:** NCBI RefSeq accessions are listed in the manuscript and in the Zenodo data archive.
- **Processed data, FlatfishBert checkpoints, and result artifacts:** https://doi.org/10.5281/zenodo.20670177

This repository does not include training data or model weights. Processed datasets, FlatfishBert checkpoints, and full data/model artifacts are archived separately in the Zenodo data/model record. This repository retains the source code together with selected lightweight result summaries, figure outputs, tokenizer files, and MMseqs2 audit summaries needed for paper-code navigation and review.

DNABERT-2 fine-tuned checkpoints are not redistributed. DNABERT-2 is used as a strong public baseline; this repository provides the scripts needed to rerun the DNABERT-2 analyses when the original DNABERT-2 base model is obtained from its public source.

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
conda activate flatfishbert-review
```

DNABERT-2 requires a separate environment; set `DNABERT2_MODEL_PATH` accordingly.

All scripts use relative paths. For custom locations:

```bash
export FLATFISH_RAW_DATA_ROOT=/path/to/raw/genomes
export DNABERT2_MODEL_PATH=/path/to/dnabert2
```

## Citation

Please cite the associated manuscript and Zenodo data archive if you use this repository:

```bibtex
@article{tang2026reproducible,
  title   = {A reproducible benchmark with sequence-similarity auditing for
             coding and non-coding genomic fragment classification across
             flatfish genomes},
  author  = {Tang, Jingwen and Zheng, Qiuhua and Zhong, Jixing
             and Cheng, Chuanhui and Hu, Qiaomu},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
