#!/usr/bin/env python3
"""
Data preparation for MMseqs2 similarity audit (Section 2.7 / 3.3).
Converts Exp4 JSON data to FASTA format for MMseqs2 alignment.

Inputs:
  - Exp4 train.json and test.json

Outputs:
  - data/test_sequences.fasta
  - data/train_sequences.fasta
  - data/test_sequences_sample10pct.fasta
  - data/metadata/test_metadata.tsv
  - data/metadata/train_metadata.tsv
  - data/metadata/sample_ids.txt
"""

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP4_DIR = Path(os.getenv(
    "FLATFISH_EXP4_DATA",
    str(PROJECT_ROOT / "data" / "benchmark" / "Exp4_cross_dedup")
))

WORK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORK_DIR / "data"
METADATA_DIR = OUTPUT_DIR / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_data(json_path):
    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"  Samples: {len(data):,}")
    return data


def write_fasta(data, fasta_path, metadata_path, dataset_name):
    """Write JSON data to FASTA and metadata TSV."""
    print(f"\nWriting FASTA: {fasta_path}")

    with open(fasta_path, 'w') as fasta_f, open(metadata_path, 'w') as meta_f:
        meta_f.write("seq_id\tlabel_name\tgene_name\tspecies\tseqname\tstart\tend\tstrand\tsequence_length\n")

        for idx, sample in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
            seq_id = f"{dataset_name}_{idx:07d}"
            sequence = sample['sequence']
            label = sample['label_name']
            gene = sample.get('gene_name', 'unknown')
            species = sample.get('species', 'unknown')
            seqname = sample.get('seqname', 'unknown')
            start = sample.get('start', 0)
            end = sample.get('end', 0)
            strand = sample.get('strand', '+')
            seq_len = sample.get('sequence_length', len(sequence))

            header = f">{seq_id}|{label}|{gene}|{species}|{seqname}|{start}|{end}|{strand}|{seq_len}"
            fasta_f.write(f"{header}\n{sequence}\n")
            meta_f.write(f"{seq_id}\t{label}\t{gene}\t{species}\t{seqname}\t{start}\t{end}\t{strand}\t{seq_len}\n")

    print(f"  Done: {len(data):,} sequences")


def create_sample_fasta(full_fasta_path, sample_fasta_path, sample_ids_path, sample_ratio=0.1):
    """Randomly sample 10% of test sequences."""
    seq_ids = []
    with open(full_fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_id = line.strip()[1:].split('|')[0]
                seq_ids.append(seq_id)

    sample_size = int(len(seq_ids) * sample_ratio)
    sampled_ids = set(random.sample(seq_ids, sample_size))

    with open(sample_ids_path, 'w') as f:
        for seq_id in sorted(sampled_ids):
            f.write(f"{seq_id}\n")

    with open(full_fasta_path, 'r') as in_f, open(sample_fasta_path, 'w') as out_f:
        write_seq = False
        for line in in_f:
            if line.startswith('>'):
                seq_id = line.strip()[1:].split('|')[0]
                write_seq = seq_id in sampled_ids
            if write_seq:
                out_f.write(line)

    print(f"  Sampled {sample_size:,} / {len(seq_ids):,} sequences ({sample_ratio*100}%)")


def main():
    train_data = load_json_data(EXP4_DIR / "train.json")
    test_data = load_json_data(EXP4_DIR / "test.json")

    write_fasta(train_data, OUTPUT_DIR / "train_sequences.fasta",
                METADATA_DIR / "train_metadata.tsv", "train")
    write_fasta(test_data, OUTPUT_DIR / "test_sequences.fasta",
                METADATA_DIR / "test_metadata.tsv", "test")
    create_sample_fasta(OUTPUT_DIR / "test_sequences.fasta",
                        OUTPUT_DIR / "test_sequences_sample10pct.fasta",
                        METADATA_DIR / "sample_ids.txt", sample_ratio=0.1)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
