"""
Cross-species generalization configuration (Section 3.6).
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERALIZATION_ROOT = Path(__file__).resolve().parent

RAW_DATA_ROOT = Path(os.getenv(
    "FLATFISH_RAW_DATA_ROOT",
    str(PROJECT_ROOT / "data" / "raw" / "generalization_species")
))

DATA_OUTPUT_DIR = GENERALIZATION_ROOT / "data"
RESULTS_OUTPUT_DIR = GENERALIZATION_ROOT / "results"

EXP4_MODEL_PATH = Path(os.getenv(
    "FLATFISH_EXP4_MODEL",
    str(PROJECT_ROOT / "checkpoints" / "flatfishbert" / "Exp4_cross_dedup")
))
TOKENIZER_PATH = PROJECT_ROOT / "pretrain" / "tokenizer" / "flatfish_tokenizer"

SPECIES_CONFIGS = {
    'hippoglossus_hippoglossus': {
        'name': 'Hippoglossus_hippoglossus',
        'common_name': 'Atlantic halibut',
        'scientific_name': 'Hippoglossus hippoglossus',
        'family': 'Pleuronectidae',
        'gff': RAW_DATA_ROOT / 'hippoglossus_hippoglossus/ncbi_dataset/data/GCF_009819705.1/genomic.gff',
        'fasta': RAW_DATA_ROOT / 'hippoglossus_hippoglossus/ncbi_dataset/data/GCF_009819705.1/GCF_009819705.1_fHipHip1.pri_genomic.fna',
        'accession': 'GCF_009819705.1',
    },
    'solea_senegalensis': {
        'name': 'Solea_senegalensis',
        'common_name': 'Senegalese sole',
        'scientific_name': 'Solea senegalensis',
        'family': 'Soleidae',
        'gff': RAW_DATA_ROOT / 'solea_senegalensis/data/GCF_019176455.1/genomic.gff',
        'fasta': RAW_DATA_ROOT / 'solea_senegalensis/data/GCF_019176455.1/GCF_019176455.1_IFAPA_SoseM_1_genomic.fna',
        'accession': 'GCF_019176455.1',
    },
    'solea_solea': {
        'name': 'Solea_solea',
        'common_name': 'Common sole',
        'scientific_name': 'Solea solea',
        'family': 'Soleidae',
        'gff': RAW_DATA_ROOT / 'solea_solea/ncbi_dataset/data/GCF_958295425.1/genomic.gff',
        'fasta': RAW_DATA_ROOT / 'solea_solea/ncbi_dataset/data/GCF_958295425.1/GCF_958295425.1_fSolSol10.1_genomic.fna',
        'accession': 'GCF_958295425.1',
    },
}

PROTEIN_CODING_LABELS = ['CDS']
NON_CODING_LABELS = ['intron', 'lncRNA_exon', 'tRNA', 'rRNA', 'snRNA', 'snoRNA']

LABEL2ID = {'Protein-coding': 0, 'Non-coding': 1}
ID2LABEL = {0: 'Protein-coding', 1: 'Non-coding'}

ORIGINAL_TO_BINARY = {}
for label in PROTEIN_CODING_LABELS:
    ORIGINAL_TO_BINARY[label] = 'Protein-coding'
for label in NON_CODING_LABELS:
    ORIGINAL_TO_BINARY[label] = 'Non-coding'

LABEL_PRIORITY = {
    'CDS': 0, 'intron': 1, 'tRNA': 2, 'rRNA': 3,
    'snRNA': 4, 'snoRNA': 5, 'lncRNA_exon': 6,
}
