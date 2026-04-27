#!/usr/bin/env python3
"""
DNABERT-2 evaluation script — save per-sample predictions for the
MMseqs2 leakage-sensitivity re-evaluation (Section 3.3, Table S6).

Environment: dnabert2_finetune
"""

import sys
sys.modules['triton'] = None
sys.modules['triton.language'] = None

import json
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DNABERT2_MODEL_PATH = Path(os.getenv(
    "DNABERT2_MODEL_PATH",
    str(PROJECT_ROOT / "external_models" / "dnabert2")
))

DNABERT2_FINETUNED_PATH = Path(os.getenv(
    "DNABERT2_FINETUNED_SEED666",
    str(PROJECT_ROOT / "checkpoints" / "dnabert2" / "exp4_seed666" / "final_model")
))

EXP4_TEST_PATH = Path(os.getenv(
    "FLATFISH_EXP4_TEST",
    str(PROJECT_ROOT / "data" / "benchmark" / "Exp4_cross_dedup" / "test.json")
))

sys.path.insert(0, str(DNABERT2_MODEL_PATH))

sys.path.insert(0, str(PROJECT_ROOT / "similarity_audit"))
import config


class DNADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sequence = sample['sequence']
        label_name = sample['label_name']
        label = 0 if label_name == 'CDS' else 1

        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_test_data():
    print(f"Loading test data: {EXP4_TEST_PATH}")
    with open(EXP4_TEST_PATH, 'r') as f:
        test_data = json.load(f)
    print(f"  Samples: {len(test_data):,}")
    return test_data


def evaluate_dnabert2(model_path, test_data):
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    dataset = DNADataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1-macro: {f1_macro:.6f}")
    print(classification_report(all_labels, all_preds,
                                target_names=['Protein-coding', 'Non-coding'],
                                digits=6))

    return {
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
    }


def main():
    test_data = load_test_data()
    results = evaluate_dnabert2(str(DNABERT2_FINETUNED_PATH), test_data)

    output_dir = PROJECT_ROOT / "similarity_audit" / "results" / "filtered_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dnabert2_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
