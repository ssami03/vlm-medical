"""
dataset.py — IU X-Ray Dataset Loading and Preprocessing
EEE3094 Dissertation — Sami Zarroug (220672267)

Handles:
  1. Downloading IU X-Ray from Kaggle (one-time)
  2. Parsing reports CSV → extracting findings text
  3. Extracting multi-hot pathology labels via keyword matching with negation handling
  4. Creating train/val/test splits (patient-stratified)
  5. PyTorch Dataset class for both classification and report generation
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from typing import List, Dict, Tuple, Optional

try:
    import clip
except ImportError:
    pass

from config import (
    DATA_DIR, IMAGES_DIR, REPORTS_CSV, PROJECTIONS_CSV, PROCESSED_JSON,
    PATHOLOGY_CLASSES, PATHOLOGY_KEYWORDS, NEGATION_WORDS, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CLASSIFIER_CFG,
)


# ══════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════

def download_iu_xray():
    """
    Downloads IU X-Ray dataset from Kaggle.
    Requires: kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME + KAGGLE_KEY env vars.

    Alternative: manually download from
    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
    and upload to Google Drive at DATA_DIR.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(REPORTS_CSV):
        print(f"[Data] IU X-Ray already downloaded at {DATA_DIR}")
        return

    print("[Data] Downloading IU X-Ray from Kaggle...")
    os.system(f'kaggle datasets download -d raddar/chest-xrays-indiana-university -p {DATA_DIR} --unzip')
    print(f"[Data] Downloaded to {DATA_DIR}")


# ══════════════════════════════════════════════════════════════════
# 2. LABEL EXTRACTION FROM REPORT TEXT
# ══════════════════════════════════════════════════════════════════

def _is_negated(text: str, keyword: str, start_idx: int) -> bool:
    """
    Checks if a keyword occurrence is negated by looking at
    the preceding 50 characters for negation words.

    Example:
        "no pleural effusion" → effusion is NEGATED
        "moderate pleural effusion" → effusion is NOT negated
    """
    prefix_start = max(0, start_idx - 50)
    prefix = text[prefix_start:start_idx].lower()
    return any(neg in prefix for neg in NEGATION_WORDS)


def extract_labels(findings: str, impression: str = "") -> np.ndarray:
    """
    Extracts a multi-hot label vector from report text using
    keyword matching with negation detection.

    Returns: numpy array of shape (NUM_CLASSES,) with 0/1 values.
    """
    combined = f"{findings} {impression}".lower()
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)

    for i, cls in enumerate(PATHOLOGY_CLASSES):
        keywords = PATHOLOGY_KEYWORDS.get(cls, [cls.lower()])
        for kw in keywords:
            idx = 0
            while True:
                idx = combined.find(kw.lower(), idx)
                if idx == -1:
                    break
                if not _is_negated(combined, kw, idx):
                    labels[i] = 1.0
                    break
                idx += len(kw)

    return labels


# ══════════════════════════════════════════════════════════════════
# 3. DATASET PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def preprocess_iu_xray(force_reprocess: bool = False) -> List[Dict]:
    """
    Parses the IU X-Ray CSVs and creates a processed JSON with:
      - image paths (frontal views only)
      - findings text
      - impression text
      - multi-hot label vectors
      - train/val/test split assignments

    Returns: list of sample dicts
    """
    if os.path.exists(PROCESSED_JSON) and not force_reprocess:
        print(f"[Data] Loading preprocessed data from {PROCESSED_JSON}")
        with open(PROCESSED_JSON, 'r') as f:
            return json.load(f)

    print("[Data] Preprocessing IU X-Ray dataset...")

    reports_df = pd.read_csv(REPORTS_CSV)
    proj_df = pd.read_csv(PROJECTIONS_CSV)

    # Keep only frontal views (AP or PA)
    frontal_df = proj_df[proj_df['projection'].str.lower().isin(
        ['frontal', 'ap', 'pa', 'anteroposterior', 'posteroanterior']
    )].copy()

    if len(frontal_df) == 0:
        print("[Data] No standard projection labels found, keeping all images")
        frontal_df = proj_df.copy()

    merged = frontal_df.merge(reports_df, on='uid', how='inner')

    samples = []
    skipped = 0

    for _, row in merged.iterrows():
        img_name = row.get('filename', '')
        if not img_name:
            skipped += 1
            continue

        # Try multiple image path locations
        img_path = None
        for candidate in [
            os.path.join(IMAGES_DIR, img_name),
            os.path.join(DATA_DIR, 'images', img_name),
            os.path.join(DATA_DIR, 'images', 'images_normalized', img_name),
            os.path.join(DATA_DIR, img_name),
        ]:
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        findings = str(row.get('findings', '')).strip()
        impression = str(row.get('impression', '')).strip()

        if findings in ['', 'nan', 'None']:
            findings = ''
        if impression in ['', 'nan', 'None']:
            impression = ''

        if not findings and not impression:
            skipped += 1
            continue

        if findings and impression:
            report_text = f"Findings: {findings} Impression: {impression}"
        elif findings:
            report_text = f"Findings: {findings}"
        elif impression:
            report_text = f"Impression: {impression}"
        else:
            skipped += 1
            continue
        labels = extract_labels(findings, impression)

        samples.append({
            'uid': str(row['uid']),
            'image_path': img_path,
            'findings': findings,
            'impression': impression,
            'report': report_text,
            'labels': labels.tolist(),
        })

    print(f"[Data] Processed {len(samples)} samples ({skipped} skipped)")

    # Patient-stratified split
    uids = list(set(s['uid'] for s in samples))
    random.seed(42)
    random.shuffle(uids)

    n_train = int(len(uids) * TRAIN_RATIO)
    n_val   = int(len(uids) * VAL_RATIO)

    train_uids = set(uids[:n_train])
    val_uids   = set(uids[n_train:n_train + n_val])

    for s in samples:
        if s['uid'] in train_uids:
            s['split'] = 'train'
        elif s['uid'] in val_uids:
            s['split'] = 'val'
        else:
            s['split'] = 'test'

    # Statistics
    train_count = sum(1 for s in samples if s['split'] == 'train')
    val_count   = sum(1 for s in samples if s['split'] == 'val')
    test_count  = sum(1 for s in samples if s['split'] == 'test')
    print(f"[Data] Split: train={train_count} | val={val_count} | test={test_count}")

    all_labels = np.array([s['labels'] for s in samples])
    print(f"\n[Data] Label distribution:")
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        pos = int(all_labels[:, i].sum())
        print(f"  {cls:<20s}: {pos:>4d} positives ({100*pos/len(samples):.1f}%)")
    normal = int((all_labels.sum(axis=1) == 0).sum())
    print(f"  {'No Finding':<20s}: {normal:>4d} ({100*normal/len(samples):.1f}%)")

    os.makedirs(os.path.dirname(PROCESSED_JSON), exist_ok=True)
    with open(PROCESSED_JSON, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"[Data] Saved to {PROCESSED_JSON}")

    return samples


# ══════════════════════════════════════════════════════════════════
# 4. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════

class IUXRayDataset(Dataset):
    """Returns: (image_tensor, label_vector, report_text)"""

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        labels = torch.tensor(sample['labels'], dtype=torch.float32)
        return img, labels, sample['report']


def get_transforms(image_size: int = 224, is_train: bool = True):
    """Returns CLIP-compatible image transforms."""
    try:
        _, preprocess = clip.load('ViT-B/32', device='cpu')
        return preprocess
    except:
        from torchvision import transforms
        if is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])


def build_dataloaders(batch_size=32, num_workers=2, seed=42, balance_train=True):
    """Builds train/val/test DataLoaders with optional balanced sampling."""
    samples = preprocess_iu_xray()
    train_transform = get_transforms(image_size=CLASSIFIER_CFG['image_size'], is_train=True)
    eval_transform = get_transforms(image_size=CLASSIFIER_CFG['image_size'], is_train=False)

    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples   = [s for s in samples if s['split'] == 'val']
    test_samples  = [s for s in samples if s['split'] == 'test']

    train_ds = IUXRayDataset(train_samples, train_transform)
    val_ds   = IUXRayDataset(val_samples, eval_transform)
    test_ds  = IUXRayDataset(test_samples, eval_transform)

    sampler = None
    shuffle = True
    if balance_train and len(train_samples) > 0:
        labels = np.array([s['labels'] for s in train_samples])
        has_disease = (labels.sum(axis=1) > 0).astype(float)
        n_healthy = int((has_disease == 0).sum())
        n_diseased = int((has_disease == 1).sum())

        if n_healthy > 0 and n_diseased > 0:
            weights = np.where(has_disease == 1, 1.0/n_diseased, 1.0/n_healthy)
            sampler = WeightedRandomSampler(
                weights=torch.FloatTensor(weights),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
            print(f"[Data] Sampler: {n_healthy} healthy, {n_diseased} diseased → 50/50")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# 5. COLLATE FUNCTION FOR DECODER TRAINING (BioGPT)
# ══════════════════════════════════════════════════════════════════

def decoder_collate_fn(batch, tokenizer, max_length=128):
    """
    Collates a batch for BioGPT decoder training.
    Prepends "Findings:" and appends EOS to each report.
    """
    images, label_vecs, reports = zip(*batch)
    images = torch.stack(images)

    # Format reports with prefix
    formatted = [report if report.lower().startswith(('findings:', 'impression:')) else f"Findings: {report}" for report in reports]

    encoded = tokenizer(
        list(formatted),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Labels: same as input_ids but -100 for padding
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return images, input_ids, attention_mask, labels
