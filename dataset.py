"""
dataset.py — IU X-Ray dataset loading and preprocessing

Handles:
1. Parsing report/projection CSVs
2. Extracting report-derived multi-label pathology targets
3. Building train/val/test splits
4. PyTorch datasets and dataloaders
"""

import os
import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import clip
except ImportError:
    clip = None

from config import (
    CLASSIFIER_CFG,
    DATA_DIR,
    IMAGES_DIR,
    NEGATION_WORDS,
    NUM_CLASSES,
    PATHOLOGY_CLASSES,
    PATHOLOGY_KEYWORDS,
    PROCESSED_JSON,
    PROJECTIONS_CSV,
    REPORTS_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
)


def download_iu_xray():
    """Optional helper if Kaggle API is configured in Colab."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(REPORTS_CSV) and os.path.exists(PROJECTIONS_CSV):
        print(f"[Data] IU X-Ray already present at {DATA_DIR}")
        return

    print('[Data] Attempting Kaggle download...')
    status = os.system(
        f'kaggle datasets download -d raddar/chest-xrays-indiana-university -p {DATA_DIR} --unzip'
    )
    if status != 0:
        raise RuntimeError(
            'Kaggle download failed. Download manually and upload to Google Drive.'
        )


def _is_negated(text: str, start_idx: int) -> bool:
    """Simple negation check using the preceding character window."""
    prefix = text[max(0, start_idx - 50):start_idx].lower()
    return any(neg in prefix for neg in NEGATION_WORDS)



def extract_labels(findings: str, impression: str = '') -> np.ndarray:
    """Extracts a weak multi-hot label vector from report text."""
    combined = f"{findings} {impression}".lower()
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)

    for i, cls in enumerate(PATHOLOGY_CLASSES):
        for keyword in PATHOLOGY_KEYWORDS.get(cls, [cls.lower()]):
            search_pos = 0
            while True:
                idx = combined.find(keyword.lower(), search_pos)
                if idx == -1:
                    break
                if not _is_negated(combined, idx):
                    labels[i] = 1.0
                    break
                search_pos = idx + len(keyword)
            if labels[i] == 1.0:
                break

    return labels



def _resolve_image_path(filename: str) -> str | None:
    candidates = [
        os.path.join(IMAGES_DIR, filename),
        os.path.join(DATA_DIR, 'images', 'images_normalized', filename),
        os.path.join(DATA_DIR, 'images', filename),
        os.path.join(DATA_DIR, filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None



def _build_report_text(findings: str, impression: str) -> str:
    findings = findings.strip()
    impression = impression.strip()

    if findings and impression:
        return f"Findings: {findings} Impression: {impression}"
    if findings:
        return f"Findings: {findings}"
    if impression:
        return f"Impression: {impression}"
    return ''



def preprocess_iu_xray(force_reprocess: bool = False) -> List[Dict]:
    """Creates cached processed dataset JSON and returns all samples."""
    if os.path.exists(PROCESSED_JSON) and not force_reprocess:
        print(f"[Data] Loading cached processed dataset from {PROCESSED_JSON}")
        with open(PROCESSED_JSON, 'r') as f:
            return json.load(f)

    if not os.path.exists(REPORTS_CSV):
        raise FileNotFoundError(f'Missing reports CSV: {REPORTS_CSV}')
    if not os.path.exists(PROJECTIONS_CSV):
        raise FileNotFoundError(f'Missing projections CSV: {PROJECTIONS_CSV}')

    print('[Data] Preprocessing IU X-Ray dataset...')
    reports_df = pd.read_csv(REPORTS_CSV)
    proj_df = pd.read_csv(PROJECTIONS_CSV)

    proj_col = proj_df['projection'].astype(str).str.lower()
    frontal_df = proj_df[proj_col.isin(['frontal', 'ap', 'pa', 'anteroposterior', 'posteroanterior'])].copy()
    if len(frontal_df) == 0:
        print('[Data] No recognised frontal projection labels found, keeping all images.')
        frontal_df = proj_df.copy()

    merged = frontal_df.merge(reports_df, on='uid', how='inner')

    samples: List[Dict] = []
    skipped = 0
    for _, row in merged.iterrows():
        img_name = str(row.get('filename', '')).strip()
        if not img_name:
            skipped += 1
            continue

        img_path = _resolve_image_path(img_name)
        if img_path is None:
            skipped += 1
            continue

        findings = '' if pd.isna(row.get('findings', '')) else str(row.get('findings', '')).strip()
        impression = '' if pd.isna(row.get('impression', '')) else str(row.get('impression', '')).strip()
        report_text = _build_report_text(findings, impression)
        if not report_text:
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

    print(f'[Data] Processed {len(samples)} samples ({skipped} skipped)')

    # Study-level split by uid.
    uids = sorted({sample['uid'] for sample in samples})
    random.seed(42)
    random.shuffle(uids)

    n_train = int(len(uids) * TRAIN_RATIO)
    n_val = int(len(uids) * VAL_RATIO)

    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:n_train + n_val])

    for sample in samples:
        if sample['uid'] in train_uids:
            sample['split'] = 'train'
        elif sample['uid'] in val_uids:
            sample['split'] = 'val'
        else:
            sample['split'] = 'test'

    train_count = sum(s['split'] == 'train' for s in samples)
    val_count = sum(s['split'] == 'val' for s in samples)
    test_count = sum(s['split'] == 'test' for s in samples)
    print(f'[Data] Split: train={train_count} | val={val_count} | test={test_count}')

    all_labels = np.array([sample['labels'] for sample in samples])
    print('\n[Data] Label distribution:')
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        positives = int(all_labels[:, i].sum())
        print(f'  {cls:<20s}: {positives:>4d} positives ({100 * positives / len(samples):.1f}%)')
    normal = int((all_labels.sum(axis=1) == 0).sum())
    print(f"  {'No Finding':<20s}: {normal:>4d} ({100 * normal / len(samples):.1f}%)")

    os.makedirs(os.path.dirname(PROCESSED_JSON), exist_ok=True)
    with open(PROCESSED_JSON, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f'[Data] Saved processed dataset to {PROCESSED_JSON}')

    return samples


class IUXRayDataset(Dataset):
    """Returns (image_tensor, labels, report_text)."""

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.tensor(sample['labels'], dtype=torch.float32)
        return image, labels, sample['report']



def get_transforms(image_size: int = 224, is_train: bool = True):
    """Returns CLIP-compatible image transforms."""
    if clip is not None:
        try:
            _, preprocess = clip.load('ViT-B/32', device='cpu')
            return preprocess
        except Exception:
            pass

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

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])



def build_dataloaders(batch_size: int = 32, num_workers: int = 2, balance_train: bool = True):
    """Builds train/val/test DataLoaders."""
    samples = preprocess_iu_xray()

    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples = [s for s in samples if s['split'] == 'val']
    test_samples = [s for s in samples if s['split'] == 'test']

    train_ds = IUXRayDataset(train_samples, get_transforms(CLASSIFIER_CFG['image_size'], is_train=True))
    val_ds = IUXRayDataset(val_samples, get_transforms(CLASSIFIER_CFG['image_size'], is_train=False))
    test_ds = IUXRayDataset(test_samples, get_transforms(CLASSIFIER_CFG['image_size'], is_train=False))

    sampler = None
    shuffle = True
    if balance_train and len(train_samples) > 0:
        labels = np.array([s['labels'] for s in train_samples])
        has_disease = (labels.sum(axis=1) > 0).astype(float)
        n_healthy = int((has_disease == 0).sum())
        n_diseased = int((has_disease == 1).sum())
        if n_healthy > 0 and n_diseased > 0:
            weights = np.where(has_disease == 1, 1.0 / n_diseased, 1.0 / n_healthy)
            sampler = WeightedRandomSampler(
                weights=torch.FloatTensor(weights),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
            print(f'[Data] Sampler: {n_healthy} healthy, {n_diseased} diseased -> balanced sampling')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader



def decoder_collate_fn(batch, tokenizer, max_length: int = 128):
    """Collates a batch for decoder training."""
    images, _, reports = zip(*batch)
    images = torch.stack(images)

    encoded = tokenizer(
        list(reports),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return images, input_ids, attention_mask, labels
