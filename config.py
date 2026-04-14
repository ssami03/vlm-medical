"""
config.py — Configuration for VLM Medical Pipeline
SW3 — Medical image analysis using visual language models (VLMs)

All paths, hyperparameters, and constants in one place.
"""

import os

# ── Paths (Colab + Google Drive) ──────────────────────────────────
DRIVE_ROOT = '/content/drive/MyDrive'
DATA_DIR = f'{DRIVE_ROOT}/iu_xray_data'
CKPT_DIR = f'{DRIVE_ROOT}/vlm_checkpoints'
RESULTS_DIR = f'{CKPT_DIR}/results'

# Raw data paths
REPORTS_CSV = f'{DATA_DIR}/indiana_reports.csv'
PROJECTIONS_CSV = f'{DATA_DIR}/indiana_projections.csv'
IMAGES_DIR = f'{DATA_DIR}/images/images_normalized'

# Processed data cache
PROCESSED_JSON = f'{DATA_DIR}/processed_dataset.json'

# Checkpoints
CLASSIFIER_CKPT = f'{CKPT_DIR}/classifier_best.pt'
DECODER_CKPT = f'{CKPT_DIR}/decoder_best.pt'

# ── Pathology Classes ─────────────────────────────────────────────
# 13 common chest X-ray findings/pathologies extracted from reports.
PATHOLOGY_CLASSES = [
    'Cardiomegaly',
    'Effusion',
    'Consolidation',
    'Edema',
    'Atelectasis',
    'Pneumothorax',
    'Pneumonia',
    'Nodule',
    'Mass',
    'Fracture',
    'Emphysema',
    'Fibrosis',
    'Opacity',
]
NUM_CLASSES = len(PATHOLOGY_CLASSES)

# ── Label Extraction Keywords ─────────────────────────────────────
PATHOLOGY_KEYWORDS = {
    'Cardiomegaly': ['cardiomegaly', 'cardiac enlargement', 'enlarged heart',
                     'enlarged cardiac silhouette', 'heart size is enlarged',
                     'heart is enlarged', 'cardiothoracic ratio'],
    'Effusion': ['effusion', 'pleural effusion', 'pleural fluid'],
    'Consolidation': ['consolidation', 'consolidative', 'airspace disease'],
    'Edema': ['edema', 'oedema', 'pulmonary edema', 'vascular congestion'],
    'Atelectasis': ['atelectasis', 'atelectatic', 'volume loss'],
    'Pneumothorax': ['pneumothorax'],
    'Pneumonia': ['pneumonia', 'infectious process'],
    'Nodule': ['nodule', 'nodular', 'nodules'],
    'Mass': ['mass', 'masses', 'mass lesion'],
    'Fracture': ['fracture', 'fractured', 'fractures'],
    'Emphysema': ['emphysema', 'emphysematous', 'hyperinflat'],
    'Fibrosis': ['fibrosis', 'fibrotic', 'scarring'],
    'Opacity': ['opacity', 'opacities', 'opacification', 'haziness'],
}

NEGATION_WORDS = [
    'no ', 'not ', 'no evidence', 'without ', 'free of', 'clear of',
    'absent', 'negative for', 'unremarkable', 'resolved', 'removed',
    'none ', 'deny', 'denied', 'rule out', 'ruled out',
]

# ── CLIP Encoder ──────────────────────────────────────────────────
CLIP_MODEL_NAME = 'ViT-B/32'
CLIP_EMBED_DIM = 512      # CLS token dimension after projection
CLIP_HIDDEN_DIM = 768     # ViT hidden size / patch token dimension
CLIP_NUM_PATCHES = 49     # 7x7 grid for ViT-B/32 @ 224x224

# ── BioGPT Decoder ────────────────────────────────────────────────
DECODER_MODEL_NAME = 'microsoft/biogpt'
DECODER_HIDDEN_DIM = 1024

# ── Classifier Hyperparameters ────────────────────────────────────
CLASSIFIER_CFG = {
    'epochs': 8,
    'batch_size': 32,
    'lr_head': 1e-4,
    'lr_backbone': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,
    'label_smoothing': 0.1,
    'dropout': 0.1,
    'seed': 42,
    'num_workers': 2,
    'image_size': 224,
}

# ── Decoder Hyperparameters ───────────────────────────────────────
DECODER_CFG = {
    'epochs': 12,
    'prewarm_epochs': 2,
    'batch_size': 8,
    'lr': 3e-5,
    'prewarm_lr': 1e-4,
    'max_length': 128,
    'gen_max_length': 80,
    'warmup_epochs': 1,
    'num_workers': 2,
    'seed': 42,
    'grad_accum_steps': 4,
    'num_beams': 4,
}

# ── Dataset Split ─────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
