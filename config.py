"""
config.py — Configuration for VLM Medical Pipeline (BioGPT version)
EEE3094 Dissertation — Sami Zarroug (220672267)

All paths, hyperparameters, and constants in one place.
"""

import os

# ── Paths (Colab + Google Drive) ──────────────────────────────────
DRIVE_ROOT  = '/content/drive/MyDrive'
DATA_DIR    = f'{DRIVE_ROOT}/iu_xray_data'
CKPT_DIR    = f'{DRIVE_ROOT}/vlm_checkpoints'
RESULTS_DIR = f'{CKPT_DIR}/results'

# Raw data paths (after download)
REPORTS_CSV     = f'{DATA_DIR}/indiana_reports.csv'
PROJECTIONS_CSV = f'{DATA_DIR}/indiana_projections.csv'
IMAGES_DIR      = f'{DATA_DIR}/images/images_normalized'

# Processed data
PROCESSED_JSON = f'{DATA_DIR}/processed_dataset.json'

# Final classifier checkpoint used by decoder training/evaluation
CLASSIFIER_CKPT = f'{CKPT_DIR}/classifier_best.pt'
DECODER_CKPT    = f'{CKPT_DIR}/decoder_best.pt'

# Extra checkpoints/results for supervisor-requested baseline + two-stage fine-tuning
CLASSIFIER_STAGE1_CKPT = f'{CKPT_DIR}/classifier_stage1_head_only.pt'
CLASSIFIER_STAGE2_CKPT = f'{CKPT_DIR}/classifier_stage2_full_finetune.pt'
CLASSIFIER_COMPARISON_JSON = f'{RESULTS_DIR}/classification_stage_comparison.json'
CLIP_ZERO_SHOT_BASELINE_JSON = f'{RESULTS_DIR}/clip_zero_shot_baseline.json'

# ── Pathology Classes ─────────────────────────────────────────────
# 13 common chest X-ray pathologies extractable from IU X-Ray reports
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
    'Cardiomegaly':  ['cardiomegaly', 'cardiac enlargement', 'enlarged heart',
                      'enlarged cardiac silhouette', 'heart size is enlarged',
                      'heart is enlarged', 'cardiothoracic ratio'],
    'Effusion':      ['effusion', 'pleural effusion', 'pleural fluid'],
    'Consolidation': ['consolidation', 'consolidative', 'airspace disease'],
    'Edema':         ['edema', 'oedema', 'pulmonary edema', 'vascular congestion'],
    'Atelectasis':   ['atelectasis', 'atelectatic', 'volume loss'],
    'Pneumothorax':  ['pneumothorax'],
    'Pneumonia':     ['pneumonia', 'infectious process'],
    'Nodule':        ['nodule', 'nodular', 'nodules'],
    'Mass':          ['mass', 'masses', 'mass lesion'],
    'Fracture':      ['fracture', 'fractured', 'fractures'],
    'Emphysema':     ['emphysema', 'emphysematous', 'hyperinflat'],
    'Fibrosis':      ['fibrosis', 'fibrotic', 'scarring'],
    'Opacity':       ['opacity', 'opacities', 'opacification', 'haziness'],
}

NEGATION_WORDS = [
    'no ', 'not ', 'no evidence', 'without ', 'free of', 'clear of',
    'absent', 'negative for', 'unremarkable', 'resolved', 'removed',
    'none ', 'deny', 'denied', 'rule out', 'ruled out',
]

# ── CLIP Encoder ──────────────────────────────────────────────────
CLIP_MODEL_NAME  = 'ViT-B/32'
CLIP_EMBED_DIM   = 512       # CLS token dimension (after projection)
CLIP_HIDDEN_DIM  = 768       # transformer hidden dimension (patch tokens)
CLIP_NUM_PATCHES = 49        # 7×7 grid for ViT-B/32 at 224×224

# ── BioGPT Decoder ───────────────────────────────────────────────
# BioGPT: 347M params, pre-trained on 15M PubMed abstracts
# Already knows medical vocabulary: "pleural effusion", "consolidation",
# "cardiothoracic ratio" etc. — unlike distilgpt2 which knows Reddit.
#
# Architecture change: Visual Prefix Injection (LLaVA paradigm)
#   Instead of cross-attention (which required hacking GPT2 config and
#   produced those MISSING keys), we prepend 49 visual tokens to the
#   text input. BioGPT's self-attention naturally attends to both.
#   No missing weights. No random initialization. No architectural hacks.
DECODER_MODEL_NAME = 'microsoft/biogpt'
DECODER_HIDDEN_DIM = 1024    # BioGPT hidden size (vs 768 for distilgpt2)

# ── Classifier Hyperparameters ────────────────────────────────────
CLASSIFIER_CFG = {
    # Supervisor-updated two-stage transfer-learning schedule
    'stage1_epochs':        15,
    'stage2_epochs':        15,
    'batch_size':           32,

    # AdamW-safe learning rates
    # Your supervisor mentioned 0.1/0.01, but those are usually too high for AdamW.
    'stage1_lr_head':       1e-3,
    'stage2_lr_backbone':   1e-6,
    'stage2_lr_head':       1e-4,

    'weight_decay':         1e-4,
    'warmup_epochs':        1,
    'label_smoothing':      0.05,
    'dropout':              0.1,
    'seed':                 42,
    'num_workers':          2,
    'image_size':           224,

    # Kept for backwards compatibility only.
    # The new training strategy does NOT use top-N unfreezing.
    'unfreeze_blocks':      0,
}

# ── Decoder Hyperparameters ───────────────────────────────────────
DECODER_CFG = {
    'epochs':           10,
    'prewarm_epochs':   2,        # 2 epochs for BioGPT (larger model needs more prewarm)
    'batch_size':       8,        # smaller batch — BioGPT is 4x larger than distilgpt2
    'lr':               3e-5,     # lower LR for larger model
    'prewarm_lr':       1e-4,
    'max_length':       128,
    'gen_max_length':   96,
    'warmup_epochs':    1,
    'num_workers':      2,
    'seed':             42,
    'grad_accum_steps': 4,        # effective batch = 8 × 4 = 32
    'num_beams':        4,
}

# ── Dataset Split ─────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
