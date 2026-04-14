"""
train.py — Complete Training Pipeline (BioGPT version)
EEE3094 Dissertation — Sami Zarroug (220672267)

Phase 1: Fine-tune CLIP encoder + classifier on IU X-Ray labels
Phase 2: Train bridge + BioGPT decoder on real IU X-Ray reports
    Phase 2a: Bridge prewarming (2 epochs) — learn visual→language mapping
    Phase 2b: Full decoder training (12 epochs) — tune BioGPT on medical reports

Run from Colab:
    !python /content/vlm-medical/train.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score

from config import (
    CKPT_DIR, RESULTS_DIR,
    CLASSIFIER_CKPT, DECODER_CKPT,
    CLASSIFIER_CFG, DECODER_CFG,
    DECODER_MODEL_NAME,
    PATHOLOGY_CLASSES, NUM_CLASSES,
)
from dataset import (
    preprocess_iu_xray, IUXRayDataset, get_transforms,
    build_dataloaders, decoder_collate_fn,
)
from model import MedicalVLM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
# PHASE 1: CLASSIFICATION TRAINING
# ══════════════════════════════════════════════════════════════════

def train_classifier(model: MedicalVLM, device: torch.device):
    """
    Fine-tune CLIP visual encoder + classification head.
    Differential LRs: 1e-6 backbone, 1e-4 head.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Classification Training")
    print(f"  Dataset: IU X-Ray | Metric: AUROC")
    print(f"{'='*60}")

    cfg = CLASSIFIER_CFG
    set_seed(cfg['seed'])

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        balance_train=True,
    )

    model.freeze_for_classification()
    param_groups = model.get_classifier_param_groups()
    optimiser = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])

    # Class-wise positive weighting to help rare labels
    train_labels_np = np.array([s['labels'] for s in train_loader.dataset.samples], dtype=np.float32)
    pos_counts = train_labels_np.sum(axis=0)
    neg_counts = len(train_labels_np) - pos_counts
    pos_weight = torch.tensor(np.clip(neg_counts / np.clip(pos_counts, 1.0, None), 1.0, 20.0),
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps  = len(train_loader) * cfg['epochs']
    warmup_steps = len(train_loader) * cfg['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')
    eps = cfg['label_smoothing']

    best_val_auc = 0.0
    log = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f"Cls {epoch}/{cfg['epochs']}"):
            images = images.to(device)
            labels = labels.to(device)
            smooth_labels = labels * (1 - eps) + (1 - labels) * eps

            optimiser.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model.classify(images)
                loss = criterion(logits, smooth_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        all_scores, all_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model.classify(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        y_score = np.concatenate(all_scores)
        y_true = np.concatenate(all_targets)

        aucs = {}
        for i, cls in enumerate(PATHOLOGY_CLASSES):
            n_pos = y_true[:, i].sum()
            if n_pos >= 2:
                try:
                    aucs[cls] = roc_auc_score(y_true[:, i], y_score[:, i])
                except:
                    aucs[cls] = float('nan')
            else:
                aucs[cls] = float('nan')

        valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        print(f"\nEpoch {epoch}: loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | AUC={mean_auc:.4f}")
        for cls, auc in aucs.items():
            if not np.isnan(auc):
                print(f"  {cls:<20s}: {auc:.4f}")

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_mean_auc': mean_auc,
                'per_class_auc': aucs,
            }, CLASSIFIER_CKPT)
            print(f"  ★ New best AUC={mean_auc:.4f} — saved")

        log.append({'epoch': epoch, 'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss, 'mean_auc': mean_auc})

    with open(f'{RESULTS_DIR}/classifier_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n  Phase 1 Complete! Best AUC: {best_val_auc:.4f}")
    return best_val_auc


# ══════════════════════════════════════════════════════════════════
# PHASE 2: DECODER TRAINING (BioGPT)
# ══════════════════════════════════════════════════════════════════

def train_decoder(model: MedicalVLM, device: torch.device):
    """
    Train BioGPT report generator on real IU X-Ray reports.

    Phase 2a — Bridge prewarming (2 epochs):
      Only the bridge MLP is trainable. This forces it to learn the
      mapping from CLIP visual features (768-dim) to BioGPT text space
      (1024-dim) before the decoder starts adjusting.

    Phase 2b — Full decoder training (12 epochs):
      Bridge + full BioGPT trainable. The decoder learns to generate
      medical reports conditioned on the visual prefix.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Report Generation Training")
    print(f"  Decoder: BioGPT (PubMed pre-trained)")
    print(f"  Dataset: IU X-Ray real reports")
    print(f"{'='*60}")

    cfg = DECODER_CFG
    set_seed(cfg['seed'])

    # BioGPT tokenizer
    tokenizer = BioGptTokenizer.from_pretrained(DECODER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Decoder] BioGPT tokenizer: {tokenizer.vocab_size:,} tokens")

    # Data
    samples = preprocess_iu_xray()
    transform = get_transforms()

    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples   = [s for s in samples if s['split'] == 'val']

    train_ds = IUXRayDataset(train_samples, transform)
    val_ds   = IUXRayDataset(val_samples, transform)

    collate = partial(decoder_collate_fn, tokenizer=tokenizer,
                      max_length=cfg['max_length'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], collate_fn=collate,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], collate_fn=collate,
                            pin_memory=True)

    print(f"[Decoder] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Phase 2a: Bridge Prewarming ───────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 2a: Bridge prewarming ({cfg['prewarm_epochs']} epochs)")
    print(f"  Only bridge MLP trainable — learning visual→language mapping")
    print(f"{'─'*60}")

    model.freeze_for_decoder_prewarm()
    prewarm_params = [p for p in model.parameters() if p.requires_grad]
    prewarm_optim = torch.optim.AdamW(prewarm_params, lr=cfg['prewarm_lr'],
                                       weight_decay=1e-4)
    prewarm_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(cfg['prewarm_epochs']):
        model.train()
        model.visual.eval()
        total_loss = 0.0

        for images, input_ids, attention_mask, labels in tqdm(
                train_loader, desc=f"Prewarm {epoch+1}/{cfg['prewarm_epochs']}"):
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            prewarm_optim.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                loss = outputs.loss

            prewarm_scaler.scale(loss).backward()
            prewarm_scaler.unscale_(prewarm_optim)
            torch.nn.utils.clip_grad_norm_(prewarm_params, max_norm=1.0)
            prewarm_scaler.step(prewarm_optim)
            prewarm_scaler.update()
            total_loss += loss.item()

        print(f"  Prewarm loss: {total_loss / len(train_loader):.4f}")

    # ── Phase 2b: Full Decoder Training ───────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 2b: Full decoder training ({cfg['epochs']} epochs)")
    print(f"  Bridge + BioGPT decoder trainable")
    print(f"{'─'*60}")

    model.freeze_for_decoder_full()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable_params, lr=cfg['lr'], weight_decay=1e-4)

    total_steps  = (len(train_loader) // cfg['grad_accum_steps']) * cfg['epochs']
    warmup_steps = (len(train_loader) // cfg['grad_accum_steps']) * cfg['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    log = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        model.visual.eval()
        total_loss = 0.0
        optimiser.zero_grad()

        for step, (images, input_ids, attention_mask, labels) in enumerate(
                tqdm(train_loader, desc=f"Dec {epoch}/{cfg['epochs']}")):

            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                loss = outputs.loss / cfg['grad_accum_steps']

            scaler.scale(loss).backward()

            if (step + 1) % cfg['grad_accum_steps'] == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimiser)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimiser.zero_grad()

            total_loss += outputs.loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images         = images.to(device)
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels         = labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Sample generations
        print(f"\nEpoch {epoch}: train={avg_train_loss:.4f} | val={avg_val_loss:.4f}")
        print("  Samples:")
        sample_batch = next(iter(val_loader))
        for i in range(min(3, len(sample_batch[0]))):
            report = model.generate_report(
                sample_batch[0][i].to(device), tokenizer,
                max_length=cfg['gen_max_length']
            )
            print(f"    [{i+1}] {report[:120]}...")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': avg_val_loss,
            }, DECODER_CKPT)
            print(f"  ★ New best val_loss={avg_val_loss:.4f} — saved")

        log.append({'epoch': epoch, 'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss})

    with open(f'{RESULTS_DIR}/decoder_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n  Phase 2 Complete! Best val_loss: {best_val_loss:.4f}")
    return best_val_loss


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  VLM Medical Pipeline — Training (BioGPT)")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    preprocess_iu_xray()

    model = MedicalVLM(device=device, freeze_encoder=True).to(device)

    # Phase 1
    best_auc = train_classifier(model, device)

    # Load best classifier before decoder training
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"[VLM] Loaded classifier (AUC={ckpt['val_mean_auc']:.4f})")

    # Phase 2
    best_loss = train_decoder(model, device)

    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Classifier AUC: {best_auc:.4f}")
    print(f"  Decoder loss:   {best_loss:.4f}")
    print(f"  Run evaluate.py for full results")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
