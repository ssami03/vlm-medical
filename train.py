"""
train.py — Full training pipeline
Phase 1: classification
Phase 2: report generation
"""

import json
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BioGptTokenizer, get_cosine_schedule_with_warmup

from config import (
    CLASSIFIER_CFG,
    CLASSIFIER_CKPT,
    DECODER_CFG,
    DECODER_CKPT,
    DECODER_MODEL_NAME,
    PATHOLOGY_CLASSES,
    RESULTS_DIR,
)
from dataset import (
    IUXRayDataset,
    build_dataloaders,
    decoder_collate_fn,
    get_transforms,
    preprocess_iu_xray,
)
from model import MedicalVLM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_classifier(model: MedicalVLM, device: torch.device):
    print(f"\n{'=' * 60}")
    print('PHASE 1: Classification training')
    print(f"{'=' * 60}")

    cfg = CLASSIFIER_CFG
    set_seed(cfg['seed'])

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        balance_train=True,
    )

    model.freeze_for_classification()
    optimiser = torch.optim.AdamW(model.get_classifier_param_groups(), weight_decay=cfg['weight_decay'])
    total_steps = len(train_loader) * cfg['epochs']
    warmup_steps = len(train_loader) * cfg['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_auc = 0.0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f'Cls {epoch}/{cfg["epochs"]}'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smooth_labels = labels * (1 - cfg['label_smoothing']) + (1 - labels) * cfg['label_smoothing']

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model.classify(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, smooth_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        all_scores, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model.classify(images)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                val_loss += loss.item()
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        y_score = np.concatenate(all_scores)
        y_true = np.concatenate(all_targets)

        aucs = {}
        for i, cls in enumerate(PATHOLOGY_CLASSES):
            if y_true[:, i].sum() >= 2:
                try:
                    aucs[cls] = roc_auc_score(y_true[:, i], y_score[:, i])
                except ValueError:
                    aucs[cls] = float('nan')
            else:
                aucs[cls] = float('nan')

        valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        print(f'\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | mean_auc={mean_auc:.4f}')

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_mean_auc': mean_auc,
                'per_class_auc': aucs,
            }, CLASSIFIER_CKPT)
            print(f'  New best classifier checkpoint saved: AUC={mean_auc:.4f}')

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'mean_auc': mean_auc,
        })

    with open(f'{RESULTS_DIR}/classifier_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_auc



def train_decoder(model: MedicalVLM, device: torch.device):
    print(f"\n{'=' * 60}")
    print('PHASE 2: Report generation training')
    print(f"{'=' * 60}")

    cfg = DECODER_CFG
    set_seed(cfg['seed'])

    tokenizer = BioGptTokenizer.from_pretrained(DECODER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = preprocess_iu_xray()
    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples = [s for s in samples if s['split'] == 'val']

    train_ds = IUXRayDataset(train_samples, get_transforms(is_train=True))
    val_ds = IUXRayDataset(val_samples, get_transforms(is_train=False))
    collate = partial(decoder_collate_fn, tokenizer=tokenizer, max_length=cfg['max_length'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], collate_fn=collate, pin_memory=True)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Phase 2a: prewarm bridge only
    model.freeze_for_decoder_prewarm()
    prewarm_params = [p for p in model.parameters() if p.requires_grad]
    prewarm_optim = torch.optim.AdamW(prewarm_params, lr=cfg['prewarm_lr'], weight_decay=1e-4)

    for epoch in range(1, cfg['prewarm_epochs'] + 1):
        model.train()
        model.visual.eval()
        total_loss = 0.0
        for images, input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Prewarm {epoch}/{cfg["prewarm_epochs"]}'):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            prewarm_optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(prewarm_optim)
            torch.nn.utils.clip_grad_norm_(prewarm_params, max_norm=1.0)
            scaler.step(prewarm_optim)
            scaler.update()
            total_loss += loss.item()
        print(f'Prewarm epoch {epoch}: loss={total_loss / max(len(train_loader), 1):.4f}')

    # Phase 2b: full decoder training
    model.freeze_for_decoder_full()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable_params, lr=cfg['lr'], weight_decay=1e-4)

    total_steps = max((len(train_loader) // cfg['grad_accum_steps']) * cfg['epochs'], 1)
    warmup_steps = max((len(train_loader) // cfg['grad_accum_steps']) * cfg['warmup_epochs'], 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        model.visual.eval()
        total_loss = 0.0
        optimiser.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f'Dec {epoch}/{cfg["epochs"]}'), start=1):
            images, input_ids, attention_mask, labels = batch
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                loss = outputs.loss / cfg['grad_accum_steps']

            scaler.scale(loss).backward()

            if step % cfg['grad_accum_steps'] == 0 or step == len(train_loader):
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                prev_scale = scaler.get_scale()
                scaler.step(optimiser)
                scaler.update()
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
                optimiser.zero_grad(set_to_none=True)

            total_loss += outputs.loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images = images.to(device, non_blocking=True)
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model.forward_decoder(images, input_ids, attention_mask, labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f'\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}')
        try:
            sample_images, _, _, _ = next(iter(val_loader))
            for i in range(min(2, len(sample_images))):
                sample_text = model.generate_report(sample_images[i].to(device), tokenizer, max_length=cfg['gen_max_length'])
                print(f'  Sample {i + 1}: {sample_text[:140]}...')
        except Exception as exc:
            print(f'  Sample generation skipped: {exc}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': avg_val_loss,
            }, DECODER_CKPT)
            print(f'  New best decoder checkpoint saved: val_loss={avg_val_loss:.4f}')

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        })

    with open(f'{RESULTS_DIR}/decoder_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_loss



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print('VLM Medical Pipeline — Training')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f"{'=' * 60}\n")

    preprocess_iu_xray()
    model = MedicalVLM(device=device, freeze_encoder=True).to(device)

    best_auc = train_classifier(model, device)
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"[VLM] Loaded best classifier checkpoint (AUC={ckpt['val_mean_auc']:.4f})")

    best_loss = train_decoder(model, device)

    print(f"\n{'=' * 60}")
    print('ALL TRAINING COMPLETE')
    print(f'Classifier AUC: {best_auc:.4f}')
    print(f'Decoder val_loss: {best_loss:.4f}')
    print('Run evaluate.py next.')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
