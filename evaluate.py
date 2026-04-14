"""
evaluate.py — Full Evaluation Pipeline (BioGPT version)
EEE3094 Dissertation — Sami Zarroug (220672267)

Covers all three SW3 objectives:
  1. Classification: AUROC, F1 (per-class optimal thresholds), Precision, Recall
  2. Report Generation: BLEU-1/2/3/4, ROUGE-L, mention recall
  3. Visualization: Grad-CAM heatmaps

Run from Colab:
    !python /content/vlm-medical/evaluate.py
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, roc_curve,
)
from transformers import BioGptTokenizer

from config import (
    CKPT_DIR, RESULTS_DIR, DECODER_CKPT,
    DECODER_MODEL_NAME,
    PATHOLOGY_CLASSES, PATHOLOGY_KEYWORDS, NUM_CLASSES,
    DECODER_CFG,
)
from dataset import preprocess_iu_xray, IUXRayDataset, get_transforms
from model import MedicalVLM


# ══════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION EVALUATION
# ══════════════════════════════════════════════════════════════════

def compute_optimal_thresholds(y_true, y_score, classes):
    """
    Per-class threshold from validation data.
    First maximise F1 over a wide threshold range.
    If that still gives F1=0, fall back to Youden's J statistic.
    """
    thresholds = {}
    for i, cls in enumerate(classes):
        true_col = y_true[:, i]
        score_col = y_score[:, i]
        if true_col.sum() < 2:
            thresholds[cls] = 0.5
            continue

        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.01, 0.99, 0.01):
            preds = (score_col >= t).astype(int)
            f1 = f1_score(true_col, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        if best_f1 == 0.0:
            fpr, tpr, thr = roc_curve(true_col, score_col)
            j = tpr - fpr
            best_t = float(thr[np.argmax(j)])
            best_t = float(np.clip(best_t, 0.01, 0.99))

        thresholds[cls] = round(best_t, 2)
    return thresholds


def evaluate_classification(model, test_loader, device, val_loader=None):
    print(f"\n{'='*60}")
    print(f"  Classification Evaluation")
    print(f"{'='*60}")

    model.eval()

    def collect(loader):
        scores, targets = [], []
        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc="Classifying"):
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model.classify(images)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(labels.numpy())
        return np.concatenate(scores), np.concatenate(targets)

    # Optimal thresholds from validation
    if val_loader:
        val_scores, val_targets = collect(val_loader)
        opt_thresh = compute_optimal_thresholds(val_targets, val_scores, PATHOLOGY_CLASSES)
        print(f"\n  Per-class optimal thresholds:")
        for cls, t in opt_thresh.items():
            print(f"    {cls:<20s}: {t:.2f}")
    else:
        opt_thresh = {cls: 0.5 for cls in PATHOLOGY_CLASSES}

    # Test evaluation
    test_scores, test_targets = collect(test_loader)

    print(f"\n  {'Class':<20s}  {'AUC':>6}  {'AP':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Thr':>5}")
    print(f"  {'-'*70}")

    results = {'per_class': {}, 'optimal_thresholds': opt_thresh}
    all_aucs, all_f1s = [], []

    for i, cls in enumerate(PATHOLOGY_CLASSES):
        true_col = test_targets[:, i]
        score_col = test_scores[:, i]
        n_pos = true_col.sum()

        if n_pos < 2:
            results['per_class'][cls] = {'auc': float('nan'), 'n_positive': int(n_pos)}
            print(f"  {cls:<20s}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  (n={int(n_pos)})")
            continue

        auc = roc_auc_score(true_col, score_col)
        ap = average_precision_score(true_col, score_col)
        t = opt_thresh[cls]
        preds = (score_col >= t).astype(int)
        f1 = f1_score(true_col, preds, zero_division=0)
        prec = precision_score(true_col, preds, zero_division=0)
        rec = recall_score(true_col, preds, zero_division=0)

        results['per_class'][cls] = {
            'auc': auc, 'ap': ap, 'f1': f1, 'precision': prec,
            'recall': rec, 'threshold': t, 'n_positive': int(n_pos),
        }
        all_aucs.append(auc)
        all_f1s.append(f1)

        print(f"  {cls:<20s}  {auc:>6.4f}  {ap:>6.4f}  {f1:>6.4f}  {prec:>6.4f}  {rec:>6.4f}  {t:>5.2f}")

    mean_auc = np.mean(all_aucs) if all_aucs else 0.0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0
    results['mean_auc'] = mean_auc
    results['mean_f1'] = mean_f1

    print(f"  {'-'*70}")
    print(f"  {'MEAN':<20s}  {mean_auc:>6.4f}          {mean_f1:>6.4f}")
    return results


# ══════════════════════════════════════════════════════════════════
# 2. REPORT GENERATION EVALUATION
# ══════════════════════════════════════════════════════════════════

def compute_bleu(reference, hypothesis, max_n=4):
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not hyp_tok:
        return {f'bleu_{n}': 0.0 for n in range(1, max_n+1)}

    scores = {}
    for n in range(1, max_n+1):
        ref_ng = Counter(tuple(ref_tok[i:i+n]) for i in range(len(ref_tok)-n+1))
        hyp_ng = Counter(tuple(hyp_tok[i:i+n]) for i in range(len(hyp_tok)-n+1))
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        scores[f'bleu_{n}'] = clipped / total if total > 0 else 0.0

    bp = min(1.0, np.exp(1 - len(ref_tok) / max(len(hyp_tok), 1)))
    for n in range(1, max_n+1):
        precs = [scores[f'bleu_{k}'] for k in range(1, n+1)]
        if all(p > 0 for p in precs):
            scores[f'bleu_{n}_cum'] = bp * np.exp(sum(np.log(p) for p in precs)/n)
        else:
            scores[f'bleu_{n}_cum'] = 0.0
    return scores


def compute_rouge_l(reference, hypothesis):
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0
    m, n = len(ref_tok), len(hyp_tok)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_tok[i-1] == hyp_tok[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n if n else 0
    r = lcs/m if m else 0
    return 2*p*r/(p+r) if (p+r) else 0.0


def compute_mention_recall(reference, hypothesis, label_vector):
    """
    What fraction of diseases in ground truth are mentioned in generated text?
    This was 0.259 before — meaning the model missed 74% of diseases.
    """
    active = [PATHOLOGY_CLASSES[i] for i, v in enumerate(label_vector) if v > 0.5]

    if not active:
        hyp_lower = hypothesis.lower()
        has_disease = any(
            kw in hyp_lower
            for kws in PATHOLOGY_KEYWORDS.values() for kw in kws
        )
        return {'mention_recall': 1.0 if not has_disease else 0.0,
                'is_normal': True, 'total': 0, 'mentioned': 0}

    hyp_lower = hypothesis.lower()
    mentioned = sum(
        1 for d in active
        if any(kw.lower() in hyp_lower for kw in PATHOLOGY_KEYWORDS.get(d, [d.lower()]))
    )
    return {'mention_recall': mentioned/len(active),
            'is_normal': False, 'total': len(active), 'mentioned': mentioned}


def evaluate_generation(model, test_samples, device, max_samples=200):
    print(f"\n{'='*60}")
    print(f"  Report Generation Evaluation (BioGPT)")
    print(f"  Samples: {min(max_samples, len(test_samples))}")
    print(f"{'='*60}")

    model.eval()
    tokenizer = BioGptTokenizer.from_pretrained(DECODER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    transform = get_transforms()

    eval_samples = test_samples[:max_samples]

    bleu_scores = {f'bleu_{n}': [] for n in range(1, 5)}
    rouge_scores = []
    mention_recalls = []
    disease_mentions = []
    normal_mentions = []
    sample_outputs = []

    for i, sample in enumerate(tqdm(eval_samples, desc="Generating")):
        from PIL import Image
        img = Image.open(sample['image_path']).convert('RGB')
        img_tensor = transform(img).to(device)

        generated = model.generate_report_beam(
            img_tensor, tokenizer,
            max_new_tokens=DECODER_CFG['gen_max_length'],
            num_beams=DECODER_CFG.get('num_beams', 4),
        )
        reference = sample['report']
        labels = np.array(sample['labels'])

        bleu = compute_bleu(reference, generated)
        for n in range(1, 5):
            bleu_scores[f'bleu_{n}'].append(bleu[f'bleu_{n}'])

        rouge = compute_rouge_l(reference, generated)
        rouge_scores.append(rouge)

        mention = compute_mention_recall(reference, generated, labels)
        mention_recalls.append(mention['mention_recall'])
        if mention['is_normal']:
            normal_mentions.append(mention['mention_recall'])
        else:
            disease_mentions.append(mention['mention_recall'])

        if i < 10:
            diseases = [PATHOLOGY_CLASSES[j] for j, v in enumerate(labels) if v > 0.5]
            sample_outputs.append({
                'labels': diseases if diseases else ['Normal'],
                'reference': reference[:150],
                'generated': generated[:150],
                'bleu_1': bleu['bleu_1'],
                'rouge_l': rouge,
                'mention_recall': mention['mention_recall'],
            })

    results = {
        'bleu_1': np.mean(bleu_scores['bleu_1']),
        'bleu_2': np.mean(bleu_scores['bleu_2']),
        'bleu_3': np.mean(bleu_scores['bleu_3']),
        'bleu_4': np.mean(bleu_scores['bleu_4']),
        'rouge_l': np.mean(rouge_scores),
        'mention_recall_overall': np.mean(mention_recalls),
        'mention_recall_disease': np.mean(disease_mentions) if disease_mentions else 0.0,
        'mention_recall_normal': np.mean(normal_mentions) if normal_mentions else 0.0,
        'n_disease': len(disease_mentions),
        'n_normal': len(normal_mentions),
    }

    print(f"\n  Generation Metrics:")
    print(f"    BLEU-1:         {results['bleu_1']:.4f}")
    print(f"    BLEU-2:         {results['bleu_2']:.4f}")
    print(f"    BLEU-3:         {results['bleu_3']:.4f}")
    print(f"    BLEU-4:         {results['bleu_4']:.4f}")
    print(f"    ROUGE-L:        {results['rouge_l']:.4f}")
    print(f"    Mention recall: {results['mention_recall_overall']:.4f}")
    print(f"      Disease:      {results['mention_recall_disease']:.4f} (n={results['n_disease']})")
    print(f"      Normal:       {results['mention_recall_normal']:.4f} (n={results['n_normal']})")

    print(f"\n  Sample Outputs:")
    for i, s in enumerate(sample_outputs[:5]):
        print(f"\n    [{i+1}] GT: {', '.join(s['labels'])}")
        print(f"    Ref: {s['reference']}")
        print(f"    Gen: {s['generated']}")
        print(f"    B1={s['bleu_1']:.3f} | RL={s['rouge_l']:.3f} | MR={s['mention_recall']:.3f}")

    results['sample_outputs'] = sample_outputs
    return results


# ══════════════════════════════════════════════════════════════════
# 3. GRAD-CAM
# ══════════════════════════════════════════════════════════════════

def generate_gradcam_figures(model, test_samples, device, n_samples=6):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
        from PIL import Image
    except ImportError:
        print("[Grad-CAM] matplotlib not available")
        return

    save_dir = f'{RESULTS_DIR}/gradcam_figures'
    os.makedirs(save_dir, exist_ok=True)
    transform = get_transforms(is_train=False)

    print(f"\n  Generating {n_samples} Grad-CAM heatmaps...")

    # Enable classification gradients for the visual encoder
    model.freeze_for_classification()
    model.eval()

    for idx, sample in enumerate(test_samples[:n_samples]):
        pil_img = Image.open(sample['image_path']).convert('RGB')
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model.classify(img_tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        gt_indices = [j for j, v in enumerate(sample['labels']) if v > 0.5]
        target_cls = gt_indices[0] if gt_indices else int(np.argmax(probs))

        score = logits[0, target_cls]
        score.backward()

        cam = None
        if model._patch_features is not None and model._patch_features.grad is not None:
            grads = model._patch_features.grad[0]      # (49, 768)
            acts = model._patch_features[0].detach()   # (49, 768)
            weights = grads.mean(dim=0)                # (768,)
            cam = torch.relu((acts * weights).sum(dim=-1)).reshape(7, 7)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cam.detach().cpu().numpy()

        if cam is None:
            cam = np.zeros((7, 7), dtype=np.float32)

        cam_resized = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(pil_img.resize((224, 224)), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Heatmap: {PATHOLOGY_CLASSES[target_cls]}')
        axes[1].axis('off')

        img_arr = np.array(pil_img.resize((224, 224))) / 255.0
        axes[2].imshow(img_arr, cmap='gray')
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
        axes[2].set_title(f'{PATHOLOGY_CLASSES[target_cls]} (P={probs[target_cls]:.3f})')
        axes[2].axis('off')

        gt = [PATHOLOGY_CLASSES[j] for j, v in enumerate(sample['labels']) if v > 0.5]
        fig.suptitle(f'GT: {", ".join(gt) if gt else "Normal"}', fontsize=12, y=0.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/heatmap_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved heatmap_{idx+1}.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Full Evaluation (BioGPT Pipeline)")
    print(f"{'='*60}\n")

    model = MedicalVLM(device=device, freeze_encoder=True).to(device)

    if os.path.exists(DECODER_CKPT):
        ckpt = torch.load(DECODER_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"[Eval] Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['val_loss']:.4f})")
    else:
        print(f"[Eval] WARNING: No checkpoint at {DECODER_CKPT}")

    samples = preprocess_iu_xray()
    transform = get_transforms()

    val_samples  = [s for s in samples if s['split'] == 'val']
    test_samples = [s for s in samples if s['split'] == 'test']

    from torch.utils.data import DataLoader
    val_loader  = DataLoader(IUXRayDataset(val_samples, transform), batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(IUXRayDataset(test_samples, transform), batch_size=32, shuffle=False, num_workers=2)

    # 1. Classification
    cls_results = evaluate_classification(model, test_loader, device, val_loader)

    # 2. Report generation
    gen_results = evaluate_generation(model, test_samples, device, max_samples=200)

    # 3. Grad-CAM
    generate_gradcam_figures(model, test_samples, device, n_samples=6)

    # Save
    save_data = {'classification': cls_results}
    save_data['generation'] = {k: v for k, v in gen_results.items() if k != 'sample_outputs'}
    save_data['generation']['sample_outputs'] = gen_results.get('sample_outputs', [])

    with open(f'{RESULTS_DIR}/full_evaluation.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Done! Results: {RESULTS_DIR}/full_evaluation.json")
    print(f"  Grad-CAM:      {RESULTS_DIR}/gradcam_figures/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
