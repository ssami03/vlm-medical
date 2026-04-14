"""
evaluate.py — Full evaluation pipeline
"""

import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BioGptTokenizer

from config import (
    DECODER_CFG,
    DECODER_CKPT,
    DECODER_MODEL_NAME,
    PATHOLOGY_CLASSES,
    PATHOLOGY_KEYWORDS,
    RESULTS_DIR,
)
from dataset import IUXRayDataset, get_transforms, preprocess_iu_xray
from model import MedicalVLM



def compute_optimal_thresholds(y_true, y_score, classes):
    thresholds = {}
    for i, cls in enumerate(classes):
        if y_true[:, i].sum() < 2:
            thresholds[cls] = 0.5
            continue
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            preds = (y_score[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[cls] = round(float(best_t), 2)
    return thresholds



def evaluate_classification(model, test_loader, device, val_loader=None):
    def collect(loader):
        scores, targets = [], []
        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc='Classifying'):
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model.classify(images)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(labels.numpy())
        return np.concatenate(scores), np.concatenate(targets)

    if val_loader is not None:
        val_scores, val_targets = collect(val_loader)
        opt_thresh = compute_optimal_thresholds(val_targets, val_scores, PATHOLOGY_CLASSES)
    else:
        opt_thresh = {cls: 0.5 for cls in PATHOLOGY_CLASSES}

    test_scores, test_targets = collect(test_loader)

    results = {'per_class': {}, 'optimal_thresholds': opt_thresh}
    all_aucs, all_f1s = [], []

    print(f"\n{'Class':<20s} {'AUC':>6s} {'AP':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Thr':>5s}")
    print('-' * 64)

    for i, cls in enumerate(PATHOLOGY_CLASSES):
        true_col = test_targets[:, i]
        score_col = test_scores[:, i]
        n_pos = int(true_col.sum())
        if n_pos < 2:
            results['per_class'][cls] = {'auc': float('nan'), 'n_positive': n_pos}
            print(f'{cls:<20s} {"N/A":>6s} {"N/A":>6s} {"N/A":>6s} {"N/A":>6s} {"N/A":>6s} {"--":>5s}')
            continue

        auc = roc_auc_score(true_col, score_col)
        ap = average_precision_score(true_col, score_col)
        threshold = opt_thresh[cls]
        preds = (score_col >= threshold).astype(int)
        f1 = f1_score(true_col, preds, zero_division=0)
        prec = precision_score(true_col, preds, zero_division=0)
        rec = recall_score(true_col, preds, zero_division=0)

        results['per_class'][cls] = {
            'auc': float(auc),
            'ap': float(ap),
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'threshold': float(threshold),
            'n_positive': n_pos,
        }
        all_aucs.append(auc)
        all_f1s.append(f1)
        print(f'{cls:<20s} {auc:>6.4f} {ap:>6.4f} {f1:>6.4f} {prec:>6.4f} {rec:>6.4f} {threshold:>5.2f}')

    results['mean_auc'] = float(np.mean(all_aucs)) if all_aucs else 0.0
    results['mean_f1'] = float(np.mean(all_f1s)) if all_f1s else 0.0
    print('-' * 64)
    print(f"{'MEAN':<20s} {results['mean_auc']:>6.4f} {'':>6s} {results['mean_f1']:>6.4f}")
    return results



def compute_bleu(reference, hypothesis, max_n=4):
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not hyp_tok:
        return {f'bleu_{n}': 0.0 for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_tok[i:i + n]) for i in range(len(ref_tok) - n + 1))
        hyp_ng = Counter(tuple(hyp_tok[i:i + n]) for i in range(len(hyp_tok) - n + 1))
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        scores[f'bleu_{n}'] = clipped / total if total > 0 else 0.0
    return scores



def compute_rouge_l(reference, hypothesis):
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0
    m, n = len(ref_tok), len(hyp_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0



def compute_mention_recall(hypothesis, label_vector):
    active = [PATHOLOGY_CLASSES[i] for i, v in enumerate(label_vector) if v > 0.5]
    hyp_lower = hypothesis.lower()

    if not active:
        has_disease = any(
            keyword in hyp_lower
            for keywords in PATHOLOGY_KEYWORDS.values()
            for keyword in keywords
        )
        return {'mention_recall': 1.0 if not has_disease else 0.0, 'is_normal': True}

    mentioned = 0
    for disease in active:
        keywords = PATHOLOGY_KEYWORDS.get(disease, [disease.lower()])
        if any(keyword.lower() in hyp_lower for keyword in keywords):
            mentioned += 1
    return {
        'mention_recall': mentioned / len(active),
        'is_normal': False,
    }



def evaluate_generation(model, test_samples, device, max_samples=200):
    tokenizer = BioGptTokenizer.from_pretrained(DECODER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    transform = get_transforms(is_train=False)

    eval_samples = test_samples[:max_samples]
    bleu_scores = {f'bleu_{n}': [] for n in range(1, 5)}
    rouge_scores = []
    mention_scores = []
    sample_outputs = []

    for i, sample in enumerate(tqdm(eval_samples, desc='Generating')):
        from PIL import Image
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = transform(image).to(device)
        generated = model.generate_report(image_tensor, tokenizer, max_length=DECODER_CFG['gen_max_length'])
        reference = sample['report']
        labels = np.array(sample['labels'])

        bleu = compute_bleu(reference, generated)
        for n in range(1, 5):
            bleu_scores[f'bleu_{n}'].append(bleu[f'bleu_{n}'])
        rouge_scores.append(compute_rouge_l(reference, generated))
        mention = compute_mention_recall(generated, labels)
        mention_scores.append(mention['mention_recall'])

        if i < 10:
            diseases = [PATHOLOGY_CLASSES[j] for j, v in enumerate(labels) if v > 0.5]
            sample_outputs.append({
                'labels': diseases if diseases else ['Normal'],
                'reference': reference[:180],
                'generated': generated[:180],
                'mention_recall': float(mention['mention_recall']),
            })

    results = {
        'bleu_1': float(np.mean(bleu_scores['bleu_1'])) if bleu_scores['bleu_1'] else 0.0,
        'bleu_2': float(np.mean(bleu_scores['bleu_2'])) if bleu_scores['bleu_2'] else 0.0,
        'bleu_3': float(np.mean(bleu_scores['bleu_3'])) if bleu_scores['bleu_3'] else 0.0,
        'bleu_4': float(np.mean(bleu_scores['bleu_4'])) if bleu_scores['bleu_4'] else 0.0,
        'rouge_l': float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        'mention_recall_overall': float(np.mean(mention_scores)) if mention_scores else 0.0,
        'sample_outputs': sample_outputs,
    }

    print('\nGeneration metrics:')
    print(f"  BLEU-1: {results['bleu_1']:.4f}")
    print(f"  BLEU-2: {results['bleu_2']:.4f}")
    print(f"  BLEU-3: {results['bleu_3']:.4f}")
    print(f"  BLEU-4: {results['bleu_4']:.4f}")
    print(f"  ROUGE-L: {results['rouge_l']:.4f}")
    print(f"  Mention recall: {results['mention_recall_overall']:.4f}")
    return results



def generate_heatmaps(model, test_samples, device, n_samples=6):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
        from PIL import Image
    except ImportError:
        print('[Heatmap] matplotlib unavailable, skipping.')
        return

    save_dir = f'{RESULTS_DIR}/heatmaps'
    os.makedirs(save_dir, exist_ok=True)
    transform = get_transforms(is_train=False)

    for idx, sample in enumerate(test_samples[:n_samples], start=1):
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = transform(image).to(device)

        model.zero_grad(set_to_none=True)
        img_input = image_tensor.unsqueeze(0).requires_grad_(True)
        logits = model.classify(img_input)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        top_cls = int(np.argmax(probs))
        logits[0, top_cls].backward()

        if model._patch_features is not None and model._patch_features.grad is not None:
            cam = model._patch_features.grad.abs().mean(dim=-1)[0]
            cam = cam.reshape(7, 7)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cam.detach().cpu().numpy()
        else:
            cam = np.ones((7, 7), dtype=np.float32) / 49.0

        cam_resized = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.resize((224, 224)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Heatmap: {PATHOLOGY_CLASSES[top_cls]}')
        axes[1].axis('off')
        axes[2].imshow(np.array(image.resize((224, 224))) / 255.0)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
        axes[2].set_title(f'{PATHOLOGY_CLASSES[top_cls]} (P={probs[top_cls]:.3f})')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/heatmap_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[Heatmap] Saved heatmap_{idx}.png')



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalVLM(device=device, freeze_encoder=True).to(device)

    if not os.path.exists(DECODER_CKPT):
        raise FileNotFoundError(f'Checkpoint not found: {DECODER_CKPT}')

    ckpt = torch.load(DECODER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"[Eval] Loaded checkpoint from epoch {ckpt['epoch']}")

    samples = preprocess_iu_xray()
    val_samples = [s for s in samples if s['split'] == 'val']
    test_samples = [s for s in samples if s['split'] == 'test']

    val_loader = DataLoader(IUXRayDataset(val_samples, get_transforms(is_train=False)),
                            batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(IUXRayDataset(test_samples, get_transforms(is_train=False)),
                             batch_size=32, shuffle=False, num_workers=2)

    cls_results = evaluate_classification(model, test_loader, device, val_loader)
    gen_results = evaluate_generation(model, test_samples, device, max_samples=200)
    generate_heatmaps(model, test_samples, device, n_samples=6)

    with open(f'{RESULTS_DIR}/full_evaluation.json', 'w') as f:
        json.dump({
            'classification': cls_results,
            'generation': gen_results,
        }, f, indent=2)

    print(f'\nDone. Results saved to {RESULTS_DIR}/full_evaluation.json')


if __name__ == '__main__':
    main()
