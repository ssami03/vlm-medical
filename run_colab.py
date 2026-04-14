"""
run_colab.py — Copy each quoted block into a separate Colab cell.
"""

# CELL 1 — Setup
"""
import os
import sys
from google.colab import drive

drive.mount('/content/drive')

REPO_URL = 'https://github.com/ssami03/vlm-medical.git'
REPO_DIR = '/content/vlm-medical'

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

!pip install -q -r /content/vlm-medical/requirements.txt

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print('✅ Setup complete')
"""

# CELL 2 — Verify IU X-Ray files exist in Drive
"""
import os
DATA_DIR = '/content/drive/MyDrive/iu_xray_data'

for f in ['indiana_reports.csv', 'indiana_projections.csv']:
    path = f'{DATA_DIR}/{f}'
    print(f"{'✓' if os.path.exists(path) else '✗'} {f}")

for candidate in ['images/images_normalized', 'images', 'images_normalized']:
    p = f'{DATA_DIR}/{candidate}'
    if os.path.exists(p):
        n = len([name for name in os.listdir(p) if name.endswith('.png')])
        print(f'✓ {n} images at {candidate}')
        break
else:
    print('✗ No images found — check your Drive upload')
"""

# CELL 3 — Preprocess and inspect label distribution
"""
from dataset import preprocess_iu_xray
samples = preprocess_iu_xray(force_reprocess=True)
print(f'Total samples: {len(samples)}')
print(f"Train: {sum(1 for s in samples if s['split'] == 'train')}")
print(f"Val:   {sum(1 for s in samples if s['split'] == 'val')}")
print(f"Test:  {sum(1 for s in samples if s['split'] == 'test')}")
"""

# CELL 4 — Smoke test before full training
"""
import torch
from transformers import BioGptTokenizer
from dataset import build_dataloaders
from model import MedicalVLM
from config import DECODER_MODEL_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _, _ = build_dataloaders(batch_size=2, num_workers=2, balance_train=False)
images, labels, reports = next(iter(train_loader))

model = MedicalVLM(device=device, freeze_encoder=True).to(device)
model.freeze_for_classification()
with torch.no_grad():
    logits = model.classify(images.to(device))
print('Classifier output shape:', logits.shape)

tokenizer = BioGptTokenizer.from_pretrained(DECODER_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sample_report = model.generate_report(images[0].to(device), tokenizer, max_length=20)
print('Generated sample:', sample_report)
print('✅ Smoke test passed')
"""

# CELL 5 — Train full pipeline
"""
from train import main
main()
"""

# CELL 6 — Run evaluation
"""
from evaluate import main as run_eval
run_eval()
"""
