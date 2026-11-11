"""Generate notebooks/01_lesion_segmentation.ipynb (keeps the notebook reviewable as code).

Usage: python scripts/make_lesion_notebook.py
"""

import json
from pathlib import Path

SAM2UNET_COMMIT = "01598e5e9912ffb23f965ecbebf4d1dfecbaa56e"  # WZH0120/SAM2-UNet main, 2026-01-16


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# AgriCL-Net, step 1: lesion segmentation (SAM2-UNet on PlantSeg)

Trains a binary **lesion vs. background** model. All PlantSeg diseases share one "lesion" class, so the model
also works on diseases it has never seen. Three diseases (soybean bacterial blight, wheat stripe rust,
apple rust) are **held out of training** and evaluated separately: that is the project's key experiment.

**Before you start**
1. Runtime > Change runtime type > **T4 GPU**.
2. Upload the project's `agricl` folder to Google Drive at `MyDrive/AGRI-NET/agricl`.
3. Run the cells in order. The first run is a **smoke test** (40 images, 1 epoch). When it finishes
   without errors, set `SMOKE_TEST = False` and choose Runtime > Run all.

Everything slow (dataset zip, SAM2 checkpoint, training checkpoints) is saved to Drive, so a disconnect only
loses the current epoch: run all cells again and training resumes.
"""),
    md("## 1. Settings"),
    code("""
SIZE = 'l'          # Hiera backbone: 't' (fastest), 's', 'b+', 'l' (most accurate, the SAM2-UNet default)
IMG_SIZE = 352      # SAM2-UNet default; must be a multiple of 32
BATCH = 4           # lower to 2 if cell 6 reports out-of-memory
EPOCHS = 20         # SAM2-UNet default
SMOKE_TEST = True   # True: 40 images, 1 epoch. Set False for the real run.
"""),
    md("## 2. GPU check"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("## 3. Google Drive and project code"),
    code("""
import os, sys
from google.colab import drive
drive.mount('/content/drive')

PROJECT = '/content/drive/MyDrive/AGRI-NET'
CACHE = f'{PROJECT}/cache'                       # dataset zip + SAM2 checkpoint, downloaded once
RUN = f"lesion_sam2unet_{SIZE.replace('+', 'p')}_{IMG_SIZE}" + ('_smoke' if SMOKE_TEST else '')
OUT = f'{PROJECT}/runs/{RUN}'
os.makedirs(CACHE, exist_ok=True)
assert os.path.isdir(f'{PROJECT}/agricl'), 'Upload the agricl folder to MyDrive/AGRI-NET/agricl first'
sys.path.insert(0, PROJECT)
print('outputs ->', OUT)
"""),
    md("## 4. PlantSeg v7 dataset (1.1 GB, cached on Drive)"),
    code("""
import hashlib
ZIP = f'{CACHE}/plantseg.zip'
if not os.path.exists(ZIP):
    !wget -q --show-progress -O "{ZIP}" "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"

h = hashlib.md5()
with open(ZIP, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
        h.update(chunk)
assert h.hexdigest() == '9358a66dff88cdd15c4fe009763c40a3', f'bad download, md5 {h.hexdigest()}: delete {ZIP} and rerun'

if not os.path.exists('/content/data/plantseg/Metadata.csv'):
    !unzip -q "{ZIP}" -d /content/data
!ls /content/data/plantseg
"""),
    md("## 5. SAM2-UNet code and SAM2 checkpoint"),
    code(f"""
!pip -q install hydra-core==1.3.2
if not os.path.exists('/content/SAM2-UNet'):
    !git clone -q https://github.com/WZH0120/SAM2-UNet.git /content/SAM2-UNet
    !git -C /content/SAM2-UNet checkout -q {SAM2UNET_COMMIT}

from agricl.models.sam2unet import CHECKPOINTS, CHECKPOINT_URL
CKPT = f'{{CACHE}}/{{CHECKPOINTS[SIZE]}}'
if not os.path.exists(CKPT):
    !wget -q --show-progress -O "{{CKPT}}" "{{CHECKPOINT_URL}}{{CHECKPOINTS[SIZE]}}"
print(CKPT, os.path.getsize(CKPT) // 2**20, 'MB')
"""),
    md("## 6. Model check: one training step on random data (shapes and GPU memory)"),
    code("""
import torch
from agricl.models.sam2unet import LesionSAM2UNet, structure_loss

m = LesionSAM2UNet('/content/SAM2-UNet', SIZE, CKPT).cuda().train()
x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device='cuda')
y = (torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE, device='cuda') > 0.8).float()
with torch.autocast('cuda', dtype=torch.float16):
    outs = m(x)
loss = sum(structure_loss(o.float(), y) for o in outs)
loss.backward()
print('outputs:', [tuple(o.shape) for o in outs], 'loss:', round(loss.item(), 4))
print(f'peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
del m, x, y, outs, loss
torch.cuda.empty_cache()
"""),
    md("## 7. Train (resumes automatically from Drive)"),
    code("""
from agricl.train_lesion import TrainConfig, train, final_report

cfg = TrainConfig(
    data_root='/content/data/plantseg', out_dir=OUT, repo_dir='/content/SAM2-UNet',
    size=SIZE, checkpoint=CKPT, img_size=IMG_SIZE, batch_size=BATCH,
    epochs=1 if SMOKE_TEST else EPOCHS, limit=40 if SMOKE_TEST else 0, workers=2,
)
model = train(cfg)
"""),
    md("""
## 8. Final evaluation at full resolution

* `test_seen`: PlantSeg test images of the diseases used in training.
* `unseen`: all images of the three held-out diseases. If these scores are close to `test_seen`,
  the lesion model works on diseases it has never seen.
* `share_r2` / `share_rmse`: how well the predicted lesion share of the image matches the true share
  (a stand-in for severity until leaf masks are added).
"""),
    code("""
results = final_report(cfg, model)
"""),
    md("## 9. Training curve and example predictions"),
    code("""
import json, matplotlib.pyplot as plt, numpy as np
from agricl.data.plantseg import load_image, load_lesion_mask
from agricl.train_lesion import build_splits, predict_mask

hist = json.load(open(f'{OUT}/history.json'))
ep = [h['epoch'] for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(ep, [h['train_loss'] for h in hist]); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')
ax[1].plot(ep, [h['val_mean_dice'] for h in hist]); ax[1].set_title('val mean Dice'); ax[1].set_xlabel('epoch')
plt.show()

samples = build_splits(cfg)['unseen'][:4]
fig, ax = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
for row, r in zip(np.atleast_2d(ax), samples):
    img = load_image(r)
    row[0].imshow(img); row[0].set_title(r.disease)
    row[1].imshow(load_lesion_mask(r), cmap='gray'); row[1].set_title('true lesion')
    row[2].imshow(predict_mask(model, img, IMG_SIZE, 'cuda'), cmap='gray'); row[2].set_title('predicted')
    for a in row: a.axis('off')
plt.tight_layout(); plt.show()
"""),
    md("""
## Next
* Smoke test passed? Set `SMOKE_TEST = False` in cell 1 and run all cells.
* Results are in `MyDrive/AGRI-NET/runs/<run name>/`: `best.pt`, `history.json`, `results.json` and per-image CSVs.
* If one epoch takes too long for a Colab session, try `SIZE = 'b+'` or `'s'` and compare the scores.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "01_lesion_segmentation.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
