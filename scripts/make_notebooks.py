"""Generate the Colab notebooks in notebooks/ (edit this file, not the .ipynb).

Usage: python scripts/make_notebooks.py
"""

import json
from pathlib import Path

REPO_URL = "https://github.com/Dev02005/AGRICL-NET.git"
SAM2UNET_COMMIT = "01598e5e9912ffb23f965ecbebf4d1dfecbaa56e"  # WZH0120/SAM2-UNet main, 2026-01-16
ROOT = Path(__file__).resolve().parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def write(name, cells):
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
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


# ---------------------------------------------------------------- shared setup
BASE_SETUP = [
    md("## GPU check"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("## Google Drive (cache and results) and project code from GitHub"),
    code(f"""
import os, sys
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/AGRI-NET'   # dataset zip, SAM2 checkpoint and all results are kept here
CACHE = f'{{DRIVE}}/cache'
os.makedirs(CACHE, exist_ok=True)

CODE = '/content/AGRICL-NET'
if os.path.exists(CODE):
    !git -C {{CODE}} pull -q
else:
    !git clone -q {REPO_URL} {{CODE}}
sys.path.insert(0, CODE)
!git -C {{CODE}} log -1 --format='code version: %h %s'
"""),
    md("## PlantSeg v7 dataset (1.1 GB, downloaded to Drive once)"),
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
DATA = '/content/data/plantseg'
"""),
]
SAM2_SETUP = [
    md("## SAM2-UNet code and SAM2 checkpoint"),
    code(f"""
!pip -q install hydra-core==1.3.2
SAM2UNET = '/content/SAM2-UNet'
if not os.path.exists(SAM2UNET):
    !git clone -q https://github.com/WZH0120/SAM2-UNet.git {{SAM2UNET}}
    !git -C {{SAM2UNET}} checkout -q {SAM2UNET_COMMIT}

from agricl.models.sam2unet import CHECKPOINTS, CHECKPOINT_URL
CKPT = f'{{CACHE}}/{{CHECKPOINTS[SIZE]}}'
if not os.path.exists(CKPT):
    !wget -q --show-progress -O "{{CKPT}}" "{{CHECKPOINT_URL}}{{CHECKPOINTS[SIZE]}}"
print(CKPT, os.path.getsize(CKPT) // 2**20, 'MB')
"""),
]
SETUP = BASE_SETUP + SAM2_SETUP

# ------------------------------------------------- 01: lesion segmentation
write("01_lesion_segmentation.ipynb", [
    md("""
# AgriCL-Net, step 1: lesion segmentation (SAM2-UNet on PlantSeg)

Trains a binary **lesion vs. background** model. All PlantSeg diseases share one "lesion" class, so the model
also works on diseases it has never seen. Three diseases (soybean bacterial blight, wheat stripe rust,
apple rust) are **held out of training** and evaluated separately: that is the project's key experiment.

**Before you start:** Runtime > Change runtime type > **T4 GPU**. The project code is cloned from GitHub;
nothing needs uploading. Run the cells in order. The first run is a **smoke test** (40 images, 1 epoch).
When it finishes without errors, set `SMOKE_TEST = False` and choose Runtime > Run all.

Everything slow (dataset zip, SAM2 checkpoint, training checkpoints) is saved to Google Drive, so a
disconnect only loses the current epoch: run all cells again and training resumes.
"""),
    md("## Settings"),
    code("""
SIZE = 'l'          # Hiera backbone: 't' (fastest), 's', 'b+', 'l' (most accurate, the SAM2-UNet default)
IMG_SIZE = 352      # SAM2-UNet default; must be a multiple of 32
BATCH = 4           # lower to 2 if the model check reports out-of-memory
EPOCHS = 20         # SAM2-UNet default
SMOKE_TEST = True   # True: 40 images, 1 epoch. Set False for the real run.
"""),
    *SETUP,
    code("""
RUN = f"lesion_sam2unet_{SIZE.replace('+', 'p')}_{IMG_SIZE}" + ('_smoke' if SMOKE_TEST else '')
OUT = f'{DRIVE}/runs/{RUN}'
print('results ->', OUT)
"""),
    md("## Model check: one training step on random data (shapes and GPU memory)"),
    code("""
import torch
from agricl.models.sam2unet import LesionSAM2UNet, structure_loss

m = LesionSAM2UNet(SAM2UNET, SIZE, CKPT).cuda().train()
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
    md("## Train (resumes automatically from Drive)"),
    code("""
from agricl.train_lesion import TrainConfig, train, final_report

cfg = TrainConfig(
    data_root=DATA, out_dir=OUT, repo_dir=SAM2UNET,
    size=SIZE, checkpoint=CKPT, img_size=IMG_SIZE, batch_size=BATCH,
    epochs=1 if SMOKE_TEST else EPOCHS, limit=40 if SMOKE_TEST else 0, workers=2,
)
model = train(cfg)
"""),
    md("""
## Final evaluation at full resolution

* `test_seen`: PlantSeg test images of the diseases used in training.
* `unseen`: all images of the three held-out diseases. If these scores are close to `test_seen`,
  the lesion model works on diseases it has never seen.
* `share_r2` / `share_rmse`: predicted vs. true lesion share of the whole image.
"""),
    code("""
results = final_report(cfg, model)
"""),
    md("## Training curve and example predictions"),
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
* Smoke test passed? Set `SMOKE_TEST = False` in the settings cell and run all cells.
* Results: `MyDrive/AGRI-NET/runs/<run name>/` (`best.pt`, `history.json`, `results.json`, per-image CSVs).
* If one epoch takes too long for a Colab session, try `SIZE = 'b+'` or `'s'` and compare the scores.
* When the full run is done, open `02_leaf_and_severity.ipynb`.
"""),
])

# ------------------------------------------------- 02: leaf masks + severity
write("02_leaf_and_severity.ipynb", [
    md("""
# AgriCL-Net, step 2: leaf masks and severity

Uses the trained lesion model from notebook 01 plus **SAM2 (pretrained, no training)** to find the leaf, then
computes severity = lesion pixels / leaf pixels.

**What this measures.** For every image, the same leaf mask is combined once with the *true* lesion mask and
once with the *predicted* one. The gap between the two severities is the lesion model's effect on severity.
It is reported separately for **seen** diseases (PlantSeg test split) and the three **unseen** diseases.
Similar scores for both groups support the project's claim. (True severity, with a true leaf mask, needs the
PDSNets dataset: a later step.)

**Needs:** a finished run of notebook 01 (`best.pt` on Drive) and a T4 GPU.
"""),
    md("## Settings"),
    code("""
SIZE = 'l'            # must match the lesion run from notebook 01
IMG_SIZE = 352        # must match the lesion run from notebook 01
LESION_RUN = f"lesion_sam2unet_{SIZE.replace('+', 'p')}_{IMG_SIZE}"   # folder in MyDrive/AGRI-NET/runs
N_PER_GROUP = 150     # images per group (seen / unseen); None = all
USE_SAM2_LEAF = True  # False: colour-index (ExG) leaf mask, the offline fallback
"""),
    *SETUP,
    md("## Load the trained lesion model and the SAM2 leaf segmenter"),
    code("""
from agricl.pipeline import load_lesion_model
from agricl.leaf import SAM2LeafSegmenter

BEST = f'{DRIVE}/runs/{LESION_RUN}/best.pt'
assert os.path.exists(BEST), f'{BEST} not found: run notebook 01 (with SMOKE_TEST = False) first'
lesion_model = load_lesion_model(SAM2UNET, SIZE, CKPT, BEST, 'cuda')
leaf_segmenter = SAM2LeafSegmenter(SAM2UNET, SIZE, CKPT, device='cuda') if USE_SAM2_LEAF else None

OUT = f"{DRIVE}/runs/severity_{LESION_RUN}_{'sam2' if USE_SAM2_LEAF else 'exg'}_leaf"
os.makedirs(OUT, exist_ok=True)
print('results ->', OUT)
"""),
    md("## Severity with true vs. predicted lesions, seen vs. unseen diseases"),
    code("""
import json, random
from agricl.data.plantseg import load_records, split_seen_unseen
from agricl.eval_severity import severity_agreement

seen, unseen = split_seen_unseen(load_records(DATA))
groups = {'seen': [r for r in seen if r.split == 'test'], 'unseen': unseen}
if N_PER_GROUP:
    rng = random.Random(0)
    groups = {k: rng.sample(v, min(N_PER_GROUP, len(v))) for k, v in groups.items()}

summary = severity_agreement(groups, lesion_model, IMG_SIZE, 'cuda', leaf_segmenter,
                             out_csv=f'{OUT}/severity_per_image.csv')
json.dump(summary, open(f'{OUT}/severity_summary.json', 'w'), indent=2)
for g, s in summary.items():
    print(g, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}))
"""),
    md("## Examples: leaf (green), lesions (red), boxes and severity"),
    code("""
import matplotlib.pyplot as plt, matplotlib.patches as patches, numpy as np
from agricl.data.plantseg import load_image
from agricl.pipeline import diagnose_severity

examples = groups['unseen'][:4] + groups['seen'][:2]
fig, ax = plt.subplots(len(examples), 2, figsize=(10, 4 * len(examples)))
for row, r in zip(ax, examples):
    img = load_image(r)
    d = diagnose_severity(img, lesion_model, IMG_SIZE, 'cuda', leaf_segmenter)
    over = np.array(img).astype(float)
    over[d.leaf > 0] = 0.6 * over[d.leaf > 0] + 0.4 * np.array([0, 200, 0])
    over[d.lesion > 0] = [255, 0, 0]
    row[0].imshow(img); row[0].set_title(r.disease)
    row[1].imshow(over.astype(np.uint8))
    row[1].set_title(f'{d.severity.percent:.1f}% {d.severity.level}, {len(d.boxes)} regions')
    for x0, y0, x1, y1 in d.boxes:
        row[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='yellow', lw=1))
    for a in row: a.axis('off')
plt.tight_layout(); plt.show()
"""),
    md("""
## Next
* Compare `seen` and `unseen` in the summary: close values support the claim that severity works on unseen diseases.
* Run again with `USE_SAM2_LEAF = False` to compare the SAM2 leaf with the colour-index fallback.
* True severity accuracy (R², RMSE against real leaf labels) comes next, with the PDSNets dataset.
"""),
])

# ------------------------------------------------- 03: disease classifier
write("03_disease_classifier.ipynb", [
    md("""
# AgriCL-Net, step 3: disease classifier (Swin-T)

Fine-tunes **Swin-T** (pretrained on ImageNet-22k) on the **112 seen PlantSeg diseases**, using a square crop
around the lesions so the model looks at the symptoms, not the background. Rare diseases are sampled more
often (square-root class balancing).

After training it fits two things on the validation set:
* a **temperature** that makes confidence scores honest (calibration);
* an **unknown threshold** that keeps 95% of known-disease images; anything less confident is answered
  "Unknown, consult an expert".

It then reports top-1/top-3 accuracy and macro-F1 on the seen test split, and how many images of the
**three unseen diseases** are correctly flagged as unknown. This model is also the "joint training" upper
bound for the continual-learning step.

**Before you start:** Runtime > Change runtime type > **T4 GPU**. Run once with `SMOKE_TEST = True`
(40 images, 1 epoch), then set it to `False` and choose Runtime > Run all. Training resumes after a disconnect.
"""),
    md("## Settings"),
    code("""
MODEL = 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'   # timm name; fixed 224 px input
BATCH = 32          # lower to 16 if out of memory
EPOCHS = 30
LR = 1e-4
SMOKE_TEST = True   # True: 40 images, 1 epoch. Set False for the real run.
"""),
    *BASE_SETUP,
    code("""
!pip -q install timm
RUN = 'classifier_' + MODEL.split('.')[0] + ('_smoke' if SMOKE_TEST else '')
OUT = f'{DRIVE}/runs/{RUN}'
print('results ->', OUT)
"""),
    md("## Model check: one batch forward and backward (GPU memory)"),
    code("""
import torch
from agricl.models.classifier import build_classifier

m = build_classifier(112, MODEL).cuda().train()
x = torch.randn(BATCH, 3, 224, 224, device='cuda')
with torch.autocast('cuda', dtype=torch.float16):
    loss = m(x).float().logsumexp(1).mean()
loss.backward()
print(f'parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M, '
      f'peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
del m, x, loss
torch.cuda.empty_cache()
"""),
    md("## Train (resumes automatically from Drive)"),
    code("""
from agricl.train_classifier import ClsConfig, train, final_report

cfg = ClsConfig(
    data_root=DATA, out_dir=OUT, model_name=MODEL, batch_size=BATCH, lr=LR,
    epochs=1 if SMOKE_TEST else EPOCHS, limit=40 if SMOKE_TEST else 0, workers=2,
)
model = train(cfg)
"""),
    md("""
## Final evaluation

* `test_seen`: accuracy on known diseases, calibration error before/after temperature scaling, and how many
  known images are wrongly flagged as unknown (about 5% by design).
* `unseen`: share of the three held-out diseases flagged as unknown (higher is better), and AUROC for
  separating known from unseen images by confidence (1.0 = perfect, 0.5 = chance).
"""),
    code("""
results = final_report(cfg, model)
"""),
    md("## Training curve, weakest diseases and confidence on known vs. unseen images"),
    code("""
import json, csv, numpy as np, matplotlib.pyplot as plt
from agricl.calibration import softmax

hist = json.load(open(f'{OUT}/history.json'))
ep = [h['epoch'] for h in hist]
fig, ax = plt.subplots(1, 3, figsize=(15, 3))
ax[0].plot(ep, [h['train_loss'] for h in hist]); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')
ax[1].plot(ep, [h['val_top1'] for h in hist], label='top-1')
ax[1].plot(ep, [h['val_top3'] for h in hist], label='top-3'); ax[1].legend(); ax[1].set_title('validation accuracy')

cal = json.load(open(f'{OUT}/calibration.json'))
lg = np.load(f'{OUT}/logits.npz')
ax[2].hist(softmax(lg['test'], cal['temperature']).max(1), bins=30, alpha=0.6, density=True, label='known (test)')
ax[2].hist(softmax(lg['unseen'], cal['temperature']).max(1), bins=30, alpha=0.6, density=True, label='unseen diseases')
ax[2].axvline(cal['unknown_threshold'], color='k', ls='--', label='unknown threshold')
ax[2].set_title('calibrated confidence'); ax[2].legend()
plt.show()

rows = sorted(csv.DictReader(open(f'{OUT}/per_class.csv')), key=lambda r: float(r['accuracy']))
print('weakest 10 diseases on the test split:')
for r in rows[:10]:
    print(f"  {float(r['accuracy']):.2f}  ({r['test_images']} images)  {r['disease']}")
"""),
    md("""
## Next
* Smoke test passed? Set `SMOKE_TEST = False` in the settings cell and run all cells.
* Results: `MyDrive/AGRI-NET/runs/<run name>/` (`best.pt`, `classes.json`, `calibration.json`, `results.json`,
  `per_class.csv`, `logits.npz`).
* The continual-learning step will compare against this model's accuracy as the upper bound.
"""),
])

# ------------------------------------------------- 04: continual learning
write("04_continual_learning.ipynb", [
    md("""
# AgriCL-Net, step 4: continual learning (learning new diseases without forgetting)

The 112 seen diseases are split **by crop into 5 rounds** (58 diseases first, then 4 rounds of 13-14).
The classifier learns round by round and, after each round, must recognise **every disease seen so far**
without being told which round an image came from.

Three methods start from the same round-0 model:

| Method | Trains each new round on | Role |
| --- | --- | --- |
| `finetune` | new round's images only | lower bound: shows forgetting |
| `replay` | + 20 saved images per old disease (herding) | common baseline |
| `replay_kd` | + knowledge distillation from the previous model + weight aligning | **AgriCL-Net** |

The upper bound is the classifier trained on all diseases at once (notebook 03).

**Before you start:** Runtime > Change runtime type > **T4 GPU**. Run once with `SMOKE_TEST = True`
(3 images per disease, 1 epoch per round), then set it to `False` and choose Runtime > Run all.
Progress is saved after every round, so a disconnect loses at most one round.
"""),
    md("## Settings"),
    code("""
MODEL = 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'
N_ROUNDS = 5
BASE_EPOCHS = 15     # round 0
INC_EPOCHS = 10      # each later round
EXEMPLARS = 20       # saved images per disease (replay methods)
BATCH = 32
SMOKE_TEST = True    # True: 3 images per disease, 1 epoch per round. Set False for the real run.
"""),
    *BASE_SETUP,
    code("""
!pip -q install timm
RUN = 'continual_' + MODEL.split('.')[0] + ('_smoke' if SMOKE_TEST else '')
OUT = f'{DRIVE}/runs/{RUN}'
IMG_CACHE = '/content/plantseg_640'   # resized copies on local disk: much faster than decoding full photos
print('results ->', OUT)
"""),
    md("## Run all rounds for all three methods (resumes automatically)"),
    code("""
from agricl.cl.train_cl import CLConfig, run

cfg = CLConfig(
    data_root=DATA, out_dir=OUT, cache_dir=IMG_CACHE, model_name=MODEL, batch_size=BATCH,
    n_tasks=N_ROUNDS,
    base_epochs=1 if SMOKE_TEST else BASE_EPOCHS,
    inc_epochs=1 if SMOKE_TEST else INC_EPOCHS,
    exemplars_per_class=2 if SMOKE_TEST else EXEMPLARS,
    limit_per_class=3 if SMOKE_TEST else 0,
    workers=2,
)
results = run(cfg)
"""),
    md("""
## Results

* **Average incremental accuracy**: mean accuracy over all rounds (higher is better).
* **Final accuracy**: accuracy on all 112 diseases after the last round.
* **Forgetting**: how much accuracy earlier rounds lost by the end (lower is better).
* **Unseen flagged**: share of the three held-out diseases answered "Unknown".
"""),
    code("""
import json, os, numpy as np, matplotlib.pyplot as plt

rounds = json.load(open(f'{OUT}/rounds.json'))
seen_counts = np.cumsum([len(r['diseases']) for r in rounds])

joint = None
JOINT = f"{DRIVE}/runs/classifier_{MODEL.split('.')[0]}/results.json"
if os.path.exists(JOINT):
    joint = json.load(open(JOINT))['test_seen']['top1']

print(f"{'method':<11} {'avg incr. acc':>13} {'final acc':>10} {'forgetting':>11} {'unseen flagged':>15}")
for m, r in results.items():
    print(f"{m:<11} {r['average_incremental_accuracy']:>13.3f} {r['final_accuracy']:>10.3f} "
          f"{r['forgetting']:>11.3f} {r['calibration']['unseen_flagged_unknown']:>15.3f}")
if joint is not None:
    print(f"{'joint':<11} {'':>13} {joint:>10.3f}   (upper bound, notebook 03)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for m, r in results.items():
    ax[0].plot(seen_counts[:len(r['step_accuracy'])], r['step_accuracy'], marker='o', label=m)
if joint is not None:
    ax[0].axhline(joint, color='k', ls='--', label='joint (upper bound)')
ax[0].set_xlabel('diseases learned so far'); ax[0].set_ylabel('accuracy on all of them'); ax[0].legend()
ax[0].set_title('accuracy as new diseases are added')

acc = np.array(results['replay_kd']['accuracy_matrix'], float) if 'replay_kd' in results else None
if acc is not None:
    im = ax[1].imshow(acc, vmin=0, vmax=1, cmap='viridis')
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if not np.isnan(acc[i, j]):
                ax[1].text(j, i, f'{acc[i, j]:.2f}', ha='center', va='center', color='w', fontsize=8)
    ax[1].set_xlabel('round whose diseases are tested'); ax[1].set_ylabel('after training round')
    ax[1].set_title('replay_kd: accuracy per round'); fig.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.show()
"""),
    md("""
## Next
* Smoke test passed? Set `SMOKE_TEST = False` and run all cells.
* Results: `MyDrive/AGRI-NET/runs/<run name>/` (`results.json`, `rounds.json`, final model of each method).
* For the report: `replay_kd` should keep accuracy closer to the joint upper bound and forget less than
  `finetune` and `replay`.
"""),
])

# ------------------------------------------------- 05: app model bundle
write("05_app_bundle.ipynb", [
    md("""
# AgriCL-Net, step 5: package the trained models for the app

Collects everything the Streamlit app needs into one folder, `MyDrive/AGRI-NET/app_bundle`, tests it on
a few photos (on GPU, and on CPU to show the speed on a laptop) and zips it for download.

**Needs:** a finished run of notebook 01 (lesion model), plus notebook 03 (joint classifier) or
notebook 04 (continual-learning classifier).
"""),
    md("## Settings"),
    code("""
SIZE = 'l'                    # lesion run settings from notebook 01
IMG_SIZE = 352
MODEL = 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'
CLASSIFIER = 'continual'      # 'continual' (notebook 04, replay_kd) or 'joint' (notebook 03)
LEAF = 'sam2'                 # default leaf method in the app: 'sam2' or 'exg'
"""),
    *BASE_SETUP[2:4],
    *SAM2_SETUP,
    md("## Build the bundle"),
    code("""
import json, shutil
!pip -q install timm

BUNDLE = f'{DRIVE}/app_bundle'
os.makedirs(BUNDLE, exist_ok=True)
short = MODEL.split('.')[0]

lesion_run = f"{DRIVE}/runs/lesion_sam2unet_{SIZE.replace('+', 'p')}_{IMG_SIZE}"
assert os.path.exists(f'{lesion_run}/best.pt'), 'run notebook 01 (SMOKE_TEST = False) first'
shutil.copy(f'{lesion_run}/best.pt', f'{BUNDLE}/lesion_best.pt')
shutil.copy(CKPT, f'{BUNDLE}/{os.path.basename(CKPT)}')
shutil.copytree(SAM2UNET, f'{BUNDLE}/SAM2-UNet', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.git', '*.jpg', '*.drawio'))

if CLASSIFIER == 'joint':
    run = f'{DRIVE}/runs/classifier_{short}'
    assert os.path.exists(f'{run}/best.pt'), 'run notebook 03 (SMOKE_TEST = False) first'
    shutil.copy(f'{run}/best.pt', f'{BUNDLE}/classifier.pt')
    shutil.copy(f'{run}/classes.json', f'{BUNDLE}/classes.json')
    shutil.copy(f'{run}/calibration.json', f'{BUNDLE}/calibration.json')
    kind = 'timm'
else:
    run = f'{DRIVE}/runs/continual_{short}'
    assert os.path.exists(f'{run}/replay_kd_final.pt'), 'run notebook 04 (SMOKE_TEST = False) first'
    shutil.copy(f'{run}/replay_kd_final.pt', f'{BUNDLE}/classifier.pt')
    rounds = json.load(open(f'{run}/rounds.json'))
    json.dump([d for r in rounds for d in r['diseases']], open(f'{BUNDLE}/classes.json', 'w'), indent=1)
    cal = json.load(open(f'{run}/results.json'))['replay_kd']['calibration']
    json.dump({'temperature': cal['temperature'], 'unknown_threshold': cal['unknown_threshold']},
              open(f'{BUNDLE}/calibration.json', 'w'), indent=2)
    kind = 'incremental'

json.dump({
    'lesion': {'size': SIZE, 'img_size': IMG_SIZE, 'weights': 'lesion_best.pt',
               'sam2_checkpoint': os.path.basename(CKPT), 'sam2unet_repo': 'SAM2-UNet'},
    'classifier': {'kind': kind, 'model_name': MODEL, 'img_size': 224, 'weights': 'classifier.pt',
                   'classes': 'classes.json', 'calibration': 'calibration.json'},
    'leaf': {'method': LEAF},
}, open(f'{BUNDLE}/agricl_bundle.json', 'w'), indent=2)
!ls -la "{BUNDLE}"
"""),
    md("## Test the bundle on unseen-disease and seen-disease photos (GPU, then CPU)"),
    code("""
import time, random
from PIL import Image
from agricl.predictor import AgriCLPredictor

if not os.path.exists('/content/data/plantseg/Metadata.csv'):
    !unzip -q "{CACHE}/plantseg.zip" -d /content/data
from agricl.data.plantseg import load_records, split_seen_unseen
seen, unseen = split_seen_unseen(load_records('/content/data/plantseg'))
samples = random.Random(0).sample(unseen, 2) + random.Random(0).sample([r for r in seen if r.split == 'test'], 2)

for device in ('cuda', 'cpu'):
    p = AgriCLPredictor(BUNDLE, device=device)
    p.diagnose(Image.open(samples[0].image))  # warm-up
    for r in samples:
        t = time.time()
        d = p.diagnose(Image.open(r.image))
        print(f"[{device}] true: {r.disease:<28} -> {d.decision.status:<10} {str(d.decision.disease):<28} "
              f"severity {d.severity.percent:5.1f}% {d.severity.level:<8} {time.time() - t:5.1f} s")
    del p
"""),
    md("## Zip for download"),
    code("""
shutil.make_archive(f'{DRIVE}/agricl_app_bundle', 'zip', BUNDLE)
print(f"{DRIVE}/agricl_app_bundle.zip", os.path.getsize(f'{DRIVE}/agricl_app_bundle.zip') // 2**20, 'MB')
"""),
    md("""
## Next
Download `MyDrive/AGRI-NET/agricl_app_bundle.zip`, unzip it to `AGRI-NET/models/bundle/` on your computer and
start the app (see the README, section "Run the app"). The CPU times above are roughly what a laptop will need
per photo; choose the colour-index leaf method in the app's sidebar to make it faster.
"""),
])

# ------------------------------------------------- 06: edge models
write("06_edge_models.ipynb", [
    md("""
# AgriCL-Net, step 6: edge models (offline, Raspberry Pi)

Distils two small **student** models from the accurate **teachers**, exports them to ONNX and quantizes them to
**INT8**, then measures size, speed and accuracy before and after:

| Task | Teacher | Student |
| --- | --- | --- |
| Lesion segmentation | SAM2-UNet (notebook 01) | LinkNet-ResNet18 (about 11.5 M parameters) |
| Disease classification | Swin-T (notebook 03 or 04) | MobileNetV3-Large |

The result is an **edge bundle** that runs with ONNX Runtime + NumPy only (no PyTorch), with the colour-index
leaf mask. The app also accepts it, so it can run fully offline.

**Needs:** finished runs of notebook 01 and of notebook 03 or 04, and a T4 GPU. Run once with `SMOKE_TEST = True`,
then set it to `False` and run all. Both trainings resume after a disconnect.
"""),
    md("## Settings"),
    code("""
SIZE = 'l'               # lesion teacher (notebook 01)
IMG_SIZE = 352
MODEL = 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'
TEACHER = 'continual'    # classifier teacher: 'continual' (notebook 04, replay_kd) or 'joint' (notebook 03)
CLS_EPOCHS = 30
SEG_EPOCHS = 40
SMOKE_TEST = True        # True: 40 images per split, 1 epoch each. Set False for the real run.
"""),
    *SETUP,
    code("""
!pip -q install timm onnx onnxruntime
import json, shutil
RUN = f'edge_{TEACHER}' + ('_smoke' if SMOKE_TEST else '')
OUT = f'{DRIVE}/runs/{RUN}'
os.makedirs(OUT, exist_ok=True)
IMG_CACHE = '/content/plantseg_640'
short = MODEL.split('.')[0]

lesion_teacher = f"{DRIVE}/runs/lesion_sam2unet_{SIZE.replace('+', 'p')}_{IMG_SIZE}/best.pt"
assert os.path.exists(lesion_teacher), 'run notebook 01 (SMOKE_TEST = False) first'
if TEACHER == 'joint':
    run = f'{DRIVE}/runs/classifier_{short}'
    teacher_weights, teacher_classes, kind = f'{run}/best.pt', f'{run}/classes.json', 'timm'
else:
    run = f'{DRIVE}/runs/continual_{short}'
    teacher_weights, kind = f'{run}/replay_kd_final.pt', 'incremental'
    teacher_classes = f'{OUT}/teacher_classes.json'
    json.dump([d for r in json.load(open(f'{run}/rounds.json')) for d in r['diseases']], open(teacher_classes, 'w'))
assert os.path.exists(teacher_weights), f'{teacher_weights} not found: run notebook 03 or 04 first'
print('results ->', OUT)
"""),
    md("## Distil the classifier student (MobileNetV3)"),
    code("""
from agricl.edge_train import EdgeConfig, distill_classifier, distill_lesion

cfg = EdgeConfig(
    data_root=DATA, out_dir=OUT, cache_dir=IMG_CACHE,
    sam2unet_repo=SAM2UNET, sam2_size=SIZE, sam2_checkpoint=CKPT, lesion_teacher=lesion_teacher, lesion_size=IMG_SIZE,
    classifier_teacher_kind=kind, classifier_teacher_weights=teacher_weights,
    classifier_teacher_classes=teacher_classes, teacher_model_name=MODEL,
    cls_epochs=1 if SMOKE_TEST else CLS_EPOCHS, seg_epochs=1 if SMOKE_TEST else SEG_EPOCHS,
    limit=40 if SMOKE_TEST else 0,
)
cls_results = distill_classifier(cfg)
"""),
    md("## Distil the lesion student (LinkNet-ResNet18)"),
    code("""
seg_results = distill_lesion(cfg)
"""),
    md("## Export to ONNX, quantize to INT8, and compare size, speed and accuracy"),
    code("""
import random, numpy as np, pandas as pd, torch
from agricl.edge import OnnxModel
from agricl.edge_eval import (classifier_inputs, classifier_top1, file_mb, latency_ms, lesion_inputs,
                              lesion_scores, quantize_int8)
from agricl.edge_train import export_onnx, splits
from agricl.models.classifier import build_classifier
from agricl.models.linknet import LinkNet18

data = splits(cfg)
EDGE = f'{OUT}/edge_bundle'
os.makedirs(EDGE, exist_ok=True)
classes = json.load(open(f'{OUT}/classifier/classes.json'))
c2i = {d: i for i, d in enumerate(classes)}

clf = build_classifier(len(classes), cfg.student_classifier, pretrained=False)
clf.load_state_dict(torch.load(f'{OUT}/classifier/best.pt', map_location='cpu', weights_only=True))
seg = LinkNet18(pretrained=False)
seg.load_state_dict(torch.load(f'{OUT}/lesion/best.pt', map_location='cpu', weights_only=True))
export_onnx(clf, f'{EDGE}/classifier_fp32.onnx', 224)
export_onnx(seg, f'{EDGE}/lesion_fp32.onnx', IMG_SIZE)

calib = random.Random(0).sample(data['train'], min(100, len(data['train'])))
quantize_int8(f'{EDGE}/classifier_fp32.onnx', f'{EDGE}/classifier_int8.onnx', classifier_inputs(calib, 224))
quantize_int8(f'{EDGE}/lesion_fp32.onnx', f'{EDGE}/lesion_int8.onnx', lesion_inputs(calib, IMG_SIZE))

shutil.copy(f'{OUT}/classifier/classes.json', f'{EDGE}/classes.json')
shutil.copy(f'{OUT}/classifier/calibration.json', f'{EDGE}/calibration.json')
json.dump({'lesion': {'onnx': 'lesion_int8.onnx', 'img_size': IMG_SIZE},
           'classifier': {'onnx': 'classifier_int8.onnx', 'img_size': 224,
                          'classes': 'classes.json', 'calibration': 'calibration.json'}},
          open(f'{EDGE}/edge_bundle.json', 'w'), indent=2)

test = random.Random(1).sample(data['test_seen'], min(300, len(data['test_seen'])))
cls_x, seg_x = classifier_inputs(test, 224), lesion_inputs(test, IMG_SIZE)
labels = np.array([c2i[r.disease] for r in test])
rows = []
for task, prec in [('classifier', 'fp32'), ('classifier', 'int8'), ('lesion', 'fp32'), ('lesion', 'int8')]:
    path = f'{EDGE}/{task}_{prec}.onnx'
    m1, mall = OnnxModel(path, threads=1), OnnxModel(path)
    x = cls_x if task == 'classifier' else seg_x
    acc = (classifier_top1(mall, x, labels) if task == 'classifier'
           else lesion_scores(mall, test, x)['mean_dice'])
    rows.append({'model': task, 'precision': prec, 'size_mb': round(file_mb(path), 1),
                 'ms_1_thread': round(latency_ms(m1, x[0]), 1), 'ms_all_threads': round(latency_ms(mall, x[0]), 1),
                 'accuracy': round(acc, 4), 'metric': 'top-1' if task == 'classifier' else 'mean Dice'})
table = pd.DataFrame(rows)
table.to_csv(f'{OUT}/edge_benchmark.csv', index=False)
print(f'Colab CPU cores: {os.cpu_count()} (a Raspberry Pi 4 has 4)')
table
"""),
    md("""
**How to read it:** INT8 files should be about 4x smaller and usually faster on CPU, with a small accuracy drop.
Teacher accuracy for comparison: `cls_results['teacher']` below, and notebook 01's `results.json`.
"""),
    code("""
print('classifier teacher vs student (test, top-1):',
      round(cls_results['teacher']['top1'], 4), 'vs', round(cls_results['student']['top1'], 4))
print('lesion student (full resolution):', json.dumps(seg_results, indent=1))
"""),
    md("## Try the edge bundle on a few photos (ONNX Runtime only)"),
    code("""
from PIL import Image
from agricl.edge import EdgePredictor

p = EdgePredictor(EDGE)
for r in random.Random(2).sample(data['unseen'], 2) + random.Random(2).sample(data['test_seen'], 2):
    d = p.diagnose(Image.open(r.image))
    print(f"true: {r.disease:<28} -> {d.decision.status:<10} {str(d.decision.disease):<28} "
          f"severity {d.severity.percent:5.1f}% {d.severity.level:<8} {sum(d.timings.values()) * 1000:5.0f} ms")
shutil.make_archive(f'{DRIVE}/agricl_edge_bundle', 'zip', EDGE)
print(f"{DRIVE}/agricl_edge_bundle.zip", os.path.getsize(f'{DRIVE}/agricl_edge_bundle.zip') // 2**20, 'MB')
"""),
    md("""
## On a Raspberry Pi
1. Raspberry Pi OS Bookworm **64-bit**, Python 3.11 to 3.14.
2. Copy the project and unzip `agricl_edge_bundle.zip` to `models/edge_bundle/`.
3. `pip install -r requirements-edge.txt`
4. `python scripts/edge_benchmark.py --bundle models/edge_bundle --images <folder of leaf photos>`
5. Or start the app: `streamlit run app/streamlit_app.py` and set the bundle folder to `models/edge_bundle`.
"""),
])

# ------------------------------------------------- 07: results report
write("07_results_report.ipynb", [
    md("""
# AgriCL-Net, step 7: results report

Collects the results of every finished run in `MyDrive/AGRI-NET/runs` (smoke tests excluded) into
`MyDrive/AGRI-NET/report/`:

* `report.md`: all results as tables, ready to copy into the documentation;
* `key_claim.png`: seen vs. unseen diseases (the project's main claim);
* `continual_learning.png`: accuracy as new diseases are added, per method;
* `edge.png`: size and speed of the edge models, FP32 vs. INT8.

Steps that have not been run yet are marked "Not run yet". No GPU needed: a CPU runtime is enough.
"""),
    *BASE_SETUP[2:4],
    md("## Build the report"),
    code("""
from IPython.display import Markdown, display
from agricl.report import build_report, collect

RUNS = f'{DRIVE}/runs'
REPORT = f'{DRIVE}/report'
os.makedirs(REPORT, exist_ok=True)
md_text = build_report(RUNS)
open(f'{REPORT}/report.md', 'w').write(md_text)
res = collect(RUNS)
display(Markdown(md_text))
"""),
    md("## Charts"),
    code("""
import json, numpy as np, matplotlib.pyplot as plt

def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{REPORT}/{name}', dpi=200); plt.show()

# 1. key claim: seen vs unseen
bars = []
for name, r in res['lesion'].items():
    bars.append(('Lesion Dice', r['test_seen']['mean_dice'], r['unseen']['mean_dice']))
for name, r in res['severity'].items():
    bars.append(('Severity R²', r['seen']['severity_r2'], r['unseen']['severity_r2']))
if bars:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(bars))
    ax.bar(x - 0.18, [b[1] for b in bars], 0.36, label='seen diseases', color='#2e7d32')
    ax.bar(x + 0.18, [b[2] for b in bars], 0.36, label='unseen diseases', color='#f9a825')
    ax.set_xticks(x, [b[0] for b in bars]); ax.set_ylim(0, 1); ax.legend()
    ax.set_title('Lesions and severity on diseases never seen in training')
    save(fig, 'key_claim.png')
else:
    print('key claim: run notebooks 01 and 02 first')

# 2. continual learning curve
for name, methods in res['continual'].items():
    rounds = json.load(open(f'{RUNS}/{name}/rounds.json'))
    seen = np.cumsum([len(r['diseases']) for r in rounds])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m, r in methods.items():
        ax.plot(seen[:len(r['step_accuracy'])], r['step_accuracy'], marker='o', label=m)
    joint = next(iter(res['classifier'].values()), {}).get('test_seen', {}).get('top1')
    if joint is not None:
        ax.axhline(joint, color='k', ls='--', label='joint (upper bound)')
    ax.set_xlabel('diseases learned so far'); ax.set_ylabel('accuracy on all of them'); ax.set_ylim(0, 1)
    ax.legend(); ax.set_title('Learning new diseases without forgetting')
    save(fig, 'continual_learning.png')

# 3. edge models
for name, r in res['edge'].items():
    if not r.get('benchmark'):
        continue
    b = r['benchmark']
    labels = [f"{x['model']}\\n{x['precision']}" for x in b]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].bar(labels, [float(x['size_mb']) for x in b], color='#2e7d32'); ax[0].set_title('size (MB)')
    ax[1].bar(labels, [float(x['ms_1_thread']) for x in b], color='#1565c0'); ax[1].set_title('ms per image, 1 CPU thread')
    save(fig, 'edge.png')
!ls -la "{REPORT}"
"""),
    md("""
## Next
Copy the tables from `report.md` and the charts into the documentation ("New Document"), in a new
"Results" section, and update the Evaluation Plan with the measured numbers.
"""),
])
