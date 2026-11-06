# EquiNexus

Working repository for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

**Team:** Arv Bali and Thai Le

**Current phase:** scorer-aligned validation baseline

## Challenge

Build a reproducible method that segments individual solar filaments in GONG H-alpha images. Predictions are instance masks: the submission contains one encoded mask per predicted filament. The organizers updated the primary metric from Dice to Panoptic Quality (PQ), which evaluates both mask overlap and whether filaments are split or merged correctly.

Use Kaggle for attached competition data and larger GPU runs. The local RTX 5070 is available for smaller GPU experiments and smoke checks. Keep shared implementation code and the Kaggle run notebook in this repository so both environments use the same split and scorer. See [the challenge brief](docs/challenge-brief.md) for the data contract, scoring, rules, deadlines, and open questions.

## Kaggle-first workflow

1. Use only competition-provided ground-truth annotations. Keep all local data under the ignored data/ folder.
2. Use the shared src/ implementation on the laptop and in Kaggle notebooks. Record the source commit and runtime details for every run.
3. Keep validation splits grouped by original observation so alternate annotator records cannot cross between train and validation.
4. Use the local RTX 5070 for small or patch-based model experiments; use Kaggle GPU sessions for larger runs and competition data access.
5. Do not commit competition data, credentials, checkpoints, generated overlays, or submission files. Keep run outputs under ignored artifacts/.

The repository's .gitignore excludes local data, environments, caches, model artifacts, credentials, and outputs. Kaggle's mounted input and working directories are remote runtime paths.

## Repository layout

```text
configs/       Experiment configuration notes
data/          Ignored local competition data, scorer reference, and split manifest
docs/          Challenge brief, decisions, and experiment log
notebooks/     Kaggle baseline and reproduction notebooks
artifacts/     Ignored local validation outputs and model checkpoints
src/           Reusable pipeline code
requirements.txt
```

## Required final materials

The organizers request a four-page technical report, a publicly accessible Git repository, pinned dependencies, and a Jupyter notebook that illustrates the complete pipeline and reproduces predictions using competition data.

See the [competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview), [data tab](https://www.kaggle.com/competitions/filament-segmentation-2026/data), [rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules), and [organizers' self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).

## Local baseline setup

All local challenge files stay inside this checkout. The downloaded dataset is stored at data/raw/MAGFiLO_1.0_Kaggle_2026/; the local scorer export and deterministic split manifest are also kept under ignored data/ paths. Generated metrics and overlays go under ignored artifacts/.

On Windows PowerShell, create the environment and keep Python's downloads and temporary files inside the repository:

    New-Item -ItemType Directory -Force .cache\pip, .cache\tmp | Out-Null
    py -3 -m venv .venv
    $env:PIP_CACHE_DIR = Join-Path $PWD '.cache\pip'
    $env:TEMP = Join-Path $PWD '.cache\tmp'
    $env:TMP = $env:TEMP
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-gpu.txt
    $env:PYTHONPATH = Join-Path $PWD 'src'
    .\.venv\Scripts\python.exe scripts\run_gpu_smoke.py
    .\.venv\Scripts\python.exe scripts\run_validation.py

The Kaggle notebook is notebooks/validation_baseline.ipynb. Attach the competition data, enable notebook internet so the notebook can clone this public repository, and select an accelerator only when GPU work is needed. The fixed classical baseline runs on CPU; the notebook records the available GPU for later experiments.
