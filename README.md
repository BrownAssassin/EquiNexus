# EquiNexus

Working repository for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

**Team:** Arv Bali and Thai Le

**Current phase:** repository setup and challenge review

## Challenge

Build a reproducible method that segments individual solar filaments in GONG H-alpha images. Predictions are instance masks: the submission contains one encoded mask per predicted filament. The organizers updated the primary metric from Dice to Panoptic Quality (PQ), which evaluates both mask overlap and whether filaments are split or merged correctly.

The competition data and GPU notebooks should be the main execution environment. Keep the source and notebook snapshots here so the final solution can be reviewed and reproduced. See [the challenge brief](docs/challenge-brief.md) for the data contract, scoring, rules, deadlines, and open questions.

## Kaggle-first workflow

1. Read the current competition rules and use the competition's attached data in Kaggle.
2. Develop, validate, and run GPU experiments in Kaggle notebooks. Save the important notebook versions and reusable code in this repository.
3. Keep validation splits grouped by the original observation so alternate annotator records of the same image cannot cross between train and validation.
4. Export the submission CSV and experiment records from Kaggle. Do not commit competition data, credentials, checkpoints, or generated submissions.
5. Before final judging, make the pipeline reproducible from a public notebook and repository, pin all dependencies in `requirements.txt`, and prepare the required technical report.

The repository's `.gitignore` excludes common local data, model, credential, and output paths. Kaggle's mounted input and working directories are outside this checkout.

## Repository layout

```text
configs/       Experiment configuration notes
docs/          Challenge brief, decisions, and experiment log
notebooks/     Kaggle notebook exports and reproduction notebook
src/           Reusable pipeline code
requirements.txt
```

## Required final materials

The organizers request a four-page technical report, a publicly accessible Git repository, a pinned `requirements.txt`, and a Jupyter notebook that illustrates the complete pipeline and reproduces the predictions using the competition data. The current `requirements.txt` is a setup placeholder and must be replaced with exact tested package versions once the implementation is chosen.

See the [competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview), [data tab](https://www.kaggle.com/competitions/filament-segmentation-2026/data), [rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules), and [organizers' self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
