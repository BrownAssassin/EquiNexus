# Solar Filament Segmentation Challenge 2026: working brief

Reviewed 2026-11-04. Kaggle pages are JavaScript-rendered, so some tab contents could not be read directly; details below are linked to the official overview, official host clarifications, and Kaggle-hosted materials where available.

## Task and data

- Segment each visible solar filament as its own mask in full-disk GONG H-alpha observations. Fine structures, low contrast, image noise/artifacts, and preserving filament continuity are called out as the main difficulties.
- The competition uses the MAGFiLO annotations. The official submission description fixes mask dimensions at 2048 x 2048 and asks for a single CSV with `filament_id` and `segmentation_rle` columns.
- Each row represents one predicted filament. The ID is the test image stem with a unique suffix (for example, `20150125172714Mh_1`). The RLE column contains the encoded counts only, without quotes or an explicit size. The organizers point to `pycocotools` mask conversion helpers.
- The Kaggle input layout and counts commonly reported for the competition are `train/train_images/`, `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`, and `test/test_images/`; 707 observations, 1,154 annotation records, 8,199 filaments, and 180 test images. These counts/layout are corroborated by a public Kaggle notebook, but should be checked against the mounted competition input before building loaders.
- The underlying MAGFiLO v1.0 research dataset is larger and contains observations outside the competition's training split. Do not treat that full release as extra training labels; see the explicit data rule below.

## Metric and evaluation

- The organizers changed the primary metric from Dice to Panoptic Quality (PQ) in an August 2026 update and re-scored the leaderboard on August 12.
- PQ is the sum of IoUs for matched instances divided by `TP + 0.5 * FP + 0.5 * FN`. It rewards accurate masks and penalizes missed, extra, split, or merged filament instances.
- The competition overview also lists Dice/IoU distributions and one-to-many or many-to-one relationships in the 70% quantitative assessment. The remaining 30% is qualitative: complete method description, morphology of masks, and code quality. The updated overview also calls out end-to-end computational efficiency.
- There is an organizer-provided [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook). Inspect and reproduce its matching rules before implementing local scoring; older overview text still contains references to Dice and public/private split behavior that predate the PQ update.
- Do not use the public leaderboard as the sole measure of model quality. The organizers describe it as a preliminary filter and say final judging considers the quality and reproducibility of the actual approach.

## Data-use rule and contamination risk

The host clarified in [the competition discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396) that external sources may be used for additional information, but ground-truth annotations must be limited to those provided through this competition. Using other MAGFiLO ground-truth annotations makes the submission liable to be disregarded.

This matters because the public MAGFiLO v1.0 release overlaps the competition test observations and exposes annotations for them. Do not download or use those outside annotations, infer test labels from them, or use a public notebook that does so. For ML inference, use only the H-alpha test images provided through the competition. Record any permitted external non-ground-truth inputs and their role in the experiment log.

## Schedule and final package

- Competition launch: July 10, 2026.
- The overview announcement gives **November 15, 2026** as the deadline to submit final reports and solutions, and November 30 for winner announcements. The Kaggle page's close countdown and the report/solution deadline may describe different milestones; confirm the exact Kaggle submission close time and the final form workflow in the competition page before planning the last run.
- Final judging materials include a four-page technical report, a publicly accessible Git repository, a `requirements.txt` with exact versions, and a Jupyter notebook showing the complete pipeline. The committee should be able to reproduce the predictions without requesting extra files/access. A form is required for final evaluation.
- The open-access policy says to make the repository public immediately after competition close and keep it public at least through winner announcements. The overview also asks for a publicly accessible repository in the final materials; plan to meet the earlier applicable requirement.
- The top three share up to $3,000. Prize acceptance requires reproducible source/configuration and a technical report. The winners are also to be announced at IEEE BigData 2026, December 14-17 in Phoenix.

## Team and workspace decisions

- Team: EquiNexus (Arv Bali and Thai Le).
- Keep large competition data, credentials, checkpoints, generated images, and submission CSVs out of Git.
- Use Kaggle for competition data and larger GPU runs. Use the local RTX 5070 for smaller or patch-based experiments and smoke checks. Keep the same source, observation split, and scorer on both runtimes.
- Keep all local competition files inside this repository. Store datasets, reference exports, and split manifests under the ignored data/ folder; store generated runs under ignored artifacts/.
- Validation must group by original observation/image filename, not by annotator-specific COCO image ID, to prevent the same H-alpha observation appearing on both sides of a split.
- Baseline dependencies are pinned in `requirements.txt`, with the CUDA-enabled PyTorch runtime in `requirements-gpu.txt`. Update these pins only when a tested experiment requires a change.

## Official references

- [Competition overview and announcements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)
- [Competition data tab](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
- [Competition rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules)
- [Organizer clarification on ground-truth use](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396)
- [Organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [MAGFiLO dataset paper](https://doi.org/10.1038/s41597-024-03876-y)
- [MAGFiLO v1.0 public record](https://doi.org/10.7910/DVN/J6JNVK) — use only in accordance with the competition's ground-truth restriction.
