# Experiment log

Record each meaningful local or Kaggle run here. Keep raw logs and large outputs in ignored local artifacts or Kaggle; copy only the concise evidence needed to reproduce decisions.

| Date | Notebook/version | Data split | Method/config | Validation PQ | Runtime/GPU | Submission/version | Notes |
|---|---|---|---|---:|---|---|---|
| 2026-11-05 | Local full-holdout run; base commit `420ec100060e7c6bfd4bd8e619d692e979e9c999`, source tree SHA-256 `0e6e282e5a92a0940bde85fe384aa8bc78e5aac914936f01ef57f8c70a47634b` | Seed 42; 20% holdout (142/707 observations), year stratification; manifest SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e` | Fixed grayscale local contrast, 31x31 blur, disk-masked Otsu, 3x3 close, 8-connected components, area 50-50,000; organizer scorer port, strict IoU > 0.5 | 0.0011154 | 145.22 s end-to-end; 12.88 s prediction + 0.36 s scoring. Baseline ran on CPU. RTX 5070 CUDA smoke passed with PyTorch 2.15.0+cu132 and CUDA 13.2. | Not submitted | Split digest matches the frozen value. There were 232 annotator-view comparisons, 184 TP edges, 192,070 FP, and 1,527 FN. Test images and labels were not evaluated. Artifacts are ignored under `artifacts/validation-baseline/`; Kaggle execution is pending. |

For each result, record the observation-level split, exact scorer version, seed, input channels/preprocessing, model/checkpoint identifier, post-processing, and any external non-ground-truth data. Do not log or commit test labels or credentials.
