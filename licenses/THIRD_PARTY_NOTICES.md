# Third-party notices

## I079 solar-filament inference source and model assets

- Source notebook: [LB 0.70 topology-safe solar filaments handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff), published by Kaggle user `phuongncn`.
- Model assets: [`phuongncn/solar-filament-i079-weights`](https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights). The model card declares Apache License 2.0. The detector and refiner files are downloaded separately into the ignored `data/reference/public-models/` directory and are not included in this repository.
- License: Apache License 2.0. A copy of the license is included at [`Apache-2.0.txt`](Apache-2.0.txt).
- EquiNexus modifications in `scripts/run_i079_topology_safe_inference.py`: local and Kaggle path discovery, a command-line interface, source and model hash recording, runtime and hardware metadata, and strict test-image, mask, and RLE validation. The pipeline remains image-conditioned and reads only the supplied H-alpha test images.

The upstream notebook and model card are the sources for the third-party material and their declared license. The EquiNexus changes are identified in the inference script header.

## Public competition-trained U-Net weights

- Dataset: [Solar Filament U-Net Weights (PQ 0.45 OOF)](https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights), published by Kaggle user `dariushafshar`; related inference notebook: [Solar Filament U-Net: OOF 0.45 vs Public 0.39](https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39).
- The dataset contains ten ResNet34/EfficientNetV2-S U-Net checkpoints and fold/provenance files, but no competition images, masks, or labels. The Kaggle dataset metadata declares `CC-BY-NC-SA-4.0`; the notebook's credits cell says the weights are `CC BY 4.0`. Because those statements conflict, EquiNexus conservatively treats the weights as `CC-BY-NC-SA-4.0` unless the author clarifies otherwise. Their use is limited to this non-commercial research competition, as authorized by the team.
- The dataset reports nested OOF PQ 0.4502 on the 707 competition training observations and public leaderboard PQ 0.39. These are upstream-reported results, not EquiNexus validation results. The checkpoints collectively train on all 707 observations, so they are not scored on the frozen 142-observation holdout; the available 142-group per-observation crossfit is not independent holdout evidence. When used for submission, inference reads only the supplied test H-alpha images.
- Local weights are stored under the ignored `data/reference/public-datasets/dariushafshar-solar-filament-unet-weights/` directory. The inference runner records model hashes, source dataset, declared license, runtime, and submission validation details.

## GONG-derived auxiliary targets

- Dataset: [GONG Multi-Channel Data for Solar Filaments](https://www.kaggle.com/datasets/vhrabar/gong-solar-filaments), which provides public GONG magnetogram data and derived magnetic-field target maps. Kaggle metadata labels its license as `other`; the dataset description states that NSO publishes no formal GONG data license and requests acknowledgment.
- Competition scope: the host clarified that participants may use external sources such as GONG magnetograms, final inference must use only the supplied H-alpha images, and ground-truth annotations must be limited to those supplied by the competition: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396.
- The local `data/reference/public-datasets/vhrabar-gong-solar-filaments/train-only-manifest.json` records the filtered training-only selection and file hashes. It contains 486 eligible magnetogram-derived target maps matched to competition training observations, with no target files from supplied test observations or frozen validation dates, and no MAGFiLO images or annotations. An earlier 504-target manifest is retained only in the ignored policy-review artifact and is ineligible. The model uses the filtered maps only as auxiliary training supervision; validation and test inference use the supplied H-alpha images.
- NSO's [GONG data-use acknowledgement](https://nso.edu/data/) should be included in reports using this data. The approved abbreviated acknowledgment is: `Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA.`
