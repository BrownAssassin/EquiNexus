# Model sprint notes

## Data and evaluation boundary

- The fixed challenge-label training and validation use the competition's 707 training observations and 1,154 annotator views. Rule 6 permits external data and models that are public and equally accessible at no cost or reasonably priced. The competition-specific restriction excludes public MAGFiLO annotations outside the supplied training annotations; final inference uses only the supplied test images.
- Keep all views of each source image in one partition. The fixed validation manifest remains seed 42, year-stratified, 20% holdout, SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- At test time, use only the 180 H-alpha images attached by the competition. Do not use public MAGFiLO labels for those observations; the host says solutions using other MAGFiLO ground truth may be disregarded: <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396>.
- Treat the public leaderboard as secondary evidence. The host describes it as a preliminary filter, and participants have documented test-label leakage that makes its scores unreliable: <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062>.

## External data and model scope (updated 2026-11-11)

- The user has clarified that EquiNexus may use public weights, models, and datasets whose terms restrict use to non-commercial research or education. Earlier exclusions based solely on a non-commercial term are superseded. Continue recording each resource's license, attribution, provenance, and any ShareAlike requirements, and keep the competition's MAGFiLO test-label restriction in force.
- Rule 6 permits qualifying public external data and models. The host's [clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396) limits MAGFiLO ground-truth annotations to those supplied through the competition; independent public data such as SWEFil may be used when it satisfies Rule 6. We do not use any public MAGFiLO annotations outside the supplied training set. Final inference uses only the supplied H-alpha test images.
- SWEFil is a public CC-BY-4.0 GONG H-alpha filament dataset with manually annotated instance masks. We use the processed images and filament masks for supervised external pretraining, while dropping the raw duplicates, sunspot masks, and category targets. Its source revision, filtered manifest, license, citation, and acknowledgment are recorded in `data/external/SWEFil/equinexus_audit.json`. The selected 241-observation subset excludes all frozen-validation and competition-test dates, exact competition observation names, and any exact byte or decoded-pixel overlap with competition images. Source: <https://huggingface.co/datasets/antonio-reche/SWEFil>.
- External unlabeled solar imagery and public pretrained backbones may also be evaluated. First candidate: the 21M-parameter DINOv2 ViT-S/14 weights, whose authors describe label-free pretraining on 142M images and release the code and weights under Apache-2.0. Source is pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; downloaded checkpoint SHA-256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Checkpoint provenance, hash, license, and any additional image data source must be recorded per run: <https://github.com/facebookresearch/dinov2> and <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>.
- The NSF National Solar Observatory provides public GONG H-alpha FITS and 2048x2048 JPG archives. Additional observations may be used without labels for representation learning or unsupervised adaptation after checking for exact overlap with the competition images; data and manifests stay under ignored `data/external/`. Final test inference still receives only the competition's H-alpha files: <https://nso.edu/data/nisp-data/h-alpha/>.
- Expanded the unlabeled public GONG H-alpha sample to 1,443 images (576,754,076 bytes) from 2026-11-01 through 2026-11-06, sampled at most once per site every 10 minutes. All files pass SHA-256 and 2048x2048 JPEG checks; exact-filename overlap is zero across the 887 local competition images. Site counts are B=273, C=325, L=326, T=189, U=330. Content-manifest SHA-256 is `e14073f882dccb945fda142371ef3a12119f27dc9c104fbbc38f98baddb653ab`. Data and manifest remain ignored under `data/external/gong_halpha_2026_11/` and can be recreated with `scripts/download_gong_halpha.py`. The adaptation loader masks station/time/credit overlays outside the fixed solar disk before sampling. Retain visible image credit `NSO/AURA/NSF` and the GONG data acknowledgement in shared artifacts: <https://nso.edu/about/image-use-policy/>.
- SWEFil external-mask pretraining is in scope under Rule 6. The verified subset has 241 unique processed observations, 1,625 filament instance masks, and 76,888,384 image bytes; its manifest SHA-256 is `f00d087ae74d0144b8101bf37eb36ac4115525fc36d9fdc04e01aa4a18f9ab69`. It shares no validation/test dates and has zero exact decoded-pixel overlaps with competition images. The local U-Net transfer run (1,000 external-pretraining steps plus 5,000 competition fine-tuning steps) scored holdout PQ 0.197632. Its Kaggle counterpart completed at PQ 0.317355 after a threshold sweep, below the existing U-Net blend, so neither transfer checkpoint is promoted.
- For external-supervision OOF experiments, the strict SWEFil subset excludes every calendar date represented in competition train or test. It contains 229 captures and 1,504 filament masks with no exact image overlap; same-day sites stay grouped. Kaggle YOLO11s pretraining and its 25-epoch competition fine-tuning screen are complete; best frozen-holdout PQ is 0.353778, below the leading YOLO/U-Net candidate, so this warm-start is not promoted. The local COCO-pretrained YOLO11m 1024px / 40-epoch holdout run scored 0.354714 alone; its 48-setting blend with the p25 U-Net probabilities reached 0.403563, below the current 0.405860 holdout / 0.415892 grouped-OOF solution. The separate Kaggle YOLO11m 1280px five-fold OOF run remains active as of 2026-11-09 02:10 UTC.
- Added a second external-label candidate from the public HSOS/BBSO archive. The reproducible sample contains 254 observations across 128 dates, after excluding every calendar date present in competition train or test; source-selection SHA-256 is `b38c84b2af37366547de0bee8e5e6c74e4d5c0da3c82cb313de18a9b8468544b`. It has binary masks rather than instance labels and no standalone license statement in its public directory listing, so it remains an experimental weak-supervision source. The conversion builds 4,409 connected-component polygons and is evaluated only on the frozen competition split. Source: <https://sun.bao.ac.cn/hsos_data/download/filaments-unet-dataset/>; method citation: Zhu et al., <https://doi.org/10.1007/s11207-019-1517-4>.
- MORDEN's Zenodo API reports CC-BY-4.0, and the local `data.zip` matches the published MD5 plus SHA-256 `14fdd527594c3723442cc6a687eff3472a871ecb666dd7753ce8e37c565e5b3e`. We selected 695 timestamped AHAS/train CHASE observations from 2024–2025 and excluded 80 anonymized samples whose dates could not be independently verified, plus all AHAS validation/test and MHAS partitions. The source observations have no calendar-date overlap with competition data dated 2011–2022; byte and normalized-pixel checks against all 887 competition images found no exact duplicates. The verified YOLO export contains 24,631 connected components (19,628 train / 5,003 date-grouped validation); the earlier raw-mask audit counted 25,741 before export filtering. Audit-manifest SHA-256: `eac620fca90febe9787e0f98b19efce5326a6426eaefe569485d449cbd4133c3`; included-export manifest SHA-256: `bde94c44d8d613c96375539b9edda4eef770b1fdaa1167b6f5a58fa0d104dc2e`. Source: <https://zenodo.org/records/21582245>; paper: <https://arxiv.org/abs/2607.24525>; code: <https://github.com/irisaltHu/MORDEN>.
- **MORDEN YOLO11s transfer screen (2026-11-11):** corrected the external split to group exact capture dates, giving 555 train / 140 validation observations with no date overlap (split SHA-256 `15be9729be03fc7e7b5fad417e5f6f65085f239593d3f32dad267f1210822008`). YOLO11s-seg initialized from its AGPL-3.0 COCO weights was pretrained 16 epochs on MORDEN CC-BY-4.0 AHAS labels (MORDEN validation mask mAP50 0.49875), then fine-tuned 40 epochs on the 565-observation competition training partition. The fixed holdout (142 observations, 232 views; digest `30cb2ec...`) was excluded from gradients but used for checkpoint selection and PQ threshold selection. Best-checkpoint PQ was 0.370888; last-checkpoint best PQ was 0.372700 at confidence 0.32 / minimum area 0 (942 TP, 540 FP, 796 FN), versus the matched competition-only YOLO11s baseline at 0.372595 (confidence 0.33). The paired observation bootstrap delta was +0.0000924 (95% CI [-0.0086501, +0.0087343]); this is an inconclusive near-tie, so MORDEN transfer is not promoted. Keep its source terms, citation, and run records with the experiment; details are in the ignored `artifacts/detector-experiments/yolo11s-seg-morden-*/` runs.
- KSO's public data policy grants non-profit scientific and educational use, prohibits profit purposes, and requests source acknowledgement, notification, and a copy of resulting research reports before publication. The user has explicitly authorized public non-commercial research/education resources under the challenge rules; preserve KSO attribution and source conditions, and do not use any public MAGFiLO labels beyond competition-provided training annotations. The 128-observation KSO weak-label pretrain (2,305 automatic filament regions; manifest SHA-256 `5a40eac8ef22354339a69cf8e869b035a76a88a82943fd39a6876e2019b72058`) followed by YOLO11s fine-tuning scored fixed-holdout PQ 0.365580, below the matched competition-only baseline 0.372595, and is not promoted. KSO policy: <https://cesar.kso.ac.at/info/datapolicy.php>.
- **Flat U-Net external labels:** the China-VO PaperData archive is public and available at no cost; its portal states that copyrights remain with the paper authors and the dataset record does not state a separate license. Under Rule 6, we use a private Kaggle compute copy with attribution and no public redistribution. The source archive is 8,375,352,019 bytes (SHA-256 `6e4cef18daedb19bcad99d590efdc3a480013b67c1bc516af3d2ad229555024e`). It contains 1,007 paired training FITS/mask observations; excluding all 801 competition train/test calendar dates removes 27 observations, and seven empty masks are excluded, leaving 973 observations on 396 dates. A deterministic sample of 256 valid images on 256 distinct, evenly spaced dates is prepared under ignored `data/external/flat-unet-101545/processed-256/`; all competition dates are excluded and the source `test_gd` partition is unused. The lossless normalized PNG images and binary semantic masks total 502,642,416 bytes. A quality-90 JPEG Kaggle derivative uses 95,163,374 bytes, has mean absolute compression error 1.065 intensity levels, and preserves the binary masks. The PNG selection manifest SHA-256 is `3a360ab58617af8e81635a9a2de3eedddfe7b2832fd4b466f2bca5938610d6d5`; the JPEG derivative manifest SHA-256 is `049bd1dda9c2d95d161b96f7fe1b29f0bb053c44ffb754b610f9b8e0f78c0a22`. The 2048x2048 FITS images are clipped at full-image p0.5/p99.5 before conversion to uint8; masks remain binary and are used only as weak semantic pretraining targets before instance-aware competition fine-tuning. Source: [China-VO dataset](https://doi.org/10.12149/101545), [PaperData rights and access](https://nadc.china-vo.org/article/20200519170135), and [Flat U-Net paper](https://doi.org/10.3847/1538-4357/adadff).
- KIS SDC's ChroTel archive exposes 1,007 daily H-alpha filament-segmentation products from 2012-04-13 through 2020-09-02. KIS states that ChroTel data are freely available for scientific and educational use, with citation and acknowledgement required. The archived masks are predictions from a semi-supervised segmentation method, not manual instance labels, so they are evaluated as weak semantic pretraining targets before fine-tuning on competition instances. Eight date-disjoint image-mask pairs were inspected visually; of 1,007 products, 774 remain after excluding all 801 calendar dates present in competition train or test. A deterministic, evenly spaced 256-observation sample is now prepared under ignored `data/external/chrotel/`; its selection manifest SHA-256 is `3ee0a0a572e3c65b19260d3dec4078cfc8035d188ed8501252c1d961acafa0fb`, with zero competition-date overlap and the frozen split digest `30cb2ec...`. The processed PNGs, masks, and provenance are attached as the private Kaggle input `mrinanksivakumar/chrotel-date-disjoint-2026` for GPU runs; raw FITS archives stay local under `data/external/chrotel/archives/`. Source: <https://archive.sdc.leibniz-kis.de/data-calendar>, <https://sdc.leibniz-kis.de/en/license-policy/citation>, and <https://doi.org/10.1051/0004-6361/202348314>.
- YOLO11-seg with public COCO-pretrained weights is a separate external-model experiment allowed under Rule 6. Ultralytics 8.4.173 and the weights are AGPL-3.0; the model is fine-tuned only on the competition training annotations and predicts only supplied competition images. On the frozen holdout, detector-only PQ is 0.369152 and detector-guided fusion with the best U-Net probability blend scores 0.405860 (the U-Net blend scores 0.366112). These holdout-selected thresholds are screening evidence; five-fold OOF detector-only evaluation scored 0.398791 on the 565-observation training partition, above the U-Net blend at 0.365637, and the completed 45-configuration five-fold OOF detector-guided fusion sweep scores 0.404464 at foreground 0.50, detector confidence 0.25, no mask growth, and minimum claimed area 400; U-Net-only OOF is 0.365637. The OOF count is 3,642 TP, 1,900 FP, and 2,846 FN over 922 annotator views. A six-setting outer-fold check chose the same configuration on all five folds and scored 0.404464; because this reduced grid followed a broad full-OOF screen, it is a stability check, not a fresh unbiased estimate. The frozen holdout was excluded from all training and scoring. The user authorized releasing EquiNexus under AGPL-3.0 if YOLO is selected; based on the completed OOF improvement, the repository now carries the AGPL-3.0 license. The OOF scorer reuses the official-PQ implementation and excludes the fixed holdout.
- Previously logged scratch-only results remain valid descriptions of those historical runs. The from-scratch-only restriction is lifted for new experiments, subject to the boundaries above and a usable, reasonably accessible model/data license.

## Public method review

- Anthony Therrien's public blended-tile U-Net notebook uses the supplied training COCO annotations, unions the annotator masks per source observation, and predicts a single 1888-pixel native-resolution solar-disk crop. Its crop and union-target ideas remain relevant; external initialization will be assessed on the frozen EquiNexus split and scorer: <https://www.kaggle.com/code/anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments>.
- Dariush Afshar's public U-Net report describes grouped five-fold training, soft consensus masks, 512-pixel full-resolution crops, and native-resolution inference, with reported 0.4502 nested OOF PQ. The checkpoint pack is CC BY-NC-SA 4.0 and is now eligible under the user's updated policy. Its published folds split the 142-observation frozen holdout across all five models, so the pack cannot be evaluated against that holdout; see the dated fold audit below. Sources: <https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39> and <https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights>.
- An earlier revision of Robina Mirbahar's public notebook described radial disk normalization, multiscale black-hat salience, robust median/MAD thresholding, hysteresis, and short-gap bridging; our independent reproduction scored PQ 0.0554 on the frozen holdout. The Kaggle page now contains a different I079-based inference handoff under the same URL, with none of those classical operations. Treat the earlier result as a historical version-specific experiment, not a description of the current notebook: <https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation>.
- The public YOLO plus U-Net notebook initializes from externally pretrained YOLO and ResNet weights. Its weights are candidates for a separate provenance and license audit before any use; architecture ideas remain a reference: <https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation>.
- A public challenge repository reports a two-channel, radially corrected U-Net combined with YOLO11-seg detectors and eight-way rotation/reflection TTA. Its MIT-licensed code is pinned for review at commit `81a739bb7786a44b989f5d76afbb6f9d22b5d1dd`. Replaying its documented 20%-by-date, seed-0 split against competition annotations shows that 112 of our 142 frozen holdout observations occur in its 563-observation training partition; these checkpoints cannot be scored on the frozen holdout. We independently evaluated the public CC BY-NC 4.0 checkpoint bundle on its separate 144-observation upstream validation set. Cross-fitted fusion PQ was 0.419160 versus 0.412857 for our selector on that same split (paired 95% interval for the difference: -0.005295 to +0.017938), so the gain is inconclusive and the candidate is not promoted. Preserve its attribution and license terms; preserve Ultralytics' AGPL terms if its code or weights are used. Sources: <https://github.com/ameyypawar/filament-seg> and <https://www.kaggle.com/datasets/ameypawar123456789/filament-seg-weights>.
- Farnoosh Memari's MIT-licensed public repository (reviewed at commit `ef37d2e93d4cc9e248595e791ee39fe520d392d0`) reports 0.389 validation PQ with a compact U-Net trained from scratch, a soft per-pixel annotator consensus, 70% foreground-centered 512-pixel crops, disk-based intensity normalization, and four-view TTA. Its month-held-out split and postprocessing search differ from the EquiNexus protocol, so that score is not directly comparable. The cited run used no external weights or labels. New pretrained candidates are assessed separately on the frozen EquiNexus split: <https://github.com/FarnooshMemari/solar-filament-segmentation>.
- A follow-up review of public challenge repositories found similar two-stage detector/refiner pipelines reporting 0.3593 five-fold mean PQ, and another YOLO-based approach reporting 0.3526 on one grouped fold. Both rely on externally pretrained detector or encoder weights, so their reported scores are context only; we retain the grouped scorer and audit external checkpoints for provenance, split integrity, and license: <https://github.com/HeShen-1/filament-segmentation-2026> and <https://github.com/Wzy633/solar-filament-segmentation-2026>.
- A closer review of HeShen-1's MIT-licensed repository at commit `b41dae215a87c6cb6dfbcf1459a4819021e0092d` and its paper reports a detector-only five-fold PQ range of 0.2868–0.3156, a five-fold mean of 0.3593 for its full two-stage pipeline, and fold 0 about 0.017 above the mean of its other detector folds. The authors also report that CLAHE did not resolve false detections, simple morphology gates hurt, and inter-annotator agreement limits mask-quality gains. These are author-reported results on a different date-grouped split and architecture, not directly comparable EquiNexus validation. This evidence reinforces grouped cross-fit and careful false-positive control; checkpoint reuse requires the external-data and model audit described above. The source-only audit clone is ignored under `artifacts/kaggle/reference-audit/heshen-public-solution/`: <https://github.com/HeShen-1/filament-segmentation-2026/blob/main/paper/paper.tex>.

## Public leaderboard and model audit (2026-11-07)

- The Kaggle API snapshot at 00:40 UTC showed the top team score at 0.70, followed by 0.64 and several 0.62 entries. The Code tab also surfaces notebooks with scores above 0.90. The host says scores above 0.35 already have high value and warns that leaderboard results can be misleading; an empty-mask scoring defect was publicly reported. We record our own scores for observation, but promote solutions only on the frozen official-scorer validation: <https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements> and <https://www.kaggle.com/discussions/product-feedback/729163>.
- Kaggle's public `phuongncn/solar-filament-i079-weights` model is Apache 2.0. Its model card says it was trained only on competition training annotations and did not use public MAGFiLO/test-overlap labels; the linked method uses COCO initialization for Mask R-CNN and ImageNet ResNet-18 initialization for its refiner. We verified both published checkpoint MD5 hashes and completed test-only inference with the independent EquiNexus runner. The publisher reports a 0.70 public score, which remains preliminary evidence; the full-fit checkpoints cannot be scored on our frozen holdout. The run and submission are recorded below: <https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights>, <https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff>, and <https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner>.
- The public CAGE R4 checkpoint card reports a three-head mask/hull/interior U-Net and a held-out PQ of 0.4022. Its ResNet-34 encoder starts from ImageNet weights, and its 100-entry record-level split allows multi-annotator observations to cross folds. The CC BY-NC 4.0 weights are eligible for consideration under the updated user policy, but the reported split is not observation-grouped and the saved checkpoint may include the EquiNexus holdout. Its score cannot be used as a comparable fixed-holdout result; the hull grouping head remains a reference idea: <https://www.kaggle.com/models/jamesmills1975/cage-filament-unet>.
- The newly released V23 package reports public LB 0.39 and uses five YOLO11 detectors, five fold-specific refiners, cross-fold association, and medoid masks. Its README says inference starts from pretrained weights, and attribution requires preserving the Ultralytics/model licenses. We use its cross-fold ensemble idea as a reference; the earlier EquiNexus U-Net folds were scratch-trained, while the COCO-initialized YOLO folds are evaluated separately under Rule 6: <https://github.com/Jeanjean59800/solar-filament-segmentation-2026>.
- Farnoosh Memari's MIT-licensed from-scratch U-Net reports PQ 0.389 on a held-out-month split and 0.32 public LB; it downscales to 1024, uses soft consensus targets, then joins nearby fragments. This is a useful scale/postprocessing reference, though the split and results are not directly comparable: <https://github.com/FarnooshMemari/solar-filament-segmentation>.
- The earlier submission batch used only EquiNexus scratch-trained checkpoints. New submissions may use audited public pretrained models or unlabeled external solar data, but never public MAGFiLO test annotations.

## EquiNexus candidates

- **U-Net:** small five-level, two-head model trained from scratch. Foreground and instance-boundary targets are built from official masks; consensus mode averages annotator masks while annotator mode trains from each view. Training uses full-resolution crops divisible by 32, geometric transforms, intensity jitter, BCE/soft-Dice loss, and mixed precision. Overlapping tile inference produces probability maps, then boundary-guided watershed extracts instances.
- **Classical:** an independent implementation of the radial-normalization and multiscale morphology approach above. Robust z-map preprocessing is cached once per validation observation; thresholds and bridge size are evaluated with the official PQ scorer.
- **Selected ensemble:** a 50:50 probability blend of a 10,000-step, 1024-pixel per-annotator U-Net with eight-way TTA and a 5,000-step, 512-pixel consensus U-Net with four-way TTA. The frozen holdout score is 0.36421; five-fold grouped OOF PQ is 0.365626, above both component models. The selected postprocessing uses foreground 0.65, boundary 0.70, boundary weight 1.5, seed threshold 0.45, seed area 8, and minimum instance area 400. The OOF score supports the blend choice without using the holdout for training.
- **Local-contrast alternative under cross-fit:** a two-channel per-annotator U-Net trained with raw intensity and local contrast scored holdout PQ 0.362937 at foreground 0.55 and boundary 0.70 (minimum area 400), after a corrected 45-point threshold sweep. Its five-fold OOF run is in progress; it will replace or join the selected ensemble only if its OOF comparison improves the score.
- **Grouped OOF comparison:** the completed Kaggle consensus512/5k/TTA4 run scored PQ 0.346691 over 565 OOF observations and 922 annotator views. The local per-annotator1024/10k/TTA8 run scored 0.356643 at the selected 0.70 boundary threshold and 0.359830 at boundary 0.80. Their 50:50 probability blend scored 0.365626 at boundary 0.70 (0.365637 at boundary 0.80), with fold PQ from 0.35258 to 0.37376. Blend weights 25:75 and 75:25 scored 0.354469 and 0.363135. The selected blend configurations produced identical TP/FP/FN counts at boundary thresholds 0.70 and 0.80. All runs used frozen manifest digest `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c` and excluded all 142 holdout observations. OOF checkpoints were selected using patch Dice on their held-out fold, so these scores are selection-aware cross-fit results rather than nested estimates. The frozen holdout is outside model fitting, but has been used for iterative candidate and postprocessing selection.
- **Solar-disk geometry audit:** a simple Otsu-contour fit on the 142 holdout images estimates radii from 880 to 1,012 px and center offsets up to 59 px horizontally and 63 px vertically. All 1,711 supplied validation instances (3,630,973 annotated pixels) remain inside the fixed 950 px circle centered at (1023, 1023), so the current clipping rule has not removed validation ground truth. Adaptive clipping is only a possible false-positive reduction experiment.
- **Runtime:** use the local RTX 5070 and Kaggle GPU where available. Keep source, split, and scorer shared; record per-run source hashes, environment, device, peak memory, and timings. Local outputs remain under ignored `artifacts/`; Kaggle outputs stay under `/kaggle/working/` and are downloaded only into this repository.

## 2026-11-06 sprint record

- **All-training-set Kaggle run:** kernel `mrinanksivakumar/equinexus-filament-final-ensemble`, version 1, completed the from-scratch per-annotator 1024-pixel U-Net (10,000 steps, TTA 8) and consensus 512-pixel U-Net (5,000 steps, TTA 4), then cached probability maps for the supplied 180 test images. The 50:50 blend submission passed coverage validation with 1,644 instances. Source commit `9668cdee92a5ef1f9150bb585b54dbe7881e7f7a`; embedded source bundle SHA-256 `b2c066c57959bc0de2880ec7e72c44c0d34c03e26dfb1307e260a939816f3522`. End-to-end runtime was 10,993.27 seconds; inference peak allocation was 4.54 GB. Kaggle reported two Tesla T4 GPUs with 15,360 MiB each; the run used `cuda:0`, Python 3.13.16, PyTorch 2.11.0+cu130, and CUDA 13.0. No external supervised weights or ground truth were used.
- **Five submissions:** refs `58075222`, `58075226`, `58075231`, `58075234`, and `58075239` completed in descending OOF-PQ order. Their OOF PQs were 0.365626, 0.363135, 0.359830, 0.356643, and 0.354469; public scores were 0.32, 0.32, 0.31, 0.31, and 0.30. Each unique CSV was checked against all 180 official test images. In the 2026-11-07 01:38 UTC leaderboard snapshot, EquiNexus ranked 962/1,411 with a best score of 0.32; the displayed top score was 0.70. The full snapshot is under ignored `artifacts/kaggle/leaderboard/`. The five daily submissions were used.
- **Local contrast-input cross-fit:** the raw-plus-local-contrast two-head U-Net remains under training on the RTX 5070 Laptop. Two of five grouped folds have completed; fold 2 scored PQ 0.364114 and fold 3/5 has started. The frozen 142-observation holdout remains excluded from every fold.
- **Five-fold test inference:** Kaggle kernel `mrinanksivakumar/equinexus-five-fold-u-net-ensemble` version 3 is running with five scratch per-annotator checkpoints, TTA 8, the exact attached OOF manifest (SHA-256 `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c`), and the frozen holdout. Its embedded source bundle SHA-256 is `dd50f46a1e1c4f3e1dab9d353fada85a377bbab7ee8bf11fee7d1d1df077b81a`. Version 1 stopped because the ignored OOF manifest was absent from Kaggle inputs; version 2 stopped because device-qualified CUDA peak-memory reset failed on T4. Version 3 packages the manifest with the private checkpoint dataset and uses the current CUDA device with a guarded memory-stat fallback. It was still running at 2026-11-07 01:32 UTC.

The submission manifest and scores are recorded under ignored `artifacts/submissions/2026-11-06-candidates-v2/`. Kaggle scores are not used as the sole candidate-selection signal; grouped validation remains primary. Inference uses only the supplied H-alpha images. Supervision may use the supplied competition training annotations and qualifying, separately licensed external datasets under Rule 6; outside MAGFiLO annotations remain excluded.

## 2026-11-07 Kaggle status

- Authenticated Kaggle CLI snapshot at 2026-11-07 03:46 UTC contains 1,411 public leaderboard entries. EquiNexus is rank 962 with public score 0.32; the displayed top score remains 0.70. The extracted CSV SHA-256 is `3e92b3c411f7e3266314ebac6d7a5525ef7bc7b1fc22f706e3496a5f5a6cf51b`; the CSV and original CLI archive are saved under ignored `artifacts/kaggle/leaderboard/2026-11-07/`.
- Kaggle reports 5/5 submissions used today and none remaining. The five submissions are confirmed complete with scores 0.32, 0.32, 0.31, 0.31, and 0.30; wait for Kaggle's quota reset before another submission batch.
- Private kernel `mrinanksivakumar/equinexus-five-fold-u-net-ensemble` version 3 completed at 2026-11-07 05:00 UTC with holdout PQ 0.360551. It predicted all 180 test images with 1,530 instances. Full holdout and test probability caches, metrics, CSVs, and overlays are downloaded under ignored `artifacts/kaggle/fold-ensemble/remote-v3-output/`. Kaggle allocated T4 x2; inference took 14,333 seconds, and the memory metric was unavailable (reported as zero), so this is not an efficiency candidate.

## 2026-11-07 external data and pretrained model update

- The user clarified Rule 6 permits reasonably accessible external data and models. The previous from-scratch-only scope is lifted for new experiments. We do not use public MAGFiLO annotations beyond the supplied competition training annotations. Separate qualifying external datasets may contribute labels under Rule 6, and final test inference uses only the supplied H-alpha files.
- Expanded the unlabeled NSO GONG sample to 1,443 images under ignored `data/external/gong_halpha_2026_11/`; see the external-data section above for the full image audit, attribution, and manifest digest. No annotation files were downloaded.
- DINOv2 ViT-S/14 is an Apache-2.0, label-free pretrained backbone (LVD-142M). Its public checkpoint is cached under ignored `artifacts/pretrained/` and has passed CPU forward/backward smoke checks. Kaggle kernel `mrinanksivakumar/equinexus-filament-model-sprint` version 3 completed its 5,000-step supervised fit and is running full-resolution inference on the frozen 142-observation holdout; latest poll was 130/142. Patch Dice was 0.61747 at step 5,000; full PQ remains pending.
- Prepared but have not uploaded notebook version 4, which downloads and checksum-verifies the six-day GONG sample, runs 1,000-step DINOv2 self-distillation without labels, then runs the 5,000-step supervised holdout experiment. Embedded-source bundle SHA-256 is `e0c2b608c70676ef23e85bea5821a2afe52e2306a89085193f635c1e66a45227`. It will be pushed after version 3 completes.
- The five-fold scratch U-Net inference kernel completed with holdout PQ 0.360551 and emitted a submission covering 180 test images with 1,530 predicted instances. Its full output bundle is now local under ignored `artifacts/kaggle/fold-ensemble/remote-v3-output/`; this candidate remains below the previous 0.364212 holdout blend and took 14,333 seconds for inference.
- The local RTX 5070 Laptop is training fold 5/5 of the raw-plus-local-contrast per-annotator OOF candidate. At the latest poll it had reached step 8,500/10,000. Kaggle is running the DINOv2 candidate independently. Both runs use the fixed manifests; neither trains on the 142-observation holdout.

## 2026-11-07 external annotation policy check

- The initial commit note in `d9e23b6` recorded an interim conservative assumption: no external supervised weights or labels. The host's [clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396), read with Rule 6, limits MAGFiLO ground-truth annotations to the competition-provided set while allowing qualifying independent public external data. Independent labels may be used when they meet Rule 6 access requirements and the source's use conditions. The active SWEFil and other external-data experiments follow this scope; no outside MAGFiLO annotations are used, and final inference reads only the supplied H-alpha test images.

## 2026-11-06 extended public-reference audit

- Downloaded ten public competition notebooks for source review into ignored `artifacts/kaggle/reference-audit/`. The `phuongncn` 0.70 handoff describes a Mask R-CNN plus crop-refiner pipeline, four-view TTA, and conservative detector-seed splitting. It depends on shared model checkpoints, so we use its design as context only and do not load its weights.
- The public `artkomissar/please-fix-the-leaderboard` notebook documented a historical defect where five empty predictions out of 180 received public score 1.00; all-empty controls also scored 0.88 and 0.70. The host subsequently clarified that final evaluation is not based on the displayed score: the leaderboard is only a preliminary filter before a quantitative and qualitative review. We will not exploit incomplete submissions, and candidate selection remains based on the official grouped scorer. <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062>
- Kaggle dataset search surfaced the official full MAGFiLO release, repackaged public MAGFiLO snapshots, and challenge-trained checkpoint datasets. We did not download or attach any of them: the full release contains annotations for test observations, and all externally shared supervised checkpoints remain outside this experiment. The host's ground-truth restriction remains controlling: <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396>.
- Added an experimental three-channel input mode combining disk-normalized H-alpha, local contrast, and a per-radius median contrast profile. It uses only each supplied image at train and inference time; no labels or parameters are derived from test observations. Feature and augmented-patch CPU smoke checks passed. Grouped five-fold training of the two-channel local-contrast model is still running; the radial mode has not yet been scored.
- Manual review of representative frozen-holdout overlays shows the U-Net often captures the central filament body but misses low-contrast branches and full filament extent. This motivates testing stronger large-context inputs and improving recall; visual review alone does not change the selected candidate.
- Prepared, but not uploaded, the next Kaggle holdout candidate: consensus targets, a 32-channel-base U-Net trained from scratch, raw plus local-contrast input, 1024-pixel patches, batch size 1, 20,000 steps, and eight-view TTA. Its generated notebook SHA-256 is `ffb4c8bd3ac786666945bfc2cb9b5b0390da1ff0d6c10209cc9ce6780c5dabfa`; embedded source bundle SHA-256 is `f91c6c31195d214781cc0fe7a742b1b53c614a70bda6b1a0960f97b6edf5a4cf`. The notebook includes a CUDA forward/backward memory smoke and uses the run's own validation paths for both postprocessing sweeps. It is waiting for the current Kaggle inference kernel to finish.

## 2026-11-07 external-data follow-up

- This follow-up initially used external solar imagery without labels and audited label-free model weights while the SWEFil eligibility question was under review. That restriction was later lifted for the separately licensed SWEFil dataset after checking the host clarification and Rule 6; no outside MAGFiLO annotations are used, and final inference uses only the supplied H-alpha test images.
- Added an audited NSO GONG sample of 3,268 unlabeled 2048x2048 images from 2023-2024. Both filename intersection and byte-identical SHA-256 overlap with the 887 competition images are zero; all image hashes and dimensions pass. Manifest SHA-256 is `c2991c659575d015bbf7e8efbe9382c01de9776b00bcfcc0331920618e468aef`; data stay under ignored `data/external/gong_halpha_2023_2024/`.
- DINOv2 self-distillation on this sample completed 2,000 steps in 471.72 seconds on the local RTX 5070, with final loss 1.675 and peak allocation 852 MB. The subsequent 5,000-step fit on competition annotations completed with patch Dice 0.60918; its best frozen-holdout threshold sweep scored PQ 0.324793 (842 TP, 913 FP, 869 FN). It was not promoted.
- Kaggle model sprint version 4 failed before training because its NSO archive requests timed out. Version 5 is running a from-scratch consensus U-Net with 1,024-pixel patches, 32 base channels, 20,000 steps, batch 1, raw plus local-contrast input, and eight-view TTA. At step 5,000 it had run 3,944.48 seconds; validation patch Dice was 0.51051. Embedded-source SHA-256: `3d8bdb9f834e41ae6ba9660c62df31ecb0773eddd34fd34f766de11a4e62393f`.
- The 5,000-step Kaggle DINOv2 candidate version 3 completed at holdout PQ 0.339394 after threshold and structure sweeps. Adding 5-25% of its probability maps to the current U-Net ensemble did not improve holdout PQ; its best tested blend reached 0.364050.

## 2026-11-08 public checkpoint audit

- Kaggle model `phuongncn/solar-filament-i079-weights`, PyTorch variation `detector-refiner` version 1, is public and identified as Apache-2.0 in its model metadata. Its model card says its Mask R-CNN and ResNet-18 crop-refiner weights were trained only on the competition training images and annotations, with no public MAGFiLO/test-overlap labels. The accompanying public inference notebook reports a 0.70 public leaderboard score; we treat this as an author-reported preliminary result, not an EquiNexus validation score.
- These weights have seen the full competition training set, including our fixed holdout, so they are excluded from grouped validation and are eligible only for final test-image inference. The private EquiNexus notebook `notebooks/external_panoptic_inference.ipynb` attaches the public model and competition inputs, independently implements the inference architecture, reads no annotations or test labels, validates RLE output, and records checkpoint hashes, runtime, hardware, and coverage. Model card: <https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights>; published inference reference: <https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation>.
## 2026-11-08 test-only public-checkpoint inference

- Private Kaggle kernel `mrinanksivakumar/equinexus-external-panoptic-inference` version 1 completed on two detected Tesla T4 devices (the inference used one), with 176.36 seconds reported runtime and 1.55 GB peak allocated GPU memory. Its source bundle SHA-256 is `2a90331207116ee758f0b315e4be30910c488ac0cbe791343abb176055d223a6`.
- The independently implemented adapter loaded the public Apache-2.0 `phuongncn/solar-filament-i079-weights` detector and refiner, applied horizontal/vertical flip TTA, processed all 180 supplied test images, and read neither training annotations nor test labels. Since the model card says the weights used all official training observations, this run is test inference only and makes no holdout-validation claim.
- The output contains 1,299 unique, nonempty 2048x2048 RLE masks over 169 images; 11 images have no predicted instances. The CSV hash is SHA-256 `8c064a0da930de9add44e63eef1b0cb4ca85325ed74f224403938232f4beb2dc` (MD5 `09c0eae5fdff29a91f8a740475ecaf36`), matching the public reference candidate output. Kaggle submission 58110863 completed with public PQ 0.34, above our previous best 0.32; this score remains secondary to grouped validation. The hash match demonstrates output parity, not independent evidence of generalization; grouped validation remains the model-selection authority.
- Checkpoint hashes and per-image diagnostics are preserved in `artifacts/kaggle/external-panoptic-inference/remote-v1-output/EquiNexus/public-i079/run_report.json`. Model card: https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights; independent inference reference: https://www.kaggle.com/code/robinamirbahar/solar-filament-panoptic-morphology-segmentation.

## 2026-11-08 Kaggle leaderboard refresh

- Kaggle submission 58110863 is complete at public score 0.34. The five-fold U-Net cache variants 58115460 and 58115525 are complete at 0.31 each; refs 58115589 and 58115626 remain pending. The prior five submissions remain complete at 0.30-0.32.
- The 2026-11-08 public leaderboard snapshot lists EquiNexus at rank 796 of 1,421 teams with score 0.34. The downloaded archive is `artifacts/kaggle/leaderboard/2026-11-08/filament-segmentation-2026.zip`, SHA-256 `d8553598fd47fe393a1bc40a8af5ca5f62fc9f73e46ab002d7e3b7f40711ddc3`. As the host describes the leaderboard as a preliminary filter, this is secondary evidence and does not replace grouped validation: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062.
## 2026-11-08 five-fold U-Net submission threshold screen

- Re-scored six postprocessing configurations from cached five-fold U-Net predictions against the unchanged 142-observation holdout (232 annotator views). The best was foreground 0.55 / boundary 0.70 at PQ 0.363726 (1,031 TP, 1,014 FP, 680 FN); the next settings scored 0.362301 (0.60/0.65), 0.360549 (0.65/0.70), 0.359309 (0.50/0.70), 0.352721 (0.55/0.60), and 0.350522 (0.65/0.60). Other settings used boundary weight 1.5, seed threshold 0.45, seed minimum area 8, minimum instance area 400, and a 950-pixel solar radius.
- The top four settings were converted into CSVs and passed local checks for all 180 supplied test images, unique IDs, non-overlapping masks, and 2048x2048 RLE dimensions. Submission refs 58115460, 58115525, 58115589, and 58115626 used 1,564, 1,555, 1,530, and 1,583 instances respectively. The first two public scores are 0.31; the last two remain pending as of 2026-11-08 00:08 Toronto time. SHA-256 and complete validation details are in each ignored `artifacts/submissions/2026-11-08-foldensemble-*/run-metadata.json` file.
- The five-fold U-Net result remains below the YOLO/U-Net fusion OOF score of 0.404464 and the public I079 inference submission score of 0.34. The holdout comparison supports retaining the detector-guided fusion as the leading validation candidate; leaderboard scores remain secondary evidence.

## 2026-11-08 public detector/refiner reference audit

- Reviewed two highly voted public notebooks: Nomannic19's Mask R-CNN plus crop U-Net refiner and Tim Gluhkikh's YOLO11s seed-refiner pipeline. The latter describes 512-pixel crops with grayscale, CLAHE, and a box seed, plus optional box-jitter test-time augmentation. These are architecture ideas only; neither notebook's model weights are being reused.
- The public notebooks do not declare a code license in their Kaggle metadata. Tim's notebook depends on `tienti/yolo-magfilo` and `tienti/seed-refiner-magfilo` checkpoints; their model metadata and training provenance were not accessible for verification, so these checkpoints are excluded. The code reports a local score near 0.409 using its own evaluation; this is not established as the official organizer scorer result.
- The referenced `deepxchat/solarfilament` Kaggle dataset is CC0 metadata for a mirror of the competition data, including the supplied test images and training annotations; we did not download or use that copy. Source notebooks are stored under ignored `artifacts/kaggle/references/` for audit only.

## 2026-11-08 external warm-start postprocessing check

- Kaggle private kernel `equinexus-external-postprocessing-sweeps` completed against the unchanged grouped holdout manifest. Its threshold and instance-structure sweeps did not improve either external warm-start model: ChroTel reached PQ 0.255020 and Flat U-Net reached 0.253396 after the instance-structure search, below their initial scored configurations and far below the YOLO/U-Net candidate. Neither is promoted.
- The two-source postprocessing job took 6,025.44 seconds for ChroTel and 5,781.61 seconds for Flat U-Net. Full summaries and selected configurations are under ignored `artifacts/kaggle/external-postprocessing-sweeps/remote-v1-output/`; source bundle SHA-256 `2c0318d61dce1530bbf940acdd012edf5cb4be677f6d9c6ce7154310af79a166`.
## 2026-11-08 YOLO11m grouped OOF experiment

- Kaggle kernel `mrinanksivakumar/equinexus-yolo11m-grouped-oof` version 4 is running YOLO11m-seg at 1024 px, 20 epochs, batch size 2, five grouped folds, and one GPU. It reuses the seed-42 OOF manifest (`f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c`) over the 565-observation training partition and excludes the frozen holdout (`30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`). Source bundle SHA-256: `116870574dd812140f60f16b2b374130f5b993cceeb364ffa61eeebd116d235e`.
- Versions 1-3 failed before training because the initial notebook omitted the ignored OOF manifest, passed `/kaggle/input` where a dataset root was required, and then assumed Kaggle's embedded workspace was a Git checkout. Version 4 includes the manifest, uses dataset-root discovery, and reads its source commit from the notebook's embedded metadata.
- Historical note: the public 10-model U-Net checkpoint set was initially excluded under an earlier user instruction because it is CC BY-NC-SA 4.0. That restriction was superseded on 2026-11-11. It is now used only for supplied-test inference because its fold map overlaps every part of the frozen holdout. Kaggle dataset and notebook: https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights and https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39.

## 2026-11-08 SMP ResNet34 grouped OOF

- The ResNet34 encoder used ImageNet-pretrained weights. The user permits public research/education resources with non-commercial terms; this run is therefore eligible under the team's policy and remains rejected only on its validation result (grouped OOF PQ 0.367271, below the incumbent). Sources: https://docs.pytorch.org/vision/main/models and https://image-net.org/accessagreement.
- The local RTX 5070 five-fold run completed with grouped OOF PQ 0.367271 (folds 0-4: 0.381588, 0.377690, 0.355633, 0.354198, 0.366462). Its frozen-holdout manifest is `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; the OOF manifest is `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c`. The artifacts remain under ignored `artifacts/oof-experiments/smp-resnet34-consensus512-3k-oof/` for transparent historical accounting, but the checkpoints are not eligible for use.
- The Kaggle CLI no longer has authenticated access to the private kernel (`kernels.get` is denied; listing kernels requires authentication), and the installed CLI has no stop command. Its current state could not be verified or cancelled; no remote outputs were used. OAuth had timed out in the earlier login attempts.

## 2026-11-08 scratch-trained YOLO-seeded crop refiner

- An independent crop-refiner implementation is in `src/filament_baseline/seed_refiner.py`; it borrows only the high-level idea of conditioning a crop U-Net on a detector seed. No reference notebook code, weights, or labels were reused. The ResNet34 backbone is random-initialized in this experiment; ImageNet initialization remains eligible under the user's resource policy, though the prior ImageNet run did not beat the incumbent.
- Training uses the 565 fixed training observations, their competition annotator masks, and five-fold out-of-fold YOLO detections. This produced 11,341 crop/view samples (8,990 matched targets, 2,351 negatives). The 142 frozen holdout observations use only their independent five-fold detector predictions; the split SHA-256 remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- The local RTX 5070 run completed 3,000 steps at 512-pixel crops, batch 8, mixed precision, and four data workers in 336.25 seconds; peak allocation was 2.10 GiB. A batch-2 512-pixel forward/backward smoke used 596 MiB. Initial fixed settings (detector confidence 0.25, crop probability 0.50, minimum crop area 200) scored holdout PQ 0.348178 (1,049 TP, 1,211 FP, 662 FN) across 232 annotator views, below the detector/U-Net OOF candidate; a small threshold screen is in progress. Output is under ignored `artifacts/model-experiments/2026-11-08-seedrefiner-yolo11s-oof-scratch/`.
- Four targeted crop-refiner tests pass. The runner writes official-scorer metrics, per-observation scores, overlays, source/config hashes, hardware details, and runtime into ignored artifacts.

## 2026-11-08 challenge rubric and leaderboard audit

- The organizer's current evaluation description weighs quantitative comparisons, morphology, and code/pipeline quality; its August announcement specifically cautions against hacking the scoring system. A recent host reply reiterates that the public leaderboard is one of three considerations and asks teams to justify model choices with validation beyond the public score: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements and https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/742143.
- A newly indexed public report claims to reverse-engineer a visible leaderboard metric. We are not adopting leaderboard-specific reverse-engineering or test-label-driven methods. Candidate selection remains based on the supplied training annotations, the frozen observation-grouped split, and the organizer's scorer.

## 2026-11-08 seed-refiner holdout screen

- The initial score at confidence 0.25, crop probability 0.50, and 200-pixel crop minimum was PQ 0.348178. A 12-setting scorer-aligned screen over detector confidence, probability threshold, and minimum crop area selected 0.45 / 0.50 / 400 at PQ 0.374363 (876 TP, 512 FP, 835 FN over 232 annotator views). It improved precision but remains below the 0.404464 grouped OOF detector/U-Net candidate; it is a holdout-selected screen and is not promoted.
- The selected CSV has columns `filament_id, segmentation_rle`, 1,379 unique IDs across 141 holdout images, and a decoded sample round-trips to 2048x2048. One holdout observation has no predicted instance. Per-setting metrics, diagnostics, and observation scores are in ignored `artifacts/model-experiments/2026-11-08-seedrefiner-yolo11s-oof-scratch/threshold-sweep/`.
- A follow-up 11-setting screen combined the best refiner masks with existing five-fold U-Net foreground probabilities. Gating and local expansion did not help; the best PQ was 0.348257 (688 TP, 263 FP, 1,023 FN), below the 0.374363 refiner-only screen. It is not promoted. Full scorer diagnostics, per-observation results, and overlays are under `artifacts/model-experiments/2026-11-08-seedrefiner-yolo11s-oof-scratch/threshold-sweep/unet-fusion/`.

## 2026-11-08 10,000-step seed-refiner follow-up

- A scratch-initialized ResNet34 crop refiner trained for 10,000 steps on 11,341 OOF-detection crops (crop size 640, batch size 6, learning rate 0.00015) on the local RTX 5070 Laptop GPU. Training took 1,345.87 seconds; end-to-end runtime was 1,438.63 seconds, and peak allocated GPU memory was 2.41 GiB.
- The fixed 0.45 detector-confidence / 0.50 crop-probability / 400-pixel minimum setting scored holdout PQ 0.392289 (910 TP, 448 FP, 801 FN). A 12-setting screen selected the same configuration. This is below the grouped OOF detector/U-Net result of 0.404464, so the refiner is not promoted. The best screen remains holdout-selected, not an unbiased estimate.
- Run artifacts are under ignored `artifacts/model-experiments/2026-11-08-seedrefiner-yolo11s-crop640-10k-scratch/`. The unchanged split digest is `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.

## 2026-11-08 YOLO11m grouped OOF comparison

- Trained YOLO11m-seg for 30 epochs at 1024 resolution with batch size 1 on each of five grouped folds, using the AGPL-3.0 COCO checkpoint and competition annotations only. All 565 training-partition observations received out-of-fold predictions; the frozen holdout digest remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e` and was excluded from fitting and selection.
- Detector-only official PQ peaked at 0.357787 (confidence 0.20; 3,887 TP, 3,146 FP, 2,858 FN). A focused 10-setting detector/U-Net OOF screen peaked at 0.384734 (confidence 0.20, no growth, minimum claimed area 200, closing radius 3; 3,490 TP, 2,083 FP, 2,998 FN), compared with 0.356643 for the U-Net OOF input. This is below the existing YOLO11s/U-Net OOF score of 0.404464, so YOLO11m is not promoted.
- The five training folds took 10,225.62 seconds total and prediction took 372.38 seconds; peak recorded GPU allocation was 2.86 GiB on the RTX 5070 Laptop. Fusion scoring took 1,634.17 seconds. Full checkpoints, per-fold metadata, predictions, and scorer outputs are under ignored `artifacts/detector-oof/yolo11m-seg-1024-30ep-coco-oof-local/`.
- The reported 0.384734 was selected from a 10-configuration OOF screen and is therefore a comparison screen rather than a fresh unbiased estimate. The OOF manifest digest is `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c`; the holdout remains untouched.
- Historical note: CC BY-NC-SA 4.0 Solar Filament U-Net weights were excluded under the user's earlier instruction. That restriction was superseded on 2026-11-11; any use still needs appropriate attribution and ShareAlike handling, plus a valid evaluation protocol.

## 2026-11-08 YOLO D4 TTA fusion OOF result and holdout correction

- Re-inferred the five grouped YOLO11s-seg checkpoints on all 565 OOF observations and the separate 142-observation holdout using eight rotations/reflections at 1280 px. The 5-fold training boundaries, 20% holdout, and split digests were checked against the frozen manifests. Test observations and labels were not accessed.
- Raw TTA produced 76,896 OOF detections and an invalidly inflated PQ of 0.505 because duplicate masks created multiple qualifying TP pairs under the organizer's scorer. We rejected that raw result and applied descending-confidence COCO mask NMS at IoU 0.30, retaining 8,130 detections. The deduplicated detector alone reached OOF PQ 0.397327.
- Fusing the deduplicated detections with the existing equal U-Net probability blend reached OOF PQ 0.408620 (3,713 TP, 1,951 FP, 2,775 FN), exceeding the earlier 0.404464 OOF screen. The fixed configuration was foreground 0.50, detector confidence 0.35, growth 0, minimum claimed area 200, closing radius 3, and solar radius 950.
- **Correction:** The previously reported 0.409776 and 0.397526 fused holdout scores, plus their per-observation diagnostics, are retracted. The U-Net probability maps in that scoring run came from two final checkpoints refit on all 707 competition training observations, including the 142-observation holdout. They are not independent holdout predictions. The YOLO fold predictions and five-fold OOF fusion score remain valid; the latter is 0.408620 (3,713 TP, 1,951 FP, 2,775 FN).
- A separate, valid holdout check used the deduplicated YOLO fold predictions and the mean probabilities from five per-annotator U-Net folds. Each fold trained on 452 observations and excluded the entire 142-observation frozen holdout. At the fixed OOF configuration (foreground 0.50, detector 0.35, growth 0, minimum claimed area 200, closing 3, solar radius 950), it scored PQ 0.406324 (1,060 TP, 687 FP, 651 FN across 232 annotator views; mean/median observation PQ 0.399696/0.413005). This is the single-U-Net variant. A follow-up independent holdout check of the 50:50 per-annotator1024 / consensus512 five-fold probability mean scored PQ 0.411672 (1,080 TP, 703 FP, 631 FN), with mean/median observation PQ 0.402923/0.415726. Each fold trained on 452 and validated on 113 observations, excluding the full 142-observation holdout. The fixed OOF configuration was used without holdout tuning; this score applies to the cross-fit fold-mean candidate and is not a direct score for the separate all-707-observation refit. No threshold was selected using this holdout. Two representative overlays showed good alignment on several long filaments, alongside faint misses and isolated extra detections.
- OOF TTA prediction took 1,971.57 seconds; the separate detector-only holdout inference took 2,585.66 seconds on the RTX 5070 Laptop GPU. The holdout detector run peaked at 0.85 GiB allocated GPU memory. Runtime details, checkpoint and split hashes, raw and deduplicated masks, and OOF scorer output are retained under ignored `artifacts/` folders and summarized in `docs/experiment-log.md`.
- At the time of this screen, the deduplicated detector / 50:50 U-Net candidate led with OOF PQ 0.408620 and independent cross-fit holdout PQ 0.411672. It was later surpassed by the 25:75 U-Net blend recorded below. The separate full-data refit was not directly evaluated on holdout. Public leaderboard feedback remains secondary to grouped validation and the organizer's qualitative review.

## 2026-11-08 public notebook family-vote review

- Reviewed the public `prvsiyan/notebook24a4dce69e` notebook. Version 18 adds an end-to-end YOLO26x-seg output as a fifth family to the Apache-2.0 Mask R-CNN/crop-U-Net, U-Net ensemble, YOLO11-refiner, and YOLO26-refiner families. It votes across public CSV outputs, keeps masks supported by at least three families at IoU > 0.5, requires IoU >= 0.6 when exactly three families support a mask, clips overlaps, and backfills empty images. The latest source snapshot is ignored at `data/reference/public-kernels/prvsiyan-current/`.
- The notebook says version 17 scored 0.40; the version 18 source does not provide a score. It reads public model submissions rather than ground-truth labels, but depends on other notebooks' prediction files and does not reproduce their pipelines from a self-contained public repository. We have not used its test-set outputs in our candidate.
- Tested the vote idea on our own cross-fitted outputs for all 565 training-partition observations: YOLO11s TTA8 at confidence 0.35, the fixed U-Net OOF instance CSV, and YOLO11m at confidence 0.20. Using a two-of-three family vote at IoU > 0.5, the notebook's overlap clipping rule, and empty-image fallback scored OOF PQ 0.399100 (3,799 TP, 2,134 FP, 2,689 FN; 3,622 instances; 10 fallback observations). This improves slightly on YOLO11s alone (0.397327), but is below the fixed YOLO11s/U-Net fusion (0.408620), so it is not promoted.
- Re-screened the current notebook's stricter rule on the same cross-fitted predictions, requiring all three families and IoU >= 0.6 for each supporting family. It scored OOF PQ 0.374876 (3,036 TP, 1,150 FP, 3,452 FN; 2,577 output instances; 31 fallback observations), below the prior two-of-three vote and the current grouped-OOF selector candidates (0.424645-0.424860). It is not promoted; see the Nov 11 experiment log and configurable scorer.
- Reviewed Ranjeet's public LightGBM cross-model consensus method as a separate reference. Its reproducible model-only result is reported as 0.39; its historical 0.55 reference fusion depends on an undistributed third-party prediction file. The repository has no explicit license file, so its code was not copied. Our independent LightGBM regression screen on grouped OOF component features scored below the current selector and is not promoted; details are in the Nov 11 experiment log.
- The vote run used the unchanged OOF manifest digest `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c` and holdout digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. CPU scoring took 211.22 seconds on the local 32-logical-CPU Windows 11 system; predictions were precomputed with fold-held-out models. Full metrics, source/input hashes, and the vote CSV are in ignored `artifacts/model-experiments/public-notebook-oof-family-vote/`.


## 2026-11-08 public notebook method screens

- Reviewed the recently run public [blueclowd/kaggle-train](https://www.kaggle.com/code/blueclowd/kaggle-train) orchestrator. It clones its GitHub repository at the moving default branch and runs YOLO11m at 1536 px, ResNet34 crop-refiner, and ResNet34 U-Net experiments. Its source points to checkpoint assets rather than pinning hashes in the notebook; those weights were not used.
- Reviewed Fususu's public [topology-safe handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff). The method adds identity / horizontal / vertical / double-flip crop-refiner TTA, accepted only when mask area, 8-connected component count, and Dice against single-view inference pass fixed guards. The published notebook reports 0.70 for a particular public submission; we use this as a method reference and did not use its checkpoints or public test outputs.
- Implemented the guarded flip-4 idea for our scratch ResNet34 refiner and screened it on the frozen holdout using the existing detector and fixed configuration. PQ was 0.391415 versus 0.392289 without TTA; 3,195 detections accepted TTA and 53 fell back. The result did not improve validation and is not promoted. Reproduction code and outputs are in ignored `scripts/evaluate_seed_refiner_flip4_tta.py` and `artifacts/model-experiments/2026-11-08-seedrefiner-yolo11s-crop640-flip4-topology-guard/`.


## 2026-11-08 public leaderboard and submission snapshot (19:53 UTC)

- The Kaggle leaderboard showed IONode Online at 0.70, Tempest at 0.64, Sayandeep Dutta, Harshit Kulkarni, Tarunkush019, and Kush Patel at 0.62, and Amit Gunjan at 0.61. Our best public score was 0.34; the four Nov 8 fold-U-Net submissions scored 0.31 each. The public i079 handoff's historical model-card submission score was 0.70 before re-scoring; its current Kaggle notebook page shows 0.34. Public scores remain secondary to grouped validation and the host's source review.
- The five Nov 8 UTC submissions are refs `58110863` (0.34), `58115460` (0.31), `58115525` (0.31), `58115589` (0.31), and `58115626` (0.31). The CLI reported the daily limit reached; the next daily quota is expected after 2026-11-09 00:00 UTC.

## 2026-11-08 Kaggle YOLO11m 1280 px / 40 epoch grouped OOF run

- Pushed private Kaggle kernel version 1 at `mrinanksivakumar/equinexus-yolo11m-1280px-40ep-grouped-oof`. Kaggle API status was `RUNNING` at 18:30 UTC. The run uses COCO-initialized YOLO11m-seg, 1280 px, 40 epochs, batch size 2, five grouped folds, GPU acceleration, and the competition training annotations only. The notebook selects one CUDA device; actual accelerator type and memory will be recorded from completed run output.
- It uses the frozen five-fold manifest `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c` over the 565-observation training partition; all folds exclude the independent 142-observation holdout `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. The generated notebook SHA-256 is `c1244a91c42e70c2c1bca96cff2eec21d770058ad1c5884354bbf9cff7ffeab2`.
- No remote output was available while the kernel was running. We will compare its OOF result with the existing YOLO11s/U-Net candidate before deciding whether to use it. The Kaggle URL slug includes `1280px` while the local kernel metadata ID omits `px`; align the metadata before any subsequent version push.

## 2026-11-08 25:75 U-Net blend fusion update

- A full 15-setting grouped OOF screen combined the 25:75 probability blend of cross-fitted per-annotator 1024/TTA8 and consensus 512/TTA4 U-Nets with deduplicated YOLO11s-seg predictions at 1280 px / TTA8. The best official PQ is 0.415892 (3,801 TP, 1,950 FP, 2,687 FN over 922 annotator views), up from 0.408620 for the equal blend. The selected fixed postprocessing is foreground 0.50, detector threshold 0.35, minimum claimed area 200, grow 0, close 3, and solar radius 950.
- The selected configuration was checked on fold-mean predictions from five U-Net folds and five YOLO folds that all excluded the 142-observation holdout. It scored PQ 0.414534 (1,088 TP, 700 FP, 623 FN over 232 views; observation mean/median 0.404646/0.416215). A six-setting holdout sensitivity screen gave 0.414612 at detector threshold 0.40 / area 200; that 0.000079 difference does not override the OOF ranking, so 0.35 / 200 stays selected. No threshold was tuned to the holdout.
- Three representative overlays show accurate alignment on several stronger filaments, with weak/small filaments missed or fragmented and a few isolated false positives. Metrics, all 15 OOF configurations, six holdout sensitivity results, and provenance are in the ranking JSON and ignored `artifacts/model-experiments/oof-fusion-weight25-postprocess-screen/` and `artifacts/model-experiments/yolo11s-p25-top5-crossfit-holdout-screen/`.
- The source output hash for the 25:75 OOF masks is `ddb27c00be48d85830815c8b895242284fbcf7b97c4e934446daface15467829`; the OOF fusion screen hash is `888c4db915b0c76d8318c673f3b8ded5552adc10cc9eafa9b0bd9d61ef1de545`. The unchanged split and OOF manifest digests remain `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e` and `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c`.

## 2026-11-08 U-Net blend-weight sensitivity and active runs

- With the same 565-observation / 922-view OOF manifest, the per-annotator:consensus U-Net probability weights 20:80, 25:75, and 30:70 produced fused PQ 0.416173, 0.415892, and 0.414475 at the fixed 0.35 detector threshold / 200-pixel claimed-area configuration. The YOLO detector source was the COCO mask-NMS file with SHA-256 `067e2becd9abbf5592e6f8bf0754442a06780f843aa85382e88e6da745a1172b`; all three runs retain the frozen OOF digest `f856708222f5747cd35760932f535e4ae2752fd5161b9cd5981fbf4bbde26a3c` and holdout digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- The p20 cross-fit holdout check scored 0.412959 (1,085 TP, 711 FP, 626 FN; per-observation mean/median 0.403484/0.417058), compared with p25's already-recorded 0.414534 (1,088 TP, 700 FP, 623 FN; mean/median 0.404646/0.416215). The p20 OOF gain is only 0.000281 and does not carry to the independent holdout, so p25 remains the primary candidate and p20 is not promoted. No detector or postprocessing threshold was changed from the OOF selection.
- One exploratory p20 run used the alternate `dedup-union-iou-030` prediction file and was invalidated; its ignored output directory contains `INVALIDATED.txt`. The corrected p20 result uses the same mask-NMS input as the p25 ranking. Corrected p20 and p30 fusion outputs are under `artifacts/model-experiments/oof-fusion-perann20-consensus80-weight-screen-corrected-20261108/` and `artifacts/model-experiments/oof-fusion-perann30-consensus70-weight-screen-20261108/`.
- The laptop YOLO11s-seg 1280 / 40-epoch all-training-groups refit later completed under `artifacts/detector-experiments/yolo11s-seg-1280-final-fulltrain-r1-20261108/`. Its metadata confirms 707 observations / 1,154 annotator views, competition annotations only, and `holdout_used_for_final_training=true`; it has no independent validation score. Kaggle YOLO11m 1280 / 40-epoch grouped OOF was still `RUNNING` at the preceding 20:32 UTC poll.
- Public reference review: `prvsiyan/a-thousand-quiet-threads` is a from-scratch U-Net run that reserves 192 observations, trains on 412 groups, and scores 103 development groups, but reports no model metric; it is reference-only. The public HeShen-1 repository reports 0.3593 five-fold mean PQ and 0.35 leaderboard for a YOLO11m / crop-refiner pipeline, and says its wider instance-voting fusion did not beat its strongest member. These results are contextual rather than directly comparable; no checkpoints or code from that repository were used. Sources: [Kaggle notebook](https://www.kaggle.com/code/prvsiyan/a-thousand-quiet-threads), [HeShen-1 reproduction repository](https://github.com/HeShen-1/filament-segmentation-2026).

- Verified the five ready p25 fallback CSVs: each has all 180 unique supplied test images, unique IDs mapped to those image stems, and RLEs that decode to nonempty 2048x2048 masks. Their OOF PQs range from 0.414754 to 0.415892. The earlier six-setting fold-mean holdout sensitivity screen includes all five settings (PQ 0.414534, 0.411943, 0.414612, 0.411634, and 0.412167 in candidate order); the 0.40/200 setting's 0.000079 edge over the OOF rank-1 setting is selection-aware, so OOF remains the primary ranking. These are under ignored `artifacts/submissions/`.

- Refreshed public evidence at 2026-11-08 21:00 UTC: the Kaggle CLI lists IONode Online at 0.70, Tempest at 0.64, then several 0.62 entries; our submissions remain at 0.31 (U-Net) and 0.34 (I079 inference). The public `prvsiyan/the-shape-of-an-honest-signal` notebook reports no competition model score and cites a controlled leaderboard report where empty output scores 0.93, versus 0.72 at 1.9 predictions/image and lower scores as more instances are emitted. That report conflicts with the official scorer copy, which counts missed GT as FN and gives empty output zero on positive GT. Public scores are therefore diagnostic only; candidate selection stays grounded in the fixed grouped scorer and task quality. Notebook source is archived under ignored `artifacts/kaggle/references/`; source thread: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809.

- Reviewed the public Ektarr YOLO11s + ResNet18 crop-refiner notebook. Its filename-grouped split selects only the most-complete annotator view, and it reports crop Dice/visuals rather than frozen-split instance PQ; we retain the architecture idea only as a reference. A brief CPU smoke of an optional ImageNet ResNet34 encoder was stopped before any competition samples, training, holdout scoring, or submission. The user has since confirmed that public non-commercial research/education resources are eligible; this pretrained encoder is not excluded by license. Scratch refiner tests pass (4/4). References: https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation, https://huggingface.co/smp-hub/resnet34.imagenet, https://image-net.org/accessagreement.

- Reviewed four fixed-holdout p25 cross-fit overlays. Stronger filament arcs mostly align, while faint branches are missed and some dense images have short false-positive fragments. These are qualitative diagnostics only; no thresholds or settings were retuned on the holdout.

## 2026-11-08 23:10 UTC current sprint status

- The local all-707 YOLO11s 1280/40-epoch refit and D4 test inference are complete. The final-fit candidate covers all 180 supplied test images, but has no direct holdout score because it trained on all observations.
- The next five-submission batch (ranks 1–5) and the following quota-window batch (ranks 6–10) are built, checksum-verified, and schema/RLE-validated. The first candidate is OOF rank 1 at 0.415892; the earlier six-setting fold-mean holdout sensitivity screen also scored the rank-3 threshold configuration 0.414612 versus 0.414534 for rank 1. Full manifests are under ignored `artifacts/submissions/`.
- Kaggle OAuth is working again. The private YOLO11m 1280/40-epoch grouped OOF kernel is still `RUNNING`; no new output is available. The submissions API confirms the five November 8 entries remain complete with public scores 0.31–0.34. The corrected leaderboard CLI command fetched a 1,440-team snapshot at 23:11 UTC: EquiNexus rank 808 at 0.34; IONode Online leads at 0.70. The initial failure came from using outdated CLI arguments.
- The exact `prvsiyan/notebook24a4dce69e` source and its separate Kaggle output bundle are archived under ignored `artifacts/kaggle/references/`. Its 3-of-4 vote produced 1,106 rows from other teams' precomputed test CSVs, but reports no ground-truth score or labeled validation. Because the output depends on those third-party predictions and the host's test-input boundary is unresolved for them, it is not used in our submissions.
- Audited the public `lamhuy8904/solar-filament-unet-segmentation-0-55` notebook and its Kaggle output. Its final CSV exactly reuses 1,342 inline RLE predictions for 178 test IDs and adds fixed square masks for the other two; the five-epoch U-Net logs validation Dice 0.0000 and is unused. The author is listed at 0.55, but the source of the static predictions cannot be established. We exclude it as a non-reproducible provenance/compliance risk. Detailed evidence is in `docs/experiment-log.md` and the ignored reference archive.
- Pulled a current I079 topology/TTA Kaggle notebook and its output. It did no training or label access, ran in 343 seconds on a T4, and produced the exact same 1,299-row submission MD5 as the upstream model; its author's 0.55 board entry is not linked to that file. Our separate flip-4 screen also failed to improve validation, so it is not a new candidate.
- The local suite passed 40 tests (12 existing pycocotools/NumPy deprecation warnings), and `git diff --check` passed.

## 2026-11-08 23:45 UTC progress update

- Kaggle still reports the YOLO11m 1280/40-epoch grouped OOF kernel as `RUNNING`; no output is available yet. The submission API reports 5/5 used today and 0 remaining. The validated rank 1–5 p25 batch is staged locally for the next quota window; its manifest and checksums are under ignored `artifacts/submissions/`.
- The Kaggle power sensor confirms the laptop is on AC (`PowerOnline=True`, `Discharging=False`); battery charge is 77% and charging is paused. Windows remains on the Balanced power plan.
- Audited the public 0.70 I079 handoff and its Kaggle Model metadata. The model is Apache 2.0 and claims training on competition data only, but its weights have seen all training observations and therefore provide no eligible grouped validation evidence. Its expected output MD5 is identical to EquiNexus submission 58110863, which scored 0.34 in the current leaderboard response, conflicting with the notebook/model claim of 0.70. We retain the result as an unresolved leaderboard discrepancy, not a candidate promotion. Details and references are in `docs/experiment-log.md`.

## 2026-11-09 00:01 UTC submission update

- Five p25 submissions were accepted by Kaggle after the quota reset: refs 58143998, 58144073, 58144089, 58144104, and 58144124. Their first status poll showed `PENDING`; no public scores are available yet. The CLI now reports 5/5 daily quota used and 15 lifetime submissions.
- The YOLO11m OOF kernel is still `RUNNING`. The local RTX 5070 is idle; AC is confirmed online, but Windows remains on Balanced. The same-CSV-hash score discrepancy for the public I079 0.70 handoff is recorded in the experiment log and does not alter validation ranking.

## 2026-11-09 00:06 UTC leaderboard update

- Four of five p25 submissions are scored: refs 58143998 and 58144089 received 0.36; refs 58144073 and 58144124 received 0.35. Ref 58144104 is still pending. The current EquiNexus best is 0.36.
- The full board snapshot contains 1,440 teams; EquiNexus ranks 666 at 0.36. The visible top scores are IONode Online 0.70, Tempest 0.64, then several teams at 0.62. The archive is under ignored `artifacts/kaggle/leaderboard/2026-11-09-0005-utc/`.

## 2026-11-09 00:11 UTC completed-score update

- All five p25 submissions are complete with public scores 0.36, 0.35, 0.36, 0.36, and 0.35 in OOF order. EquiNexus' best public score is 0.36; the higher OOF-ranked first candidate and candidates 3–4 tie at that score.
- The refreshed 1,440-team leaderboard lists EquiNexus at rank 651/1,440 with 15 evaluated submissions. IONode Online remains at 0.70, Tempest at 0.64, and four teams follow at 0.62. Snapshot: ignored `artifacts/kaggle/leaderboard/2026-11-09-0010-utc/`.

## 2026-11-09 00:18 UTC local GPU run

- Started the YOLO11m-seg 1280px/40-epoch frozen-holdout experiment on the laptop RTX 5070, using batch size 1 to fit its 8 GiB VRAM and the same COCO-pretrained weights and core settings as the Kaggle OOF run. The frozen split SHA remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; 565 observations train and 142 holdout observations / 232 annotator views validate. The first epoch has completed without an out-of-memory error; official holdout scoring will follow training and inference.
- The laptop GPU reports 6.0 GiB use at 91% utilization, 75 C, and 102 W while training. Kaggle's separate five-fold YOLO11m OOF kernel remains `RUNNING`.

## 2026-11-09 00:27 UTC local memory adjustment

- Stopped the 1280px laptop experiment after 3 complete epochs and epoch 4 at step 884/922. WDDM showed 7,733/8,151 MiB VRAM use, Ultralytics estimated 13.6 GiB, the Python working set exceeded 10 GiB, and free host memory fell to 1.3 GiB as throughput dropped sharply. It has no official scorer result and is excluded. Resources returned to idle after stopping it.
- Restarted the same frozen grouped holdout with YOLO11m at 1024px, 40 epochs, batch 1 under a new ignored run directory. Kaggle continues the matching 1280px five-fold OOF run separately.

## 2026-11-09 02:10 UTC model screen update

- SWEFil remains an eligible external source under Rule 6: use only the audited 229 date-disjoint captures / 1,504 filament masks, with attribution and CC-BY-4.0 provenance under `data/external/SWEFil/`. The verified source revision is `bc818c69a79266ad0fa91a539359485af57967cf`; the source manifest accompanies the canonical local data folder.
- The Kaggle YOLO11s SWEFil warm-start screen scored 0.353778 on the frozen holdout; local COCO-pretrained YOLO11m scored 0.354714. Fusion with p25 U-Net probabilities reached 0.403563. All remain below the current 0.405860 holdout screen / 0.415892 grouped-OOF candidate and are not promoted.
- The laptop is idle after the holdout run. AC was verified online during the later part of training; charge remained near 77% and Windows stayed on Balanced. The Kaggle YOLO11m five-fold OOF kernel still reports `RUNNING`; check its status before the next training or submission decision.

- Refreshed the public leaderboard at 2026-11-09 02:15 UTC: EquiNexus rank 652/1,440 at 0.36; leading teams remain at 0.70 and 0.64. Snapshot and SHA-256 are recorded in `docs/experiment-log.md`.

## 2026-11-09 02:47 UTC YOLO11m D4 holdout screen

- Eight-view D4 inference on the local 1024px YOLO11m checkpoint produced 19,016 masks in 503 seconds. Applying the established confidence-ordered mask NMS at IoU 0.30 retained 2,019 detections and removed 16,997 repeats. The fixed confidence 0.30 / area 400 detector score was 0.348379, below the single-view 0.354714 result.
- The fixed 25:75 U-Net fusion configuration scored 0.407802 on this holdout, below the 0.415892 grouped-OOF candidate. The holdout-selected local checkpoint makes this a screening result only; it is not promoted. Raw undeduplicated TTA scoring was invalidated and marked in its artifact folder.
- The source and full scorer details are in `docs/experiment-log.md`; all D4 predictions and reports remain under ignored `artifacts/`. Kaggle's five-fold YOLO11m grouped-OOF kernel remains `RUNNING`; the laptop is idle on AC.

## 2026-11-09 06:30 UTC sprint status

- Kaggle's YOLO11m 1280px grouped-OOF kernel completed. Its best detector-only OOF PQ was 0.400640; best p25 U-Net fusion was 0.397230, and D4/TTA8 fusion was 0.408422. YOLO11s p25 remains the stronger validated family at 0.415892 OOF.
- The aligned p25 holdout rerun reproduced PQ 0.414534. p22.5 scored 0.413165. Confidence 0.37 scored 0.414659 versus 0.414534 at 0.35, while its OOF gain is only 0.000019; paired observation-bootstrap uncertainty spans zero, so no new variant is promoted.
- Two earlier holdout scores were invalidated because they used mismatched per-annotator caches. Their artifact folders now contain `INVALIDATED.md` markers. The frozen split and OOF digests remain unchanged.
- The user reports Kaggle GPU quota is exhausted until after the sprint cutoff; the latest OAuth approval timed out while they were away. No new Kaggle run or submission is pending, and no further authorization will be requested while they are unavailable. Local RTX 5070 AC power was confirmed.

## 2026-11-09 12:45 UTC local seed-100 result

- Completed a five-fold scratch per-annotator U-Net OOF run with a new training seed and the unchanged frozen split. The p25 YOLO fusion OOF PQ was 0.416005 versus 0.415892 for the incumbent; the aligned holdout was 0.413370 versus 0.414534. Paired observation bootstrap uncertainty crossed zero, so the new seed was not promoted.
- The local grouped OOF runner now accepts `--training-seed` without changing the seed-42 split or fold manifest. All new checkpoints and predictions remain ignored under `artifacts/`; no submission or all-data refit was created.
- Kaggle GPU quota is unavailable before the cutoff and OAuth timed out while the user was away. No additional Kaggle work is pending.

## 2026-11-09 13:16 UTC cross-seed ensemble screen

- The 50:50 seed-42/seed-100 per-annotator model average improved standalone OOF PQ to 0.357821 from 0.356643 for seed 42. With consensus maps at the established 25:75 weight and the incumbent YOLO fusion setting, OOF PQ was 0.415972 versus 0.415892.
- The paired observation bootstrap interval for the fused OOF change was [-0.000756, 0.000932]; the 0.000080 gain is inconclusive. Keep the current p25 candidate and do not spend holdout budget on this branch.
- Kaggle quota resets in 113h41m according to the user, which is after the sprint cutoff. The local laptop remains the available runtime; no OAuth prompts or Kaggle jobs were started.

## 2026-11-09 14:25 UTC component selector and submission candidate

- New leading scorer-aligned candidate: a logistic keep/drop selector trained only on grouped OOF components and competition annotations. At its OOF-selected threshold 0.40, it scores 0.423498 OOF and 0.430531 on the fixed holdout, versus 0.415892 and 0.414534 for the incumbent. The paired holdout improvement is +0.015997 with 95% observation-bootstrap interval [0.010623, 0.021313]. It reduces holdout FP from 700 to 511 while increasing FN from 623 to 656.
- The selector's component deletion targets, holdout aggregates, and organizer-scorer samples were audited. A byte-to-text RLE serialization bug in its first saved prediction files was fixed in the shared writer; the corrected run reproduces the in-memory metrics and all emitted RLE CSVs round-trip correctly.
- Rebuilt the earlier three-family OOF vote CSV through the corrected writer; its PQ remains 0.399100, so it stays below the promoted selector candidate. Only the explicitly invalidated first selector run retains malformed RLE CSVs.
- Paired overlays are saved for the largest holdout gain, median change, and largest loss. One image shows the selector deleting the only correct match; keep the OOF-selected 0.40 cutoff fixed and treat this over-pruning case as a model limitation.
- Full-fit test inference is ready at `artifacts/submissions/component-selector-fullfit-test-20261109/`: 1,022 masks across all 180 supplied test images, compared with 1,124 from the base candidate. It is not submitted and has no public score yet. Kaggle GPU quota resets after this sprint, and the user is unavailable for sign-in while asleep.
- A grouped-OOF-only screen of three logistic regularization strengths and two tree models found no meaningful gain over C=1: the best alternative reached 0.423515 versus 0.423498. Keep C=1 / 0.40; no holdout tuning was done.
- Meta-selector work ran CPU-only on Windows 11 / Python 3.12.10 with the RTX 5070 idle; package versions, source hashes, and per-run times are recorded in the ignored metrics artifacts.

## 2026-11-09 15:30 UTC fallback candidate and refreshed Kaggle status

- A separately evaluated empty-output fallback raises OOF PQ from 0.423498 to 0.423636 and fixed-holdout PQ from 0.430531 to 0.430617. Both point estimates favor it, but paired observation-bootstrap intervals include zero and the holdout gain is only 0.000086. Keep the standard selector as the primary candidate and preserve the fallback build as a backup; do not describe it as a robust gain.
- The backup full-fit test CSV is validated and contains 1,023 instances across 180 supplied images; it restores one component where the selector would otherwise leave an image empty. The output is ignored under `artifacts/submissions/component-selector-fullfit-test-20261109-empty-fallback/` and has not been submitted.
- Kaggle CLI authentication is valid. All five submissions for November 9 are complete, so today's submission allowance is used. The full leaderboard now has 1,456 teams; EquiNexus is rank 663 at 0.36. The public top remains 0.70, with a new 0.62 submission from Sayandeep Dutta at 14:15 UTC. Snapshot and input SHA are recorded in `docs/experiment-log.md`.
- The revised sprint goal now continues through Friday, November 13 at 8:00 PM Toronto time. The Kaggle session is valid and daily submission capacity resets separately, but the GPU quota reset estimate remains after Friday's cutoff; use the local laptop for training before then. No OAuth request is needed.

## 2026-11-09 15:43 UTC external-method and selector screen

- Audited the recent public vote notebook and the V14 low-confidence YOLO sweep. The vote relies on cached test outputs from external model families and includes a CC BY-NC-SA-derived model output (now eligible under the user's policy) plus a checkpoint of unverified provenance. We still exclude those cached predictions because the ensemble has no comparable cross-validation and one source model's provenance remains unknown. The V14 idea of caching detections at a low floor before threshold sweeps is eligible for an independent grouped-OOF experiment, but its own local 15% unstratified split/scorer is not comparable. The author also published a scratch U-Net using annotator-union targets without an official PQ result; our existing consensus branch already covers that target idea.
- Regressed continuous component PQ contribution across the five OOF folds and screened ridge, histogram boosting, and extra trees. Extra trees scored 0.421691 OOF and 0.432926 holdout. Since it trails the current classifier by 0.001807 OOF and the holdout paired interval includes zero, keep the classifier as the primary model.
- Full details, source hashes, and ignored reference notebook copies are in `docs/experiment-log.md`. The current selector and empty-fallback test CSVs remain the only validation-supported full-fit submission candidates; today's five slots are complete.

## 2026-11-09 16:00 UTC low-confidence OOF screen

- Tested detector confidence 0.25–0.50 with the existing 0.05-floor YOLO11s TTA8 OOF cache and a fixed U-Net-mask fusion setup under the official scorer. The best was 0.35 at PQ 0.404799, below the 0.415892 incumbent probability-map fusion. Higher thresholds reduce false positives but lose too many true positives; no holdout was used.
- This confirms the public low-confidence sweep is a useful screening idea, but the alternate U-Net-mask fusion is weaker than our current probability-map fusion. The six-setting output and hash are recorded in `docs/experiment-log.md`.

## 2026-11-09 16:16 UTC next submission batch

- Prepared five complete test CSV variants for the next daily submission window: the standard selector (C=1, 0.40), the separately validated empty fallback, C=1 thresholds 0.35 and 0.50, and the grouped-OOF-screened C=4 / 0.45 variant. Counts are 1,022, 1,023, 1,045, 965, and 991 instances respectively. All have complete 180-image coverage and passed full-resolution submission validation.
- OOF PQ values are 0.423498, 0.423636, 0.421799, 0.423053, and 0.423515. Only the standard selector and fallback have fixed-holdout scores; keep the standard selector as primary because the fallback point gain is tiny and the C=4 OOF gain over C=1 is only 0.000017. Do not interpret the test-only CSVs as scored.
- Today's five Kaggle submissions are complete, so this batch is queued locally. Its run metadata is ignored under `artifacts/submissions/`; no data or predictions are tracked.

## 2026-11-09 18:40 UTC LightGBM selector screen

- Cross-fitted three regularized LightGBM component selectors across the five frozen observation-grouped folds, then selected one blend and threshold using OOF predictions only. The selected 50% LightGBM / 50% incumbent probability blend at threshold 0.425 scored 0.425026 OOF, a small gain over the RF blend's 0.424791.
- A single evaluation on the frozen holdout scored 0.431953 versus 0.434930 for the RF incumbent. The paired observation-bootstrap 95% interval for LightGBM minus RF was [-0.006400, 0.000249], with 3.64% positive resamples. Retain as a documented negative result; do not promote or submit.
- Full inputs, hashes, per-observation metrics, and overlays remain ignored under `artifacts/model-experiments/component-selector-lightgbm-20261109-reviewed/`. The CPU run used LightGBM 4.7.0, pinned as an optional experiment dependency in `requirements-experiments.txt`.

## 2026-11-09 19:49 UTC local YOLO11s validation

- Completed a 25-epoch, 1280-pixel YOLO11s-seg run on the local RTX 5070 Laptop GPU. It trained on 565 training-partition observations / 922 annotator views and evaluated on the frozen 142-observation holdout / 232 views; the split digest remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. Best mask mAP50 was 0.5961 at epoch 25.
- Applied the single confidence threshold 0.35 selected in the earlier grouped OOF YOLO11s screen, with minimum area 0. The official scorer returned PQ 0.368679 (838 TP, 371 FP, 909 FN over 1,711 GT instances). This is well below the current selector blend at 0.434930; do not promote or submit it.
- Ultralytics selected the checkpoint by mask mAP on this repeatedly used holdout, so treat its PQ as exploratory fixed-split evidence rather than an untouched generalization estimate. The run used the Ultralytics COCO YOLO11s-seg weights (AGPL-3.0, SHA-256 `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3`), PyTorch 2.15.0+cu132, CUDA 13.2, and Ultralytics 8.4.173; training took 3,274 s, best/last holdout inference 104 s, and Torch reported 3.26 GB peak allocated GPU memory. Checkpoint, predictions, scorer output, and complete metadata are ignored under `artifacts/detector-experiments/yolo11s-seg-1280-25ep-holdout-20261109/`.

## 2026-11-09 20:04 UTC XGBoost selector screen

- Cross-fitted three regularized CPU XGBoost configurations across the same five observation-grouped folds, then selected one blend and threshold using OOF predictions only. XGBoost 3.4.1 at 50% weight with the existing logistic/ExtraTrees core scored 0.424315 OOF, below the RF blend's 0.424791.
- The single OOF-selected candidate scored 0.433302 on the fixed holdout versus 0.433977 for the core blend and 0.434930 for the RF incumbent. The paired holdout delta against the RF blend was -0.001628 with 95% observation-bootstrap interval [-0.004282, 0.000943]. Do not promote or submit it.
- The largest-gain overlay adds one small match; the largest-loss overlay drops a short matched trace. Inputs, hashes, per-observation metrics, and overlays are under ignored `artifacts/model-experiments/component-selector-xgboost-20261109/`. The reusable screen is `scripts/screen_component_selector_xgboost.py`, and XGBoost is pinned as an optional CPU experiment dependency.

## 2026-11-09 20:12 UTC leaderboard refresh

- Refreshed the complete Kaggle leaderboard: 1,460 teams, EquiNexus rank 663 at 0.36. IONode Online remains at 0.70, Tempest at 0.64, and four teams are at 0.62. The public score ordering does not change validation decisions.
- All 15 EquiNexus submissions are scored. The five hash-locked component-selector CSVs for Nov 10 remain ready for the scheduled submission window; no new candidate was promoted by the YOLO or XGBoost screens.

## 2026-11-09 20:27 UTC site/year context screen

- Added one-hot capture site and year to the component selector's existing numeric features, then cross-fitted C=0.25/1/4 logistic models across the same five observation-grouped folds. OOF-only blend selection assigned the context model 0% weight; the best score remained the core selector at 0.424245, below the RF blend at 0.424791.
- Because no context model improved OOF, the fixed holdout was not scored. No submission candidate changed. Full probabilities and input hashes are under ignored `artifacts/model-experiments/component-selector-context-20261109/`; reusable code is `scripts/screen_component_selector_context.py`.

## 2026-11-09 20:37 UTC morphology-aware selector screen

- Added ten component features from skeleton geometry, estimated width and branching, contour perimeter, interior intensity quantiles, and local ring contrast. A random forest was cross-fitted over the same five frozen observation groups, then blended with the incumbent selector using OOF predictions only.
- The best morphology blend scored 0.424284 OOF, below the existing RF blend at 0.424791. The fixed holdout was not evaluated and no submission candidate changed.
- The run took 75.19 seconds on CPU across 3,533 components. A guarded rerun reproduced the enriched features and candidate sweep byte-for-byte in 74.83 seconds. Inputs, features, probabilities, scores, and hashes are under ignored `artifacts/model-experiments/component-selector-morphology-20261109/` and `...-recheck/`; reusable code is `scripts/screen_component_selector_morphology.py` (SHA-256 `0805dda59e7c014241087c4a7545c526975a004bd07c1d66ef1c6ad3a6b2519a`).

## 2026-11-09 20:45 UTC public MAGFiLO V11/V12 audit

- Pulled the public V11 inference notebook and V12 crop-fine-tuning notebook under ignored `artifacts/kaggle/reference-audit/`. V11 reports a local PQ near 0.4347, but its model selection comes from a 15% random photo split with repeated PQ evaluation during crop-model fine-tuning. This is selection-aware and differs from our frozen five-fold grouped OOF evidence.
- The inference notebook's detector/crop checkpoint paths and versions are inconsistent across cells, and Kaggle returned HTTP 403 when asked for metadata on its referenced checkpoint datasets. Their license and full training provenance therefore remain unverified. We did not download or use either checkpoint, and the notebook's reported score was not compared as an equivalent validation result.
- Its confidence-first detection and crop-refinement pipeline is already represented by our YOLO plus crop-refiner experiments. No new experiment or submission candidate was justified by this audit.

## 2026-11-09 20:48 UTC leaderboard refresh

- Refreshed the complete leaderboard: 1,460 teams; EquiNexus remains rank 663 at 0.36, with 15 submissions. The public leaders remain at 0.70 and 0.64; four teams are at 0.62. Recent activity includes 0.56 and 0.55 scores, but no new high score that changes our validation decisions.
- The full snapshot and SHA-256 are in `artifacts/kaggle/leaderboard/202611092055-utc/`; the five prepared Nov 10 CSVs remain unchanged.

## 2026-11-09 20:53 UTC observation-level count calibration

- Cross-fitted ridge, random-forest, and histogram-gradient count regressors across the five frozen observation folds. Features summarized the OOF component probabilities and morphology; targets were mean annotated filament counts across each observation's competition-provided annotator views.
- The best top-K candidate scored 0.417146 OOF, below the global RF blend at 0.424791. It lost in all five folds, by 0.0047 to 0.0140 PQ, so the holdout was not scored and no candidate changed.
- The 25.08-second CPU run, count features, cross-fitted counts, candidate sweep, and input hashes are under ignored `artifacts/model-experiments/component-selector-count-calibration-20261109/`. Reusable code is `scripts/screen_component_selector_count_calibration.py` (SHA-256 `05769d4afa2eab6d4765d3475a295fbee4272cbf620eb11f9f3237b5db1ce334`).

## 2026-11-09 21:03 UTC marginal-utility weighted selector

- Weighted component-classifier samples by the absolute marginal PQ effect of keeping or deleting that component, normalized so each observation retained equal total weight. Cross-fitted logistic and random-forest variants across the five grouped folds, screening four utility exponents, blends, and thresholds.
- The best OOF setting gained only 0.000044 over the incumbent RF blend. The 20,000-resample paired observation bootstrap interval was [-0.000869, 0.000946], with 54.22% positive draws; the holdout was not scored and no submission candidate changed.
- The CPU run took 63.23 seconds. Fold comparisons, OOF probabilities, candidate sweep, bootstrap output, and hashes are under ignored `artifacts/model-experiments/component-selector-utility-weighted-20261109/`. Reusable code is `scripts/screen_component_selector_utility_weighted.py` (SHA-256 `6f5b7aac7f420708c4350a718a02f06151d7f1ee32c91aa11db0b4b027089769`).

## 2026-11-09 21:14 UTC pairwise component ranking screen

- Cross-fitted three XGBoost `rank:pairwise` configurations over within-observation ranks of component marginal PQ contribution, using the same five grouped OOF folds and frozen split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- The best candidate blended the existing RF selector with the depth-3 ranker at 30% ranker weight and threshold 0.40, reaching 0.424907 OOF PQ versus 0.424791 for the RF incumbent. Four folds improved and one fell by 0.001930. A 20,000-resample paired observation bootstrap estimated +0.000116, with 95% interval [-0.000647, 0.000929] and 61.18% positive draws; evidence is inconclusive, so the holdout was not scored and no candidate changed.
- The CPU run took 37.52 seconds with XGBoost 3.4.1; inputs, OOF probabilities, fold scores, candidate sweep, bootstrap and metrics are under ignored `artifacts/model-experiments/component-selector-pairwise-ranker-20261109-full565/`. Reusable code is `scripts/screen_component_selector_pairwise_ranker.py`.

## 2026-11-09 21:27 UTC rank-consensus selector ensemble

- Scored 96 combinations of per-observation rank consensus, raw probability means, and blends across seven frozen grouped-OOF selector probability sources. The best candidate was the equal probability mean at threshold 0.45, with 0.424435 PQ versus 0.424791 for the RF incumbent.
- The paired 20,000-resample observation bootstrap delta was -0.000354, with 95% interval [-0.001644, 0.000951] and 29.13% positive draws. Rank-only candidates scored at most 0.393263. The fixed holdout was not scored and no submission candidate changed.
- Full cross-fitted probability inputs, candidate sweep, hashes, and metrics are under ignored `artifacts/model-experiments/component-selector-rank-consensus-20261109/`. Reusable code is `scripts/screen_component_selector_rank_consensus.py`.

## 2026-11-09 21:42 UTC CatBoost pairwise ranker screen

- Cross-fitted three CatBoost 1.2.10 `PairLogitPairwise` rankers on the five frozen grouped OOF folds, using competition-provided annotation-derived relevance labels. The best 30% ranker blend scored 0.424590 OOF PQ, below the RF incumbent at 0.424791.
- Its 20,000-resample paired observation bootstrap delta was -0.000201, with 95% interval [-0.001163, 0.000774] and 34.75% positive draws. It lost in four folds and gained in one. The fixed holdout was not scored, and no submission candidate changed.
- The 293.35-second CPU run, cross-fitted rank scores, candidate sweep, input hashes, and metrics are under ignored `artifacts/model-experiments/component-selector-catboost-ranker-20261109/`. Reusable code is `scripts/screen_component_selector_catboost_ranker.py`; CatBoost 1.2.10 is pinned in `requirements-experiments.txt`.

## 2026-11-09 21:48 UTC public report metric audit

- Reviewed the v0.3 (July 20) public report from FELIPEACASTRO/solar-filament-2026, which claims a hidden `instance_wise_dice` metric and a 0.71 public score.
- Kaggle's official overview records an Aug 7 evaluation update replacing the original mean-Dice wording with Panoptic Quality. The report's metric claim and score predate that update, so neither is comparable to current scorer-aligned validation. We adopted no code, checkpoint, or public test labels; the frozen split and scorer remain unchanged.
- Sources: [historical report](https://github.com/FELIPEACASTRO/solar-filament-2026/blob/master/report/TECHNICAL_REPORT.md) and [official evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

## 2026-11-09 22:05 UTC component-selector seed-ensemble screens

- Averaging five cross-fitted random-forest seeds scored 0.424763 OOF versus 0.424791 for the current blend. The paired observation-bootstrap interval was [-0.000218, 0.000171], so the ensemble was not promoted.
- Averaging five cross-fitted ExtraTrees seeds scored 0.424872 OOF, a +0.000081 point difference; its 95% paired interval [-0.000102, 0.000352] includes zero. The fixed holdout was not used, and neither ensemble changed the prepared five-candidate submission batch.
- Both screens covered all 565 OOF observations under the frozen grouped split, used only competition training annotations, and ran CPU-only. Source scripts and detailed metrics are recorded in `docs/experiment-log.md` and ignored `artifacts/model-experiments/component-selector-*-seed-ensemble-20261109/` folders.

## 2026-11-09 22:42 UTC cross-fold medoid association review

- Adapted the public V23 cross-fold association and medoid idea using EquiNexus predictions only. On the frozen holdout it scored 0.369707 PQ versus 0.434930 for the incumbent; the paired 95% interval for the delta was [-0.077360, -0.052939]. Recall improved, but false positives nearly tripled, so the approach was rejected without test inference or submission.
- The fixed split and scorer are unchanged. Full metrics, input hashes, per-observation diagnostics, and the reproduction script are recorded in `docs/experiment-log.md` and ignored `artifacts/model-experiments/crossfold-medoid-unet-fusion-20261109/`.

## 2026-11-09 22:57 UTC seed-100 selector-feature screen

- Replaced the per-annotator probability-map features with the 50:50 seed-42/seed-100 map ensemble while retaining the existing 75% consensus-map contribution and fixed base predictions. On grouped OOF, the best selector blend scored 0.423946 versus 0.424791 for the incumbent. The 142-observation holdout and test images were not used; this candidate is rejected.
- The feature builder, four screens, split digest, input hashes and metrics are recorded in `docs/experiment-log.md` and ignored `artifacts/model-experiments/component-selector-seed100-*20261109/`.

## 2026-11-09 23:22 UTC high-score public model audit

- The live 1,460-team leaderboard places EquiNexus at rank 664 / 0.36 and IONode Online at rank 1 / 0.70. Because the organizer has warned that the leaderboard is preliminary and public MAGFiLO includes test labels, use this only as a signal to inspect methods; do not infer that a high score is compliant or well validated.
- Inspected [Fususu's public 0.70 inference handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) and its [Kaggle model card](https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights). The model is listed as Apache 2.0; its publisher claims training used only the competition-provided training images and annotations and excluded public test-overlap labels. Under the user's policy, its stated non-commercial/academic-use restriction is not by itself a reason to exclude it. The full-fit weights cannot validate observations included in their training set; test-only inference was reproduced and compared with our prior submission, with no improvement evidence, so it is not promoted. The publisher's 0.70 reports remain separate claims.
- Its most promising design is instance detection followed by native-crop refinement: Mask R-CNN on 1024-square grayscale input, grouping positive-overlap detections, a ResNet18 crop U-Net on 256-square crops, then flip-four TTA accepted only if area, component-count and overlap guards preserve topology. A conservative detector-seed split runs only behind crowding gates before merging overlaps and emitting compressed COCO RLE.
- The [credited source training notebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner) is also Apache 2.0. Its training code uses torchvision's COCO-pretrained Mask R-CNN and ImageNet-pretrained ResNet18. Before adoption, reproduce training using observation-grouped folds, inspect its existing split protocol and dependencies, and evaluate every checkpoint with our unchanged scorer. The final released weights were not downloaded or used.
- Both source notebooks and the full leaderboard/model metadata audit are retained under ignored `data/reference/upstream/kaggle-kernels/` and `artifacts/kaggle/reference-audit/2026-11-09-2320-utc/`. The five hash-locked candidates for the next window remain unchanged. The split digest remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- The November 10 batch remains scheduled for 20:20 Toronto time after a single-docs-only commit/push checkpoint at 20:05; the task updates the ignored manifest's source commit and runs its dry-run before pushing. The submission wrapper verifies the pushed remote head and clean worktree before uploads. The leaderboard/status refresh is scheduled at 21:00.
- The local 1536-pixel YOLO11m-seg five-fold OOF run is active on fold 1 with batch size 1 and early stopping. Ultralytics epoch metrics are only training diagnostics; no scorer result or candidate decision is available yet. Artifacts remain ignored under `artifacts/detector-oof/yolo11m-seg-1536-30ep-coco-oof-20261109/`.

## 2026-11-10 00:28 UTC local YOLO OOF progress

- The first grouped fold completed epoch 5 at cumulative training time 4,573.6 seconds; mask mAP50 was 0.35911 and mask mAP50-95 was 0.10141. Epoch 6 is active. These are detector diagnostics only; scorer-aligned OOF results and any candidate decision remain pending. Training uses batch size 1, a 30-epoch cap, and patience 10; outputs and checkpoints remain ignored under `artifacts/detector-oof/yolo11m-seg-1536-30ep-coco-oof-20261109/`.

## 2026-11-10 01:12 UTC submission and training status

- The November 10 UTC batch completed with five public scores of 0.37; EquiNexus moved to rank 455 of 1,460 teams with 20 scored submissions. The public leaderboard is secondary evidence; validation decisions continue to use the frozen grouped OOF split and official scorer.
- Five hash-locked RF blend variants for November 11 UTC passed submission preflight. Their weights, thresholds, OOF scores, manifest, and paths are recorded in `docs/experiment-log.md`; holdout PQ is intentionally absent. One-time tasks are set for a commit/push and manifest update at 20:05 Toronto time, upload at 20:20, and leaderboard refresh at 21:00 on November 10.
- The team roster includes all three expected Kaggle accounts: `arvbali`, `lolmrdabbydab`, and `mrinanksivakumar`.
- The local YOLO11m-seg run remains active at epoch 6 after epoch 5 reached mask mAP50 0.35911 and mAP50-95 0.10141. GPU usage is near full capacity on the 8 GB RTX 5070; AC sleep is disabled, so the run can continue while unattended. No scorer result is available yet.

## 2026-11-10 01:24 UTC YOLO OOF progress

- The first grouped fold completed epoch 6 at 8,327.09 seconds cumulative training time. Mask mAP50 reached 0.40753 and mask mAP50-95 0.11446; epoch 7 is active. These are detector diagnostics, not challenge PQ, and the fixed holdout remains excluded.
- The 1536-pixel, batch-size-1 run is using about 7.8 GiB of the RTX 5070's 8.15 GiB and 95-100% GPU utilization. It remains active on AC power. Scorer-aligned OOF results will be measured after all folds finish.

## 2026-11-10 01:33 UTC leaderboard status

- The latest 1,460-team snapshot is unchanged: EquiNexus rank 455 at 0.37 from 20 submissions. The official team row lists all three expected users. The top public score remains 0.70; no new high-scoring submission or public kernel was found in this refresh.
- Snapshot path and SHA-256 are recorded in `docs/experiment-log.md`.

## 2026-11-10 01:36 UTC public kernel review

- Reviewed three public approaches: a crop-trained semantic U-Net, grouped RF-DETR-Seg, and YOLO plus ResNet18 crop refinement. Their last runs were in July; they do not report comparable PQ on the current scorer. The grouped RF-DETR split and crop-refinement design are useful references. Its Dice-proxy threshold still needs scorer-aligned reproduction; the official RF-DETR repository confirms Apache-2.0 code and segmentation weights.
- The official metric changed from Dice to PQ on Aug 7, with leaderboard rescoring announced Aug 12. I downloaded notebook source only; no public predictions, checkpoints, or test labels were used. Audit paths and source links are in `docs/experiment-log.md`.

## 2026-11-10 01:50 UTC Kaggle YOLO OOF score

- Applied the official scorer to the completed Kaggle YOLO11m 1024px grouped OOF predictions. Best PQ is 0.365169 at confidence 0.25 / minimum area 400, below the incumbent grouped OOF selector result of 0.424791; reject this YOLO configuration.
- The sweep used the 565-observation OOF partition and 922 annotator views only, with no holdout or test labels. Prediction and manifest hashes, scorer output, and site/year summaries are in ignored `artifacts/model-experiments/yolo11m-kaggle-1024-oof-score-20261110/`.
- Stopped the local YOLO11m 1536px run after seven epochs on fold 0: matched-fold PQ was 0.325708 versus 0.348854 for the completed Kaggle 1024px model. The larger model increased TP but also produced substantially more FP; reject it and preserve the checkpoint plus fold comparison under ignored `artifacts/model-experiments/yolo1536-fold0-diagnostic-20261110/`.

## 2026-11-10 02:10 UTC selector sweep and next batch

- A 273-setting fine sweep improved the highest grouped OOF score only slightly, from 0.424791 to 0.424860 at RF blend weight 0.35 / threshold 0.455. The difference is marginal and requires leaderboard evidence; no holdout or test labels were used.
- Prepared five distinct, unseen candidates for the Nov 12 UTC submission window. Preflight confirms 180 supplied test images per CSV, five unique hashes, the expected submission schema, and frozen split digest. The dynamic batch builder reads the complete sweep and records past settings so later batches do not repeat earlier submissions.
- Latest leaderboard remains rank 455 / 0.37 for EquiNexus among 1,461 teams and 20 submissions. The full snapshot, hashes, candidate settings, and paths are in `docs/experiment-log.md`.
- The local torchvision Mask R-CNN ResNet-50 FPN V2 run is training from official COCO weights on 565 grouped training observations and scoring all views of the fixed 142-observation holdout. At epoch 2 it reached PQ 0.311350 (1,047 TP, 1,378 FP, 754 FN), up from 0.267843 at epoch 1 but still below the RF selector's 0.434930 holdout score. The eight-epoch run remains active; it has not been used to generate a submission. Details and provenance are in the dated experiment-log entry.

## 2026-11-10 03:05 UTC model and submission status

- The grouped Mask R-CNN finished with best holdout PQ 0.331323 at epoch 4 and was rejected against the 0.434930 RF selector baseline. Its full diagnostics, per-observation scores, checkpoint hashes, and visual overlays are archived under the ignored run directory. Public crop-refiner code remains a possible follow-up only if refinement improves official grouped PQ.
- Five unique grouped-OOF candidates passed preflight for each of Nov 13 and Nov 14 UTC. The final Nov 14 UTC batch is scheduled for 7:20 PM Toronto time on Nov 13, before the 8:00 PM sprint cutoff. The source commit will be updated by the scheduled daily checkpoint before upload.
- The official team row lists `arvbali`, `lolmrdabbydab`, and `mrinanksivakumar`; the user confirmed the three-account roster is expected.

## 2026-11-10 03:45 UTC crop-refiner follow-up

- An eight-epoch, random-initialized ResNet-18 U-Net trained on 4,449 grouped training-instance crops raised Mask R-CNN holdout PQ from 0.331323 to 0.389061. It still trails the 0.434930 selector baseline; the remaining 1,184 FP and visual 18-versus-11 prediction count on a dense sample confirm the quality gap. No test output was generated and no candidate was submitted.
- The ResNet-18 refiner used random initialization, with no external refiner weights or supervised labels. Its Mask R-CNN proposal source used official Torchvision COCO initialization and was fine-tuned only on the grouped competition-training partition. Checkpoint and complete scorer artifacts are in the ignored `maskrcnn-r50fpn-v2-gt-crop-refiner-256-holdout-20261110` run directory. The first failed attempt is archived separately and has no metric or candidate status.

## 2026-11-10 03:59 UTC detector-selector fusion screen

- A small holdout-only fusion screen of the crop-refiner detector with the incumbent RF selector peaked at PQ 0.435486 versus 0.434930 for the selector. Observation-grouped bootstrap uncertainty spans zero (95% CI -0.006000 to +0.007184; improvement probability 0.5704), so the apparent lift is noise-level and is not a submission candidate. No test images or labels were used; see the experiment log for hashes and the full sweep.

## 2026-11-10 04:27 UTC inference review and ImageNet refiner

- A fresh leaderboard snapshot places EquiNexus at rank 456 of 1,462 teams, score 0.37, with 20 submissions. All three expected accounts are listed. The public archive and hash are recorded in the experiment log.
- Reviewed the Apache-2.0 Mask R-CNN/crop-refiner notebook and topology-safe inference handoff. Reproduced their four-flip TTA and guard ideas using our own checkpoints. The best TTA mode scored 0.390995 versus 0.389061 single-pass, with a paired observation-bootstrap 95% interval of +0.000147 to +0.003680; because this is best-of-modes on the fixed holdout, it remains exploratory. The refiner still trails the selector at 0.434930, so no submission is warranted.
- The Torchvision ImageNet ResNet18 initialization passed a mixed-precision GPU smoke step. Full grouped training is running against the fixed split; the local RTX 5070 has 8.15 GiB and the run output/checkpoint remain under ignored `artifacts/`. It uses no external labels and does not access public MAGFiLO labels.

## 2026-11-10 04:52 UTC ImageNet refiner eligibility decision

- The eight-epoch ImageNet-initialized crop refiner scored 0.392410 PQ, a paired +0.003349 versus the scratch refiner (95% interval -0.000171 to +0.007006) and -0.042520 versus the 0.434930 RF selector incumbent (95% interval -0.057001 to -0.028055). The small scratch comparison is inconclusive and the candidate remains well below the incumbent.
- The official ImageNet agreement restricts the source database to non-commercial research and education, which is allowed under the user's updated resource policy. This run is eligible by license but is not promoted because its fixed-holdout PQ (0.392410) is below the 0.434930 incumbent. No submission or test prediction used it. Its fine-tuned checkpoint and downloaded ImageNet weights remain in ignored workspace paths; metadata records their hashes and prior policy decision.
- The committed refiner runner is scratch-only, and its smoke check passes. The holdout, seed, official scorer, and test data boundary remain unchanged.

## 2026-11-10 05:21 UTC all-annotator crop-refiner result

- Training on all 922 annotator views of the 565 grouped training observations produced 6,488 crops and 0.390305 holdout PQ at epoch 8. The selected-view scratch refiner reached 0.389061; their paired delta is +0.001245 with a 95% observation-bootstrap interval of -0.001513 to +0.003952, so the increase is inconclusive.
- The all-view refiner remains 0.044625 below the 0.434930 selector incumbent. It is rejected as a submission candidate; the model, per-observation comparisons, overlays, runtime, and GPU memory stay under ignored `artifacts/`.

## 2026-11-11 02:45 UTC all-view Mask R-CNN and selector fusion

- Training the detector on all 922 annotation views improved its holdout PQ by 0.019979 over selected-view training (paired 95% interval +0.009158 to +0.030748), but it still scored 0.351302 versus 0.434930 for the selector. It is not a submission candidate.
- Four selector/detector fusion policies at two IoU gates did not improve the selector. The best was identical to the selector at 0.434930; intersection, union, and replacement reduced PQ. No test data were accessed, and no candidate was submitted.
- See the experiment log for checkpoint and script hashes, paired-bootstrap details, and artifact paths.

## 2026-11-11 03:16 UTC leaderboard refresh

- The latest public snapshot after the Nov 11 batch places EquiNexus at rank 467 of 1,479 teams, score 0.37 from 25 submissions. All five new candidates scored 0.37; the leading score remains 0.70. Snapshot details are in `docs/experiment-log.md`.

## 2026-11-11 public non-commercial resource clarification

- The user clarified that public resources restricted to non-commercial research or education may be used in this challenge. Earlier exclusions based only on a non-commercial restriction are no longer current. We will still honor source attribution, ShareAlike conditions, Rule 6 accessibility, and the host's restriction against using MAGFiLO annotations beyond those supplied by the competition.
- This makes the existing ImageNet-initialized refiner eligible by resource terms; its 0.392410 fixed-holdout PQ remains well below the 0.434930 selector incumbent, so it is not a candidate on performance.
- The user's update also supersedes the earlier ImageNet ResNet34 exclusion. That independent grouped-OOF run scored 0.367271, well below the incumbent, and remains rejected on performance.
- The public CC BY-NC-SA 4.0 ten-model U-Net pack may now be evaluated under its stated terms. Its author reports 0.4502 nested OOF PQ and 0.39 public score, and says it used only competition-provided training labels. The published fold map covers all 707 competition observations; our 142 fixed-holdout observations distribute across folds 0-4 as 32/30/22/30/28. No single checkpoint excludes the full holdout, so its metrics cannot be recomputed on our frozen holdout. The author's OOF result remains separate evidence for a test-only candidate.

## 2026-11-11 public U-Net candidate

- The 10-model CC BY-NC-SA 4.0 U-Net ensemble completed inference on all 180 supplied test images in 1,100.98 seconds, with 3.81 GB peak GPU allocation. It produced 1,177 instances on 177 images; three images had no predictions. Full run metadata and an exact RLE validation report are stored under ignored `artifacts/submissions/public-unet-ensemble-20261112/`.
- All 1,177 RLEs decoded to 2048x2048 masks and re-encoded byte-for-byte; IDs were unique and contiguous per image, with no pixel overlap between instances. The output hash is `109a1abe8ea7d888ab301af91d0cc68d9060625ef6f08b6dce620dbefdacfbfa`.
- Replaced the weakest of the five Nov 12 selector candidates with this test-only U-Net result. The candidate's own OOF and fixed-holdout fields are null because its folds overlap the frozen holdout; its author-reported nested OOF PQ (0.4502) and public score (0.39) remain clearly separate evidence. All labels used upstream are reported as competition-provided training labels; public MAGFiLO test labels remain excluded.
- Attribution and ShareAlike provenance are recorded. Under the user's Rule 6 policy, public resources restricted to non-commercial research/education are eligible for challenge evaluation; the checkpoint remains outside Git and its license terms remain attached to any use.

## 2026-11-11 I079 inference and public-resource scope

- The user confirms that public resources limited to non-commercial research/education may be used under the challenge rules. We retain attribution and each license's specific terms; the host's restriction on public MAGFiLO test labels remains unchanged.
- Downloaded the public I079 model instance under `data/reference/phuongncn-i079/`; both checkpoint MD5 values matched the published card. Its Apache-2.0 model card reports training only on competition training images/annotations and no public MAGFiLO/test-overlap labels. Its source notebook's final mode refits on the full 707 training observations, so its checkpoints are not eligible for our fixed-holdout score.
- Test-only inference completed on all 180 supplied H-alpha images: 1,299 masks over 169 images, 106.30 seconds runtime, and 1.20 GB peak GPU allocation. All 1,299 masks passed exact 2048x2048 RLE round trips, with unique contiguous IDs, no empty masks, and no overlap. The output SHA-256 is `1812d164291d89ad0700233f920a33d7367aa408fac1cd3a828aae5548e62b3a`.
- Compared with our prior I079 submission 58110863 (public PQ 0.34), the new run has the same 1,299 IDs and 0.9963 mean mask IoU. It is a near-repeat and is rejected for resubmission; the publisher's 0.70 claim remains uncorroborated by our result. Its full-fit training also prevents a fixed-holdout score.
- The public U-Net occupies slot 5/5 in the Nov 12 batch; Nov 13 retains five distinct RF selector variants. Both batches pass manifest dry-run checks.
- The CC BY-NC 4.0 Amey Pawar checkpoint set is eligible under the user's policy, but it is not eligible for our frozen holdout: replaying the documented seed-0 date split finds 112/142 holdout observations in the checkpoint training partition. Its 144-observation upstream validation set is independent of those weights. Cross-fitted fusion scored 0.419160 versus 0.412857 for the EquiNexus selector on that set; the paired interval includes zero, so it is not promoted. No test labels were accessed.

## 2026-11-11 05:10 UTC public non-commercial resource policy reaffirmed

- The user reaffirmed that public weights, models, and datasets restricted to non-commercial research or education may be used for this challenge. Do not exclude a resource solely because of that restriction; record attribution, provenance, accessibility, and any license-specific conditions such as ShareAlike. The prohibition on public MAGFiLO test annotations remains unchanged.
- Completed the independent Amey Pawar CC BY-NC 4.0 checkpoint evaluation on its 144-observation upstream validation set. Cross-fitted U-Net/YOLO fusion achieved PQ 0.419160 versus 0.412857 for the EquiNexus selector; paired 20,000-observation bootstrap delta was +0.006303 with 95% interval [-0.005295, +0.017938]. This is promising but inconclusive, so the model is not promoted. Full results are in ignored `artifacts/external-data/ameypawar-upstream-validation/`.
- Corrected the current model notes: previous non-commercial-only exclusions of ImageNet-initialized refiners, the Fususu public weights, and the CC BY-NC-SA U-Net are superseded. Their existing validation or split limitations still determine whether they are candidates. No public MAGFiLO test labels were accessed.

## 2026-11-11 04:12 UTC leaderboard refresh

- The latest snapshot has 1,479 teams; EquiNexus is rank 468 at 0.37 after 25 submissions. All five Nov 11 UTC RF candidates completed at 0.37. The top rows are 0.70, 0.64, 0.62, and 0.62. Snapshot hash and archive path are in `docs/experiment-log.md`.
- The Nov 12 batch now allocates one slot to the public U-Net and the Nov 13 batch keeps five distinct unseen RF selector settings. The I079 near-repeat is not scheduled for another submission.

## 2026-11-11 05:25 UTC current leaderboard and metric audit

- The fresh public leaderboard has 1,479 teams. EquiNexus is rank 468 at 0.37 with 25 submissions; the five most recent submissions all scored 0.37. The current top rows are 0.70, 0.64, and several 0.62 entries. The leaderboard snapshot and SHA-256 are recorded in `docs/experiment-log.md`.
- Reviewed a public July 20 report that claims a historical 0.71 result and describes `instance_wise_dice`. The current competition page was updated Aug 7 and announced a leaderboard re-score Aug 12. Later organizer replies on Oct 29 confirm the leaderboard ranks by PQ, and on Nov 5 clarify that leaderboard scoring blocks duplicate-mask inflation. We will keep selecting for the official PQ and task quality, and avoid duplicate-mask or empty-output scoring behavior.
- The Nov 12 UTC batch has five hash-locked candidates, including the public CC BY-NC-SA U-Net test ensemble; the Nov 13 UTC batch has five distinct RF-selector candidates. Both manifest preflights pass with the frozen split digest. Kaggle's five-submission allowance is exhausted for the current UTC day, and the next batch is scheduled for Nov 11 at 20:20 Toronto time.
- RLE validation now passes for all ten queued CSVs: each covers all 180 supplied test images, every mask decodes to 2048x2048, round trips are exact, IDs are unique, and no predicted masks overlap. Reports are under ignored `artifacts/submissions/rle-validation-20261112/` and `artifacts/submissions/rle-validation-20261113/`.

## 2026-11-11 06:31 UTC Kaggle refresh

- Current account snapshot: 1,480 teams; EquiNexus rank 468 at public score 0.37 with 25 submissions. The five latest EquiNexus submissions all remain at 0.37. The latest public-source review did not produce a validation-supported candidate to promote; details, IDs, hashes, and archive paths are in `docs/experiment-log.md`.

## 2026-11-11 06:45 UTC SOTA checkpoint review

- Audited a newly published CC0 checkpoint bundle. Its YOLO11s metadata identifies AGPL-3.0 code but does not establish data or fold provenance; supporting files have no training manifest. We keep it as a source reference and do not validate, train, fuse, or submit from it until the label provenance can be verified. Full static inspection and SHA-256 values are in `docs/experiment-log.md`.

## 2026-11-11 07:13 UTC public U-Net family probe

- Initially added one ResNet34-only test-inference candidate to the Nov 14 batch. It was later replaced by CAGE r5 and YOLO26x to add independently trained model families; its test-only artifacts remain for comparison. The EfficientNetV2-S-only output closely matches the public full ensemble and was not scheduled. Full hashes and test-only comparisons are in `docs/experiment-log.md`.

## 2026-11-11 07:36 UTC CAGE r5 and public prediction-vote review

- Public CC BY-NC 4.0 resources remain eligible under the user's policy. CAGE r5 is now a checksum-pinned, safe-loaded test-only candidate; its exact training fold is unknown and its reported validation is annotator-leaky, so it has no OOF or fixed-holdout score. Local RTX 5070 inference and all 1,234 RLE round trips passed. The final Nov 14 batch uses CAGE r5 in place of the ResNet34-only U-Net subset.
- Reviewed the current `prvsiyan` five-family vote notebook. Its use of other teams' submitted prediction files is not adopted because their model-data provenance is not fully audited. Our own cross-fitted family vote was already tested and trails the leading YOLO11s/U-Net fusion. Keep the organizer restriction on public MAGFiLO test annotations, and use public leaderboard movement only as preliminary evidence.

## 2026-11-11 07:52 UTC YOLO26x and high-OOF U-Net ensemble review

- Added a test-only YOLO26x candidate from a public Apache-2.0 model whose card says it used the 707 competition training images. The exact training folds are unavailable, so it receives no validation score. The local 2048px inference and full RLE check passed; the final batch now retains three grouped-OOF RF candidates plus CAGE r5 and YOLO26x.
- Reviewed a public U-Net/UNet++ + RF-DETR + crop-refiner notebook reporting nested OOF PQ 0.4591. Its model/data dependencies and licenses are not publicly retrievable through Kaggle metadata, so this remains an architecture reference only. Our cross-fitted candidate choices and scorer remain unchanged.

## 2026-11-11 12:34 UTC Ektarr YOLO11s and crop-refiner screen

- Audited and reproduced the Apache-2.0 Ektarr YOLO11s-seg plus ResNet18 crop-refiner approach. The source notebook's Kaggle run reports a 142/565 random observation split and PQ 0.415533; 30 of those validation observations overlap our frozen holdout, and its validation selected checkpoints, so that result is screening evidence only. The output-checkpoint licenses are not separately stated, so those weights were used only for this source-split screen and not for the EquiNexus training run or a submission. Its outputs are archived under ignored `data/reference/public-notebooks/ektarr-yolo-unet/`.
- Retrained the recipe on the 565-observation frozen training partition and scored all 142 holdout observations across 232 annotator views. The official scorer reports PQ 0.415717 (1,071 TP, 645 FP, 690 FN), below the 0.434930 selector incumbent. A paired observation bootstrap (20,000 resamples, seed 42) gives delta -0.019214, 95% CI [-0.033518, -0.005489], and probability of improvement 0.00255. The holdout received no gradient updates but selected detector/refiner checkpoints, so this remains a selection-biased screen; the candidate is not promoted.
- The local run used YOLO11s-seg COCO initialization and an ImageNet-initialized ResNet18 refiner, competition annotations only, confidence 0.25, and minimum predicted area 20. Training took 2,800.49 seconds and inference 46.98 seconds on the RTX 5070 Laptop. PyTorch reported 3,287,865,856 bytes peak allocated during training; an external `nvidia-smi` sample during the run reached 7,786 MiB used. Fixed low-, median-, and high-PQ overlays are in ignored `artifacts/detector-experiments/ektarr-style-yolo11s-crop-refiner-frozenholdout-20261111/overlays/`.
- Source commit `dd5d77c21ddb8ff8022907d38e8d440c9d7e2325`, runner SHA-256 `9d9465f1f961816a390525ff87792a6fa9852a7422f33367f65e97d4de163c65`, and frozen split SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e` are recorded in the ignored run metadata. Ultralytics is AGPL-3.0; the source notebook is Apache-2.0. No public MAGFiLO test labels, test-image predictions, or submissions were used in this experiment.

## 2026-11-11 12:59 UTC fold-aware public U-Net validation

- Audited the newly run [Huendao inference notebook](https://www.kaggle.com/code/huendao/solar-filament-segmentation-challenge-2026). It uses Dariush Afshar's 10-model U-Net dataset, D4 TTA, and the dataset's fixed component keep rule; it reports author-provided OOF PQ 0.4523 and public score 0.3945. The CC BY-NC-SA 4.0 dataset README states all ten weights were trained only on the 707 competition training observations and labels, grouped into five image-level folds. The user permits non-commercial research/education resources; preserve attribution and ShareAlike obligations.
- Added `scripts/evaluate_public_unet_fold_aware.py`. For each frozen holdout observation it loads only the ResNet34 and EfficientNetV2-S checkpoints for the source fold that held that observation out. All 707 stems map exactly once in the source fold file; the 142 holdout observations map as 32/30/22/30/28 across folds. The seed-42 split digest and upstream keep rule are unchanged, and all 232 annotator views are scored. No holdout gradients, checkpoint selection, or threshold tuning were used.
- Official PQ is 0.447799 (1,099 TP, 564 FP, 612 FN), versus 0.434930 for the selector incumbent. The paired bootstrap delta is +0.012869, 95% CI [-0.000142, +0.026097], P(improvement)=0.97405 over 20,000 observation-group resamples. The interval narrowly includes zero. This is strong independent evidence for the model family, while the frozen holdout predictor uses two fold-held-out models and the supplied test candidate uses all ten models.
- The full ten-model test CSV was already RLE-validated for all 180 supplied images and remains slot 5/5 in the Nov 12 UTC batch. Its batch entry now records the fold-aware holdout evidence. Local inference used the RTX 5070 Laptop GPU for 194.75 seconds at 3,405,122,048 bytes PyTorch peak allocation. Run outputs, per-observation results, and fixed overlays remain ignored under `artifacts/model-experiments/dariushafshar-unet-fold-aware-holdout-20261111/`; run metadata records all weight hashes, the fold-manifest SHA-256, scorer, packages, and hardware. No public MAGFiLO test labels were used.

## 2026-11-11 13:20 UTC latest public notebook screen

- Rejected the newly run ASTH-Net notebook as a candidate. Its own source-run log reports mean PQ 0.0258 on its separate chronological validation and, due a missing `all_rows.extend(img_rows)` call, writes only two synthetic fallback rows after processing all 180 test images. Notebook and log hashes, paths, and details are in the experiment log.
- The V18 multi-annotator YOLO fine-tuning notebook has an interesting annotation-consolidation idea, but the current kernel run is cancelled, its public export has no metric outputs, and its checkpoint datasets are unavailable to this account. Its 107-photo comparison has also been used repeatedly across the author's model iterations. No candidate or submission is derived from it.

## 2026-11-11 15:49 UTC public U-Net plus YOLO family vote

- The fixed cross-family agreement candidate scored holdout PQ 0.457392 over 142 observations and 232 annotator views. It improves on the public U-Net by +0.009592 (paired 95% CI +0.003384 to +0.015692) and on the incumbent by +0.022461 (+0.010721 to +0.034402); each interval comes from 20,000 observation-grouped bootstrap resamples.
- Its operating rules were frozen before scoring: five-fold YOLO11s TTA8 at confidence 0.35 / dedup IoU 0.30, upstream public U-Net extraction, agreement IoU > 0.5, fixed clipping priority, and deterministic empty-image fallback. The 142 observations were not used for fitting, checkpoint selection, or threshold tuning.
- Test inference used only the 180 supplied H-alpha images. The candidate passed exact RLE round trips for all 1,057 masks, has zero overlaps and unique IDs, and covers all 180 files. The Nov 13 submission batch now pairs four grouped-OOF RF settings with this validation-supported candidate. Full provenance and score diagnostics are in `docs/experiment-log.md` and ignored run metadata.
- Fixed low-, median-, and high-PQ overlays show continuous agreed filaments overall, with misses and short fragments in lower-scoring observations; the reviewed cases do not show widespread disk-wide false positives.

## 2026-11-11 15:56 UTC RF-DETR-Seg screen

- The six-epoch RF-DETR-Seg Small run scored PQ 0.141106 on the frozen holdout, far below the 0.434930 incumbent. Its paired bootstrap delta is -0.293824 (95% CI -0.312863 to -0.274494; P(improvement)=0). Overlay review shows missed filaments and fragmented predictions, so the candidate is rejected and not submitted.
- The run used only the competition-training partition, an 85-observation inner grouped validation split, and confidence 0.40 selected by its internal PQ sweep. The fixed holdout was used only for final evaluation. Runtime was 3,113 seconds with 3.93 GB peak GPU allocation; full logs and hashes are in the ignored RF-DETR experiment folder.

## 2026-11-11 16:16 UTC cross-fitted family-vote validation

- Public U-Net source-fold predictions on all 565 frozen training observations scored OOF PQ 0.453552 across 922 annotator views. The fixed two-family agreement rule scored 0.455022 (3,838 TP, 1,278 FP, 2,650 FN), with 3,123 masks on 557 observations.
- Against cross-fitted U-Net alone, paired OOF delta is +0.001470 (95% CI -0.002730 to +0.005512; P(improvement)=0.7567). Against cross-fitted YOLO11s, delta is +0.057695 (95% CI +0.051184 to +0.064224; P(improvement)=1.0). This supports the combined rule over the detector while leaving its incremental benefit over the U-Net uncertain; the independent frozen-holdout gain remains positive.
- OOF predictions use only fold checkpoints that excluded each scored observation; the frozen holdout and test observations are absent. The Nov 13 slot-5 manifest now records OOF PQ 0.455022 and holdout PQ 0.457392. Outputs, provenance, and hashes are in the ignored OOF run folders and experiment log.

## 2026-11-11 16:50 UTC OOF-only vote tuning and holdout confirmation

- An eleven-variant OOF-only screen around the confidence-0.35 baseline found the best family-vote setting at YOLO confidence 0.25 and strict cross-family IoU > 0.5: PQ 0.459116. This is +0.004094 over confidence 0.35 (paired 95% CI +0.001526 to +0.006762; P(improvement)=0.9996). Higher and lower confidence, stricter agreement, and one-family union all scored lower; full settings are in the experiment log.
- The preselected 0.25 setting was evaluated once on the holdout: PQ 0.454668, +0.006869 versus U-Net alone (95% CI +0.001837 to +0.012024), +0.019738 versus the incumbent (+0.007713 to +0.031900), and -0.002723 versus confidence 0.35 (-0.005857 to +0.000562). We retain both, since confidence 0.25 is strongest under OOF and 0.35 has the higher holdout point estimate.
- Its test CSV passed all 1,084 exact RLE round trips, unique-ID and overlap checks, and full 180-image coverage. The Nov 13 batch now has three top OOF RF candidates plus both U-Net/YOLO thresholds.

## 2026-11-11 17:21 UTC native-resolution U-Net candidate

- Audited Robin Amir Bahar's public 1888px U-Net notebook. Its one semantic mask per original observation is a useful complement to instance detectors, but its random 7.5% validation and symmetric best-instance-Dice selection are not equivalent to the frozen split and official PQ scorer.
- Added `scripts/train_native_unet_grouped.py` as an independent implementation. It uses five grouped folds within the fixed 565-observation training partition for OOF predictions, compares fixed epoch snapshots 16/20/24 and connected-component settings by official PQ on those OOF predictions, then measures the selected model and postprocessing once on the 142-observation frozen holdout. The holdout never enters model fitting or parameter selection.
- An architecture-only full-crop backward pass used 6.56 GiB of the local RTX 5070 Laptop GPU's 7.96 GiB and completed. Training and scorer-aligned evaluation are pending; no result or submission candidate is claimed yet.
- Retrieved only the public source run log and its 53-image validation-distribution CSV. The run selected epoch 20 at custom symmetric Dice 0.6187, with mean best-overlap IoU 0.4966; these are not official PQ. No test prediction output was retrieved or reused.

## 2026-11-11 18:53 UTC leaderboard and public-source refresh

- The latest public snapshot has 1,492 teams. EquiNexus is rank 477 at 0.37 after 25 submissions; all five Nov 11 submissions remain at 0.37. The top score is 0.70, followed by 0.64 and four teams at 0.62.
- Queried Kaggle's public kernels for accounts attached to the top-ten rows. Available notebooks from the 0.70 leader IONode Online, Sayandeep Dutta, Tensura, Kush Patel, Amit Gunjan, and AINOJOB are unrelated; other queried accounts exposed no public kernels. IONode has no public model and only unrelated datasets. No reproducible top-entry pipeline was available to adopt, so selection stays grounded in grouped OOF and frozen-holdout evidence.

## 2026-11-11 19:05 UTC Kaggle quota and cross-runtime comparison

- Kaggle GPU quota is 0 hours until Nov 13 at 7:00 PM Toronto time. A fold-aware U-Net evaluator using the same code and frozen split is scheduled for a Kaggle T4 at 7:10 PM, after the final commit/push and before the 7:20 PM submission batch. It will skip if the quota has not refreshed or Kaggle cannot provide a GPU.
- The local 5070 native-resolution U-Net OOF run remains in progress; it is separate from the same-evaluator public U-Net comparison scheduled for the final T4 window.

## 2026-11-11 19:09 UTC submission batch preflight

- Dry-run preflight passed for all three prepared batches (Nov 12, Nov 13, and Nov 14 UTC), each with five candidates and the frozen split digest. The Nov 13 batch includes both grouped-OOF-selected U-Net/YOLO fusions; the final Nov 14 UTC batch is scheduled Friday at 7:20 PM Toronto time.

## 2026-11-11 19:17 UTC historical high-score source audit

- Audited two public references titled 0.70/0.71. Their scores predate Kaggle's August change to PQ: current rows are Felipe KNBIS at 0.55 and Fususu at 0.36. The GitHub repo does not publish trained weights; the Kaggle handoff has no saved outputs, uses public I079 weights for test-only inference, and reports no current-PQ validation. Neither title's historical score is evidence for the current metric, and neither becomes a submission candidate.
- The report's Mask R-CNN + CropUNet architecture and warning about losing fine structure at half resolution are reference ideas only. Our native-resolution grouped U-Net run is already testing the relevant resolution hypothesis under the fixed scorer.

## 2026-11-11 19:37 UTC refreshed leaderboard and public-source audit

- Current leaderboard snapshot: 1,494 rows; EquiNexus rank 479 at 0.37 after 25 submissions. The five latest EquiNexus scores remain 0.37. IONode Online is at 0.70, but searches of its public Kaggle account and other current top-account handles found no relevant public solar-filament model or pipeline. We keep grouped validation as the model-selection signal.
- Reviewed prvsiyan's five-family test-prediction voting notebook. Its method is an inference-only vote over other public outputs, with strict IoU support requirements; it reports 0.40 but does not provide grouped validation or reproduce all underlying weights. Our U-Net/YOLO family vote already tests the comparable idea using reproducible sources on the frozen grouped split.
- Reviewed Muhammad Shoaib Altaf's Apache-2.0 V18 detector experiment. It proposes consolidating cross-annotator detections into one target set per physical photo before YOLO fine-tuning, but its 107-photo evaluation split has been reused and the pulled export has no attached V10/V12 checkpoints. Retain this as a future hypothesis; do not infer it improves PQ.
- Huendao's recent notebook reuses the same ten-model public U-Net family already scored locally. No new candidate was added from that source review.
- The native-resolution grouped U-Net remains live on local RTX 5070, fold 0 epoch 9/24. Kaggle GPU quota is exhausted until Nov 13 at 7 PM Toronto; the scheduled T4 comparison remains conditional on quota refresh.
- Its EfficientViT-B2 ImageNet encoder weights are Apache-2.0 in the timm model table, with timm advising that ImageNet's non-commercial research limitation applies to ImageNet-pretrained weights. This fits the user's explicit permission for public non-commercial research weights; retain the usage restriction in any final model documentation.

## 2026-11-11 19:53 UTC rank-7 public model dataset

- Found the CC0-1.0 checkpoint bundle from leaderboard rank 7 (`daydreamerthinks`, public score 0.61). Two assets are one-channel ResNet-34 U-Net state dictionaries; a third is a YOLO11s detector checkpoint with AGPL-3.0 metadata. The small meta-learner file contains RSNA knee labels and is unrelated.
- No model source or training-split provenance is public, so it cannot be used for a valid frozen-holdout score, training, or a reproducible submission. Keep it isolated in ignored `data/reference/` with hashes; do not consume test annotations. License permission alone does not resolve the training-data provenance gap.

## 2026-11-11 20:15 UTC GONG auxiliary-task candidate

- Use only the 504 GONG magnetogram-derived target maps matched to the frozen 565-observation training partition. The inventory exposes 124 additional maps matching holdout observations; the downloader never selects or retrieves them. The split digest remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- Train an independent native-resolution U-Net with competition-mask segmentation loss plus a fixed 0.20 auxiliary regression loss for PIL proximity and strong-flux maps. OOF fold validation maps are not passed into the fold's optimizer. Compare cross-fitted predictions with the native U-Net using a paired observation bootstrap, use OOF-only epoch/postprocessing selection, and evaluate each competitive candidate once on the frozen holdout. Test inference uses only supplied H-alpha images.
- Source: [Kaggle GONG dataset](https://www.kaggle.com/datasets/vhrabar/gong-solar-filaments). Include the [NSO GONG acknowledgement](https://nso.edu/data/) in any archived report or solution documentation. No model score exists yet; the original native-resolution U-Net is still training on the local RTX 5070.

## 2026-11-11 20:39 UTC GONG target preparation

- The training-only download is complete: 504 arrays, 396,426,240 bytes, per-file hashes verified, manifest SHA-256 `af7ed77847698d1179fc7f13ffc493fac07f1698f1a05ed2ff478e18d2d3dcc0`, and combined target digest `2751fe4c75f0b5c5f964284e08ec2d94e7ca8ac41d744b9422ea8d670b506da2`.
- Holdout target files downloaded: 0. The split remains unchanged. The scheduled local task will wait for the current native U-Net OOF run, then run the GONG OOF model and paired bootstrap; it proceeds to one holdout comparison only if the OOF score is not lower.

## 2026-11-11 submission batch adjustment

- Promoted the OOF-selected confidence-0.25 U-Net + YOLO11s family vote to Nov 12 slot 5. It scored OOF PQ 0.459116 and holdout PQ 0.454668, with a paired holdout gain of +0.019738 versus the incumbent (95% CI [+0.007713, +0.031900], P(improvement)=0.9995). The confidence-0.35 vote (OOF 0.455022, holdout 0.457392) remains scheduled for Nov 13 as a complementary preselected candidate; the standalone 10-model U-Net (holdout 0.447799) shifts to Nov 13 slot 4. Both five-candidate manifests pass hash and split preflight. The Kaggle submission listing has no family-vote upload yet.
- The user clarified that public datasets, model weights, and models licensed for non-commercial research or education may be used. Retain each resource's attribution and non-commercial/share-alike conditions in our use and reporting.
- The native-resolution local U-Net OOF run remains active; latest progress at 21:29 UTC was fold 0 epoch 21/24, step 350/452.
## 2026-11-11 21:43 UTC current public-method review

- Kaggle's current top scores are IONode Online 0.70, Tempest 0.64, four teams at 0.62, then 0.61. EquiNexus's five most recent submissions remain at 0.37 each.
- Rechecked the [ASTH-Net notebook](https://www.kaggle.com/code/koushikkumardinda/asth-net-anisotropic-spine-tangent-hysteresis): the source hash matches the version previously rejected for non-grouped validation and its empty-output bug. Its radial absorption, hysteresis, and tangent-aligned gap-bridging ideas remain references only. The saved family-vote OOF predictions are discrete instances, so testing its probability hysteresis requires new fold-held-out probability maps; no candidate was added.
- The public CAGE r5 model remains eligible for non-commercial research under CC BY-NC 4.0 per the user's clarification, but its undocumented training membership and reported duplicate overlap make its prepared output test-only. It appears once in the Nov 14 UTC submission batch without a frozen-holdout score.
## 2026-11-11 CAGE r5 submission validation

- The prepared CAGE r5 CSV passed full validation against all 180 supplied test images: 1,234 exact RLE round trips, 1,234 unique contiguous instance IDs, zero empty masks and zero inter-instance overlap. It has 1,234 predictions across 177 images; three images have no instances. SHA-256: `287c306837f138ed90aa29352dc45d37511037331015b8444b8d5d9c853c395d`.
- This checks format and coverage, not segmentation accuracy. Because upstream training membership overlaps its reported validation split, no frozen-holdout evaluation is used; the candidate stays test-only for Nov 14 UTC.

## 2026-11-11 native and GONG task schedule

- The native U-Net OOF run is active at fold 0 epoch 24/24. The GONG task waits for its complete OOF summary, uses the same frozen split, and skips holdout comparison if its OOF score is lower. Wake-on-run is enabled while stop-on-battery remains enabled.
## 2026-11-11 22:01 UTC native U-Net progress

- Fold 0 completed training and generated its selected-epoch OOF predictions for all 113 held-out observations. Fold 1 is training on the complementary 452 observations. No aggregate OOF score is available yet.

## 2026-11-11 22:13 UTC five-family public checkpoint audit

- Inspected the public [notoverkil five-family weight dataset](https://www.kaggle.com/datasets/notoverkil/solar-filament-weights). Its description claims public PQ 0.41 and five-fold OOF PQ 0.4747, and lists 25 checkpoints across YOLO11m, Mask R-CNN, YOLO26m, and ConvNeXt U-Net, plus five refiners and a rescorer. The description states AGPL-3.0; Kaggle's machine-readable license field is `other`. This license is within the user's stated use policy, subject to retaining AGPL terms.
- The published fold map contains the 707 competition training observations. The five model-fold training complements include 112, 121, 113, 113, and 109 of our 142 frozen-holdout observations, respectively. No checkpoint provides a valid prediction for the complete fixed holdout, so the advertised OOF result is not comparable to our scorer run.
- The dataset card points to `infra/reproduce-submission.py` but does not identify or include the source repository; Kaggle's public notebook search found no reproducer. We downloaded only four small model configs and the fold metadata, not checkpoint weights or `submission.csv`. No public MAGFiLO test labels or third-party predictions were accessed. Defer use until reproducible inference code and training-data provenance are available; do not score these checkpoints on the frozen holdout.

## 2026-11-11 22:19 UTC native U-Net fold 1 update

- Fold 1 reached epoch 2/24 at mean training loss 0.69131 (551 seconds for the epoch) on the local RTX 5070. This is training progress only; aggregate grouped OOF metrics await all five folds.

## 2026-11-11 22:20 UTC submission batch preflight

- Ran the real hash and CSV validation dry-run for the Nov 12, Nov 13, and Nov 14 UTC manifests. All three passed with five candidates each and the unchanged frozen split digest. The Nov 13 batch includes the public CC BY-NC-SA U-Net and OOF-selected U-Net/YOLO vote; the Nov 14 batch includes test-only YOLO26x and CAGE candidates. No upload occurred in this preflight.

## 2026-11-11 22:39 UTC OOF-ranked submission refresh

- Replaced the three lower-scoring RF-selector slots in the Nov 13 and Nov 14 manifests with additional public U-Net/YOLO vote settings already evaluated on the frozen cross-fitted training partition. Nov 13 now includes confidence 0.20 (OOF PQ 0.458104), 0.30 (0.458100), and 0.15 (0.457062), alongside the selected 0.35 family vote and the fold-aware standalone U-Net. Nov 14 now includes support IoU >0.55 (0.452149), confidence 0.40 (0.450632), and support IoU >0.60 (0.448411), alongside test-only YOLO26x and CAGE outputs.
- Generated each new CSV from the supplied test images and existing public U-Net/YOLO test predictions. All new masks passed exact 2048x2048 RLE round-trips, unique-ID, nonempty-mask, and zero-overlap checks; each file covers all 180 test images, with predictions on 179–180. No holdout threshold sweep was performed.
- Updated both five-candidate manifests with OOF scores, selection settings, run metadata, RLE validation reports, and new CSV hashes. Hash/dataset-format preflight passed for Nov 13 and Nov 14 with the frozen split digest unchanged.

## 2026-11-11 22:51 UTC submission batch resequencing

- Reallocated all three remaining batches by OOF rank so the Nov 12 UTC upload contains the five distinct confidence settings 0.15, 0.20, 0.25, 0.30, and 0.35 (OOF PQ 0.457062–0.459116). This replaces four lower-scoring RF selector variants in tonight's batch.
- The Nov 13 UTC batch now contains support 0.55, confidence 0.40, support 0.60, the fold-aware standalone U-Net, and support 0.65. The Nov 14 UTC batch contains confidence 0.45, the confidence-0.35 union, support 0.70, and the test-only YOLO26x and CAGE candidates. Each test CSV is unique across the three batches.
- The newly generated support-0.65, confidence-0.45, support-0.70, and union CSVs passed exact RLE validation. All three five-candidate manifests passed hash and schema preflight with the frozen split digest unchanged; no holdout threshold sweep was performed.

## 2026-11-11 22:52 UTC native U-Net progress

- The local grouped OOF run remains live on fold 1, epoch 6/24, step 250/452. Mean training loss is 0.55467. This is training progress only; no aggregate OOF result is available.


## 2026-11-11 22:58 UTC leaderboard and local OOF checkpoint

- Downloaded the current public leaderboard into ignored `artifacts/kaggle/leaderboard/manual-20261111-1757/`. The snapshot contains 1,497 teams; EquiNexus is rank 481 at 0.37 after 25 submissions, and the highest public row is IONode Online at 0.70. All five latest EquiNexus submissions remain at 0.37. Snapshot SHA-256: `4648e32200dc00420fc0d50dae11923ad107d4ccb688bbf3e9cfae51823085b2`.
- The public score is secondary evidence: Kaggle's [public diagnostic notebook](https://www.kaggle.com/code/artkomissar/please-fix-the-leaderboard) shows a public score of 1.00 from five empty masks, and a [competition discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809) reports empty submissions scoring 0.93. The competition [official evaluation overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/acknowledgment) specifies PQ, so candidate selection remains based on the frozen grouped scorer and genuine segmentation outputs.
- The local grouped U-Net is training fold 1, epoch 7/24, step 100/452, at mean training loss 0.53308. The RTX 5070 is under sustained load; this is training progress only, with no aggregate OOF score yet. The next five-candidate batch is scheduled for 8:20 PM Toronto time after the 8:05 PM daily commit/push task.


## 2026-11-11 23:07 UTC native U-Net progress

- Fold 1 completed epoch 7/24 in 563.194 seconds with mean training loss 0.515594 and began epoch 8 on the local RTX 5070. All five grouped OOF folds remain necessary before the aggregate scorer result is available; this is not a validation metric.


## 2026-11-11 23:11 UTC public U-Net++ notebook audit

- Audited Ruhul Amin's [Apache-2.0 U-Net++ / EfficientNet-B4 notebook](https://www.kaggle.com/code/ruhul20/solar-segmentation-u-net-efficientnet-b4). Its Kaggle metadata attaches only the competition data; the source uses ImageNet encoder weights, a three-channel raw/CLAHE/local-contrast representation, soft targets averaged across annotator passes, and a BCE/Tversky/focal/clDice objective. It trains three random physical-image folds and one complementary DeepLabV3+ model.
- The source's validation is not selection-safe: it tunes on fold 0 but averages all three primary models for those predictions, so two models have trained on fold 0. Its internal PQ uses greedy one-to-one matching and `IoU >= 0.5`, which differs from the official strict `IoU > 0.5` all-qualifying-pairs scorer. The [competition Code tab](https://www.kaggle.com/competitions/filament-segmentation-2026/code?competitionId=143678&excludeNonAccessedDatasources=true&sortBy=scoreDescending) surfaced a score of 0.67, while the [version 339473185 detail page](https://www.kaggle.com/code/ruhul20/solar-segmentation-u-net-efficientnet-b4?scriptVersionId=339473185) reports 0.27 current and 0.29 best; neither score is selection evidence.
- The page reports a 7h18 T4x2 run; Kaggle's current kernel status is cancelled. Reproducing its full multi-model setup would contend with the live local OOF run, and Kaggle GPU quota is exhausted until Nov 14. No checkpoint or notebook output was downloaded or used. Retain the multi-channel and consensus-target ideas for later experiments, but do not promote this notebook's score or outputs.


## 2026-11-11 23:26 UTC grouped-OOF confidence bracket

- Rechecked the public U-Net/YOLO11s two-family vote using the same cross-fitted 565-observation partition and official scorer. YOLO11s confidence 0.24 scored PQ 0.458726 (154.8 s); 0.25 scored 0.459116 (139.0 s, existing run); 0.26 scored 0.458816 (153.7 s). All use split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; the holdout and test labels were not used. Keep 0.25 as the OOF-selected setting; no candidate or manifest changed.
- The first 0.24 run used a different three-family vote (the default perann10k/consensus512 U-Net OOF blend, YOLO11s, and YOLO11m), not the public-U-Net/YOLO11s method above. It scored PQ 0.398480 on the same OOF partition and is not comparable to the 0.459116 two-family result; it is not promoted.
- The local native-resolution U-Net has begun fold 1 epoch 10/24, step 50/452, at mean training loss 0.40982. Aggregate OOF scoring is still pending.


## 2026-11-11 23:44 UTC native U-Net progress

- The grouped native-resolution U-Net completed fold 1 epoch 11/24 in 564.232 seconds at mean training loss 0.32768. The RTX 5070 remains under training load (82% utilization, 7,413/8,151 MiB, 69.4 C). This is training progress only; official OOF metrics remain pending all five folds.


## 2026-11-11 23:54 UTC native U-Net progress

- Fold 1 completed epoch 12/24 in 564.237 seconds at mean training loss 0.28625 and began epoch 13 on the RTX 5070. GPU utilization is 100% at 7,413/8,151 MiB and 69.8 C; the laptop remains on AC at 79%. This is training progress only; OOF scoring awaits all five folds.


## 2026-11-12 00:00 UTC native U-Net progress

- The live grouped OOF run is on fold 1, epoch 13/24, step 350/452, with mean loss 0.25846. This is training progress only; five-fold OOF scoring is pending.


## 2026-11-12 00:02 UTC native U-Net progress

- Fold 1 completed epoch 13/24 in 563.708 seconds at mean training loss 0.25675. The run remains on the RTX 5070; this is training progress, not an OOF result.


## 2026-11-12 00:12 UTC submission RLE revalidation

- Revalidated the five Nov 12 batch CSVs against all 180 supplied test images. Their row counts are 1,106, 1,098, 1,084, 1,074, and 1,057 (5,419 masks total); every RLE decoded to 2048x2048 and re-encoded exactly, all IDs are unique, masks are nonempty, and there are zero inter-instance overlap pixels. Each file processes all 180 images; four omit predictions for one image. Hash-locked batch dry-run passes with the frozen split SHA-256. No submission has been uploaded yet.


## 2026-11-12 00:12 UTC native U-Net progress

- Fold 1 completed epoch 14/24 in 563.323 seconds at mean training loss 0.23543. The run remains on the local RTX 5070; OOF scoring is still pending all five folds.


## 2026-11-12 00:21 UTC native U-Net progress

- Fold 1 completed epoch 15/24 in 565.250 seconds at mean training loss 0.21937. This is training progress only; official OOF metrics remain pending all five folds.


## 2026-11-12 00:28 UTC native U-Net progress

- The live grouped OOF run reached fold 1, epoch 16/24, step 350/452 at mean training loss 0.20952. This is training progress only; OOF scoring is pending all folds.


## 2026-11-12 00:30 UTC native U-Net progress

- Fold 1 completed epoch 16/24 in 563.722 seconds at mean training loss 0.20675. This is training progress only; the grouped OOF score is pending all five folds.


## 2026-11-12 00:40 UTC native U-Net progress

- Fold 1 completed epoch 17/24 in 563.996 seconds at mean training loss 0.19888. OOF scoring remains pending completion of all five folds.


## 2026-11-12 00:49 UTC native U-Net progress

- Fold 1 completed epoch 18/24 in 564.250 seconds at mean training loss 0.19119. The run remains live; no aggregate OOF metric is available yet.
