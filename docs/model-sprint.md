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
- Dariush Afshar's public U-Net report describes grouped five-fold training, soft consensus masks, 512-pixel full-resolution crops, and native-resolution inference, with reported 0.4502 nested OOF PQ. The checkpoint pack is CC BY-NC-SA 4.0 and is now eligible under the user's updated policy. The 142 frozen-holdout observations span all five published folds (32/30/22/30/28). Per-observation cross-fitted scoring is possible with the matching held-out fold checkpoints, but other holdout observations appear in those models' training sets; treat the resulting metric as crossfit, not an independent holdout. Sources: <https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39> and <https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights>.
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
- Historical note: the public 10-model U-Net checkpoint set was initially excluded under an earlier user instruction because it is CC BY-NC-SA 4.0. That restriction was superseded on 2026-11-11. It supplies the test candidate and per-observation fold-crossfit evidence on the frozen holdout subset; that crossfit metric is not an independent holdout because peer holdout groups occur in the other folds' training sets. Kaggle dataset and notebook: https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights and https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39.

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
- The public CC BY-NC-SA 4.0 ten-model U-Net pack may now be evaluated under its stated terms. Its author reports 0.4502 nested OOF PQ and 0.39 public score, and says it used only competition-provided training labels. The published fold map covers all 707 competition observations; our 142 fixed-holdout observations distribute across folds 0-4 as 32/30/22/30/28. No single checkpoint excludes the full holdout. We can evaluate each target observation with its matching held-out fold checkpoints, but the resulting 142-sample crossfit is not an independent frozen-holdout estimate because peers occur in the other folds' training data. The author's OOF result remains separate evidence for a test-only candidate.

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
- Added `scripts/evaluate_public_unet_fold_aware.py`. For each frozen holdout observation it loads only the ResNet34 and EfficientNetV2-S checkpoints for the source fold that held that observation out. All 707 stems map exactly once in the source fold file; the 142 holdout observations map as 32/30/22/30/28 across folds. The seed-42 split digest and upstream keep rule are unchanged, and all 232 annotator views are scored. Each target observation is excluded from the two checkpoints used for its prediction; however, the source fold models were trained on other frozen-holdout groups. No holdout checkpoint selection or threshold tuning was used.
- Official PQ is 0.447799 (1,099 TP, 564 FP, 612 FN), versus 0.434930 for the selector incumbent. The paired bootstrap delta is +0.012869, 95% CI [-0.000142, +0.026097], P(improvement)=0.97405 over 20,000 observation-group resamples. The interval narrowly includes zero. Interpret 0.447799 as a per-observation crossfit score over the frozen-holdout sample, not independent evidence: fold models used some peer holdout groups for training. The paired interval crosses zero and is not a promotion gate; the supplied test candidate uses all ten models.
- The full ten-model test CSV was already RLE-validated for all 180 supplied images and remains slot 5/5 in the Nov 12 UTC batch. Its batch entry records the per-observation crossfit evidence separately from independent holdout scores. Local inference used the RTX 5070 Laptop GPU for 194.75 seconds at 3,405,122,048 bytes PyTorch peak allocation. Run outputs, per-observation results, and fixed overlays remain ignored under `artifacts/model-experiments/dariushafshar-unet-fold-aware-holdout-20261111/`; run metadata records all weight hashes, the fold-manifest SHA-256, scorer, packages, and hardware. No public MAGFiLO test labels were used.

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

- Promoted the OOF-selected confidence-0.25 U-Net + YOLO11s family vote to Nov 12 slot 5. It scored OOF PQ 0.459116 and holdout PQ 0.454668, with a paired holdout gain of +0.019738 versus the incumbent (95% CI [+0.007713, +0.031900], P(improvement)=0.9995). The confidence-0.35 vote (OOF 0.455022, holdout 0.457392) remains scheduled for Nov 13 as a complementary preselected candidate; the standalone 10-model U-Net (142-group crossfit PQ 0.447799; not independent holdout) shifts to Nov 13 slot 4. Both five-candidate manifests pass hash and split preflight. The Kaggle submission listing has no family-vote upload yet.
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


## 2026-11-12 02:00 UTC submissions, leaderboard, and local OOF status

- The daily checkpoint was pushed as `8a61f528bcafb1632a81f127fb3df6ef57c845a6` (`feat: add public model and external data evaluations`). The verification report passed with a clean worktree, five successful uploads, and matching source, remote, manifest, and frozen split digests.
- All five public-U-Net/YOLO11s vote submissions completed at public PQ 0.39: refs 58256453 (confidence 0.15; OOF PQ 0.457062), 58256458 (0.20; 0.458104), 58256468 (0.25; 0.459116; holdout 0.454668), 58256478 (0.30; 0.458100), and 58256483 (0.35; 0.455022; holdout 0.457392). The public score ties at Kaggle's displayed precision, so it does not distinguish these thresholds; retain the frozen-split evidence as the selection basis.
- The 21:00 Toronto leaderboard snapshot contains 1,499 teams. EquiNexus is rank 321 at 0.39 after 30 submissions, up from rank 481 at 0.37 after 25 in the previous archived snapshot. IONode Online remains first at 0.70. Snapshot: `artifacts/kaggle/leaderboard/20261112020000-utc/filament-segmentation-2026.zip`, SHA-256 `c4084df592d8dc6d7ade4c97ea3d4279e7c12fc5708f675692aaa8648130b0c9`. Treat rank and public score as secondary because of the documented empty-output and test-label leakage issues.
- The Kaggle T4 validation was safely skipped because GPU quota was 0.00 hours, refreshing at 2026-11-14 00:00 UTC. The local native-resolution grouped U-Net completed fold 1 and is training fold 2, epoch 2/24, step 100/452, at mean loss 0.68320; aggregate OOF scoring remains pending.


## 2026-11-12 02:16 UTC public submission 58067341 source audit

- Pulled LadyFaye's [reproduction notebook](https://www.kaggle.com/code/ladyfaye/solar-filament-reproduction-58067341) and its source bundle pinned to GitHub commit `07871ef5b04b8e1d562005ea00f845f95af4a2e5`. The notebook SHA-256 is `0803da49b25090beaf87398e739282a4c980df1c793dbe1f1a0b991d47a26185`; the downloaded source archive SHA-256 is `bf4882312cc55adee1b4bd822e1d8db4acd1f90fe9b12c8027be338b6337d42`. Both exports are ignored under `data/reference/public-kernels/`.
- The upstream release is AGPL-3.0 and packages eight detectors (seven YOLOv8-seg and one Mask R-CNN) plus three U-Net refiners. Its hash-locked replay receipt reports that the 11 published weights reproduce submission 58067341 byte-for-byte: 1,270 disjoint instances, CSV SHA-256 `83139e60ad3392fbdf4c06e1a287fdfe78c79dc23d7c31114934f97c3119bbff`, public PQ 0.39. This is upstream evidence; we did not download the weights, run the replay, or submit its CSV.
- The release documents both full-707 training and a 565/132 date-purged partition. The latter's training IDs overlap 109 of our 142 frozen holdout groups; its 132 validation IDs overlap 30 holdout and 102 frozen-training groups. Therefore the released full ensemble cannot be scored against our frozen holdout. Some individual protocols explicitly record competition-training annotations and test-image exclusion; the full-neighbor model lists 2,683 added neighbor images across 1,649 frames, but its label-generation and exclusion path is not established well enough for us to rely on it. We did not download or use the weights. No candidate or validation metric was added from this source.


## 2026-11-12 02:24 UTC public segmentation characterization notebook audit

- Pulled `deepmalyakoley/solar-filament-segmentation-characterization` to ignored `data/reference/public-kernels/deepmalyakoley-characterization-20261110/`; notebook SHA-256 `124e47457775935037005f044499a982a20db860b84d7ce60231b73c48c3e416`. It is an EDA and rasterization notebook with no embedded outputs, trained model, or measured submission score.
- The notebook proposes 512x512 patches at stride 384, positive-tile filtering, and overlap-averaged reconstruction. Those are ideas only; they were not validated here and no code was reused. Its validation cell applies `StratifiedGroupKFold` with COCO `image_id` as the group. The official training JSON has 1,154 annotator image records for 707 distinct filenames; 296 filenames have multiple image IDs, up to three. Therefore annotator views of the same physical observation can cross folds, and the code's image-ID leakage check does not prove physical-observation separation. Keep the frozen filename-grouped split for all model selection.


## 2026-11-12 02:26 UTC Nov 13 and Nov 14 batch preflight

- Refreshed both ignored batch manifests to current source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6` and ran the hash-locked dry-run. Both five-candidate batches passed expected columns, candidate hashes, ID/count checks, 180-image coverage declarations, and frozen split SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- Nov 13 candidate row counts are 1,041, 1,022, 1,017, 1,177, and 990. Nov 14 counts are 981, 1,529, 896, 1,436, and 1,234. Their exact RLE round-trip reports are recorded above for the external YOLO26x and CAGE outputs and for the selected component-vote CSVs. The daily checkpoints will refresh the respective manifest source commit again after pushing their documentation commits and rerun the preflight before upload.


## 2026-11-12 02:54 UTC updated scorer and public-model audit

- Pulled the current organizer self-evaluation notebook. Its 11 cell sources are identical to the previously downloaded scorer (source SHA-256 `c6c029a1f955f9867668146bf3842d5b86ba2c37e33cdf98697782a984c147a1`). The frozen scorer contract remains current: every pair with IoU > 0.5 contributes a TP, unmatched predictions and GT contribute FP/FN, predictions are reused for each annotator record, and many-to-one diagnostics use IoU > 0.
- Audited the public I079 Mask R-CNN + U-Net refiner handoff and its model card. The reproducible inference path is test-only, verifies checkpoint MD5s, and does not train or read test labels. Its version-1 report verifies 180 test images and CSV MD5 `09c0eae5fdff29a91f8a740475ecaf36`, with 1,299 rows; its own report says predictions cover 169 images and leave 11 with no predicted instances. The author/model-card text cites two historical 0.70 submissions, while the currently indexed version-1 notebook shows 0.34 and the competition notebook listing has shown 0.70. This score conflict is unresolved, and the model was trained on all competition training data, so it cannot be evaluated fairly on our frozen holdout. Do not promote or submit without a comparable validation basis.
- The source Mask R-CNN + U-Net notebook and I079 handoff declare Apache 2.0. The model card says its weights use only competition training data and no test-overlap labels; the user has authorized non-commercial research/education weights. The weights were not downloaded or used.
- Audited the CAGE r5 weights dataset, licensed CC BY-NC 4.0. Its model card reports PQ 0.405 on a 100-image split with multi-annotator duplicates crossing train/holdout, and no leaderboard score. The metadata and ignored 181 MB package are under `data/reference/public-datasets/`; SHA-256 sums match the included manifest. It is not a valid frozen-holdout candidate.
- Audited the new 1888-resolution U-Net notebook: a centered 1888x1888 crop, ImageNet EfficientViT-B2 encoder, 24 epochs, and a 7.5% file-name validation split. It unions all annotator masks and chooses post-processing against that union, rather than scoring each annotator with our official scorer. Treat its architecture and normalization as references only; no result or model is adopted.
- The native grouped U-Net process remains live at fold 2, epoch 7, step 350/452, mean training loss 0.44598. Aggregate OOF scoring is still pending. No candidate was added or submitted from this audit.


## 2026-11-12 03:17 UTC submission access and native U-Net status

- The Nov 12 five-candidate batch completed with five successful uploads at displayed public PQ 0.39; submission references and scores are recorded in the 02:00 UTC entry. A current read-only Kaggle submissions query now returns `Authentication required`. No OAuth flow was started while the user is asleep.
- Re-ran the local no-upload preflight for the Nov 13 five-candidate batch. All five files pass manifest and submission validation with source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6` and frozen split SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; its scheduled upload is Thu Nov 12 at 20:20 Toronto time and needs Kaggle authentication restored.
- The native grouped U-Net remains live on the RTX 5070 at fold 2, epoch 9/24, mean training loss 0.30753, with epoch runtime 564.914 seconds. This is training progress only; aggregate OOF metrics are pending.
- GitHub read access is working and local `main` matches `origin/main` at `8a61f528bcafb1632a81f127fb3df6ef57c845a6`. The daily documentation checkpoint remains scheduled for Thu Nov 12 at 20:05 Toronto time.


## 2026-11-12 03:22 UTC high-score notebook source audit

- Reviewed L?m Huy's current Kaggle notebook, version 35, which displays public PQ 0.55. The public source-only release `ranjeet258/solar-filament-consensus-2026` reports that the earlier 0.55 reference result came from an embedded prediction CSV from this notebook; its generating weights are not provided there. That release separately records its own model-only submission at public PQ 0.39 (Kaggle reference 57745515). Treat the 0.55 figure as unverified reference-output evidence, not the reproducible score of the notebook's independently trained U-Net. Sources: [Kaggle notebook](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation), [public provenance report](https://github.com/ranjeet258/solar-filament-consensus-2026).
- The separate consensus source describes three U-Nets, YOLO11m, and a learned cross-model selector, while explicitly calling its four-fold meta-selection estimate non-nested. Its reported 0.55 variants also include the reference CSV. We did not download or use its CSV, weights, model-output cache, or code; no submission candidate or validation metric was added from this audit.


## 2026-11-12 03:37 UTC public leaderboard empty-output audit

- The publicly indexed Kaggle notebook list includes a notebook titled [Please fix the Leaderboard](https://www.kaggle.com/code/artkomissar/please-fix-the-leaderboard), whose report says five empty masks received public LB 1.00. A separate [Kaggle product-feedback report](https://www.kaggle.com/discussions/product-feedback/729163) says PPP8-only empty-mask submissions received about 0.90-0.93.
- In [competition discussion 729809](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809), a participant reports controlled post-processing runs with the same model: empty output scored 0.93, 1.9 instances per image scored 0.72, and 6.4 scored 0.67. The discussion includes the host's direction to use the updated evaluation, self-evaluation, and leaderboard-ranking sections; it does not establish that the public board is a valid segmentation-quality signal.
- Treat public scores and rank as secondary diagnostics. Do not use empty-mask outputs or this leaderboard behavior for selection; retain the official frozen grouped PQ scorer and genuine filament predictions.


## 2026-11-12 03:39 UTC native U-Net progress

- The grouped native-resolution OOF run is live at fold 2, epoch 12/24, step 250/452 on the RTX 5070. GPU memory is 7,773/8,151 MiB at 72 C. This is training progress only; all-fold OOF scoring remains pending.
- Kaggle CLI read-only access still returns `Authentication required`. No OAuth flow was started while the user is asleep. The next prepared batch remains local and its scheduled upload will require restored authentication.


## 2026-11-12 03:42 UTC native U-Net epoch update

- Fold 2 completed epoch 12/24 in 564.153 seconds at mean training loss 0.21235. The process remains live on the RTX 5070; grouped OOF evaluation still awaits all five folds. This loss is not used to select the training duration.


## 2026-11-12 03:45 UTC recent public notebook audit

- The recent public listing for [solar filament segmentation improved pipeline](https://www.kaggle.com/code/sivapnlr80/solar-filament-segmentation-improved-pipeline) shows version 44, public score 0.36, a 54-minute T4x2 run, a `Private Dataset` input, and a dependent `solar filament train refiner` notebook. The private dependency prevents a public-data replay or provenance audit, so this score and source were not adopted.
- The [MAGFiLO V9 notebook](https://www.kaggle.com/code/muhammadshoaibaltaf/magfilo-v9-evaluation-aligned-solar-filament-s) lists public score 0.36 and a 5h43 T4x2 run, with hundreds of run outputs. The publicly indexed page does not establish its training provenance or provide a frozen-group OOF comparison. We did not adopt its artifacts.
- Both public scores are below our displayed 0.39 submissions, and neither provides comparable evidence to change selection. Preserve the frozen OOF and scorer as the selection basis.


## 2026-11-12 03:47 UTC leaderboard and submission refresh

- Kaggle CLI access is working again. Downloaded the complete public leaderboard with 1,498 teams to `artifacts/kaggle/leaderboard/20261112-034700-utc/filament-segmentation-2026.zip` (SHA-256 `ecf4f5c920d1e41bac501fc94207683ecac1dd6a21769b0ecbd2befe4fa3e8cd`) and wrote a parsed snapshot summary beside it.
- EquiNexus is rank 321/1,498 at displayed public score 0.39 with 30 submissions. The leaderboard leaders are IONode Online at 0.70 and Tempest at 0.64; the complete top 20 and team roster are in the ignored snapshot summary.
- Refreshed the submissions API: all five latest refs (58256453, 58256458, 58256468, 58256478, 58256483) are complete and display public score 0.39. Their OOF scores remain the basis for threshold selection because the public board does not distinguish them.


## 2026-11-12 03:55 UTC public peer experiment audit

- Pulled the public MIT-licensed [HeShen-1 research repository](https://github.com/HeShen-1/filament-segmentation-2026) to `data/reference/public-kernels/heshen-filament-segmentation-20261112/`, pinned at commit `b41dae215a87c6cb6dfbcf1459a4819021e0092d`. It includes code and reports, but no competition data, checkpoints, or submissions.
- Its reported paired five-fold YOLO11m comparison found the 100-epoch recipe scored 0.3581 mean PQ versus 0.3593 for the 50-epoch recipe (delta -0.0012), despite a +0.0065 gain on fold 0. This is the author's result, not an independent reproduction; it supports selecting duration from multi-fold validation rather than a single fold or training loss.
- Its multi-scale detector TTA sweep also fell from 0.3661 without TTA to 0.3552 with two views and 0.3484 with three views at the reported operating point. Do not assume more views help.
- Pulled Fususu's public Apache-2.0 [topology-safe inference handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) to `data/reference/public-kernels/phuongncn-i079-handoff-20261112/` (notebook SHA-256 `b59aea47798e2ccbaa6f14f0e3bd7505dddec20ceb7d4c37704521bbde02e49c`). Its four-flip crop-refiner TTA accepts the average only when area ratio stays in [0.60, 1.70], component count does not increase beyond `max(single + 1, 2)`, and mask Dice is at least 0.62 for control areas >=32; otherwise it falls back to the single view. This is a candidate ablation for our own OOF maps, not an adopted result.
- The handoff's Kaggle page displays 0.34 while its markdown claims two 0.70 submissions. Its weights were trained on the full competition training set, so they cannot provide leakage-free frozen-holdout evidence. We did not download or use the weights or submission CSV.


## 2026-11-12 04:10 UTC prepared-batch and TTA audit

- Re-ran the hash-locked dry-run for the Nov 13 and Nov 14 Toronto-time batches. Both contain five valid candidates, expected submission columns, complete 180-image coverage declarations, matching file hashes, and the frozen split SHA-256 `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; no upload occurred.
- Added an explicit selection-bias caveat to the native-U-Net TTA guard report: its epoch and post-processing settings were selected on the same cross-fitted training observations, so this comparison is exploratory until checked on the independent frozen holdout. Python AST parsing and the scheduled PowerShell checkpoint-script parse both pass.
- The GONG auxiliary OOF task remains scheduled for Nov 12 at 09:15 Toronto time and waits for the live native U-Net OOF run to finish before starting.


## 2026-11-12 04:11 UTC native U-Net epoch update

- Fold 2 completed epoch 15/24 in 542.699 seconds at mean training loss 0.18276. The training session remains live; no aggregate OOF metric is available yet. Training loss is not used to choose duration.


## 2026-11-12 04:12 UTC experiment sequencing

- Updated the ignored GONG orchestrator to run the native-U-Net single-view versus TTA8 topology-guard ablation after the native OOF summary is complete and before GONG OOF begins. This serializes GPU work; an optional TTA failure is logged and does not prevent the GONG experiment.
- PowerShell parsing passes for the updated GONG orchestrator and daily checkpoint script. The current U-Net run remains live; no OOF validation result is available yet.


## 2026-11-12 04:15 UTC leaderboard refresh

- Downloaded the full 1,498-team public leaderboard to `artifacts/kaggle/leaderboard/20261112-0414-utc/filament-segmentation-2026.zip` (SHA-256 `6233fa4f377dfc2b2234db27946d70bb7553133d22af92b7a08f27ccb4e6cf82`) and saved the parsed top 20 and EquiNexus record beside it.
- EquiNexus remains rank 321 at displayed score 0.39 after 30 submissions. IONode Online leads at 0.70, Tempest is 0.64, and the next three entries are 0.62. The public score remains secondary to frozen grouped validation and the organizer's stated evaluation rules.


## 2026-11-12 04:19 UTC native U-Net checkpoint update

- Fold 2 saved its epoch-16 checkpoint after 542.909 seconds at mean training loss 0.17867. Fold 2 is continuing toward epoch 24; the five-fold OOF score sweep is still pending and no training duration has been selected from loss values.
- Verified the cached GONG auxiliary package is present under `data/reference/public-datasets/vhrabar-gong-solar-filaments/`: 504 target files, 396,696,305 bytes, training-only partition, zero validation target files, and the expected frozen split digest. The scheduled GONG run has its required local input.


## 2026-11-12 04:23 UTC historical top-solution report audit

- The public [FELIPEACASTRO technical report](https://github.com/FELIPEACASTRO/solar-filament-2026/blob/master/report/TECHNICAL_REPORT.md) is version 0.3 dated July 20 and reports a 0.71 public score for a Mask R-CNN R50-FPNv2 plus CropUNet pipeline. Its conclusions and 44-submission ablations use the then-described `instance_wise_dice` leaderboard metric.
- Kaggle's [official leaderboard-ranking page](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking) records an Aug 7 change to Panoptic Quality and an Aug 12 leaderboard re-score. The report predates that change, so its 0.71 score and metric-specific effects are not comparable to our current submissions or frozen PQ validation. Preserve our scorer and selection protocol; no model or report code was adopted from this audit.


## 2026-11-12 04:29 UTC native U-Net checkpoint update

- Fold 2 completed epoch 17/24 in 542.432 seconds at mean training loss 0.17217. The grouped OOF run remains live; aggregate epoch comparisons are not available until the remaining folds complete, and loss is not used to select duration.


## 2026-11-12 04:34 UTC annotation-spine audit

- Inspected the competition training JSON's per-filament `spine` field: all 8,199 annotations have a valid even-length coordinate list, totaling 102,289 points. Polylines contain 2–122 points (median 10, p95 29), all within the 2048-pixel image canvas; 73 extend beyond their annotation bbox by more than 2 pixels.
- These labels could support a centerline auxiliary loss or improve morphology diagnostics. No target rasterization or model change was made; evaluate alignment to segmentation masks and a grouped-validation ablation only after the current native OOF and scheduled GONG comparison free the GPU.


## 2026-11-12 04:48 UTC annotation-spine diagnostic (superseded before training)

- A second seeded 150-annotation alignment check (1,871 points) found 93.48% of spine points inside their instance mask, 99.52% within one pixel, and all within three pixels. This remains a morphology diagnostic only; no spine-supervised training run was started.
- The official [competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements) identifies spines as extra ground-truth metadata and says training must focus on masks rather than other ground-truth metadata. We therefore removed spine targets, the spine head, and spine-based promotion fields from the GONG experiment before its scheduled run. The corrected candidate uses mask supervision plus public GONG proxy targets only.
- The single daily commit-and-push checkpoint is scheduled for 23:45 Toronto time. The GONG orchestrator records the run-start commit, dirty paths, and trainer SHA-256 before waiting for native OOF; the committed source hash can be reconciled after the daily push.


## 2026-11-12 04:50 UTC training and access check

- The native U-Net completed fold 2 epoch 19/24 and is partway through epoch 20. Grouped OOF comparisons for saved epochs 16, 20, and 24 remain pending; falling training loss alone will not justify longer runs.
- Kaggle CLI authentication is active. The latest five YOLO11s vote submissions are complete at public score 0.39; public scores remain secondary to grouped validation.
- Scheduled the end-of-day source checkpoint for 23:45 Toronto time and the gated GONG auxiliary run for 09:15. The corrected GONG trainer compiles and its CLI exposes OOF, holdout, and final-fit modes.


## 2026-11-12 04:56 UTC leaderboard and public-method review

- Refreshed the complete Kaggle leaderboard at 04:55:57 UTC: 1,498 teams; EquiNexus is rank 321 at displayed score 0.39 after 30 submissions, with the expected three-account roster. IONode Online leads at 0.70, Tempest is 0.64, and the next group is at 0.62. The full snapshot and parsed top 20 are under ignored `artifacts/kaggle/leaderboard/20261112-0454-utc/`; ZIP SHA-256 `779245a98aafe871bdde0c729a1a2157f52debcd8216ddf68af219281c6a868d`.
- Kaggle's current public-code list includes notebooks that still describe the superseded Dice metric and report empty-output scores. The host says the leaderboard is only a preliminary filter; keep selection on our grouped PQ and visual quality ([official evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)).
- Reviewed the public [zlyx2377 spine-guided watershed repository](https://github.com/zlyx2377/filament-segmentation-2026). Its README also describes stale Dice scoring and suggests checking public test labels; no license file or independently verifiable official-PQ result is available, so its code, checkpoint, and score are excluded. The spine-head idea is also excluded because the organizer prohibits training on non-mask ground-truth metadata.


## 2026-11-12 05:26 UTC native U-Net duration and promotion gate

- The native grouped run completed all 24 training epochs and OOF prediction maps for fold 2; fold 3 training has started. Completed folds continue to show lower training loss through epoch 24, but training loss does not determine duration; compare grouped OOF PQ at 16/20/24 before deciding on an explicit longer schedule.
- The gated GONG auxiliary run is set to 20 epochs, comparing OOF checkpoints 16 and 20. This leaves time after its grouped evaluation for both holdout fits and, if it strictly improves OOF and holdout, the all-707 final fit and test inference before the challenge cutoff; the native baseline run remains at 24 epochs.
- Added a promotion-gated final-fit mode for GONG auxiliary: it must strictly beat native U-Net on both OOF and the frozen holdout before fitting all 707 training observations and predicting supplied test H-alpha images. The orchestrator will replace a Friday test-only candidate only after strict improvement and full RLE validation.
- Python compilation/CLI, PowerShell parsing, GONG-manifest subset checks, supplied-image input routing, synthetic 2048x2048 RLE decoding, and `git diff --check` pass. The 23:45 Toronto time daily commit task and 09:15 gated experiment remain scheduled.
- An in-memory check of the Friday batch promotion preserves five candidates and replaces only the test-only CAGE entry; the on-disk manifest was not changed.


## 2026-11-12 05:44 UTC annotation policy correction and epoch plan

- Rechecked the organizer's competition overview: the extra spine field is ground-truth metadata, and training is restricted to the mask target. No spine-supervised training was run; the previously prepared head was removed before the 09:15 task. The GONG auxiliary uses public magnetogram-derived proxy targets and does not consume the competition's spine field.
- Keep the native 24-epoch run intact. Its epoch-16, epoch-20, and epoch-24 OOF maps provide the relevant comparison. If PQ and instance diagnostics still improve at epoch 24, longer training is worth a fresh, explicitly scheduled run, but only with per-epoch OOF evidence and overfit checks. Training loss alone is not enough, and extending the current run is not possible from its weights because its checkpoint omits optimizer state and the cosine schedule reaches zero at epoch 24.
- User is asleep/unavailable; continue unattended work without auth prompts or questions. Current native grouped OOF process remains live on fold 3.


## 2026-11-12 06:06 UTC submission and commit sequencing

- The Nov 13 batch submits at 20:20 Toronto time, before the 23:45 daily push. Its wrapper now permits only the current experiment files to be dirty while checking that local `main`, remote `main`, and the hash-locked manifest use the same source commit. All five candidate files pass the dry-run preflight; no upload occurred.


## 2026-11-12 06:10 UTC native U-Net progress

- The live grouped OOF run reached fold 3, epoch 4/24, step 200/452. Its partial training loss is not a validation score; grouped comparisons at epochs 16, 20, and 24 will decide whether a longer run is warranted. The RTX 5070 remains on AC at 98% utilization and 72.5°C.

## 2026-11-12 07:09 UTC submission sequencing correction

- The inner submission command requires a clean worktree, so the earlier 20:20 Toronto upload time conflicted with the 23:45 daily commit/push despite the outer wrapper allowing only known experiment files. Kept the single daily checkpoint at 23:45 and moved the five-candidate upload to 23:58, after the commit updates the ignored manifest to the pushed source commit. Both scheduled tasks were verified Ready with those start times; the hash-locked dry run passes and no upload occurred.

## 2026-11-12 07:13 UTC leaderboard refresh sequencing

- Moved the Nov 13 status refresh to 00:20 Toronto, after the 23:58 submission batch, so the refresh captures those five submissions and their public scores. The scheduled task is Ready at the verified time.

## 2026-11-12 08:43 UTC longer-training decision

- The native U-Net's partial grouped OOF comparison improves at epochs 16, 20, and 24 on all three completed folds. Epoch 20 to 24 adds 0.005452 PQ (paired observation bootstrap 95% interval 0.000481 to 0.010429). This supports testing longer schedules, but does not establish that training beyond 24 epochs will help.
- Extended the queued GONG-auxiliary OOF, holdout, and final-fit schedule from 20 to 24 epochs. The trainer snapshots epochs 16, 20, and 24, so model choice remains based on grouped OOF PQ; the separate frozen holdout remains a promotion gate. The scheduled run waits for native OOF and uses the same single local GPU.

## 2026-11-12 08:49 UTC public-consensus compliance finding

- Excluded the public `prvsiyan` consensus CSV after auditing its upstream notebooks: two source pipelines train with the competition's extra `spine` ground-truth metadata. The consensus uses their masks in both agreement voting and final selection. The host explicitly limits training to the supplied segmentation masks; it disallows other ground-truth metadata. We downloaded the notebook and CSV for audit only; neither was used in a candidate or submission.

## 2026-11-12 09:10 UTC submission-candidate update

- Replaced the final batch's unverified YOLO26x model with a new U-Net + fold-trained YOLO11s confidence-0.50 candidate. It scores 0.431139 on the frozen OOF split and passes full test RLE validation; its OOF gain over the weakest batch candidate was not paired-bootstrap tested. The batch is now four validation-backed variants plus the competition-training-only CAGE r5 model for test inference. No upload yet.
- The latest public leaderboard is unchanged: EquiNexus rank 321 at 0.39 after 30 submissions.


## 2026-11-12 09:53 UTC final-window public U-Net gate

- Scheduled `EquiNexus_Public_Unet_Weights_20261113` for Friday 14:00 Toronto. It waits for local grouped/GONG GPU work to finish, then runs the ten-model eight-view test inference only if the GONG model has not already passed both OOF and frozen-holdout promotion gates.
- The task validates the full 180-image RLE output and may replace only the remaining CAGE test-only slot. It has a 90-minute GPU-busy wait limit and an 18:45 hard stop, leaving time for the 19:05 documentation checkpoint, 19:10 Kaggle comparison, 19:20 five-submission batch, and 20:00 sprint cutoff. Task Scheduler confirms it is Ready, wakes the laptop, and refuses to run on battery.


## 2026-11-12 09:48 UTC public U-Net candidate preparation

- Downloaded the public ten-checkpoint U-Net dataset to the ignored `data/reference/public-datasets/dariushafshar-solar-filament-unet-weights/` folder (939,216,137 extracted bytes). All ten checkpoint sizes and SHA-256 hashes match the dataset receipt. Its README documents mask-only training on the competition's 707 training observations; license CC BY-NC-SA 4.0 is allowed by the user's non-commercial-use clarification.
- The dataset reports nested OOF PQ 0.450163 for the box postprocess, public 0.39. Its 0.455236 learned-filter result uses a postprocess threshold calibrated with a sparsity estimate inferred from the public leaderboard; prefer the nested-OOF box rule for this candidate. These are source-reported metrics, not EquiNexus frozen-holdout results.
- Added `scripts/run_public_unet_weights_inference.py` for supplied-test-only inference, with checkpoint receipt verification, eight D4 views, the source's nested-OOF box rule, exact RLE round trips, run metadata, and model-level resume checkpoints. It explicitly declines to score the frozen holdout because the ten public fold models collectively train on all 707 observations.
- Updated the Thursday daily checkpoint allowlist and description to include the public U-Net runner. Its Conventional Commit summary is `feat: add GONG and public U-Net pipelines`; the script still stages only the five expected source and documentation paths.
- Audited the displayed-0.55 Lâm Huy notebook: it emits a compressed, embedded table of fixed RLE predictions rather than image-conditioned test inference. Exclude it and its payload under the host's inference requirements; no stored prediction rows were used.
- Smoked the new runner on one supplied test image using all five ResNet34 fold checkpoints, one TTA view, and CPU inference (13.2 seconds). The official RLE validator passed both masks with exact 2048x2048 round trips and zero overlap; the output is marked partial and is not a candidate. Both encoder families load the published state dictionaries successfully.

## 2026-11-12 10:25 UTC I079 public model audit

- Audited the public I079 topology-safe handoff notebook (https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) and its attached model card (https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights). The notebook uses only the 180 supplied H-alpha test images during inference and applies four-view refiner TTA. The model-card usage field says its detector and refiner were trained on the competition training images and annotations only; the model and notebook declare Apache 2.0.
- Downloaded the two checkpoints into ignored data/reference/public-models/phuongncn-solar-filament-i079-weights/. Their MD5 hashes match the notebook receipt: detector dcd3053561e40ba78718546c31525d26, refiner 74991b8d95f7922f0ca6fd75f8ab4f5f. No annotation or test-label file was downloaded or read.
- The handoff claims a 0.70 public score, but Kaggle's search/listing currently shows inconsistent score fields for this notebook. Treat its reported score as unverified, make no frozen-holdout claim because the training-fold membership is undocumented, and keep this as a test-only candidate pending our own image-conditioned run and RLE validation.
- Added scripts/run_i079_topology_safe_inference.py, adapting the published inference pipeline for workspace and Kaggle paths. The runner records model/source hashes, runtime, device, peak GPU memory, expected-output hash comparison, and full test RLE validation. It will replace only the CAGE test-only slot if the GONG candidate does not pass both frozen OOF and holdout gates.
- The run remains scheduled after the current local GPU experiments; no I079 inference or submission has been run yet.
## 2026-11-12 10:30 UTC I079 task and checkpoint update

- Registered EquiNexus_I079_Candidate_20261113 for Friday 14:00 Toronto with the same AC-only, wake-on-sleep, five-hour limit as the superseded public-U-Net task. It waits up to 90 minutes for native/GONG GPU work, then validates the I079 run and replaces only the CAGE test-only slot if GONG has not passed both gates.
- Updated Thursday's exact-path checkpoint allowlist to include the I079 inference runner. The scheduled Conventional Commit is now feat: add GONG and external inference pipelines.
- Native grouped OOF remains live on the RTX 5070; fold 5 completed epoch 7/24 and is training epoch 8. Training loss is not used to select the schedule.
- The local CPU smoke strictly loaded both checkpoints with the declared tensor counts (432 detector tensors, 182 refiner tensors), verified their MD5s, and discovered exactly the supplied 180 test images. This checked loading and input discovery only; no predictions or submission were generated.
## 2026-11-12 10:40 UTC sprint cutoff protection

- The queued GONG task previously allowed a 36-hour run, which could continue past the Friday 20:00 sprint cutoff. Capped it at 34 hours and added a 19:10 Friday hard stop that terminates any active Python stage; the task ends before the 19:20 submission batch.
- Extended the I079 job's GPU-busy wait to three hours. If GONG still owns the GPU at 17:00 Friday, the I079 job skips; if the GPU is free and GONG has not promoted a candidate, it can run before its 18:45 hard stop.
- PowerShell parsing, Python compilation, and Git whitespace checks pass. Both scheduled jobs are Ready, wake the laptop, and require AC power.
## 2026-11-12 10:48 UTC leaderboard refresh and native OOF progress

- Refreshed the Kaggle API snapshot at 10:47 UTC. The 1,499-row board lists EquiNexus at rank 321/1,499 with public score 0.39 after 30 submissions. All five Nov 12 submissions (58256453, 58256458, 58256468, 58256478, 58256483) are complete at 0.39. Leaders remain IONode Online 0.70, Tempest 0.64, Sayandeep Dutta 0.62, and Harshit Kulkarni 0.62. Snapshot: `artifacts/kaggle/leaderboard/20261112104703-utc/filament-segmentation-2026.zip`, SHA-256 `43523e4acc24863a3da381da218b1aa51c23a80de3fb568d7e4bda6ef9149e4e`.
- The latest competition-kernel search exposed no new reproducible code from the top leaderboard teams. The newly run ASTH-Net and Swin-UNet references were already source-audited; ASTH-Net reads prohibited spine supervision, while Swin-UNet uses a different, non-comparable validation protocol. The `cracker1war` account has no public competition kernel in the API response.
- The native grouped OOF run remains live on the RTX 5070 at fold 5, epoch 10/24, step 200/452. Training loss is progress telemetry only; selection still depends on official per-annotator OOF PQ at epochs 16, 20, and 24. The GPU is at 99% utilization, 7,773/8,151 MiB, and 75.16 C.

## 2026-11-12 11:00 UTC Kaggle TPU availability probe

- Kaggle quota reports 0.00 GPU hours through the sprint deadline and 20.00 TPU hours. The official Kaggle CLI documents `TpuV5E8` (TPU v5e-8) as a kernel accelerator: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md.
- Pushed private kernel `mrinanksivakumar/equinexus-kaggle-tpu-v5e-smoke-20261112` to run a tiny PyTorch XLA forward/backward check with no competition data. Its notebook and metadata are under ignored `artifacts/kaggle/tpu-smoke-20261112/`. Kaggle reports `KernelWorkerStatus.QUEUED`; TPU quota remains unused. This is an availability probe only, not a validation result.
- If Kaggle allocates the TPU, adapt the existing fold-aware public U-Net evaluator to `torch_xla` and run the same fixed scorer/split there. Keep its per-observation fold-cross-fit metric labeled separately from the strict independent holdout; if no TPU is allocated, the local grouped OOF and queued local GONG run remain the compute path before cutoff.

## 2026-11-12 11:13 UTC public U-Net crossfit interpretation correction

- Re-audited the fold map for the public ten-model U-Net. The 142 observations in EquiNexus's frozen validation subset distribute across upstream folds as 32/30/22/30/28. Each target observation is excluded from its own predictor, but the corresponding fold-model training sets contain 110/112/120/112/114 other frozen-holdout groups.
- Corrected the earlier description: PQ 0.447799 is per-observation five-fold crossfit over the 142 selected holdout observations, not a strict independent frozen-holdout metric. The paired comparison with the strict-holdout incumbent (delta +0.012869; 95% CI [-0.000142, +0.026097]) is exploratory and is not a promotion gate.
- Updated the evaluator metadata, Kaggle notebook description, batch manifest, submission audit fields, and earlier summary references to preserve that distinction. The test CSV is unchanged; its RLE checks and full 180-image coverage remain valid. An ignored correction sidecar preserves the original run-metadata hash and corrected interpretation.

## 2026-11-12 11:18 UTC native OOF and compute status

- The native grouped OOF run is live on the RTX 5070 at fold 5, epoch 13/24, step 250/452. Training loss remains progress telemetry; official OOF PQ at epochs 16/20/24 is the duration gate.
- AC power is online; battery is at 89% and charging. GPU telemetry: 100% utilization, 7,591/8,151 MiB, 77.16 C.
- Kaggle reports 0.00 GPU hours until 2026-11-14 00:00 UTC and 20.00 TPU hours; the private TpuV5E8 availability probe remains queued and has used no TPU quota.

## 2026-11-12 11:24 UTC Kaggle leaderboard and submission refresh

- Refreshed the Kaggle submission list and public leaderboard. All five Nov 12 submissions (58256453, 58256458, 58256468, 58256478, 58256483) are complete at 0.39 public score, despite grouped OOF PQ ranging from 0.455022 to 0.459116. The public score did not distinguish them.
- EquiNexus remains rank 321/1,499 at 0.39 with 30 submissions; leading scores remain 0.70, 0.64, and 0.62. Snapshot SHA-256: `ca1669bcf20bd8929f3877e6190c426d35968141069f6004503c3511f592f59e`.

## 2026-11-12 11:32 UTC public model provenance audit

- Audited new Nov 11 kernel `ladyfaye/solar-filament-reproduction-58067341`, which replays submission 58067341 at public PQ 0.39 using seven YOLOv8l-seg members, a native Mask R-CNN bank, and three crop refiners. Its pinned source commit is `07871ef5b04b8e1d562005ea00f845f95af4a2e5`; source license is AGPL-3.0 and initializers are generic COCO weights. Notebook and source are preserved under ignored `artifacts/kaggle/public-kernels/ladyfaye-20261112/` with SHA-256 values recorded in `experiment-log.md`.
- Its selected 565/132/10 train/validation/excluded-date partition overlaps our frozen 142-observation holdout as 109 upstream training, 30 upstream validation, and 3 outside both. Its 707-photo neighbor model trained on every competition observation. Do not use its released weights for independent frozen-holdout scoring or promotion. Its recorded 1,649-frame GONG neighbor experiment remains a method reference; our fixed split/scorer remains unchanged.
- Native grouped OOF is live at fold 5, epoch 14/24, 452/452 steps; GPU is at 100%, 7,621/8,151 MiB, and 72.56 C. Wait for complete OOF scoring at epochs 16/20/24 before deciding whether a longer native schedule is warranted.
- Our GONG auxiliary targets are independently sourced from the public `vhrabar/gong-solar-filaments` dataset: 504 magnetogram-derived targets matched to frozen training observations, zero downloaded holdout targets, and no MAGFiLO images or labels. They are used only in training supervision; validation and inference use H-alpha. The host explicitly permits external GONG sources subject to these boundaries, and NSO requests a GONG acknowledgement in reports; the approved abbreviated form is in `licenses/THIRD_PARTY_NOTICES.md`.

## 2026-11-12 11:38 UTC public submission-vote notebook audit

- Inspected `prvsiyan/notebook24a4dce69e`, the public notebook previously flagged by the team. It runs an existing detector/refiner and votes over six other participants' public test-prediction CSVs; it trains no model and reports no validation-label score for the current fifth-family vote. The notebook and metadata are saved under ignored `artifacts/kaggle/public-kernels/prvsiyan-20261112/`.
- Treat this as a public-consensus analysis only. It is not a stand-alone model or evidence on the frozen grouped holdout, so it does not enter the promotion or submission pipeline.

## 2026-11-12 11:49 UTC final-day submission and checkpoint ordering

- The Friday five-candidate batch remains scheduled for 19:20 Toronto time. Its wrapper requires clean `main` and a pushed HEAD, refreshes the ignored manifest's `source_commit` to that verified HEAD if needed, validates all five CSVs, then submits.
- Moved the one Friday docs commit/push to 19:50, after the submission batch and before the 20:00 sprint cutoff, so the final commit can include the run and leaderboard record. It remains AC-only and wakes the laptop.

## 2026-11-12 11:52 UTC native epoch-16 checkpoint and final-batch dry run

- The native grouped OOF run saved epoch 16 and is training epoch 17/24 on fold 5. Its launch metadata preserves the exact entrypoint hash, which matches the current source file; the recorded source commit predates tracking that file, so the Thursday checkpoint will archive the exact code used.
- The final five-candidate batch passed dry-run validation on the frozen split and current pushed HEAD. Four entries have grouped OOF scores but no independent holdout score; the CAGE r5 entry is test-only. Submit this batch only after its preflight succeeds, and replace CAGE only if the GONG candidate beats the native model on both grouped OOF and strict holdout PQ.

## 2026-11-12 11:59 UTC Kaggle leaderboard refresh

- Public results are unchanged: EquiNexus rank 321/1,499 at 0.39, all five Nov 12 submissions complete at 0.39, and the six leading teams remain at 0.70/0.64/0.62/0.62/0.62/0.62. Snapshot SHA-256: `6eb1d5231d999b64768e7064612e76ca663e1f8012d485c22a8988ec7e034fda`.

## 2026-11-12 12:04 UTC Kaggle quota and Friday evaluation window

- Kaggle reports 0 GPU hours until Friday 19:00 Toronto and 20 TPU hours, with the TPU smoke still queued. The fold-aware public U-Net GPU evaluation now starts Friday 19:05 on T4, after reset and before the 19:20 batch; treat its crossfit score as exploratory only.
- Keep tracked files unchanged after Thursday's commit through the Friday 19:20 batch preflight; record Friday run and leaderboard evidence afterward for the 19:50 daily commit.

## 2026-11-12 12:12 UTC Nov 13 batch preflight and native OOF progress

- The Nov 13 five-candidate manifest passes dry-run validation against the current pushed source and frozen split. Candidate scores are recorded in `docs/experiment-log.md`; none is an independent frozen-holdout result, and no submission was made.
- Native OOF training is active at fold 5, epoch 19/24, step 200/452. On the partial OOF subset of folds 0-2, epoch 24 beats epoch 20 by +0.005452 PQ (paired 95% CI +0.000481 to +0.010429); this supports finishing the longer schedule but remains provisional until all folds are scored.

## 2026-11-12 12:18 UTC leaderboard and recent public-kernel audit

- Kaggle remains rank 321/1,499 at 0.39; all five Nov 12 submissions are complete at 0.39. The latest top seven include a new 0.61 entry, but no public competition kernel was found under that team's Kaggle username.
- Audited the completed public CS2-style Swin-U-Net run and its released checkpoint. It reports 0.362831 on its own connected-component PQ proxy; no official-scorer result is reported. Its 14-day split trained on 118 of our 142 frozen-holdout observations and its postprocessing sweep used the other 24, so it cannot provide independent fixed-holdout evidence. Its 512-pixel tiling and clDice loss remain unvalidated ideas, not promoted changes. The initial source-only note incorrectly called the run unexecuted; Kaggle CLI outputs and logs confirmed it completed.

## 2026-11-12 12:36 UTC correction: public CS2 Swin-U-Net output and split audit

- The public run completed and released a checkpoint; the Kaggle-pulled notebook source omitted the execution outputs. Its 0.362831 score is a custom proxy on its own split; 118 of our 142 frozen-holdout observations were in its training set and its postprocessing sweep used the other 24. Do not use it for frozen-holdout comparison or promotion; the verified overlap and output hashes are in the ignored audit JSON.

## 2026-11-12 13:23 UTC native U-Net OOF progress

- All five OOF folds have completed training through epoch 24, with probability maps generated at epochs 16, 20, and 24. The official-scorer postprocessing sweep is still running (20/324 settings at last poll); its current best is provisional. Wait for the full sweep and paired crossfit comparison before deciding whether to train longer.

## 2026-11-12 13:35 UTC GONG sweep efficiency update

- The live native OOF official-scorer sweep reached 50/324 settings; its provisional best is PQ 0.381859 at epoch 16. The process remains active, with no completed summary yet.
- Updated the pending GONG OOF candidate to center its postprocessing sweep on the native OOF-selected configuration. It evaluates five nearby thresholds, the selected and adjacent minimum-area values, and the selected and adjacent closing kernels, for up to 45 settings per epoch / 135 total instead of 324. The exact center, native summary digest, axes, and setting count will be recorded in the GONG OOF summary. Its own grouped OOF and independent frozen-holdout promotion gates remain unchanged.
- Python compilation and a deterministic grid smoke check passed; the temporary check file was removed. This reduces avoidable scorer time while preserving the selected config and a local neighborhood for the candidate.

## 2026-11-12 13:39 UTC leaderboard and submission audit

- Refreshed Kaggle's 1,500-row leaderboard: EquiNexus is rank 321 with public score 0.39 after 30 submissions. All five Nov 12 submissions (58256453, 58256458, 58256468, 58256478, 58256483) are complete at 0.39. The top scores remain 0.70, 0.64, 0.62 x4, then 0.61.
- Saved the ignored leaderboard archive at `artifacts/kaggle/leaderboard/202611121340-utc/filament-segmentation-2026.zip` (SHA-256 `11771d823237d7eb21852a2bdc5a9bc09cfd0a413659986190f470c8c95e1ea6`) and the 30-entry account-submission snapshot at `artifacts/kaggle/submissions/202611121340-utc/equinexus-submissions.json` (SHA-256 `5136ae77d0af9032559e2490c34f5c1b6734798788a66c56ecb475d388fc056d`).

## 2026-11-12 13:40 UTC native OOF scorer progress

- The live full OOF scorer sweep advanced to 60/324 settings; provisional best PQ is 0.386102 at epoch 16. The official-scorer sweep and paired epoch comparison remain incomplete, so this does not select the final checkpoint.

## 2026-11-12 13:45 UTC native OOF scorer progress

- The live full OOF scorer sweep advanced to 70/324 settings; provisional best PQ is 0.388489 at epoch 16. Training remains complete and the summary is still pending; no holdout observations have entered selection.
- Targeted Kaggle CLI searches for competition kernels under the usernames of the top three leaderboard teams (`ionodeonline`, `manshanirapure`, `tensura911`, `duttasam`) returned no public kernels. No new source artifact from those teams is available to audit; the separate public I079 handoff remains an already-reviewed reference.

## 2026-11-12 13:50 UTC native OOF scorer progress

- The native full OOF sweep advanced to 80/324 settings; provisional best PQ is 0.389509 at epoch 16. The process is active, and its summary remains pending.

## 2026-11-12 13:57 UTC overlapping GONG OOF with native scoring

- The native scorer advanced to 90/324 settings; provisional best PQ is 0.391565 at epoch 16.
- Started the GONG auxiliary OOF process early (PID 34532) because the native sweep is CPU-bound and the RTX 5070 was idle. It is rasterizing only the 565 frozen training observations; the GONG target manifest excludes all 142 holdout observations.
- Updated the scheduled orchestrator to detect and reuse an already-running/completed GONG OOF job, and to skip the optional native TTA guard while GONG owns the GPU. This prevents duplicate runs and GPU contention. The trainer hash at launch is `5864229b1ebb38d48e7b869ca306541ef4d471db04e2562475db3dd2bb36007a`; the wrapper parses successfully.

## 2026-11-12 13:59 UTC GONG auxiliary OOF training started

- Training-only mask/auxiliary cache preparation completed for all 565 OOF training observations in 100.4 seconds; zero foreground pixels were cropped away. Fold 0 has started at 452 observations, batch size 1, accumulation 4, 24 epochs.
- The RTX 5070 reports 4,930/8,151 MiB allocated during startup; there are no CUDA errors, and free system RAM is 7.4 GiB. Monitor for steady epoch progress and memory pressure.
- At 14:00 UTC, GPU utilization reached 100% and memory use rose to 7,723/8,151 MiB (57 C); free system RAM was 6.9 GiB. The first epoch is still in progress, with no CUDA error in stderr.
- At 14:03 UTC the native CPU scorer reached 100/324, interim best PQ 0.394829 at epoch 16. GONG fold 0 is actively using the GPU at 100%, 7,739/8,151 MiB, 59 C; no training epoch has completed yet and stderr remains clear.
- At 14:04 UTC the GONG worker is still active in fold 0; GPU use is 7,753/8,151 MiB at 100% and 61 C, with 5.4 GiB system RAM free. No CUDA errors are present; the only stderr line is the HF Hub unauthenticated-rate-limit notice.
- At 14:08 UTC the native sweep reached 110/324, interim best PQ 0.397316 at epoch 16. GONG remains active in fold 0 after about ten minutes, at 7,753 MiB GPU memory and 100% utilization; its first epoch has not yet reported, but no error is present.
- At 14:14 UTC the native sweep reached 120/324 with the same interim best (PQ 0.397316 at epoch 16). GONG remains active with no first-epoch report after roughly 17 minutes; GPU utilization remains full without CUDA errors.
- At 14:16 UTC Task Scheduler started `EquiNexus_GONG_Auxiliary_20261112`; its hidden PowerShell orchestrator recorded the source snapshot and is waiting for the live native OOF process. The GONG trainer remains active on the GPU, still without an epoch-completion line or CUDA error.

## 2026-11-12 14:31 UTC GONG patch training and duration gate

- Replaced the stalled full-resolution GONG minibatches with aligned random 1024-pixel training patches. The original 1888-pixel center crop is retained for validation and inference; patch origins are multiples of four so the quarter-resolution auxiliary map stays aligned. The training-only 565-observation cache was reused.
- The sampler passed an in-memory compile and coordinate-coded shape/alignment smoke check. At 14:31 UTC the new run had reached 250/452 steps in epoch 1, used about 2.9 GiB of 8 GiB GPU memory, and reported no CUDA errors. The native CPU OOF scorer had reached 150/324 settings, provisional best PQ 0.397316 at epoch 16.
- Keep the existing 16/20/24 checkpoints. Longer training is justified only if the complete grouped OOF comparison and independent frozen holdout continue to improve; partial-fold evidence alone does not select a longer schedule.
- By 14:34 UTC, patch-run fold 0 had completed two epochs in 146.8 and 130.7 seconds. GPU use remained about 2.9 GiB and 76 C, with no CUDA error; the workspace battery reports AC power.
- By 14:43 UTC, fold 0 had completed epoch 6 and reached epoch 7 step 150/452. Epoch 6 took 129.5 seconds; the RTX 5070 remained at about 2.9 GiB and 76 C. Native official-scorer OOF had last reported 160/324 settings, still provisionally led by epoch 16 at PQ 0.397316.
- The new Kaggle run of `juandavidvanegas/cs2-swin-unet-finetune-fixed` is byte-identical in source and outputs to the earlier audited run; it provides no changed model evidence. It remains excluded from holdout selection because its workflow trained on 118 frozen-holdout observations and used the other 24 for its own validation/postprocessing.
- Dry-run preflight passed for both upcoming five-candidate batches on the current pushed source and frozen split. The Thursday batch includes the ten-model public U-Net as explicitly labeled 142-observation fold-crossfit evidence; the Friday batch retains four OOF-scored variants and the test-only CAGE candidate, which GONG may replace only after strict OOF and independent-holdout gains.
- Added a one-time Windows cutoff task for Friday 19:10 Toronto. Its preview matched only the local native and GONG training process pairs; the task wakes the laptop and can run on battery. The scheduled Kaggle T4 run retains its 2,400-second runtime limit.
- At 14:54 UTC, GONG fold 0 had completed epoch 11 and entered epoch 12; epoch 11 took 128.9 seconds. GPU use remained 2.9 GiB at 77 C. The native OOF scorer remained live and CPU-active, with last confirmed progress 170/324 settings at 14:47 UTC.
- The native scorer advanced at 14:56 UTC to 180/324 settings and found a new interim best of PQ 0.398085 at epoch 20. Longer training is outperforming epoch 16 in this partial sweep; the complete OOF comparison and independent holdout still govern duration and promotion.
- Checked public datasets and model repositories for the current leading leaderboard members. None exposed a relevant filament dataset or public model; IONode Online's listed datasets concern unrelated agent-paper artefacts. No new competitor resource is eligible to add.
- At 14:58 UTC local `main` matched `origin/main` at `8a61f528`; the eleven active worktree paths exactly match tonight's checkpoint allowlist and `git diff --check` passes.

## 2026-11-12 15:04 UTC ASTH-Net public-kernel audit

- Audited `koushikkumardinda/asth-net-anisotropic-spine-tangent-hysteresis` (T4; reported complete Nov 12 01:46 UTC). Its source and metadata use only the competition source, with no external dataset or model. Source SHA-256: `8930cd90379251c2095061f459bc7208198345e6fa53bb25036f75eae9381b9e`; execution-log SHA-256: `ac228cc7dfce27debe33211f2931092967cdb7efcf8c59702521cd2c0be992d4`.
- The notebook trains on the first 160 chronologically ordered observations for three epochs and reports custom, one-to-one per-image PQ 0.0258 after hysteresis/spine bridging (baseline 0.0058). This is not the official grouped per-annotator scorer. Its train set overlaps 33 of our 142 frozen holdout groups; its validation set overlaps another 5, so its reported metric cannot select our model.
- It processed the 180 supplied test images but predicted zero instances. Its two-row CSV is a synthetic verification fallback keyed to `20150125172714Mh`, which is an actual competition training image and is absent from the supplied test-image directory. CSV SHA-256: `00bec355ae6ccd88b0292864988300e414413c92ec05c49d231763740bd58cbe`. Do not use this output.
- Decision: exclude this kernel from model selection and submissions. Preserve the source/output audit under ignored `artifacts/kaggle/public-kernels/`; the corrected audit JSON records the training-image/test-image check.

## 2026-11-12 15:06 UTC longer-run progress

- The native model has completed training for all five OOF folds at epochs 16, 20, and 24. Its official-scorer postprocessing sweep reached 190/324 settings and provisionally favors epoch 20 at PQ 0.399435. The sweep is incomplete; this partial result supports keeping the longer schedule in the comparison, but does not yet select a final checkpoint.
- The patch-based GONG auxiliary OOF run remains active and has written its fold-0 epoch-16 checkpoint. It still has to finish all folds and produce grouped OOF scores before holdout evaluation. The RTX 5070 is at 2,909/8,151 MiB and 96% utilization; battery telemetry shows AC at 99%.

## 2026-11-12 15:28 UTC LadyFaye reproduction and external-data audit

- Pulled `ladyfaye/solar-filament-reproduction-58067341` and its pinned GitHub source commit `07871ef5b04b8e1d562005ea00f845f95af4a2e5` into ignored `artifacts/`. The source archive SHA-256 is `bf4882312cc55adee1b4bd822d18e1db4acd1f90fe9b12c8027be338b6337d42`; notebook SHA-256 is `0803da49b25090beaf87398e739282a4c980df1c793dbe1f1a0b991d47a26185`. All 34 code-file hashes and 40 asset hashes in its manifest match.
- The AGPL-3.0 release documents eight detectors and three refiners; its prior submission 58067341 scored public PQ 0.39 and its release claims byte-identical replay over 180 images, 1,270 instances, and CSV SHA-256 `83139e60ad3392fbdf4c06e1a287fdfe78c79dc23d7c31114934f97c3119bbff`. The current Kaggle kernel metadata disables GPU and its output contains only an nbconvert log, no inference CSV; those replay claims were not independently rerun here. We downloaded no weights and adopted no inference code.
- Its historical `long565` fit overlaps 109/142 of our frozen holdout groups, while its own validation overlaps 30; its full-707 deployment members train on all 142. Therefore its public score and package do not provide independent validation for our split.
- The release's `solar/external_data_policy.py` cites Kaggle discussion `748045#3536644` and applies a one-hour all-site test buffer plus frozen development-date/time exclusions to external GONG frames. That discussion URL returned HTTP 404 during direct lookup, so this is recorded as the upstream author's interpretation; we applied the stricter filter conservatively.
- The previously running GONG auxiliary job had used 504 training-target files, including `20130208063154Uh` and `20150913082014Th`, each 20 seconds from a supplied test observation at another site, and 16 targets on frozen validation dates. Stopped it at fold-0 epoch 16 before OOF scoring or holdout evaluation. Its checkpoints are retained but marked ineligible in ignored `artifacts/model-experiments/native-unet-gong-aux-20261112/policy-review/eligibility-audit.json`; the original manifest is preserved alongside it.
- Added a shared temporal-exclusion audit used by the downloader and trainer. Regenerated the training-only manifest with 486 eligible target maps; the trainer now independently recomputes the 180 test and 142 development timestamp rules and fails closed if the manifest differs. The filtered manifest SHA-256 is `91307d0a9c28aa240de585e301d8aa6a540aef130f9492cc9754fe9235a5904`. The frozen split digest is unchanged.

## 2026-11-12 15:31 UTC filtered GONG OOF restarted

- Restarted the 24-epoch GONG auxiliary OOF run in a separate ignored output directory, `artifacts/model-experiments/native-unet-gong-aux-filtered-20261112/`, using the regenerated 486-target manifest. The training loader independently verified all 486 target files and the unchanged split digest before building its 565-observation cache. The previous checkpoint directory remains quarantined as ineligible.
- Restarted the cutoff-aware orchestrator against the filtered candidate path; it waits for the full native grouped OOF sweep, then applies the same OOF and independent-holdout promotion gates. The Friday 19:10 local stop remains in place.
- The native official-scorer sweep had reached 230/324 settings with provisional PQ 0.404830 at epoch 20. The 24-epoch comparison is still in progress.
- Updated tonight's exact-path Conventional Commit checkpoint to include the temporal-policy helper and GONG target downloader guard, with the filter rules included in the commit description.

## 2026-11-12 15:36 UTC paired training progress

- Filtered GONG OOF fold 0 completed epoch 1 in 144.2 seconds and advanced to epoch 2 step 250/452; mean loss is 0.73573. GPU use is 2.9/8.2 GiB at 79% utilization and 77 C; no CUDA error appears. Battery telemetry is 98% and charging on AC. System free RAM is about 2.0 GiB while the native scorer runs, so monitor memory during the next epochs.
- The native grouped official-scorer sweep is still CPU-active at 230/324 settings, with its last confirmed best PQ 0.404830 at epoch 20. No OOF summary or holdout comparison is available yet.

## 2026-11-12 15:40 UTC pending-batch checkpoint repair

- Current dry-runs pass for both five-candidate manifests on source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6`, with the frozen split digest unchanged.
- Found that tonight's checkpoint only refreshed the Nov 13 manifest; the Nov 14 manifest would otherwise fail its strict source-commit check after tonight's push. Updated the scheduled checkpoint to stamp the new commit into both manifests and run both dry-runs. PowerShell syntax parsing passed. No submission has been made.
- The native official-scorer sweep advanced to 240/324 settings, still provisionally led by epoch 20 at PQ 0.404830. The filtered GONG run is training fold 0, epoch 3.

## 2026-11-12 15:54 UTC live OOF and leaderboard check

- Directly polled the native scorer session: 250/324 official-scorer settings completed; current provisional best remains PQ 0.404830 at epoch 20. The scorer has not completed its full sweep or written the OOF summary.
- The filtered GONG auxiliary OOF process remains active with its last fold-0 checkpoint write at 10:50 Toronto time. Hardware snapshot: RTX 5070, 89% utilization, 2.9/8.2 GiB allocated, 75.8 C, 5.09 GiB system RAM free. No CUDA error was observed.
- Five entries labeled "Nov 12" in their descriptions completed at 2026-11-12 01:20 UTC / 2026-11-11 20:20 Toronto, all at public score 0.39. No Nov 12 Toronto batch has been submitted yet; it remains scheduled for 23:58 Toronto. The Friday batch is scheduled for 19:20 Toronto on Nov 13. The refreshed leaderboard top rows are 0.70, 0.64, and 0.62 x4; keep public score secondary to the unchanged grouped OOF and holdout gates.

## 2026-11-12 16:10 UTC paired OOF run progress

- The native grouped scorer's last confirmed position is 270/324 postprocessing settings, with provisional best PQ 0.404830 at epoch 20; its worker remains CPU-active without a final summary.
- The filtered GONG auxiliary run reached fold 0, epoch 18, step 250/452. The GPU remains at about 2.9/8.2 GiB and 78 C, with 5.05 GiB system RAM free; battery is 98% and charging. The current stderr contains only the Hugging Face unauthenticated-rate notice and no CUDA error.

## 2026-11-12 16:18 UTC paired training progress

- The native scorer remains active at its last reported 280/324 settings, with provisional best PQ 0.404830 at epoch 20. CPU time continues to rise; the final summary is not yet available.
- Filtered GONG fold 0 reached epoch 21, step 400/452. Epoch 20 took 125.8 seconds. Current GPU use is 2.9/8.2 GiB at 76 C, system RAM free is 4.77 GiB, and battery is 98% and charging. Only the HF Hub rate notice appears in stderr; no CUDA error is present.
- The Thursday five-entry submission task and daily checkpoint remain Ready for 23:58 and 23:45 Toronto time respectively. Friday's submission and checkpoint remain Ready for 19:20 and 19:50.

## 2026-11-12 16:20 UTC Thursday batch preflight

- Dry-run preflight passed all five candidates against source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6` and frozen split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- The batch contains four grouped-OOF family votes (PQ 0.452149, 0.450632, 0.448411, 0.441988) and one public ten-model U-Net (142-group per-observation crossfit PQ 0.447799, not independent holdout evidence). All five candidate files are hash-locked. Submission is scheduled for 23:58 Toronto time after the 23:45 checkpoint refreshes the manifest source commit.

## 2026-11-12 16:54 UTC native duration result and runner repair

- The native 5-fold OOF postprocess sweep completed 324/324 settings. Best grouped crossfit PQ is 0.397316 at epoch 16, 0.404830 at epoch 20, and 0.405374 at epoch 24 (threshold 0.70, min area 400, closing 9). The epoch-24 gain over epoch 20 is +0.000544 and still needs the frozen holdout comparison.
- The complete native summary and sweep CSV were saved before a final `json.dumps(flush=True)` logging typo exited the process with status 1. Corrected the native OOF and holdout log calls. The active GONG OOF run is left on its original source snapshot; after it finishes, fix the same logging typo in the GONG OOF, holdout, and final paths before those stages are launched. Its fold 1 is at epoch 14/24; no GONG validation result is available yet.


## 2026-11-12 17:16 UTC topology-loss screen preparation

- The native grouped official-scorer OOF sweep completed at 324/324. Its selected configurations score 0.397316 at epoch 16, 0.404830 at epoch 20, and 0.405374 at epoch 24; the 20-to-24-epoch increase is only +0.000544 OOF PQ. This supports testing the longer run but does not establish a holdout gain, so the frozen holdout remains the promotion gate and no further epoch extension is planned without evidence.
- Added an optional soft-clDice term to the native U-Net loss, fixed at weight 0.25 for one randomly sampled patch up to 512x512 and 10 skeletonization iterations. The planned single holdout screen uses epoch 24 and postprocessing selected by native OOF, with no holdout tuning. Reference: [Shit et al., clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.html).
- Syntax and CLI checks passed for both runner scripts. CPU backward smoke checks on empty and filament targets produced finite losses and gradients. The holdout training has not started because the filtered GONG grouped OOF still owns the GPU.
- At this checkpoint the GONG OOF worker is on fold 1, epoch 24/24, step 250/452; GPU utilization is 96%, 2.69/8.15 GiB memory, 79.2 C, with no CUDA errors. Battery telemetry reports 98% on AC and the active AC sleep timeout is disabled. The original GONG trainer hash still matches the run metadata, and tonight's exact 15-path checkpoint allowlist matches the worktree.


## 2026-11-12 17:23 UTC leaderboard refresh and I079 source audit

- Downloaded a fresh 1,502-row public leaderboard snapshot to ignored `artifacts/kaggle/leaderboard/202611121717-utc/`; archive SHA-256 is `9d972e820643c351eb3a351604b3ab6324424fb3f1b2024bd46cbca8a80c0f1b`. EquiNexus remains rank 322 at 0.39 with 30 submissions. The current third-place row is 0.62; its account has no public competition kernel listed. These public scores remain preliminary.
- Re-read the highly voted [I079 topology-safe inference handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff). It claims two 0.70 submissions from a normal 1,299-instance output, explicitly excludes public test labels, and describes an image-only Mask R-CNN plus refiner with guarded flip-4 TTA. Its attached model card says training used only competition training annotations and declares Apache-2.0; these remain upstream provenance claims.
- Verified the local detector/refiner checkpoints match the handoff's MD5 locks. The existing `scripts/run_i079_topology_safe_inference.py` already implements this test-only pipeline. Its full-data weights include the frozen validation observations, so I will not score them on the holdout or treat them as independently validated. Run it only on supplied test H-alpha images and label any submission as an external test-only experiment.
- Pulled the I079 and Dariush public notebooks to ignored `artifacts/kaggle/public-kernels/202611121717-utc/` for source review. Dariush's learned keep-rule table combines all 707 training observations, so it cannot be used for a clean frozen-holdout evaluation; any use would require rebuilding from crossfit predictions on the 565 training observations only.
- Audited the newer Huendao, Muhammad Shoaib Altaf, and Robina Mirbahar kernels. Huendao reports 0.4523 OOF / 0.3945 public using the ten-model U-Net family and learned keep rule, but it reuses the full-data weights and is not frozen-holdout evidence. Muhammad's V18 detector experiment uses a different 107-photo split and states that its 67-photo subset has prior project-selection exposure; its V10/V12 checkpoints are not valid for our frozen holdout. Robina's recent panoptic notebook duplicates the I079 handoff rather than adding an independently validated method. No checkpoints or labels from these notebooks were used.

- GONG OOF has since completed fold 1 and advanced to fold 2, epoch 2/24, step 350/452. GPU use is 2.69/8.15 GiB at 80% utilization and 77.4 C; the training and orchestrator processes remain active with no CUDA errors.


## 2026-11-12 17:34 UTC post-GONG run queue

- Started ignored local watcher PID 28144. It waits for the GONG/native orchestrator to release the GPU, runs the fixed clDice holdout screen only if GONG was not promoted and the Friday 16:00 Toronto cutoff has not passed, then runs I079 only on the supplied H-alpha test images before 18:00. It does not submit or edit manifests. The queue will skip existing output folders and stops if the GONG pipeline remains active at the 18:00 cutoff.


## 2026-11-12 17:35 UTC GONG training progress

- Filtered GONG OOF is training fold 2, epoch 7/24, just completed 452/452 steps. GPU use is 2.69/8.15 GiB at 84% utilization and 77.3 C. Stderr contains only the Hugging Face unauthenticated-rate notice; no CUDA or training error is logged. The post-GONG queue remains active.


## 2026-11-12 17:38 UTC submission and frozen-source preflight

- Re-ran the non-submitting preflight for tonight's and Friday's five-candidate manifests. Both passed with source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6`, the frozen split digest, five candidates, and validated hash-locked CSVs. The scheduled checkpoint will refresh the source commit after its push and rerun both checks. No new Kaggle submission was made; the account still has 30 completed submissions.


## 2026-11-12 17:42 UTC quota and daylight-saving verification

- The authenticated Kaggle submission-limits API reports `numToday=5`, `numAllowedNow=0`, and `numTotal=30`; the latest five uploads at 01:20 UTC consumed the Nov 12 UTC quota. No change is needed to tonight's schedule: its 23:58 Toronto task runs at 04:58 UTC on Nov 13. Friday's 19:20 Toronto task is stored with the correct `-05:00` offset and runs at 00:20 UTC on Nov 14, after the next UTC reset. This preserves five submissions in each UTC day.
- Corrected the post-GONG watcher cutoffs for Toronto standard time: clDice may start before Friday 16:00 local (21:00 UTC), and I079 inference before 18:00 local (23:00 UTC). Restarted the waiting watcher as PID 17676; the active GONG run was still live.


## 2026-11-12 17:44 UTC live quota guard

- Added a live `KaggleApi.competition_get_submission_limits` preflight to the batch uploader. A real batch now starts only when the entire batch fits the remaining UTC quota; it records the returned counters and refuses partial uploads otherwise. Verified the guard against the current full quota (`5` used, `0` available): it failed closed without invoking Kaggle submit. The manifest dry-run still validates all five candidates, and the scheduled Conventional Commit script parses successfully with the new bullet in its commit description.


## 2026-11-12 17:44 UTC live run check

- Filtered GONG OOF completed fold 2 epoch 11 and is training epoch 12, step 100/452. GPU use is 2.69/8.15 GiB at 97% utilization and 77.9 C; the training process and corrected post-GONG queue are live. The 15 changed repository paths still exactly match tonight's commit allowlist.


## 2026-11-12 17:50 UTC submission preflight and GONG update

- Both five-candidate manifests pass the updated uploader's post-change dry-run with source commit `8a61f528bcafb1632a81f127fb3df6ef57c845a6` and the frozen split digest. Dry-run mode does not query or consume Kaggle quota.
- Filtered GONG OOF completed fold 2, epoch 14/24; GPU use is 2.69/8.15 GiB at 85% utilization and 77.5 C. The GONG orchestrator and corrected post-GONG watcher remain active.


## 2026-11-12 17:53 UTC GONG OOF progress

- Filtered GONG OOF is training fold 2, epoch 16/24, step 300/452. GPU use remains 2.69/8.15 GiB at 92% utilization and 79.9 C. The source-hash repair watcher and post-GONG queue are live; stderr shows only the Hugging Face unauthenticated-rate notice.


## 2026-11-12 18:35 UTC training duration and Friday automation

- The live GONG auxiliary OOF run uses 24 epochs on each of five grouped folds of 113 observations; it is now on fold 3, epoch 10/24. Epoch losses continue to fall, but the promotion decision will use grouped OOF PQ and the paired frozen-holdout result. We will extend training only if those validation results support it, since lower training loss alone does not establish better segmentation.
- The RTX 5070 is actively training on AC at 89% utilization, 2.69/8.15 GiB VRAM, and 81.3 C. A workspace-ignored helper keeps the system awake only while both training PIDs run and AC remains connected; the display can turn off and the helper releases the request when training stops.
- Kaggle CLI authentication is valid, but GPU quota is exhausted until Friday 19:00 Toronto. The fold-aware T4 check starts at 19:05; the five-candidate batch submits at 19:20. I moved the Kaggle status refresh from 19:50 to 19:35 because it previously collided with the final checkpoint; it stops waiting for the T4 runner by 19:43, leaving time for the single 19:50 Conventional Commit and push.
- After tonight's checkpoint, the Friday runner permits only unstaged edits to the two sprint logs; all code, notebook, and submission-building changes must remain at the pushed source commit. The final manifest checkpoint records the submission-producing source commit before it updates the manifest to the documentation commit.
- The Friday Kaggle refresh now records that preserved submission-source commit and all five candidate CSV SHA-256 values alongside their references and public scores, so the final archive can identify the exact uploaded files.


## 2026-11-12 18:42 UTC public leaderboard and notebook review

- The latest leaderboard archive is `artifacts/kaggle/leaderboard/20261112T1842Z-utc/filament-segmentation-2026.zip` (SHA-256 `c203ad35ff07f0b7ed9104d17034a45bc25fd6e861775240be42a695935d517a`). EquiNexus remains rank 322 at 0.39; IONode leads at 0.70, followed by Tempest at 0.64 and four teams at 0.62. The top eight and 50-kernel vote-sorted index are preserved under ignored `artifacts/kaggle/`.
- The public U-Net notebook reports 0.4502 base nested OOF, 0.4552 with its learned keep rule, and 0.39 public. We downloaded and receipt-verified its ten CC BY-NC-SA 4.0 checkpoints under the user's non-commercial research authorization. The models span all competition observations, so the 0.447799 score on our 142 frozen-holdout observations is crossfit evidence, not an independent holdout; this candidate is used only for supplied-test inference. Any selector experiment must be trained on our own grouped OOF predictions.
- The V18 detector-fusion notebook's per-photo target consolidation is not independently verified: its V10/V12 checkpoint sources are absent from Kaggle metadata, four cells failed, and its evaluation data are not our frozen split. ASTH-Net's hysteresis and axis-aligned bridge are reference concepts only; its evaluation protocol is a 20-record chronological screen, and its test writer never appends normal per-image rows. Neither notebook changes the current candidates or their validation claims.
- The 49-vote [topology-safe I079 handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) reports its exact test CSV received 0.70 twice and claims a 1,299-row, label-free run. Its four-flip, guarded TTA is already implemented in the local I079 inference runner and is staged as a test-only candidate if the validation-gated GONG replacement does not happen. The score remains attributed public evidence only; the top 0.70 row in our current leaderboard snapshot belongs to IONode Online.
- All four source notebooks and their Kaggle metadata are archived under `artifacts/kaggle/public-kernels/20261112T1842Z-utc/`; their SHA-256 values are in the experiment log. The latest audit also corrected an earlier progress note: the live GONG run has five grouped folds of 113 observations each, not four.

## 2026-11-12 19:05 UTC submission and training update

- The five latest submissions (refs `58256453`, `58256458`, `58256468`, `58256478`, `58256483`) are all complete at public score 0.39. Grouped OOF PQ ranges 0.455022–0.459116 across the five confidence settings; public scores do not resolve their small differences. The UTC quota is full at five submissions today and 30 total.
- GONG auxiliary OOF has saved epoch 24 for fold 3. Folds 0–2 are complete; fold-3 prediction generation is ongoing and fold 4 remains. Wait for complete grouped OOF scoring and paired frozen-holdout evidence before commissioning a longer schedule. The learning-rate schedule reaches zero at epoch 24, so any extension must use a newly scheduled run.
- The 23:58 Toronto batch remains prepared with five distinct hash-locked candidates and will run after the UTC quota reset. The nightly Conventional Commit and push are scheduled for 23:45 Toronto time.
- Added an AC-only keep-awake helper for the long-lived GONG orchestrator so later holdout/final subprocesses remain protected after the current OOF Python process exits. The helper releases when the orchestrator ends or AC is disconnected.

## 2026-11-12 19:38 UTC next-batch preflight

- Both scheduled five-candidate manifests pass the local uploader dry run. All ten CSV hashes, schemas, nonempty rows, unique IDs, instance counts, source commit, and frozen split digest match; each manifest declares complete 180-image coverage. The dry run does not contact Kaggle or consume quota.

## 2026-11-12 19:46 UTC recent public-kernel audit

- The refreshed 100-kernel vote-sorted index and two newest notebooks are archived under ignored `artifacts/kaggle/public-kernels/20261112T1942Z-utc/`. The Swin notebook's 512px overlap, deep supervision, and clDice ideas are references only: it tunes on a separate 15% date-grouped split with Dice and a custom PQ proxy, and Kaggle attaches no pretrained model sources. The standalone U-Net demo claims MAGFiLO-trained CC BY-NC 4.0 weights, but its dataset source metadata is blank; we leave the weights unused and the claim unverified.
- During this scan, GONG OOF was on fold 4, epoch 19/24 after completing folds 0–3. Its full five-fold PQ sweep remains pending.


## 2026-11-12 20:13 UTC leaderboard refresh and GONG sweep

- The latest Kaggle leaderboard snapshot is `artifacts/kaggle/leaderboard/20261112201340-utc/filament-segmentation-2026.zip`, SHA-256 `7b88f7f4e67aede76e8a81038c8c4d669aeec258cdad23e33a54d2460131bcc8`. It contains 1,504 teams; EquiNexus is rank 322 at 0.39 with 30 submissions. The top eight scores are 0.70, 0.64, 0.62, 0.62, 0.62, 0.62, 0.61, and 0.60. All five Nov 12 submissions are complete at 0.39.
- GONG's five grouped folds and prediction snapshots are complete. Its fixed 36-setting postprocessing sweep has reached 20/36 settings; the current best is OOF PQ 0.385348 at epoch 20, below the native grouped OOF baseline 0.405374. The scorer process is still active on CPU under memory pressure. No OOF summary, paired comparison, holdout comparison, or final candidate is available yet.
- Continue with the already scheduled gates. Do not promote or extend epochs until the full sweep is complete; only run the independent holdout if grouped OOF exceeds baseline. The separate Nov 13 UTC submission batch is scheduled for 04:58 UTC after its quota reset. Kaggle GPU quota resets at 00:00 UTC Nov 14; the T4 fold-aware run is scheduled at 00:05 UTC then. The Nov 12 daily commit and push remain scheduled for 23:45 Toronto.

## 2026-11-12 20:21 UTC GONG grouped OOF result

- The fixed 36-setting sweep completed on the frozen split. Best GONG OOF PQ by epoch is 0.375957 (epoch 16), 0.387088 (epoch 20), and 0.380154 (epoch 24); the selected epoch-20 configuration (threshold 0.70, minimum area 400, closing 9) is below the native U-Net OOF baseline of 0.405374. The paired 20,000-resample comparison remains in progress to quantify uncertainty; the lower point PQ already fails the strict improvement gate, so holdout evaluation and promotion are skipped.
- No longer GONG run is planned from these results: epoch 24 underperforms epoch 20, and neither approaches native OOF. Run metadata records 565 competition observations, 922 annotator views, 486 filtered GONG targets, RTX 5070 8.15 GiB, eight-view inference, and 4.78 hours. The temporal target manifest SHA-256 is `91307d0a9c28aa240de585e301d8aa6a540aef130f9492cc9754fe9235a5904f`; the frozen split digest remains unchanged and no validation/test targets were fetched.

## 2026-11-12 20:36 UTC paired GONG comparison and clDice screen

- Paired grouped OOF is complete: native PQ 0.405374, GONG PQ 0.387088, delta -0.018285; bootstrap mean -0.018320, median -0.018292, 95% interval [-0.024716, -0.011986], with 0/20,000 resamples favoring GONG. The orchestrator validates the artifact against the frozen split and metric/provenance fields and records the GONG holdout skip. Epoch 24 is below epoch 20, so no longer GONG schedule or final fit is planned.
- Epoch length is model-specific in the current evidence: native grouped OOF best PQ rises modestly from 0.397316 at epoch 16 to 0.404830 at 20 and 0.405374 at 24, while GONG falls from 0.387088 at epoch 20 to 0.380154 at 24. Keep 24 epochs for the already-running clDice screen, but base any extension on grouped validation rather than training loss.
- Repaired `compare_gong_aux_oof.py` so its final status print flushes correctly. The exact source used by the GONG OOF run is preserved under `artifacts/model-experiments/native-unet-gong-aux-filtered-20261112/source/` and its SHA-256 matches run metadata; post-run completion logging changed only three print calls.
- The clDice screen is now training on the 565 frozen training observations for 24 epochs with weight 0.25. It uses native OOF-selected postprocessing and eight-view inference; holdout threshold tuning is disabled, external data are unused, and test labels are not read. RTX 5070 training is active under an AC-only keep-awake helper. I079 test-only inference remains queued for after this independent holdout result.

## 2026-11-12 23:01 UTC leaderboard and submission refresh

- Refreshed the public leaderboard to `artifacts/kaggle/leaderboard/20261112T2300Z-utc/filament-segmentation-2026.zip` (SHA-256 `aa167bb6a2beade185469d6299f9bb108eb70d2f2f373bb33d5fd9535d2082be`). It lists 1,504 teams; EquiNexus is rank 324 at 0.39 with 30 submissions. IONode Online leads at 0.70, and the next scores are 0.64, 0.62, 0.62, 0.62, 0.62, 0.61, 0.60, and 0.60.
- Confirmed the five latest EquiNexus submissions are complete at public 0.39 (refs `58256453`, `58256458`, `58256468`, `58256478`, `58256483`). Their grouped OOF PQ values range from 0.455022 to 0.459116; the two-decimal public score does not distinguish them. The next five-candidate batch is scheduled after the UTC quota reset.
- The independent clDice run remains healthy on the RTX 5070 Laptop GPU at epoch 13/24, step 300/565; the frozen-holdout score and paired comparison are pending.

## 2026-11-12 23:06 UTC submission-batch preflight

- Dry-ran both prepared five-entry manifests against their CSV hashes and RLE validators. The Nov 13 UTC batch and Nov 14 UTC batch each pass with the frozen split digest; neither command submitted anything. Tonight's checkpoint refreshes each manifest's source commit and repeats the preflight before pushing. The Nov 13 batch is scheduled for 23:58 Toronto, after the UTC quota reset.

## 2026-11-12 23:10 UTC gated clDice test runner prepared

- Added `scripts/compare_native_unet_cldice_holdout.py` for the reproducible 20,000-resample seed-42 observation bootstrap against the best independent incumbent, `scripts/visualize_native_unet_cldice_holdout.py` for lowest/median/highest-PQ overlays against the first annotation view, and `scripts/run_native_unet_cldice_test_inference.py` for preliminary train-565 inference or a final fit on all 707 competition training groups. Inference is gated by a positive paired delta, a 95% interval above zero, and at least 97.5% of resamples favoring the candidate; the runner verifies checkpoint training groups and validates the resulting RLEs. Syntax checks pass; test inference runs only if the holdout gate clears. Updated the nightly allowlist and Conventional Commit body.
- Added `scripts/workspace_runtime.py` to force Hugging Face, Torch, pip, Ultralytics, Matplotlib, and temporary-file paths into ignored `.cache/` and `.tmp/` folders before framework imports. The local EfficientViT cache is present under the repository's `.cache/`; the already-running process predates this temporary-path override and its training code writes checkpoints and data caches under `artifacts/`.

## 2026-11-12 23:38 UTC clDice training progress

- The fixed 24-epoch clDice run is on epoch 16/24, step 450/565, with training loss around 0.267. RTX 5070 use remains stable at roughly 7.4/8.0 GiB and 74 C on AC. No holdout metric has been read; completion is expected around 20:10 Toronto before scoring and the paired comparison.

## 2026-11-12 23:43 UTC clDice training progress

- The same fixed run reached epoch 17/24, step 150/565, with mean training loss 0.2686. RTX 5070 remains stable on AC at 99% utilization, 7.57/8.15 GiB, and 73.7 C. The frozen-holdout score and paired comparison remain pending; training loss is not selection evidence.

## 2026-11-12 23:45 UTC Kaggle refresh

- The saved 1,504-team leaderboard ranks EquiNexus 324th at 0.39 (30 submissions), behind IONode Online at 0.70 and Tempest at 0.64. All five latest submissions remain complete at 0.39. Their frozen grouped-OOF scores range from 0.455022 to 0.459116; the public leaderboard rounds to two decimals. The authenticated submissions and leaderboard archive is under `artifacts/kaggle/leaderboard/20261112234512-utc/`.

## 2026-11-12 23:49 UTC public-kernel audit

- The most-voted notebook is a 1024px Mask R-CNN plus ResNet-18 U-Net refiner. It does split one selected record per observation, but its threshold selection uses a custom Dice objective and a Hungarian IoU-`>=0.5` diagnostic rather than official PQ. Its code and metadata are archived under `artifacts/kaggle/public-kernels/20261112T2348Z-utc/top-voted-maskrcnn/`; absent model/dataset-source metadata and an explicit license, its weights/code stay unused. The Apache-2.0 I079 detector/refiner candidate remains test-only.
- Notebook and Kaggle metadata SHA-256 values are `c6b004aa5dca9edde71ba73972d0c69e9e97173430abac1ab1af8661b93fed6f` and `e97369189d343122f8e55640db6f51b0c672619d741fbc681bc36db8dc2eb43e`.
- Kaggle exposes no competition notebook from IONode Online, the current 0.70 leader; its two listed public kernels concern unrelated tasks.

## 2026-11-12 23:52 UTC clDice gate review

- Fixed the paired-comparison reader to use the holdout CSV's `tp` column. Syntax compilation and a small workspace-local schema smoke check pass. The training run is still underway; no holdout score or model-promotion decision exists yet.

## 2026-11-13 00:02 UTC submission quota reset

- Kaggle's authenticated limits endpoint confirms a fresh UTC quota: zero submissions used today, five allowed, and 30 total so far. The prepared batch will recheck the limit before upload at 23:58 Toronto.

## 2026-11-13 00:17 UTC clDice training progress

- Epoch 19/24 completed in 711.6 seconds; mean training loss was 0.25425. Five epochs remain. The holdout has not been scored, and this loss is not promotion evidence.

## 2026-11-13 00:30 UTC pretrained encoder provenance

- The clDice run's `efficientvit_b2.r288_in1k` encoder uses the `timm/efficientvit_b2.r224_in1k` pretrained ImageNet-1K checkpoint, whose [model card lists Apache-2.0](https://huggingface.co/timm/efficientvit_b2.r224_in1k). Recorded revision `891530d0c5a400dd2d4549c81d433661b192abe1` and checkpoint SHA-256 `f5726851b5c55fab49f8302a79c3a898814e11bc20df76cc23cab28ae15c7328` in run metadata.

## 2026-11-13 00:35 UTC external-weight metadata correction

- Run metadata now marks external data as used through the pretrained ImageNet encoder, while separately recording that no external filament-segmentation labels or test labels were used. Test inference enforces the recorded Apache-2.0 license and encoder weight hash.
- Launch-time evaluator and base-trainer hashes are recorded, but their exact bytes were not saved before workspace-runtime and completion-log edits. Three post-start snapshots are archived; the run metadata states that launch-time source cannot be reconstructed from them, while the live process continues with its already loaded code.

## 2026-11-13 00:38 UTC clDice training progress

- Epoch 20/24 completed in 712.7 seconds with mean training loss 0.25060. Epoch 21 is at step 500/565; three epochs remain. No holdout score is available.

## 2026-11-13 00:40 UTC clDice training progress

- Epoch 21/24 completed in 712.1 seconds with mean training loss 0.25177. Three epochs remain; no holdout score is available.

## 2026-11-13 00:50 UTC clDice training progress

- Epoch 22/24 completed in 687.9 seconds with mean training loss 0.24883. Two epochs remain; no holdout score is available.

## 2026-11-13 01:02 UTC clDice training progress

- Epoch 23/24 completed in 687.6 seconds with mean training loss 0.24631. One epoch remains; no holdout score is available.

## 2026-11-13 01:14 UTC clDice training complete

- The fixed 24-epoch training run completed; epoch 24 took 687.4 seconds with mean training loss 0.25023. Frozen-holdout prediction is in progress at 40/142 observations; no score has been read yet.

## 2026-11-13 01:16 UTC clDice holdout result

- Official grouped holdout PQ is 0.372570, below the independent incumbent 0.454668. Paired seed-42 bootstrap (20,000 observation resamples): delta -0.082098, 95% CI [-0.095056, -0.069966], probability of improvement 0.0. The fixed promotion gate fails; clDice test inference and the all-707 refit are skipped. Overlays are visual diagnosis only.

## 2026-11-13 01:17 UTC clDice visual review

- Reviewed seven paired overlays. The median candidate observation scores 0.385 versus incumbent 0.434; the weakest drops from 0.470 to 0.000 with extra false-positive fragments, while the strongest is tied around 0.757. No parameters were tuned from these views; the hashed overlay manifest is saved under the ignored clDice run directory.


## 2026-11-13 01:24 UTC I079 supplied-test inference

- Ran the Apache-2.0 I079 detector/refiner on all supplied test H-alpha images with four-view flip TTA. The full RLE validator passed 1299 rows across 180 test images with no overlaps; CSV SHA-256 is 2c07cf2f06ee85b721ccb11631d6876f693ac6516277ed2d1dcd179ee7eb0794. The output MD5 differs from the upstream notebook's expected hash. This remains test-only: the model card states competition-mask-only training but does not document fold membership, so no frozen-holdout PQ is claimed.

## 2026-11-13 01:52 UTC Kaggle source audit and Swin-T grouped OOF

- Refreshed the full public leaderboard archive at artifacts/kaggle/leaderboard/20261113T013149Z-utc/ (CSV SHA-256 4b88fd0761906cd6d19b5ca64c3b9b6cc6b7f53631caac445cceaab7b3b292e5). It contains 1,505 teams; EquiNexus is rank 326 at 0.39 with 30 submissions. IONode Online leads at 0.70, followed by Tempest at 0.64. The five Nov 12 submissions remain complete at 0.39.
- Archived metadata and source for three newly active public notebooks. The [Nov 12 Swin-T notebook](https://www.kaggle.com/code/juandavidvanegas/cs2-swin-unet-finetune-fixed) groups records into 14-day bins, chooses thresholds with per-image Dice, and reports a connected-component PQ proxy over annotator-union masks; it does not implement our official scorer or frozen observation split. The [PRVSiyan notebook](https://www.kaggle.com/code/prvsiyan/a-thousand-quiet-threads) is an explicitly unscored from-scratch experiment with a separate seed-20261108 split and a 192-image reserved cohort. The new [Solar Filament Demo v2](https://www.kaggle.com/code/manurupanm/solar-filament-demo-v2) states its weights were trained on public MAGFiLO v1.0, uses CC BY-NC 4.0 weights, and does not document image-level separation from competition test labels; no code or weights were adopted.
- Audited the [daydreamerthinks SOTA model package](https://www.kaggle.com/datasets/daydreamerthinks/solar-filament-sota-models). The Kaggle dataset is CC0 but has no training-data description. Static checkpoint inspection identifies an Ultralytics YOLO11s model under AGPL-3.0, while its bundled meta_learner.pkl contains unrelated knee-label feature names. The checkpoint training split cannot be verified, so these weights remain reference-only. The Apache-2.0 [I079 fold-0 detector package](https://www.kaggle.com/datasets/lolarriven/solar-filament-i079-fold0-detector) likewise has no fold-membership details and is unused.
- Started an independent Swin-T encoder U-Net OOF run using the MIT [timm ImageNet-1K weights](https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k) (revision 4cc3a7275b50b53a7bec45f32c236ebe64227cff, SHA-256 fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1). The exact launch script is snapshotted under artifacts/model-experiments/native-unet-swin-tiny-oof-20261113/source/; run metadata records the RTX 5070, dependencies, source hash, and unchanged split digest.
- The Swin experiment trains 24 epochs across five grouped folds with one foreground-biased 512-pixel patch per observation and eight-view, 512-pixel tiled inference at stride 384. Its cache includes only 565 frozen training observations and 922 annotator views; the 142-observation holdout is not read. The 512-pixel CUDA forward/backward and tiled inference smoke check passed at 0.72 GiB peak allocation. Fold 0 is at epoch 9/24; training loss is not selection evidence.
- Candidate holdout evaluation, OOF threshold selection, and submission remain gated on the complete official grouped OOF scorer. No notebook score from the audited sources is treated as comparable evidence.

## 2026-11-13 02:05 UTC Swin-T progress and Kaggle preflight

- The live Swin-T OOF process remains active on the RTX 5070. Fold 0 completed its epoch-16 prediction pass over all 113 held-out observations; the epoch-20 pass has started. The model is still training/evaluating against grouped folds only; no holdout score is available.
- Re-ran the five-candidate submission dry-run. All five CSVs passed validation against source commit `9345dce3f010547bf0d0b37dd867c048f8a3be51` and frozen split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. Candidate OOF PQs range from `0.441988` to `0.452149`; the public ten-model U-Net uses a per-group crossfit protocol and is explicitly not an independent holdout.
- Kaggle reports 0 of 5 submissions used today, 30 lifetime submissions, and GPU quota refresh at `2026-11-14T00:00:00Z`. The Friday 19:05 Toronto GPU task will read quota before launch; the current remaining GPU hours are zero until that refresh.

## 2026-11-13 02:10 UTC leaderboard refresh and RF-DETR reference audit

- Saved a new full leaderboard archive at `artifacts/kaggle/leaderboard/20261113T020910Z/`; ZIP SHA-256 is `590c3ca0d5910bcb00246ad05382f6915f74d51572c11fd1576d36ee7be44c65`. It contains 1,505 teams. EquiNexus remains rank 326 at 0.39 with 30 submissions; IONode Online is first at 0.70, Tempest second at 0.64, and Sayandeep Dutta third at 0.62.
- Inspected the public [RF-DETR segmentation notebook](https://www.kaggle.com/code/jirkaborovec/filamentsegm-eda-rf-detr-seg-ptl), listed at 0.58 on the [competition code page](https://www.kaggle.com/competitions/filament-segmentation-2026/code?competitionId=143678&excludeNonAccessedDatasources=true&sortBy=scoreDescending). Its split groups physical JPEGs with a 15% `GroupShuffleSplit`; training keeps the most-annotated record per observation while validation retains all views. It calibrates confidence with a one-to-one matched-Dice proxy and adds an intensity or crop-U-Net refiner, so its validation is not directly comparable to our frozen official-PQ evaluation. The notebook's prose calls the model Nano while its executable configuration selects RF-DETR-Seg Small at 768px; no checkpoint from that run was adopted.
- Queried public kernels for current leaderboard leaders IONode Online (`ionodeonline`) and Sayandeep Dutta (`duttasam`); neither account exposes a solar-filament competition notebook, so no new top-team source or weights were available to audit.
- Upstream documents RF-DETR package code and Seg Nano/Small/Medium/Large as Apache-2.0. Our prior local fixed-holdout screen already tested RF-DETR-Seg Small at 384px for six epochs with mask downsample ratios 2 and 4; the best official PQ was `0.141106` versus incumbent `0.434930`, with 3.93 GB peak allocation. This public notebook remains architecture context only; we are not repeating the low-resolution run. [Upstream license and model table](https://github.com/roboflow/RF-DETR).

## 2026-11-13 02:19 UTC Swin-T OOF progress

- Fold 0 has completed its 16-, 20-, and 24-epoch prediction snapshots over all 113 held-out training observations. Fold 1 is training and has reached epoch 9/24. The run has not read the frozen 142-observation holdout and has no selected OOF threshold or holdout result yet.

## 2026-11-13 02:21 UTC Swin-T OOF progress

- The live process remains healthy and fold 1 has reached epoch 14/24. Fold 0 is the only completed OOF fold so far; the five-fold scorer sweep and frozen-holdout gate remain pending.

## 2026-11-13 02:25 UTC Swin-T OOF progress

- The process advanced through fold 1 epoch 18 and is training epoch 19/24. Kaggle/local submission tasks remain scheduled with wake-on-availability; the training helper is still holding the system awake while AC is connected.

## 2026-11-13 02:34 UTC Swin-T OOF progress

- Fold 1 finished all 24 epochs and completed its epoch-16 predictions for all 113 validation observations; epoch-20 predictions are underway. No holdout scoring has occurred.
- Independently checked the generated OOF manifest against the frozen split: all 565 training observations appear exactly once (113 in each of five folds), with zero overlap against the 142 holdout observations. Frozen split SHA-256 remains `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`; OOF fold-map SHA-256 is `18f4f3ed4ba96a7f6f218b662bfe028c358e2a6ff2ed2025fa7fc586743d7c99`.

## 2026-11-13 02:38 UTC fold and submission preflight

- Fold 1 has reached epoch-24 prediction 50/113; fold 0 remains the only completed fold. Re-ran the five-candidate batch dry-run: all files and hashes pass with source `9345dce3f010547bf0d0b37dd867c048f8a3be51` and the frozen split digest. The scheduled upload remains set for 23:58 Toronto.

## 2026-11-13 02:41 UTC Swin-T OOF progress

- Fold 1 completed all three prediction snapshots for 113 observations and wrote its fold-completion manifest. Fold 2 training has started; folds 3 and 4 and the official OOF sweep remain pending.
- Kaggle's live limits endpoint still reports 0/5 submissions used today (30 lifetime submissions). GPU time is exhausted until the published refresh at `2026-11-14T00:00:00Z`, five minutes before the scheduled Friday 19:05 Toronto T4 run.

## 2026-11-13 02:49 UTC prepared-submission visual review

- Rendered test-only overlays for the three supplied H-alpha images with the largest per-image instance-count spread across the five prepared candidates: `20140420195854Bh`, `20131027223034Lh`, and `20220714185352Th`. The public ten-model U-Net predicted more instances and short fragments (up to 17 vs. 11 on the first image); no test labels were used, so the batch remains unchanged and selection continues to rely on training-partition OOF evidence.
- Overlay PNG and exact-area review JSON are saved at `artifacts/submissions/filament-20261113-component-selector-rf-oof-batch/qualitative-review/`; JSON SHA-256 is `c1fbcafeadfa87af923477cf152371a83a9833b280932cb9592207aa55b8fd01`.

## 2026-11-13 02:51 UTC Swin-T OOF progress

- Fold 2 reached epoch 18/24. Two folds are complete; three folds and the official OOF postprocessing sweep remain pending. No holdout scoring has occurred.

## 2026-11-13 02:54 UTC training and submission check

- Swin-T fold 2 reached epoch 21/24. Training loss decreased from 0.2937 at epoch 18 to 0.2644 at epoch 20; this is training evidence only. The 24-epoch run saves predictions at epochs 16, 20, and 24 so longer training can be judged by grouped OOF metrics rather than training loss alone.
- Kaggle CLI authentication works from the repository virtual environment. The latest five submissions completed with public PQ 0.39 each. The next five-candidate batch passes dry-run validation and is scheduled for 2026-11-13 04:58 UTC (2026-11-12 23:58 Toronto). No new submission has been sent yet.

## 2026-11-13 02:56 UTC Swin-T OOF progress

- Fold 2 completed its 24 training epochs and started OOF inference; the epoch-16 snapshot reached 10/113 fold observations at the last check. Training loss was 0.2703 at epoch 24 versus 0.2644 at epoch 20, so it is not a reliable basis for extending training. Wait for grouped OOF scores at epochs 16, 20, and 24 before deciding whether to train longer. The independent 142-observation holdout remains untouched.

## 2026-11-13 03:07 UTC leaderboard and public-kernel audit

- Downloaded a fresh public leaderboard archive to `artifacts/kaggle/leaderboard/20261113T030708Z/filament-segmentation-2026.zip` (SHA-256 `1c07e585b25feb30d7bdc167af5dae5faa3d321e4f76ad0d745d16baac3fa590`). It contains 1,505 teams; EquiNexus is rank 326 at 0.39 with 30 submissions. Current top scores are IONode Online 0.70, Tempest 0.64, and Sayandeep Dutta 0.62; new Nov 12–13 entries appear at 0.56–0.58.
- Queried public competition kernels for six current leaderboard accounts; Kaggle returned no public competition notebooks for those accounts. Downloaded three public references under ignored `artifacts/kaggle/kernels/` for source review. SHA-256s: Phuong's 0.70 I079 handoff `b59aea47798e2ccba6af14f0e3bd7505dddec20ceb7d4c37704521bbde02e49c`; ASTH-Net `8930cd90379251c2095061f459bc7208198345e6fa53bb25036f75eae9381b9e`; and the user-linked `prvsiyan/notebook24a4dce69e` vote notebook `0ab2dc04f0e20d3687cd52405874977003b853609371fd03f7a32abde02e8a90`.
- Phuong's handoff consumes the public I079 detector/refiner weights and competition test images only. Its model card says training used only competition training images and annotations, with no public MAGFiLO/test-overlap labels. The notebook's `0.70` refers to a pre-August metric; the current leaderboard snapshot places team Fususu (`phuongncn`) at 0.36, rank 722. The local I079 test candidate follows the same family and thresholds but has a different CSV MD5 (`e115badd59b863f77a10d9be49ca21ed` vs. the T4 notebook's `09c0eae5fdff29a91f8a740475ecaf36`). Across all 1,299 matching IDs the local and Kaggle-T4 masks have mean RLE IoU 0.99875 (minimum 0.61847); its frozen-holdout score remains unknown, so keep it test-only and do not treat the old 0.70 title as current evidence.
- ASTH-Net uses competition training annotations and supplied test images, but its code sorts records by filename, trains on the first 160 records, and validates on 20 records after the 88% index. That is not the frozen observation-grouped split; the pulled notebook has no cached metric outputs despite comments claiming OOF tuning. Its high/low hysteresis and tangent-guided fragment bridging are ideas to screen on proper grouped OOF probabilities, not validation evidence to adopt. Its pulled metadata has no license field, so no code was copied.
- The current leaderboard places team `prvsiyan` at 0.40, rank 164. Its notebook is inference-only, but its vote includes the Tushhaaarrrr 0.37/0.40 U-Net outputs; the upstream training code uses MAGFiLO spine metadata, which the host prohibits as training ground truth. The vote therefore is not a compliant reference candidate, and its 0.40 should not guide our selection. The notebook has no grouped validation result.

## 2026-11-13 03:28 UTC Swin-T progress and submission preflight

- Verified the live five-fold Swin-T OOF process is using the RTX 5070 with AC power connected. Folds 0-2 have completion manifests; fold 3 finished training epoch 24 and its epoch-16 OOF inference, and the epoch-20 pass reached 50/113 observations. Fold 4 and the aggregate OOF scorer sweep remain pending. The 142-observation frozen holdout has not been evaluated.
- Revalidated both scheduled five-candidate batches in dry-run mode. All ten candidate files and hashes pass; the manifests retain pushed source `9345dce3f010547bf0d0b37dd867c048f8a3be51` and split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. The first batch is scheduled for Nov 12 23:58 Toronto; the Friday batch is scheduled for Nov 13 19:20 Toronto, after the Kaggle UTC quota refresh.
- Kaggle still shows the prior five submissions complete at public PQ 0.39 and no newer batch submitted. The keep-awake monitor remains active for the GPU run; the daily Conventional Commit and push task remains scheduled for Friday 19:50 Toronto.

## 2026-11-13 03:37 UTC Swin-T OOF fold progress

- Fold 3 completed all 16-, 20-, and 24-epoch predictions for its 113 validation observations and wrote its completion manifest. Fold 4, the fifth and final grouped fold, has started training and reached epoch 4/24. The aggregate OOF postprocessing sweep and independent holdout evaluation have not started; training loss remains diagnostic only.

## 2026-11-13 04:02 UTC Swin-T grouped OOF predictions complete

- All five folds now have completion manifests and epoch-16, epoch-20, and epoch-24 predictions for the 565 frozen training observations; each observation appears in one 113-observation validation fold. The fold-map SHA-256 is `18f4f3ed4ba96a7f6f218b662bfe028c358e2a6ff2ed2025fa7fc586743d7c99`, and the frozen split remains unchanged.
- Rechecked `oof-folds.csv` against `observation_oof_seed42_5fold.csv`: 565/565 training names covered, 0 fold-map mismatches, 0 holdout overlaps, and 113 observations in each fold.
- The runner is entering its 324-setting official grouped-scorer sweep. Its process is live and the GPU has been released; no OOF score summary or independent holdout score exists yet.

## 2026-11-13 04:16 UTC Friday batch manifest correction

- The current Nov 14 Toronto submission manifest contains four grouped-OOF public U-Net/YOLO11s fusion variants (PQ 0.443089, 0.441954, 0.427073, 0.431139) plus a test-only HDJoJo YOLO26x-seg candidate. All five passed the latest dry-run with source commit `9345dce3f010547bf0d0b37dd867c048f8a3be51` and the frozen split digest. No batch has been submitted yet.
- YOLO26x test inference passed full RLE validation for 1,436 instances over all 180 supplied test images. Its public model card reports training on the 707 competition training images, but exact fold membership is unknown; no holdout metric is claimed. The HDJoJo team score of 0.55 (rank 25 in the archived leaderboard) is context for selecting this test-only candidate, not a score for our CSV.
- Older Nov 11/12 notes describe CAGE r5 in this slot. The current manifest replaced it with YOLO26x. CAGE remains a license-eligible CC BY-NC 4.0 reference under the user's policy clarification, but its model card reports a biased 0.405 PQ split with duplicated annotator observations across train and validation, so that number is not independent validation evidence. Its test-only CSV is not in the current batch.
- The Swin-T OOF scorer sweep has advanced to 20/324 settings, with partial best PQ 0.294108 at epoch 16. This is not a final selection score; the sweep and independent holdout evaluation remain incomplete.

## 2026-11-13 04:20 UTC Kaggle API authentication check

- Kaggle API calls for submissions, quota, competition kernels, and competition listings all returned `Authentication required`. The local five-candidate manifests still pass dry-run validation, but the 23:58 Toronto upload task cannot be assumed to succeed until Kaggle authentication is restored. No submission upload was attempted after the authentication failure.

## 2026-11-13 04:24 UTC Swin-T scorer sweep progress

- The live CPU-side grouped scorer sweep reached 40/324 settings. Its current partial best is PQ `0.313127` at epoch 16, below the incumbent grouped OOF PQ `0.452149`; no selection decision is made until the sweep completes. The independent holdout remains unevaluated.

## 2026-11-13 04:41 UTC public notebook score audit

- The competition Code listing indexes "Solar Segmentation U-Net++ / EfficientNet-B4" at 0.67, while its linked `ruhul20/solar-segmentation-u-net-efficientnet-b3` notebook page reports public score 0.27 and best score 0.29 (version 2). The notebook lists a 7h 18m 48s T4x2 runtime and Apache-2.0 license. These values do not reconcile, so the listing score is not comparable evidence for model selection. No code or weights were adopted. [Competition Code listing](https://www.kaggle.com/competitions/filament-segmentation-2026/code?competitionId=143678&excludeNonAccessedDatasources=true&sortBy=scoreDescending) and [notebook page](https://www.kaggle.com/code/ruhul20/solar-segmentation-u-net-efficientnet-b3)

## 2026-11-13 04:47 UTC public leaderboard integrity audit

- A competition discussion reports that the same model and postprocessing scored 0.93 with zero predictions, compared with 0.72 at 1.9 predictions per image and 0.67 at 6.4. The host later replied that the Evaluation, Self Evaluation, and Leaderboard Ranking guidance had been updated. This explains why indexed public scores, including the 0.93 notebook listing, are not reliable evidence of segmentation quality. We continue selecting with the fixed PQ scorer and use public scores only as secondary context. [Discussion and host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809) and [updated leaderboard metric](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)

## 2026-11-13 05:08 UTC submission batch and uploader fix

- The 04:58 UTC scheduled task validated all five CSVs but stopped before any upload: `submit_prepared_batch.py` stripped the leading porcelain-status column and misread the first allowed sprint-log path. Fixed this in pushed commit `48f6487` by preserving leading whitespace and trimming only line endings. Updated both ignored manifests to source commit `48f6487f3caa97f4df95db417131e87d929201aa`; dry-run validation passed for both five-candidate batches with the frozen split digest unchanged.
- Retried the Nov 13 batch at 05:06-05:07 UTC. Kaggle accepted all five submissions: `58296190` (OOF PQ 0.452149), `58296201` (0.450632), `58296209` (0.448411), `58296219` (per-observation crossfit PQ 0.447799; not an independent holdout), and `58296226` (0.441988). The immediate status refresh showed all five as pending. The previous five completed submissions remain at public PQ 0.39 each.
- Swin-T grouped OOF postprocessing reached 120/324 settings with partial best PQ `0.349162` at epoch 16. This remains below the incumbent 0.452149 OOF candidate; the sweep and independent holdout gate are still incomplete.

## 2026-11-13 05:14 UTC submission scores and leaderboard snapshot

- Kaggle marked all five new submissions complete: `58296190` scored 0.39, `58296201` 0.38, `58296209` 0.38, `58296219` 0.39, and `58296226` 0.38. The detailed status refresh is archived in `artifacts/submissions/filament-20261113-component-selector-submissions.jsonl`.
- Downloaded the full 1,507-team [public leaderboard](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard) to `artifacts/kaggle/leaderboard/20261113T051300Z/`. The EquiNexus row is rank 327 at 0.39 with 35 submissions. The top three are IONode Online at 0.70, Tempest at 0.64, and Sayandeep Dutta at 0.62. ZIP SHA-256: `942743b3858ae5599ec5b3ff3b7e6ab03683f6e2bad937260fa0f30346bf2309`; CSV SHA-256: `ddb1dac25e1307b943c5e9544a95f76ff63e6f7b22cdc7be2d142b9dec14fac1`.
- These public scores are secondary to grouped validation: three candidates scored 0.38, two matched the team's existing 0.39, and none improved the prior public best. The public leaderboard's empty-prediction issue remains documented above.

## 2026-11-13 05:20 UTC Kaggle status refresh

- The scheduled Kaggle refresh confirms all five Nov 13 submissions remain complete at scores 0.39, 0.38, 0.38, 0.39, and 0.38. The 1,507-team leaderboard is unchanged: EquiNexus rank 327 at 0.39 with 35 submissions. The refreshed archive is in `artifacts/kaggle/leaderboard/20261113052000-utc/` (ZIP SHA-256 `d76d011d79435196b97b0779d74d3a95a9db10ee9ca9437b6cdfad171e9c4ae9`; CSV SHA-256 `ee4342b027bf72b12737ba62401c0feaecdddb47a1e33dd1dedd5ca2cf9bb204`).

## 2026-11-13 05:22 UTC Swin-T OOF sweep progress

- The live grouped scorer sweep reached 150/324 settings and improved its partial best to PQ `0.352145` at epoch 20. This remains below the incumbent `0.452149`; the epoch-24 settings, full sweep, and independent holdout gate are pending.

## 2026-11-13 05:28 UTC Swin-T OOF sweep progress

- The live grouped scorer sweep reached 160/324 settings and improved its partial best to PQ `0.357087` at epoch 20. It remains below the incumbent `0.452149`; the remaining settings and independent holdout gate are pending.

## 2026-11-13 05:32 UTC public 0.55 notebook source audit

- Pulled version 35 of the Apache-2.0 `Solar Filament Unet Segmentation | 0.55+` notebook into ignored `artifacts/kaggle/kernels/lamhuy8904-solar-filament-unet-segmentation-v35-20261113/` (SHA-256 `906841cbc0e1f50b89bda6648b1c7f85eff16975e62c3679a120141aa172a687`). Its Kaggle metadata lists no dataset or model sources and reports a 0.55 public score. [Notebook page](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55)
- Static source inspection found a literal compressed `CHAMPION_PAYLOAD`, decompressed into a dataframe, and the test submission loop copies its `segmentation_rle` entries by test-image stem. The loop's output does not come from the trained U-Net. A notebook comment alleges the payload contains ground-truth labels. We did not decode or use the payload; the notebook and score are excluded from model selection. [Comment reporting the leak](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55/comments)

## 2026-11-13 05:34 UTC Swin-T OOF sweep progress

- The live sweep reached 170/324 settings and improved its partial best to PQ `0.362439` at epoch 20. It remains below the incumbent `0.452149`; epoch-24 settings, the final OOF summary, and the holdout gate remain pending.

## 2026-11-13 05:38 UTC Swin-T OOF sweep progress

- The live sweep reached 180/324 settings and improved its partial best to PQ `0.366852` at epoch 20. It remains below the incumbent `0.452149`; the remaining epoch-20 and epoch-24 configurations are pending.

## 2026-11-13 05:47 UTC evaluation-aligned public model audit

- Audited the [public evaluation-aligned MAGFiLO notebook](https://www.kaggle.com/code/muhammadshoaibaltaf/magfilo-v1-evaluation-aligned-solar-filament-s). Its cross-annotator mask-IoU consensus boxes and crop refiner are candidate ideas, but its baseline checkpoint provenance is unknown, its random 15% split differs from our frozen validation, and the referenced checkpoints are not attached or discoverable as public Kaggle datasets. No weights or source code were adopted. Source SHA-256: `EE91EBBACF63BABD0B828CA9D285290BDE6260AD22945B4F33EE1D3931B28D3F`.

## 2026-11-13 05:48 UTC Swin-T OOF sweep progress

- The local RTX 5070 run reached 190/324 OOF postprocessing settings (best partial PQ `0.368644`, epoch 20); its promotion gate remains the incumbent grouped OOF PQ `0.452149`.

## 2026-11-13 06:14 UTC public learned-filter candidate

- Audited the latest public [Dariush Afshar U-Net notebook](https://www.kaggle.com/code/dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39), which uses the authorized CC BY-NC-SA 4.0 weights and a learned 16-feature component filter. Its receipt reports nested grouped OOF PQ `0.455236` and public PQ `0.39`; these are upstream figures, not EquiNexus frozen-split results.
- Archived its exact public test output as a batch candidate after the local RLE validator passed 1,148 masks over 180 supplied test images (177 with rows), including exact 2048×2048 round trips and zero overlaps. The exact CSV differs from the author's previously submitted file and remains unscored; no local holdout claim is made. Notebook SHA-256: `48a1a2f5266fe794886d3a875cee87dbab656c78405c52d863f264e9ccc9c94a`.

## 2026-11-13 06:40 UTC live leaderboard and public-method check

- Kaggle API lists EquiNexus at rank 327/1,507, score 0.39, with 35 submissions; the Nov 13 five-file batch scored 0.38-0.39. The archived 06:24 UTC leaderboard snapshot is under ignored artifacts/kaggle/leaderboard/20261113T062458Z/.
- Reviewed [prvsiyan's public five-family vote](https://www.kaggle.com/code/prvsiyan/notebook24a4dce69e): its test-only output is schema- and RLE-valid, but its attached model predictions have incomplete training provenance. The accessible Tushhaaarrrr 0.37 source uses competition spine annotations as an auxiliary training target, so the merged output is excluded under the restriction on other ground-truth metadata. The new pipeline may reuse only image-derived features and mask supervision.
- The local Swin-T grouped OOF postprocessing sweep has reached 290/324 settings with best partial PQ 0.377095 at epoch 20, below the 0.452149 incumbent. The independent holdout remains untouched.

## 2026-11-13 06:58 UTC local OOF results

- The completed five-fold Swin-T run selected epoch 24 and postprocessing threshold 0.70, minimum area 400, closing kernel 9, at grouped OOF PQ `0.379023` on the 565 training observations. This is below the existing `0.452149` incumbent, so it is rejected without scoring the independent holdout.
- The mask-only annotator-mean EfficientViT-B2 run is still training on competition mask annotations only. Epoch 1 loss was `0.769184`; epoch 2 was at step 300/452. It has no completed OOF metric yet and is not eligible for promotion.
- Split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e` remains unchanged. The Nov 13 Kaggle batch is complete (five submissions, public scores 0.38–0.39); the next UTC-day five-candidate batch is prepared for the 19:20 Toronto submission window after Kaggle quota reset.
- The Apache-2.0 I079 detector/refiner has confirmed competition-training-only provenance, but its fold map is undocumented. Our valid test-only run differs from the notebook's hash-locked artifact; Kaggle records disagree between 0.70 in the competition code index and 0.34 on the version-1 notebook page, while the prior EquiNexus output scored 0.34. It stays out of the next batch.
- Audited Huendao's recent ten-model U-Net output from the authorized Dariush CC BY-NC-SA weights: 1,222 masks passed exact RLE round trips across the full test set, with 177 images receiving predictions. Its cited OOF/public metrics are upstream only; we kept the currently prepared batch unchanged because it already contains a Dariush-derived candidate.

## 2026-11-13 07:33 UTC EfficientViT-B2 grouped OOF relaunch and evening task check

- The earlier 24-epoch soft-mean run had completed three epochs in its first fold but its observed epoch time (about 578 seconds) made a full five-fold run unlikely to fit the remaining sprint window. Preserved that partial output and launched a fresh 16-epoch OOF run from the prepared cache at 07:25 UTC, writing to `artifacts/model-experiments/native-unet-softmean-oof-20261113-e16/`. It uses the same 565-observation training partition, 922 competition annotator views, mask-only per-pixel mean targets, and frozen split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. No epoch metric or OOF score is available yet; the independent holdout remains unused.
- At 07:32 UTC the local RTX 5070 was at 100% utilization with 7,302/8,151 MiB allocated and AC connected. The 19:10 Toronto cutoff preview now identifies both Python processes for this run, so they will be stopped if still active at the authorized sprint cutoff.

## 2026-11-13 07:33 UTC submission and checkpoint preflight

- Revalidated the five Nov 14 candidate files against the current manifest. Local HEAD, `origin/main`, and manifest source all match `48f6487f3caa97f4df95db417131e87d929201aa`; the frozen split digest is unchanged, and all five candidates pass the uploader dry-run. The full preflight record is in `artifacts/automation/submit-batch-20261114.jsonl`.
- Disabled the expired Nov 13 one-time submission task, which fired early on Nov 12 at 23:58 Toronto and failed before upload. The five Nov 13 candidates were submitted separately with the fixed uploader and verified in the Kaggle API. The Nov 14 batch uses its own manifest and scheduled task at Nov 13 19:20 Toronto, after the UTC-day Kaggle quota reset.
- Corrected the ignored cutoff script to include `train_native_unet_softmean_grouped.py` and the daily-checkpoint script to read the new 16-epoch run's summary path. PowerShell syntax checks passed; tracked changes remain limited to the two sprint logs.
## 2026-11-13 07:55 UTC patch-resolution redesign and topology ablation

- Stopped the full-frame EfficientViT-B2 grouped OOF attempt after it reached only epoch 1, step 50/452, over about 20 minutes; no epoch checkpoint or validation score was produced. Its partial cache and logs remain archived. The sprint cutoff script still covers the trainer family.
- Reworked the local runner to sample 512×512 training patches from the cached 1888×1888 training crop: 80% of patches center on a pixel with soft target value at least 32, and the rest are sampled uniformly. A local CUDA forward/backward smoke measured 0.071 and 0.069 seconds for the two warmed-up 512px steps after a 2.48-second first step, with 0.71 GiB peak allocation. Full-crop 1888px inference also fit at 0.89 GiB and took 0.13 seconds. The frozen split and training labels are unchanged; the new OOF run will record checkpoint metrics at multiple epochs.
- Evaluated the public connected-component cleanup idea using our grouped OOF support-0.55 masks. The baseline reproduced PQ `0.4521491778`; largest-component and secondary-island thresholds of 5%, 10%, and 20% all changed zero masks and retained the same score, so it offers no improvement for this candidate. Output is under `artifacts/model-experiments/component-island-oof-20261113/support055/`.
- Audited the public Felipe KNBIS solution repository. Its July 20 technical report describes the older `instance_wise_dice` leaderboard metric; the competition announced a PQ methodology update on Aug 7, so its historical 0.71 should not guide our current official-PQ selection. The timm model card documents the EfficientViT-B2 ImageNet weights as Apache-2.0. We used the public report only as an idea source and independently implemented the connected-component ablation. [Historical report](https://github.com/FELIPEACASTRO/solar-filament-2026/blob/master/report/TECHNICAL_REPORT.md), [official scoring update](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements), [timm weight license](https://huggingface.co/docs/timm/main/en/models/efficientvit-mit).
- The ignored daily checkpoint script now reads the new patch-OOF output directory so the scheduled Conventional Commit will include its final result when available.
## 2026-11-13 08:00 UTC active run

The 40-epoch, 512px patch-based EfficientViT-B2 five-fold OOF run is active on the local RTX 5070. As of 10:06 UTC, folds 0-2 have completed training and all four eight-view inference passes for their 113 held-out training observations each, with completion manifests present. Fold 3 has saved epoch-16 and epoch-20 checkpoints and is training epoch 21/40. The official grouped OOF sweep is pending, and no validation metric is available yet. Training uses only competition-provided mask annotations from the fixed 565-observation training partition and leaves the 142-observation holdout untouched. See the detailed launch record in `docs/experiment-log.md`. AC power is connected, and sleep-on-AC is disabled for the unattended run.

- Promotion comparator for this run: the strongest leakage-free grouped U-Net/YOLO11s family vote is OOF PQ `0.459116` and independent frozen-holdout PQ `0.454668`. A standalone patch model or an OOF-only selected fusion must first exceed the grouped-OOF value; only then run one preselected holdout evaluation and a paired observation bootstrap against the `0.454668` incumbent. Threshold and fusion selection will not inspect the holdout.


## 2026-11-13 09:10 UTC public-kernel reference audit

- Archived four additional public references and a 100-row Kaggle code listing. The recent YOLO26 full-resolution notebook uses a random grouped 20% split and Hungarian PQ; the EdgeAttNet and ASTH-Net notebooks also use one-to-one/averaged PQ variants. Spongeboi's ResNet-18 U-Net uses one annotator view for validation and an FP/FN formula that can become negative on split/merge cases. None supplies comparable evidence on the frozen split, so no weights or outputs were promoted.
- Full source hashes, method details, and exclusion reasons are in `docs/experiment-log.md`; downloaded public source and metadata remain in ignored `artifacts/kaggle/reference-audit/leaderboard-kernels-20261113/`.


## 2026-11-13 09:28 UTC leaderboard refresh

- Kaggle now lists 1,511 teams; EquiNexus is rank 327 at `0.39` with 35 submissions. The five Nov 13 submissions completed at `0.38-0.39` and did not improve the public best. Current leaders are at `0.70`, `0.64`, and `0.62`; the archived leaderboard and submission history are in `artifacts/kaggle/leaderboard/20261113T092802Z-utc/`.


## 2026-11-13 10:03 UTC submission preflight

- The Nov 14 five-candidate batch passed a fresh source/split/hash dry run. All five stored RLE reports match their CSVs, cover all 180 test images, round-trip every mask, and report no duplicate IDs, empty masks, or pixel overlaps. Three candidates have EquiNexus frozen OOF PQ below the `0.459116` incumbent; two remain test-only external-model outputs without our frozen-split scores. Batch details and the replacement gate are in `docs/experiment-log.md`.


## 2026-11-13 10:23 UTC patch OOF progress

- Fold 3 completed 40 training epochs and started the epoch-16 OOF inference pass (20/113 observations at the latest check). The five-fold map covers the frozen 565-observation training partition exactly once, with no overlap against the 142-observation holdout. The patch run has no aggregate score yet; folds 0-2 are complete, while fold 3 inference, fold 4, and the OOF scorer sweep remain pending.
## 2026-11-13 10:26 UTC patch OOF progress

- Fold 3 is complete with all four eight-view prediction snapshots and a fold-completion manifest. Fold 4 has started training. No aggregate OOF score exists yet; the frozen holdout remains untouched.
## 2026-11-13 10:28 UTC public leaderboard refresh

- EquiNexus remains rank 327/1,511 at public score `0.39`, with 35 submissions. The current leaders are at `0.70`, `0.64`, and `0.62`. All five Nov 13 submissions are complete at `0.38-0.39`; the fresh leaderboard and submission archive is recorded in `docs/experiment-log.md`.
## 2026-11-13 10:31 UTC patch trainer archived

- The exact active patch trainer is now archived at `scripts/experiments/train_native_unet_softmean_patches.py` with source SHA-256 `2c89dffea15071af547106d42298360ad0adc92e2c92f486b0f12f28b560d8c6`; its command-line help smoke check passed. The scheduled daily checkpoint now includes this file.
## 2026-11-13 10:49 UTC evening task guards verified

- The scheduled Kaggle T4 and five-candidate submission jobs now permit only the pending sprint logs and exact local patch-trainer archive. Their PowerShell syntax and path-guard simulations pass; no source commit or submission timing changed.
## 2026-11-13 10:58 UTC patch OOF final fold training complete

- All five folds have completed their 40 training epochs. Fold 4 is now generating its four eight-view OOF snapshots; aggregate grouped scoring and the independent-holdout gate remain pending.
## 2026-11-13 11:03 UTC patch OOF prediction complete

- All five folds now have complete training and four-snapshot OOF predictions for all 565 training observations. The official grouped-PQ sweep is underway; the fixed 142-observation holdout remains untouched.
## 2026-11-13 11:12 UTC grouped OOF sweep started

- The four-snapshot postprocessing sweep is running CPU-only and has evaluated 10/432 settings so far (partial best PQ `0.071089`, epoch 16). This is incomplete and is not used to judge promotion; the holdout remains untouched.
## 2026-11-13 11:14 UTC Kaggle quota check

- GPU quota remains exhausted until midnight UTC; TPU quota is 20 hours. The scheduled T4 run remains five minutes after the reset, and the five-submission and checkpoint tasks are still enabled.

## 2026-11-13 11:23 UTC leaderboard refresh

- The latest authenticated leaderboard snapshot contains 1,510 teams. EquiNexus is rank 326 at `0.39` with 35 submissions; the five Nov 13 submissions completed at `0.38-0.39`. The archived ZIP and extracted CSV are in `artifacts/kaggle/leaderboard/20261113T112245Z/`.
- The local cross-fitted predictions are complete; the CPU scorer sweep remains live at 20/432 on its last emitted progress line. The frozen holdout has not been used for this model.
- Refreshed the top-100 Kaggle code list; it has the same 100 references as the 10:35 UTC snapshot. The I079 topology-safe handoff source is byte-identical to the previously audited version, so there is no newly published technique to add.

## 2026-11-13 11:30 UTC grouped OOF progress

- The cross-fitted local scorer sweep reached 30/432 settings; partial best PQ is `0.080135` at epoch 16 and is not used for promotion. The fixed holdout is untouched while the sweep runs.

## 2026-11-13 11:38 UTC promotion path

- Prepared a final-fit/test runner gated on OOF improvement, a better independent holdout, and a positive paired observation-bootstrap interval. If all gates clear, it trains on all 707 competition training observations, validates all 180 supplied test images/RLEs, and replaces only the weakest OOF-scored candidate in the Nov 14 batch. Its watcher is live; no test predictions or holdout tuning have occurred.
- The OOF scorer reached 40/432 settings with partial PQ `0.083138`; the sweep is incomplete and not a promotion metric.

## 2026-11-13 11:44 UTC OOF sweep progress

- The live cross-fitted scorer sweep advanced to 50/432 settings. Its partial best PQ is `0.085606`, with no selection or holdout result available yet.

## 2026-11-13 11:51 UTC OOF sweep progress

- The cross-fitted scorer sweep advanced to 60/432 settings; partial best PQ is `0.088203` at epoch 16. This is not a selection result.

## 2026-11-13 11:59 UTC OOF sweep progress

- The active scorer sweep reached 70/432 settings; partial best PQ remains `0.088203` and is not used for promotion.

## 2026-11-13 12:06 UTC OOF sweep progress

- The scorer reached 80/432 settings; partial best PQ is `0.089747` at epoch 16. The holdout is still untouched.

## 2026-11-13 12:13 UTC OOF sweep progress

- The scorer reached 90/432 settings; partial best PQ is `0.090753` at epoch 16. No final OOF result is available.

## 2026-11-13 12:21 UTC OOF sweep progress

- The scorer reached 100/432 settings; partial best PQ is `0.092743` at epoch 16. No promotion decision is available.

## 2026-11-13 12:29 UTC OOF sweep progress

- The scorer reached 110/432 settings with partial best PQ `0.096411`. Available RAM was 4.37 GiB and the GPU remained idle; there is no final OOF score yet.

## 2026-11-13 12:36 UTC OOF sweep progress

- The scorer reached 120/432 settings; partial best PQ rose to `0.127452` at epoch 20. The full sweep is still in progress.

## 2026-11-13 12:43 UTC OOF sweep progress

- The scorer reached 130/432 settings; partial best remains `0.127452` at epoch 20 and is not a promotion metric.

## 2026-11-13 12:47 UTC source preflight

- `main` and `origin/main` both remain at `48f6487f3caa97f4df95db417131e87d929201aa`. The current pending files are within the scheduled checkpoint allowlist and pass `git diff --check`.

## 2026-11-13 12:51 UTC OOF sweep progress

- The cross-fitted sweep reached 140/432 settings; partial best remains `0.127452` at epoch 20. The fixed holdout remains untouched.

## 2026-11-13 12:59 UTC grouped OOF sweep progress

- The active grouped OOF postprocessing sweep reached 150/432 settings; partial best PQ remains `0.127452` at epoch 20. The run remains CPU-active and responsive, with about 4.27 GiB RAM available and the RTX 5070 idle. No holdout evaluation has started.

## 2026-11-13 13:09 UTC grouped OOF sweep progress

- The active grouped OOF postprocessing sweep reached 160/432 settings; partial best PQ remains `0.127452` at epoch 20. Process CPU time continues to increase, RAM availability remains stable near 4.3 GiB, and the GPU is idle. The holdout remains untouched.

## 2026-11-13 13:14 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 170/432 settings; partial best PQ remains `0.127452` at epoch 20. The Python process remains CPU-active, about 4.4 GiB RAM is available, and no holdout evaluation has started.

## 2026-11-13 13:22 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 180/432 settings; partial best PQ remains `0.127452` at epoch 20. The run remains active; no holdout evaluation has started.

## 2026-11-13 13:31 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 190/432 settings; partial best PQ remains `0.127452` at epoch 20. The CPU process is active and the independent holdout remains untouched.

## 2026-11-13 13:40 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 200/432 settings; partial best PQ remains `0.127452` at epoch 20. The process continues to accrue CPU time, and the holdout watcher remains gated on the final OOF result.

## 2026-11-13 13:48 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 210/432 settings; partial best PQ remains `0.127452` at epoch 20. The Python process is still CPU-active, with no holdout evaluation yet.

## 2026-11-13 13:56 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 220/432 settings; partial best PQ remains `0.127452` at epoch 20. The scorer process is CPU-active and the holdout remains untouched. Available RAM is about 3.5 GiB, so memory is being watched.

## 2026-11-13 14:03 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 230/432 settings. Partial best PQ increased to `0.138875` at epoch 24. The process remains CPU-active; about 3.47 GiB RAM is available, and the RTX 5070 remains idle. The independent holdout is untouched.

## 2026-11-13 14:09 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 240/432 settings; partial best PQ increased to `0.142573` at epoch 24. The scorer remains CPU-active. Available RAM is about 2.9 GiB on this poll and has fluctuated; the process has stayed responsive, but memory remains under observation. The holdout is untouched.

## 2026-11-13 14:16 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 250/432 settings; partial best PQ remains `0.142573` at epoch 24. The CPU process remains active, its private memory is about 5.2 GiB, and available system RAM is about 2.9 GiB. The process is responsive and the holdout remains untouched.

## 2026-11-13 14:24 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 260/432 settings; partial best PQ remains `0.142573` at epoch 24. The Python process continues to accrue CPU time, and the holdout remains untouched.

## 2026-11-13 14:32 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 270/432 settings; partial best PQ remains `0.142573` at epoch 24. The process remains active, with about 2.8 GiB RAM available on the latest check. The independent holdout remains untouched.

## 2026-11-13 14:38 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 280/432 settings; partial best PQ remains `0.142573` at epoch 24. The run remains CPU-active and memory has stayed near 2.8 GiB available over the last checks. The holdout is still untouched.

## 2026-11-13 14:47 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 290/432 settings; partial best PQ remains `0.142573` at epoch 24. The scorer process remains active and the holdout remains untouched.

## 2026-11-13 14:53 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 300/432 settings; partial best PQ remains `0.142573` at epoch 24. The scorer remains responsive, about 2.8 GiB RAM is available, and the independent holdout remains untouched.

## 2026-11-13 15:02 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 310/432 settings; partial best PQ remains `0.142573` at epoch 24. The process remains CPU-active, and the holdout gate is still waiting for completion.

## 2026-11-13 15:08 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 320/432 settings; partial best PQ remains `0.142573` at epoch 24. The process remains active, with about 2.9 GiB RAM available. The holdout remains untouched.

## 2026-11-13 15:16 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 330/432 settings; partial best PQ remains `0.142573` at epoch 24. Available RAM is about 1.2 GiB while the Python working set is about 19.5 GiB; both were stable across two checks, the process advanced, and roughly 16.7 GiB page-file space remains free. The holdout remains untouched.

## 2026-11-13 15:25 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 340/432 settings; partial best PQ remains `0.142573` at epoch 24. The Python working set is about 19.5 GiB and available RAM about 1.3 GiB, stable across checks with page-file capacity available. The run remains active and the holdout untouched.

## 2026-11-13 15:34 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 350/432 settings; partial best PQ remains `0.142573` at epoch 24. The process is active, and memory remains tight but stable. The holdout remains untouched.

## 2026-11-13 15:41 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 360/432 settings; partial best PQ remains `0.142573` at epoch 24. About 1.3 GiB RAM is available with roughly 16.7 GiB page-file capacity remaining; the scorer is responsive and the holdout remains untouched.

## 2026-11-13 15:46 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 370/432 settings; partial best PQ remains `0.142573` at epoch 24. Available RAM and process working set remain stable at roughly 1.3 GiB and 19.4 GiB. The holdout is still untouched.

## 2026-11-13 15:53 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 380/432 settings; partial best PQ remains `0.142573` at epoch 24. The scorer is still CPU-active, with about 1.3 GiB RAM available; the holdout remains untouched.

## 2026-11-13 16:00 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 390/432 settings; partial best PQ remains `0.142573` at epoch 24. The CPU process remains active and available RAM has held near 1.3 GiB. The holdout remains untouched.

## 2026-11-13 16:08 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 400/432 settings; partial best PQ remains `0.142573` at epoch 24. Available RAM is about 1.2 GiB, with 16.7 GiB page-file capacity remaining. The process remains CPU-active and the holdout untouched.

## 2026-11-13 16:15 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 410/432 settings; partial best PQ remains `0.142573` at epoch 24. The run is active, and the independent holdout remains untouched.

## 2026-11-13 16:22 UTC grouped OOF sweep progress

- The grouped OOF postprocessing sweep reached 420/432 settings; partial best PQ remains `0.142573` at epoch 24. The process remains responsive, available RAM is about 1.25 GiB, and the holdout is untouched.

## 2026-11-13 16:29 UTC grouped OOF sweep complete

- The five-fold grouped OOF run finished all 432/432 postprocessing settings in 30,615.611 seconds. It used the frozen split (`30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`), 565 training observations, and 922 annotator views. Training ran on the RTX 5070 Laptop GPU; the largest fold peak was about 806 MiB, with 2.71 GPU-training hours in total.
- The best setting was epoch 24, probability threshold `0.35`, minimum area `400`, and closing kernel `9`. OOF PQ was `0.142573` across the 922 views (TP 1,865; FP 9,018; FN 4,623), below the incumbent OOF gate of `0.459116`. The promotion gate correctly recorded `holdout_not_run`; this candidate was not evaluated on or selected by the fixed holdout.
- Run summary and sweep are under `artifacts/model-experiments/native-unet-softmean-patches512-oof-20261113/`; the gate decision is in `artifacts/automation/patch-holdout-gate-20261113.jsonl`. Source commit was `48f6487f3caa97f4df95db417131e87d929201aa`, with trainer SHA-256 `2c89dffea15071af547106d42298360ad0adc92e2c92f486b0f12f28b560d8c6`. System RAM recovered to about 20.8 GiB after the process exited.

## 2026-11-13 16:35 UTC live Kaggle refresh

- The Kaggle API reports all five Nov 13 submissions complete with public scores `0.38`, `0.39`, `0.38`, `0.38`, and `0.39`. EquiNexus is rank `327/1512` at score `0.39`, with 35 team submissions. Current leaders include IONode Online `0.70`, Tempest `0.64`, and four teams at `0.62`.
- Archived the full leaderboard and submission listing under `artifacts/kaggle/leaderboard/20261113163508-utc/`; archive SHA-256 is `e435aae24dd4d4f9f418133dd5e84114ba9c75b6906267f2596b5f76342d7c61`. The five-entry Nov 14 batch still passes its dry run; its public-model candidates are not eligible for promotion to the separate local final-fit path without fresh frozen-split/holdout evidence.

## 2026-11-13 16:38 UTC submission preflight refresh

- Revalidated the five-entry Nov 14 batch against source commit `48f6487f3caa97f4df95db417131e87d929201aa` and frozen split `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`. All five hash-locked CSVs pass dry-run validation; this did not upload them. The learned-filter and YOLO26x entries remain marked with their public-source/test-only validation limits.

## 2026-11-13 16:42 UTC review of the updated prvsiyan notebook

- Pulled `prvsiyan/notebook24a4dce69e` v18, source SHA-256 `622c847e5c715e091a4146c1cff777d8ef1511ae7f60c1d25b6192d709412629`. Its Nov 13 update combines six public model families, retaining rows supported at IoU `>0.5` by at least three families and requiring IoU `>=0.6` from every supporter when exactly three agree. The author reports public PQ `0.40` and explicitly says the notebook trains no models and reads no labels.
- The attached model is Apache-2.0 I079 weights and the cited public outputs are reported as Apache-2.0. This is a test-time public-output vote, so its public-board result is exploratory evidence; the notebook itself cautions that a vote over other authors' CSVs cannot reproduce a final model-and-weights deliverable. Kaggle reported this kernel still `RUNNING` at 16:42 UTC, and no output files were available to evaluate or add to the batch.
- Source and metadata are archived under `artifacts/kaggle/kernels/20261113163900-utc/prvsiyan/`.

## 2026-11-13 17:06 UTC public vote candidate added to the Nov 14 batch

- The updated `prvsiyan` Kaggle kernel completed on CPU in 5,792.47 seconds with training disabled. Its six-family test-output vote produced 1,274 instances; source notebook SHA-256 is `622c847e5c715e091a4146c1cff777d8ef1511ae7f60c1d25b6192d709412629`. The exact output's MD5 matches its consensus report; SHA-256 is `2a24d525c22c3f0891a31ac5d060f73fb2399f8b3c1aff3526d423cc1c04cd9c`.
- Independent checks decoded all 1,274 masks at 2048×2048, confirmed unique IDs, pixel-disjoint masks, and coverage of all 180 supplied test images. No test labels were read. The exact CSV has no frozen OOF/holdout or public score; the notebook's `0.40` statement refers to an earlier five-family version and is not attributed to this output.
- Replaced the weakest frozen-OOF-scored entry (public U-Net + YOLO11s confidence 0.50, OOF PQ `0.431139`) with this explicitly exploratory public-vote candidate. The five-entry manifest passed dry-run validation again; no submission has been uploaded yet. Candidate and provenance: `artifacts/submissions/prvsiyan-six-family-consensus-20261114/`.

## 2026-11-13 17:08 UTC public-output source consistency check

- The six-family notebook's own Mask R-CNN 0.70-cut CSV reproduces its stated upstream MD5 exactly (`09c0eae5fdff29a91f8a740475ecaf36`). This strengthens the source lineage for the vote but does not establish a score for the final consensus CSV.

## 2026-11-13 17:17 UTC Kaggle access and Nov 14 batch preflight

- The saved Kaggle refresh credential renewed the expired access token without opening a browser. Competition submissions, submission limits, and accelerator quota are readable again.
- Kaggle reports 0.00 GPU hours remaining until `2026-11-14T00:00:00+00:00` (19:00 Toronto), so the scheduled T4 run remains queued for the quota reset. The current competition submission limit is also exhausted for today and resets at the same time.
- Revalidated all five hash-locked Nov 14 CSVs against source commit `48f6487f3caa97f4df95db417131e87d929201aa` and the frozen split digest. The dry run passed; no submissions were uploaded in this check.
## 2026-11-13 17:21 UTC leaderboard and public-source refresh

- The archived public leaderboard contains 1,512 teams. EquiNexus remains rank 327 at 0.39 with 35 submissions; the leading scores are unchanged at 0.70, 0.64, and 0.62. The archive is under `artifacts/kaggle/leaderboard/20261113172000-utc/` (CSV SHA-256 `9bdae09e504282cf393e023674f2cddaeb45bb84b66524fb675ac79c46f6e1a2`).
- Rechecked the public I079 topology-safe handoff. Its generated CSV SHA-256 `8c064a0da930de9add44e63eef1b0cb4ca85325ed74f224403938232f4beb2dc` matches EquiNexus submission 58110863, which scored 0.34; it is a repeat, not a new 0.70 result. The six-family vote remains a separate, unscored test-only candidate.
- Audited the latest public notebook releases. The updated Huy notebook uses the already reviewed ten-model public U-Net weights; Jamie Nojek publishes a scratch-U-Net reproducibility release with a non-comparable split; and the MAGFiLO-trained standalone demo does not document separation from competition test labels. None replaces a candidate in the current five-entry batch.

## 2026-11-13 17:53 UTC ASTH-Net public notebook review

- Pulled `koushikkumardinda/asth-net-anisotropic-spine-tangent-hysteresis` from Kaggle. The source notebook is archived at `artifacts/kaggle/kernels/20261113-asth-net-source/` with SHA-256 `8930cd90379251c2095061f459bc7208198345e6fa53bb25036f75eae9381b9e`.
- The notebook proposes limb-darkening correction, dual-threshold connected components, tangent-based fragment bridging, and a small ResUNet. Its local screen trains for three epochs on at most the first 160 chronologically sorted annotation records and evaluates 20 later records; this is not our frozen observation-grouped split.
- Its custom metric performs one-to-one matching, while the organizer's scorer counts every qualifying pair above IoU 0.5. The notebook provides no EquiNexus frozen-split result or retrievable trained checkpoint, so it does not qualify for the submission batch. We did not copy its code or adopt its reported score.
- The second local SWEFil download under `data/raw/SWEFil` matches all 557 source files under the canonical `data/external/SWEFil` path byte-for-byte; the external directory also contains the audit manifests. A recursive cleanup attempt was blocked by the environment, so the redundant ignored copy remains inside `data/` pending workspace cleanup.
## 2026-11-13 18:00 UTC review of public V18 multi-annotator detector notebook

- Pulled `muhammadshoaibaltaf/magfilo-v1-evaluation-aligned-solar-filament-s`; archived source SHA-256 `ee91ebbacf63babd0b828ca9d285290bde6260ad22945b4f33ee1d3931b28d3f` under `artifacts/kaggle/kernels/20261113-magnfilo-v1-eval-source/`.
- V18 groups duplicate image records by physical filename and merges cross-annotator filament instances when mask IoU is at least 0.40, then fine-tunes a YOLO detector while keeping the crop U-Net fixed. Its scorer counts every IoU > 0.5 pair and reuses predictions against each annotator view, matching the organizer's key PQ semantics.
- The notebook reserves 40 calibration and 67 comparison photos from a 107-photo validation group, but acknowledges those images were used in earlier project selection. It depends on public V10/V12 checkpoints whose exact provenance and license were not audited here. Kaggle reports the kernel as `CANCEL_ACKNOWLEDGED`; no run artifacts or submission CSV were downloadable, so the method is recorded for future evaluation and is not promoted.

## 2026-11-13 18:24 UTC public five-family consensus rebuild (later withdrawn)

- Audited the six public CSV inputs used alongside the notebook's Mask R-CNN output. Excluded the YOLO11 crop-refiner output because its inference depends on Tienti checkpoints whose training-data provenance could not be independently verified. Updated the Kaggle notebook inputs and code so the excluded source is not loaded.
- The first local build stopped at its hash guard because it had selected our newer topology-safe I079 variant. Switched to the archived `submission_thr070.csv` from the reviewed prvsiyan run, which matches the notebook's expected MD5 `09c0eae5fdff29a91f8a740475ecaf36`.
- Rebuilt the vote with five families: U-Net ensemble, Mask R-CNN, YOLO26 + U-Net++, YOLO26x-seg, and EdgeAttNet. This was test-only; the local run did not train or read competition test annotations. It used strict IoU > 0.5, required at least three families and IoU >= 0.6 from every supporter when exactly three agreed, clipped overlaps, and backfilled the best-voted mask only for otherwise-empty images.
- The candidate contains 1,227 nonempty 2048x2048 RLE masks over all 180 test images. Masks are pixel-disjoint per image; 6 masks were clipped and 2 images backfilled. The CSV SHA-256 is `857ecf2f6ae2fca5b9192a8a9d415413ab2bc87848cf966525f2fc7ce6916075` (MD5 `0551032eb99248efaacbc024cbce8378`). No frozen-split score or leaderboard result is claimed.
- Replaced slot 4 in the Nov 14 submission manifest with this candidate. The complete five-candidate manifest dry-run passed at 2026-11-13 18:24 UTC with the frozen split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- Visual QA is archived as a montage and a full-resolution example. Spot checks covered images `20110130110334Ch`, `20140118180214Mh`, and `20140519195834Ch`; the masks render as thin, localized candidates. This confirms rendering and coverage only, not completeness or expert-quality continuity.

## 2026-11-13 18:37 UTC submission and checkpoint readiness

- Kaggle API authentication is working. The live read-only quota check reports 5 submissions used and 0 available until 2026-11-14 00:00 UTC (19:00 Toronto); the five-candidate upload remains scheduled for 19:20 Toronto.
- The submit runner requires the solution source commit to be pushed and the code worktree clean. Scheduled the day's single commit and push for 18:55, before the Kaggle run and upload; the later 19:50 task now succeeds as a no-op if the repository is already clean. The Swin-T trainer snapshot is archived under `scripts/experiments/` so the source-window task does not create a second tracked change.


## 2026-11-13 19:12 UTC audit of newly surfaced public kernels

- Reviewed the current Kaggle code list sorted by public score. Lamhuy's notebook advertises 0.55, but its current source decodes an embedded base85/lzma `CHAMPION_PAYLOAD` CSV and copies its precompiled RLE masks to the test image IDs; the trained U-Net does not generate the submitted masks. A public comment alleges the payload is hardcoded test ground truth. Its provenance is unverified, so the source and output are excluded under the host's test-label restriction.
- Foysal's Apache-2.0 notebook trains an ImageNet-initialized ResNet34 U-Net on binary unions of competition training annotations, with a random 80/20 filename split and Dice/BCE validation. It does not report frozen-split official instance PQ, so it remains a reference only.
- Pavlo's 0.90 public-score baseline also uses a random image split and emits empty masks when no components survive thresholding. The public score is not treated as segmentation evidence. Ruhul's U-Net++/EfficientNet-B4 page reports 0.27 and a 7h18m runtime. None of these sources replaces a validation-supported candidate.
- Notebook source snapshots and SHA-256 provenance are archived under `artifacts/kaggle/reference-notebooks/20261113-public-kernel-audit/`; the structured audit is `source-audit.json`. No embedded payload, public MAGFiLO test labels, or candidate CSV from these notebooks was used.

## 2026-11-13 20:00 UTC training-duration decision

- The five-fold EfficientViT-B2 patch U-Net run already evaluated checkpoints through 40 epochs and consumed 30,615.6 seconds (about 8.5 hours). Its best grouped OOF PQ was 0.142573 at epoch 24; the best epoch-40 configuration scored 0.112639. This is far below the incumbent OOF PQ 0.459116, so more epochs on this same run are not justified and the model remains unpromoted.
- Longer training can help when validation metrics and training diagnostics show underfitting, but epoch count alone is not a goal. For this model the OOF trend declined by epoch 40, and the run has already had a substantial training budget. Future longer runs should first show improving fold-level evidence on the frozen OOF protocol before consuming a larger budget.

## 2026-11-13 20:08 UTC leaderboard and public-source refresh

- Archived a fresh Kaggle snapshot with 1,514 leaderboard teams. EquiNexus is rank 328 at 0.39 with 35 total submissions. The displayed leaders are IONode Online at 0.70, Tempest at 0.64, and several teams at 0.62. The snapshot, full leaderboard, latest notebook list, and our submission history are in artifacts/kaggle/leaderboard/20261113-1503-toronto/.
- Kaggle reports all five Nov 13 submissions complete: IDs 58296190, 58296201, 58296209, 58296219, and 58296226 scored 0.39, 0.38, 0.38, 0.39, and 0.38. This public score signal remains secondary to our frozen grouped scorer.
- Reviewed the newly surfaced huendao/solar-filament-segmentation-challenge-2026 notebook (SHA-256 bbac8f2e65ccb0a11a24f73d04d3cd4b6579ad5c3bbc92fd5b771951741204e). It performs inference with the public ten-model Dariush U-Net weight set and connected-component filtering; it does not train or read labels. Its quoted OOF/public scores are upstream results, not our frozen-split evidence. This model family is already represented by the unscored public-model candidate in the Nov 14 batch.
- Checked current public competition-kernel listings for the leading owners IONode Online, Kush Patel, ctogaurav, cracker1war, eunseobsim, refomi, and santmani; Kaggle returned no competition-tagged public kernels for them. HDJoJo's competition notebook remains the only public source among the checked account list and was already audited. No new reproducible top-account method was identified.

## 2026-11-13 20:23 UTC Notoverkil public ensemble artifact (later excluded)

- Reassessed the public notoverkil/solar-filament-weights release after the user authorized use of public model artifacts under their stated license terms. Its dataset card says AGPL-3.0 and reports public PQ 0.41 plus five-fold OOF PQ 0.4747. The current leaderboard independently lists the owner account at rank 140 with public score 0.41; we do not claim that the downloaded CSV is byte-identical to the owner's uploaded file.
- The Kaggle dataset manifest hash-locks 25 model checkpoints across YOLO11m, Mask R-CNN, YOLO26m and ConvNeXt-Small U-Net, five refiners, a rescorer and a date-grouped five-fold map over the 707 competition training observations. Its source card still does not identify or include the advertised inference repository. We did not download the model weights and did not reproduce their inference code.
- The dataset's published submission.csv hash matches its manifest (1b39b8d9447c1e7e703d4e848056bd9471e50a7b2616cb3027ad5f0972b8d73c). It passes the local RLE validator against all 180 supplied test images: 1,287 nonempty 2048x2048 masks on 178 images, two empty images, unique IDs, exact RLE round trips and zero within-image overlap. We read no public MAGFiLO test labels.
- Added this public output as an exploratory test-only candidate in slot 2, replacing the weaker confidence-0.35 union candidate. Its upstream OOF and public scores remain attributed claims; no EquiNexus frozen-split or holdout score is claimed because the upstream folds differ and inference code is unavailable. The five-candidate hash preflight passes with the unchanged frozen split digest.


## 2026-11-13 20:52 UTC Jnojek reproducibility release audit

- Reviewed the competition notebook [Solar Filament Segmentation 2026](https://www.kaggle.com/code/jamienojek/solar-filament-segmentation-2026) and its pinned [GitHub release](https://github.com/jnojek/solar-filament-segmentation-2026/tree/ae5aceb337ddc3aacffa915b19e081f97d8b2d07). The notebook identifies submission 58294715 but does not state its public score.
- The release describes a native-resolution grayscale U-Net trained from scratch using only competition training annotations, with test inference on the supplied H-alpha images. Its custom observation partition has 336 training, 115 development, 80 confirmation, 101 held-out Udaipur, and 75 purged observations; it is not our frozen seed-42 split.
- The source reports PQ 0.3365 on development, 0.2931 on confirmation, and 0.3385 on the Udaipur site holdout, plus a reproducible 1,293-instance test CSV. These scores use a different split and are not directly comparable to our frozen grouped OOF or holdout results, so this model was not promoted.
- The pinned GitHub tree contains no LICENSE file, and we did not establish a reuse license for its code or checkpoint. We did not run or use it in the submission batch. The 17.5 MB checkpoint remains only in the ignored source-audit archive under artifacts/kaggle/reference-refresh/; an attempted removal was rejected by the file-operation review.


## 2026-11-13 21:06 UTC Kaggle CPU consensus reproduction

- The Kaggle kernel mrinanksivakumar/equinexus-public-output-consensus-20261113 completed inference-only on CPU in 6,545.4 seconds. It produced a 1,227-row consensus submission covering all 180 supplied test images; no training or test-label evaluation was performed. The report records 2 backfilled rows and 6 clipped masks.
- The first local CSV used CRLF while Kaggle emitted LF. An ordered comparison proved all 1,227 IDs and RLE strings were identical. We set the deterministic local CSV writer to LF and rebuilt into artifacts/submissions/public-five-family-consensus-20261114-lf-verify/ in 71.68 seconds. The rebuilt, prepared, and Kaggle CSVs now match byte-for-byte at SHA-256 9269d4bbe2556b98e07de59705565039bf06f541c757c6d8404d4b6929ae6d8c (MD5 656c9327ad7706fe938f0f13bd909d0d). Candidate metadata and the five-item manifest are hash-locked to that output.
- The local validator confirms 1,227 exact 2048x2048 RLE round trips, unique nonempty IDs, zero within-image overlap, and coverage of all 180 images. The remote output and deterministic local rebuild are archived under artifacts/kaggle/runs/public-output-consensus-kaggle-20261113-retry/ and artifacts/submissions/public-five-family-consensus-20261114-lf-verify/.
- The initial monitor download encountered PowerShell native-stderr handling. A direct Kaggle CLI download recovered all outputs; subsequent byte comparison and validation succeeded.

## 2026-11-13 21:23 UTC cross-runtime public U-Net diagnostic

- Completed the fixed-rule, fold-aware evaluation locally on the RTX 5070: 142 observations, 232 annotator views, official PQ 0.447797, runtime 189.81 seconds, peak allocation 3.17 GiB. The paired bootstrap delta versus the incumbent was +0.012867 (95% CI -0.000146 to +0.026095).
- Interpret this as exploratory cross-fit evidence, not an independent holdout: upstream fold models were trained across all 707 competition training observations, so other validation observations appeared in the training data for different fold predictors. The scored image itself was excluded from its predictor; the upstream threshold was fixed; no test labels were used.
- A Kaggle T4 reproduction is scheduled for 19:05 Toronto. Outputs remain under ignored `artifacts/`.

## 2026-11-13 21:36 UTC I079 source audit

- The Apache-2.0 I079 detector/refiner model card says training used competition training images and annotations only, with no public MAGFiLO test-overlap labels. The original [handoff notebook](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) claims public score 0.70, but the current Kaggle page displays 0.34; the discrepancy is unresolved.
- The notebook is archived with its exact source hash. We did not run it as a standalone candidate or replace the YOLO26x slot. Its public output remains one documented source in the five-family consensus; no frozen-split score is claimed.

## 2026-11-13 21:50 UTC submission preflight

- All five hash-locked submission CSVs passed fresh full RLE validation: 6,139 masks total, exact 2048x2048 round trips, 180/180 test-image coverage per candidate, unique IDs, nonempty masks, and no overlaps.
- The submission dry run passed with the unchanged frozen split digest.

## 2026-11-13 21:52 UTC public detector source review

- Audited a recent consensus-trained YOLO notebook. Its required V10 and V12 checkpoint datasets are not identified in its downloaded kernel metadata, so their data provenance and license remain unknown. We did not download or use those weights or outputs.

## 2026-11-13 21:54 UTC confidence-fusion provenance

- Candidate 1's YOLO11s component is our own five-fold COCO-initialized run, trained only on competition training annotations; frozen holdout groups were excluded. It does not use the unverified Tienti checkpoints. The AGPL-3.0 model terms follow the user's prior authorization.

## 2026-11-13 22:18 UTC leaderboard and public-source review

- The full public leaderboard snapshot contains 1,515 teams. EquiNexus is rank 329 at 0.39; the top score is 0.70. The public [YOLO26x model](https://www.kaggle.com/models/hdjojo/yolo26x-seg-solar-filament) used in candidate 5 is attached from its Apache-2.0 Kaggle model card, which reports training on the competition's 707 training observations. The exact folds are unpublished, so no frozen-holdout metric is claimed for it.
- Audited recent public kernels before promotion: a 0.55 U-Net notebook embeds an opaque prediction CSV that bypasses its visible training cell, and was excluded for unverifiable prediction provenance; a 0.55 morphology notebook reproduces I079 code; ASTH-Net is associated with a 0.26 public score. Candidate 5 remains the strongest attributable high-scoring external model artifact in today's batch.
- Source snapshots, model metadata, and the leaderboard CSV are archived under ignored `artifacts/kaggle/reference-refresh/` and `artifacts/leaderboard/`. No public MAGFiLO test labels were used.


## 2026-11-13 22:34 UTC [HDJoJo YOLO26x output ablation](https://www.kaggle.com/code/hdjojo/solar-filament-segmentation) and submission gate

- Added the exact public HDJoJo kernel output as candidate 3 and retained our confidence-ordered local inference as candidate 5. Both have separate SHA-256 hashes and complete RLE reports; the current public team score of 0.55 is not attributed to either exact file.
- The local and public outputs differ substantially in mask pixels despite sharing most IDs. They are submitted as separate postprocessing variants without frozen-holdout evaluation because the model card says the source was trained on all 707 competition training observations.
- Fixed the submitter's worktree allowlist mismatch and verified its submit-state guard without uploading. Kaggle/API source-head checks remain enforced; final batch preflight is repeated immediately before submission.


## 2026-11-13 22:40 UTC training-length decision

- Epoch sweeps show that extra training can help through epoch 20-24, but it is not monotonic: EfficientViT-B2 fell from 0.142573 at epoch 24 to 0.112639 at epoch 40, and the GONG auxiliary run peaked at epoch 20. Swin-T improved through epoch 24 but remained below the incumbent.
- Use grouped OOF plateau/early-stop evidence to choose epochs; do not spend the remaining quota on long runs of models that remain well below the incumbent.

## 2026-11-13 23:00 UTC withdrawal of public-output consensus candidate

- Re-auditing the five-family vote revealed that its U-Net inputs included both Tushhaaarrrr 0.37 and 0.40 outputs. The 0.37 training source uses prohibited MAGFiLO spine metadata; the 0.40 training recipe is unavailable. The vote also included the V11 output whose checkpoint provenance was not established. The prior section's “compliant” label was incorrect: local consensus code did not train or read labels, but these upstream predictions are not eligible under the host's mask-only training rule. The 1,227-row candidate was not submitted.
- Moved the copied Kaggle consensus notebook and metadata to ignored `artifacts/kaggle/reference-audit/excluded-public-output-consensus-20261113/`. Its Kaggle CPU run and CSV remain archived for audit only and are excluded from the batch.
- Replaced batch slot 4 with the existing local I079 topology-safe detector/refiner inference on supplied test H-alpha images. The Apache-2.0 model card limits training to competition training images and annotations; the local run report confirms no training annotations or public test labels were read during inference. Its 1,299 masks pass exact 2048x2048 RLE validation, cover 169 of 180 images, and have SHA-256 `2c07cf2f06ee85b721ccb11631d6876f693ac6516277ed2d1dcd179ee7eb0794`. No frozen-split score is claimed.
- The five-candidate manifest dry run passes with all hashes and the unchanged split digest `30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e`.
- Also removed the Notoverkil precompiled submission from slot 2. The public dataset includes hash-locked model files and reports competition-training-only folds, but does not identify or include the referenced inference repository, so we could not independently establish the exact CSV's image-inference lineage. Replaced it with the public Dariush ten-model U-Net learned-filter output (SHA-256 `f264ce76fff59ac146b10dbbe196c54d4eda80fbb2a34246540d137a6499ca55`). Its source notebook, CC BY-NC-SA 4.0 weight receipt, inference log, and 1,148 exact RLE round trips are archived. The publisher-reported nested OOF PQ is 0.455236 on a different protocol; no frozen-split score is claimed.

## 2026-11-13 23:33 UTC public consensus reference and batch candidate

- The latest [prvsiyan public notebook](https://www.kaggle.com/code/prvsiyan/notebook24a4dce69e) adds an EdgeAttNet model family to its label-free prediction-agreement vote. We recorded its hash and audit notes; its full vote remains excluded because some upstream sources have disallowed or unverified training provenance.
- Built a separate 1,156-mask consensus from three distinct, hash-locked candidate families already in the batch. The scorer and frozen grouped split were not used to tune or claim a metric; this is a test-only exploratory submission, and no test labels or pixels were read.
- The new CSV passes full RLE checks and covers 180/180 supplied images. It replaces the second overlap-ordering output from the same YOLO26x model, keeping five distinct batch entries. The batch dry-run passes with the unchanged split digest.
- Revalidated all five prepared CSVs with `scripts/validate_submission_rles.py`: 6,079 total masks round-trip exactly at 2048x2048, with unique IDs and no overlap. Image prediction coverage is 179, 177, 179, 169, and 180 out of 180 respectively; reports are archived under `artifacts/automation/rle-preflight-20261114/`.
- Updated all source-path allowlists for the tracked consensus builder and confirmed Python compilation, PowerShell parsing, `git diff --check`, and batch dry-run. The shared split digest is unchanged.

## 2026-11-13 23:49 UTC CAGE U-Net reference

- Screened James Mills's public [CAGE Filament U-Net R4 model](https://www.kaggle.com/models/jamesmills1975/cage-filament-unet). It uses mask, derived hull, and interior heads; the model card reports training only on competition masks and a 0.4022 result on a random split with duplicate-image leakage. Its CC BY-NC 4.0 license is compatible with the user's research-use direction, but the companion code could not be downloaded through the Kaggle API (403), and its split is not the frozen grouped validation.
- Kept it as a future architecture reference; no weights were downloaded or added to today's candidates. A later experiment can implement the derived-hull target from the allowed masks and assess it with grouped OOF before promotion.

## 2026-11-14 00:20 UTC Kaggle run and quota-limited batch

- Recovered the completed Kaggle T4 crossfit report after the automated output transfer failed. The Dariush ten-model U-Net scored PQ 0.447800 (TP 1,099; FP 564; FN 612) on the 142 frozen-split groups and 232 annotator views. The target group was excluded from its predictor, but other validation groups appeared in other fold models' training sets; this is diagnostic crossfit, not an independent holdout. Inference took 446.97 seconds; peak allocation was 1.66 GB. Source commit and split digest match the frozen run.
- The live controller and one-shot Task Scheduler submission trigger overlapped. Five uploads consumed the daily quota: refs 58325842/58325843 are duplicate candidate 1 uploads, 58325859/58325860 duplicate candidate 2 uploads, and 58325868 is candidate 3. Candidates 4 and 5 could not be submitted. Disabled the already-fired duplicate one-shot tasks; status checks and final checkpoint remain scheduled.

## 2026-11-14 00:52 UTC Kaggle and submission status refresh

- Public leaderboard: EquiNexus rank 330/1519, score 0.39, 40 submissions. Leading six: IONode Online 0.70, Tempest 0.64, Sayandeep Dutta 0.62, Harshit Kulkarni 0.62, Tarunkush019 0.62, Kush Patel 0.62. Archive: artifacts/kaggle/leaderboard/20261114005201-utc/filament-segmentation-2026.zip, SHA-256 15e23345a6695e325b2fcaf85f9214749ecf1376f773e450c7352ff449d9f27a.
- Kaggle status output: artifacts/kaggle/leaderboard/20261114005201-utc/submissions.txt.
- Nov 14 submissions were produced from source commit 48f6487f3caa97f4df95db417131e87d929201aa.
- Kaggle ref 58325868: COMPLETE, public score 0.38; EquiNexus Nov 14 3/5; exact public HDJoJo YOLO26x kernel output; team public score 0.55 (exact file score unverified) (submitted 2026-11-14 00:20:31.423000).
- Kaggle ref 58325860: COMPLETE, public score 0.39; EquiNexus Nov 14 2/5; public ten-model Dariush U-Net + learned filter; upstream nested OOF 0.455236; no frozen-split score (submitted 2026-11-14 00:20:17.970000).
- Kaggle ref 58325859: COMPLETE, public score 0.39; EquiNexus Nov 14 2/5; public ten-model Dariush U-Net + learned filter; upstream nested OOF 0.455236; no frozen-split score (submitted 2026-11-14 00:20:17.787000).
- Kaggle ref 58325843: COMPLETE, public score 0.39; EquiNexus Nov 14 1/5; public U-Net + YOLO11s confidence 0.38; grouped OOF PQ 0.452845; holdout PQ 0.454046 (submitted 2026-11-14 00:20:04.507000).
- Kaggle ref 58325842: COMPLETE, public score 0.39; EquiNexus Nov 14 1/5; public U-Net + YOLO11s confidence 0.38; grouped OOF PQ 0.452845; holdout PQ 0.454046 (submitted 2026-11-14 00:20:04.287000).
- Prepared candidate SHA-256 (manifest entry EquiNexus Nov 14 1/5; public U-Net + YOLO11s confidence 0.38; grouped OOF PQ 0.452845; holdout PQ 0.454046): 0f58b488ee56cbab5eddac5432d6749d67b85a91afdfd8a3bbffd39eeb8eea03. This identifies a prepared file, not a successful upload; consult the Kaggle references above for actual submissions.
- Prepared candidate SHA-256 (manifest entry EquiNexus Nov 14 2/5; public ten-model Dariush U-Net + learned filter; upstream nested OOF 0.455236; no frozen-split score): f264ce76fff59ac146b10dbbe196c54d4eda80fbb2a34246540d137a6499ca55. This identifies a prepared file, not a successful upload; consult the Kaggle references above for actual submissions.
- Prepared candidate SHA-256 (manifest entry EquiNexus Nov 14 3/5; exact public HDJoJo YOLO26x kernel output; team public score 0.55 (exact file score unverified)): 035d5da13bc5898e8c331fc0e5ee21606e3e39e4419708a72529147f0a1ed9b6. This identifies a prepared file, not a successful upload; consult the Kaggle references above for actual submissions.
- Prepared candidate SHA-256 (manifest entry EquiNexus Nov 14 4/5; I079 Apache-2.0 topology-safe refiner; 1,299 test predictions; no holdout score): 2c07cf2f06ee85b721ccb11631d6876f693ac6516277ed2d1dcd179ee7eb0794. This identifies a prepared file, not a successful upload; consult the Kaggle references above for actual submissions.
- Prepared candidate SHA-256 (manifest entry EquiNexus Nov 14 5/5; audited-source three-family test consensus; no test labels): 45688ce9ad0921aaa4fd4206e4fada2e0c62f33b896a7ddc2bdd729c7c28dcf6. This identifies a prepared file, not a successful upload; consult the Kaggle references above for actual submissions.
- Kaggle T4 fold-aware crossfit completed from source commit 27ece8f76d8bde0e13ebc88e42f0649b28a7f039, split 30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e, PQ 0.44780036508507; runtime 446.969343289s, peak GPU allocation 1659777536 bytes. Outputs: artifacts/kaggle/runs/public-unet-fold-aware-20261113/dariushafshar-unet-fold-aware-holdout.
