# Chest X-ray Foundation Model Comparison (NIH ChestX-ray14)

Frozen embeddings from two chest X-ray foundation models, scored with a linear probe on
**normal vs. abnormal** classification.

## Study design

| | |
|---|---|
| **Dataset** | NIH ChestX-ray14 (public). Image archives `images_01` and `images_02` plus the `Data_Entry_2017` labels CSV; 13,163 frontal images used. |
| **Task / labels** | Binary. **Label 0 = normal**: the image's `Finding Labels` is exactly `No Finding`. **Label 1 = abnormal**: any other value, i.e. one or more of the 14 NIH findings. Labels were NLP-mined from radiology reports, so they're noisy. |
| **Split** | Patient-wise 80/20. **Test:** 2,635 images (1,503 normal, 1,132 abnormal) from 715 patients. **Train:** 10,528 images from 2,850 patients. The train patients are split again into a fit slice (8,890 images) and a 15% patient-level **validation** slice (1,638 images). No patient appears in more than one set (asserted in code). |
| **Models** | **RAD-DINO**: Microsoft's chest X-ray DINOv2 ViT-B/14 at 518 px (`microsoft/rad-dino`). **CheXFound**: a chest X-ray foundation model, ViT-L/16 with 4 registers, at 512 px (`DIAL-RPI/CheXFound`). Both are frozen. |
| **Method** | Embedding = [CLS token, mean of patch tokens]. The probe is logistic regression on standardized embeddings with `class_weight='balanced'`. C is picked by validation AUROC. Decision thresholds (Youden J, and sensitivity ≥ 0.90) are picked **on validation only** and then applied once to test. The models are compared with a **paired patient-level bootstrap** of the test-AUROC difference (2,000 resamples of patients). A small MLP head (256→64) was also tried. |

## Results (test set: 2,635 images, 715 patients)

| Model | Linear probe test AUROC |
|---|---|
| RAD-DINO | **0.760** |
| CheXFound | **0.755** |

- The paired patient-level bootstrap shows a statistical tie: the 95% CI for the difference crosses zero.
- MLP heads gave no gain over the linear probes.
- These headline AUROCs were computed on the October 27, 2026 embeddings, before the October 31 re-extraction that fixed the row-alignment issue; on the re-extracted embeddings the linear probes scored 0.758 (RAD-DINO) and 0.759 (CheXFound), still effectively a tie with CheXFound a hair ahead.

Caveat: both models were pretrained on image sets that include NIH ChestX-ray14 images (without these labels), so the evaluation images may have been seen during unlabeled pretraining.

## Data note: row alignment (resolved)

An embedding/label row-alignment issue was found and then resolved on **Oct 31, 2026** by re-extracting
the embeddings from the original images. The fresh embeddings passed a row-alignment spot check: rows 0, 4,
6000, 12662 and 13000 were re-extracted, and each had a cosine similarity of 1.0 with the saved row.
See `src/alignment_check.py`.

## Nested cross-validation (November 8, 2026)

To check that the single-split result was not an artifact of one split, the same frozen RAD-DINO and CheXFound embeddings (13,163 NIH ChestX-ray14 images, 3,565 patients; label 0 = "No Finding", 1 = any finding) were re-evaluated with nested cross-validation:

- Outer loop: StratifiedGroupKFold, 5 folds x 5 repeats, grouped by patient (25 outer folds).
- Inner loop: 3-fold patient-grouped CV on the outer training data only, tuning logistic-regression C and MLP learning rate x dropout. MLP seeds 0, 1, 2.
- Row alignment re-verified first (5 rows re-extracted from images, cosine >= 0.99999 for both models).
- Confidence intervals: Nadeau-Bengio corrected repeated-CV t interval.

| Model | Mean AUROC | Fold SD | Corrected 95% CI | Single split |
|---|---|---|---|---|
| RAD-DINO linear probe | 0.759 | 0.014 | 0.743 - 0.774 | 0.760 |
| CheXFound linear probe | 0.757 | 0.014 | 0.741 - 0.772 | 0.755 |
| RAD-DINO MLP | 0.756 | 0.014 | 0.740 - 0.771 | |
| CheXFound MLP | 0.755 | 0.015 | 0.738 - 0.771 | |

Paired difference (CheXFound minus RAD-DINO): probe -0.002 (CI -0.007 to +0.002), MLP -0.001 (CI -0.005 to +0.003). C = 0.0003 was selected in all 25 folds; the MLP selected lr 0.0001 in every fold, mostly with dropout 0.5. Runtime 54 minutes on a Colab L4, no errors. Script: `src/nested_cv.py`.

**Conclusion:** the single-split headline holds. Both models score within 0.002 of their single-split AUROC, they remain a statistical tie, and MLP heads give no gain over linear probes.

## Repository layout

```
src/data.py                download NIH images 01-02 + labels CSV; labels; patient-level splits
src/extract_embeddings.py  frozen RAD-DINO and CheXFound embedding extraction (GPU)
src/probe_8020.py          linear probe + MLP head, validation-only thresholds, test AUROC
src/paired_bootstrap.py    paired patient-level bootstrap, RAD-DINO minus CheXFound
src/alignment_check.py     row-alignment spot check (re-extract rows, cosine vs saved)
```

The code was adapted from the original Google Colab analysis notebook. It was cleaned up for this
repository and has not been re-run in this form.

## Getting the data

NIH ChestX-ray14 is public: <https://nihcc.app.box.com/v/ChestXray-NIHCC>. `python src/data.py` downloads
`images_01.tar.gz` and `images_02.tar.gz` from NIH's official Box links. It also downloads the
`Data_Entry_2017_v2020.csv` labels file from a Hugging Face mirror (`alkzar90/NIH-Chest-X-ray-dataset`) into
`./cxr_data` (override with `CXR_ROOT`). Images and embeddings are **not** committed.

```bash
pip install -r requirements.txt
cd src
python data.py                 # download + split summary
python extract_embeddings.py   # writes ./fm_emb/*.npy (override with EMB_DIR)
python probe_8020.py
python paired_bootstrap.py
```

The models are distributed under their own licenses: RAD-DINO under the MSRLA (research-only) and CheXFound under Apache-2.0.

## License

The code is under the MIT License (see `LICENSE`).
